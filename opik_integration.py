"""
Opik observability integration for Email Agent.

This module provides LLM observability without sending any PII to external services.
All data is scrubbed before being sent to Opik.
"""

import time
from contextlib import contextmanager
from dataclasses import asdict
from functools import wraps
from typing import Any, Callable, Optional

import structlog

from pii_scrubber import PIIScrubber, get_scrubber

logger = structlog.get_logger()

# Opik client - lazily initialized
_opik_client = None
_opik_enabled = False


def init_opik(api_key: str, project: str, workspace: str = "") -> bool:
    """
    Initialize Opik client.
    
    Args:
        api_key: Opik API key
        project: Project name in Opik
        workspace: Optional workspace name
        
    Returns:
        True if initialization successful, False otherwise
    """
    global _opik_client, _opik_enabled, _opik_project
    
    if not api_key:
        logger.info("Opik disabled - no API key provided")
        _opik_enabled = False
        return False
    
    try:
        import opik
        
        # Configure Opik with API key and workspace
        opik.configure(
            api_key=api_key,
            workspace=workspace if workspace else None,
            force=True,  # Overwrite existing config
            automatic_approvals=True  # Don't prompt for confirmation
        )
        _opik_client = opik
        _opik_project = project  # Store project name for use in track() calls
        _opik_enabled = True
        logger.info("Opik initialized", project=project)
        return True
        
    except ImportError:
        logger.warning("Opik not installed - observability disabled")
        _opik_enabled = False
        return False
    except Exception as e:
        logger.error("Failed to initialize Opik", error=str(e))
        _opik_enabled = False
        return False


# Store project name globally
_opik_project = "email-agent"


def is_opik_enabled() -> bool:
    """Check if Opik is enabled and initialized."""
    return _opik_enabled


class OpikTracker:
    """
    Tracks LLM operations to Opik without sending PII.
    
    All email content, sender addresses, and other PII is scrubbed
    before being sent to Opik.
    """
    
    def __init__(self):
        self.scrubber = get_scrubber()
        self._current_trace = None
        self._opik_instance = None
    
    def _get_opik(self):
        """Get or create Opik client instance."""
        if self._opik_instance is None and _opik_enabled:
            try:
                import opik
                self._opik_instance = opik.Opik(project_name=_opik_project)
            except Exception as e:
                logger.debug("Failed to create Opik instance", error=str(e))
        return self._opik_instance
    
    @contextmanager
    def trace_session(self, session_name: str = "email_processing"):
        """
        Create a trace for an email processing session.
        
        Usage:
            with tracker.trace_session("batch_processing"):
                # Process emails
                pass
        """
        if not _opik_enabled:
            yield None
            return
        
        try:
            opik_client = self._get_opik()
            if opik_client:
                trace = opik_client.trace(name=session_name)
                self._current_trace = trace
                start_time = time.time()
                
                yield trace
                
                duration_ms = (time.time() - start_time) * 1000
                trace.update(metadata={"duration_ms": duration_ms})
                trace.end()
            else:
                yield None
            
        except Exception as e:
            logger.debug("Opik trace failed", error=str(e))
            yield None
        finally:
            self._current_trace = None
    
    def track_classification(
        self,
        email: Any,
        importance: str,
        confidence: float,
        category: str,
        suggested_action: str,
        is_whitelisted: bool,
        latency_ms: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        error: Optional[str] = None
    ):
        """
        Track a single email classification (PII-free).
        
        Args:
            email: Email object (PII will be scrubbed)
            importance: Classification result
            confidence: Confidence score
            category: Email category
            suggested_action: Action taken
            is_whitelisted: Whether sender was whitelisted
            latency_ms: Time taken for classification
            tokens_input: Input tokens used
            tokens_output: Output tokens used
            error: Error message if any (should be generic, no PII)
        """
        if not _opik_enabled:
            return
        
        try:
            # Scrub email metadata
            email_meta = self.scrubber.scrub_email(email)
            
            # Scrub classification
            class_meta = self.scrubber.scrub_classification(
                email_id=email.id,
                importance=importance,
                confidence=confidence,
                category=category,
                suggested_action=suggested_action,
                is_whitelisted=is_whitelisted
            )
            
            # Build safe metadata dict
            safe_data = {
                # Email metadata (no PII)
                "sender_domain": email_meta.sender_domain,
                "sender_hash": email_meta.sender_hash,
                "subject_length": email_meta.subject_length,
                "body_length": email_meta.body_length,
                "date_hour": email_meta.date_hour,
                "date_weekday": email_meta.date_weekday,
                
                # Classification result
                "importance": class_meta.importance,
                "confidence": class_meta.confidence,
                "category": class_meta.category,
                "suggested_action": class_meta.suggested_action,
                "is_whitelisted": class_meta.is_whitelisted,
                
                # Performance
                "latency_ms": latency_ms,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                
                # Error (generic only)
                "has_error": error is not None,
                "error_type": self._categorize_error(error) if error else None
            }
            
            # Build safe input/output for Opik (no PII)
            safe_input = {
                "sender_domain": email_meta.sender_domain,
                "subject_length": email_meta.subject_length,
                "body_length": email_meta.body_length,
                "date_hour": email_meta.date_hour,
                "date_weekday": email_meta.date_weekday,
            }
            
            safe_output = {
                "importance": class_meta.importance,
                "confidence": class_meta.confidence,
                "category": class_meta.category,
                "suggested_action": class_meta.suggested_action,
                "is_whitelisted": class_meta.is_whitelisted,
            }
            
            # Log to Opik
            opik_client = self._get_opik()
            if opik_client:
                if self._current_trace:
                    self._current_trace.span(
                        name="classify_email",
                        input=safe_input,
                        output=safe_output,
                        metadata=safe_data
                    )
                else:
                    # Create a standalone trace for this classification
                    trace = opik_client.trace(
                        name="classify_email",
                        input=safe_input,
                        output=safe_output,
                        metadata=safe_data
                    )
                    trace.end()
                
        except Exception as e:
            logger.debug("Failed to track classification", error=str(e))
    
    def track_batch_classification(
        self,
        batch_size: int,
        classifications: list,
        total_latency_ms: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        error: Optional[str] = None
    ):
        """
        Track a batch classification operation (PII-free).
        
        Args:
            batch_size: Number of emails in batch
            classifications: List of classification results
            total_latency_ms: Total time for batch
            tokens_input: Total input tokens
            tokens_output: Total output tokens
            error: Error message if any
        """
        if not _opik_enabled:
            return
        
        try:
            # Aggregate stats only - no individual email data
            importance_counts = {"important": 0, "not_important": 0, "uncertain": 0}
            category_counts = {}
            total_confidence = 0.0
            
            for c in classifications:
                imp = c.importance.value if hasattr(c.importance, 'value') else str(c.importance)
                importance_counts[imp] = importance_counts.get(imp, 0) + 1
                category_counts[c.category] = category_counts.get(c.category, 0) + 1
                total_confidence += c.confidence
            
            avg_confidence = total_confidence / len(classifications) if classifications else 0.0
            
            safe_data = {
                "batch_size": batch_size,
                "emails_classified": len(classifications),
                "importance_important": importance_counts.get("important", 0),
                "importance_not_important": importance_counts.get("not_important", 0),
                "importance_uncertain": importance_counts.get("uncertain", 0),
                "avg_confidence": round(avg_confidence, 3),
                "unique_categories": len(category_counts),
                "latency_ms": total_latency_ms,
                "latency_per_email_ms": total_latency_ms / batch_size if batch_size > 0 else 0,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "has_error": error is not None,
                "error_type": self._categorize_error(error) if error else None
            }
            
            # Build safe input/output for batch
            safe_input = {
                "batch_size": batch_size,
                "email_count": len(classifications),
            }
            
            safe_output = {
                "important_count": importance_counts.get("important", 0),
                "not_important_count": importance_counts.get("not_important", 0),
                "uncertain_count": importance_counts.get("uncertain", 0),
                "avg_confidence": round(avg_confidence, 3),
                "has_error": error is not None,
            }
            
            opik_client = self._get_opik()
            if opik_client:
                if self._current_trace:
                    self._current_trace.span(
                        name="classify_batch",
                        input=safe_input,
                        output=safe_output,
                        metadata=safe_data
                    )
                else:
                    trace = opik_client.trace(
                        name="classify_batch",
                        input=safe_input,
                        output=safe_output,
                        metadata=safe_data
                    )
                    trace.end()
                
        except Exception as e:
            logger.debug("Failed to track batch classification", error=str(e))
    
    def track_session_complete(
        self,
        batch_size: int,
        emails_processed: int,
        emails_whitelisted: int,
        emails_important: int,
        emails_not_important: int,
        emails_uncertain: int,
        emails_trashed: int,
        emails_kept: int,
        errors_count: int,
        total_latency_ms: float,
        avg_confidence: float,
        dry_run: bool
    ):
        """
        Track completion of an email processing session (PII-free).
        """
        if not _opik_enabled:
            return
        
        try:
            session_meta = self.scrubber.scrub_session(
                batch_size=batch_size,
                emails_processed=emails_processed,
                emails_whitelisted=emails_whitelisted,
                emails_important=emails_important,
                emails_not_important=emails_not_important,
                emails_uncertain=emails_uncertain,
                emails_trashed=emails_trashed,
                emails_kept=emails_kept,
                errors_count=errors_count,
                total_latency_ms=total_latency_ms,
                avg_confidence=avg_confidence,
                dry_run=dry_run
            )
            
            safe_data = asdict(session_meta)
            
            # Build safe input/output for session
            safe_input = {
                "batch_size": batch_size,
                "dry_run": dry_run,
            }
            
            safe_output = {
                "emails_processed": emails_processed,
                "emails_trashed": emails_trashed,
                "emails_kept": emails_kept,
                "emails_important": emails_important,
                "emails_not_important": emails_not_important,
                "avg_confidence": round(avg_confidence, 3),
                "errors_count": errors_count,
            }
            
            opik_client = self._get_opik()
            if opik_client:
                trace = opik_client.trace(
                    name="session_complete",
                    input=safe_input,
                    output=safe_output,
                    metadata=safe_data
                )
                trace.end()
                # Flush to ensure all traces are sent
                self.flush()
            
        except Exception as e:
            logger.debug("Failed to track session completion", error=str(e))
    
    def track_llm_call(
        self,
        model: str,
        system_prompt: str,
        email_count: int,
        response_text: Optional[str],
        latency_ms: float,
        success: bool,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        error: Optional[str] = None
    ):
        """
        Track an LLM API call (PII-free).
        
        The system prompt is safe to send - it's our instructions, not user data.
        The response text is scrubbed to remove any PII the LLM might have echoed.
        
        Args:
            model: Model name (e.g., "gemini-2.5-flash-lite")
            system_prompt: The system prompt (safe - contains our instructions)
            email_count: Number of emails in the prompt
            response_text: Raw response from LLM (will be scrubbed)
            latency_ms: API call latency
            success: Whether the call succeeded
            temperature: Model temperature setting
            max_tokens: Max output tokens setting
            error: Error message if failed
        """
        if not _opik_enabled:
            return
        
        try:
            # System prompt is safe to send (it's our instructions, not PII)
            # But we truncate if very long
            safe_system_prompt = system_prompt[:2000] if len(system_prompt) > 2000 else system_prompt
            
            # Scrub any PII from response (LLM might echo names/emails)
            safe_response = None
            if response_text and success:
                safe_response = self.scrubber.scrub_text(response_text)
            
            safe_input = {
                "model": model,
                "system_prompt": safe_system_prompt,
                "email_count": email_count,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            safe_output = {
                "response": safe_response,
                "response_length": len(response_text) if response_text else 0,
                "success": success,
            }
            
            safe_metadata = {
                "latency_ms": round(latency_ms, 2),
                "has_error": error is not None,
                "error_type": self._categorize_error(error) if error else None,
            }
            
            opik_client = self._get_opik()
            if opik_client:
                trace = opik_client.trace(
                    name="llm_call",
                    input=safe_input,
                    output=safe_output,
                    metadata=safe_metadata
                )
                trace.end()
                
        except Exception as e:
            logger.debug("Failed to track LLM call", error=str(e))
    
    def track_security_test(
        self,
        test_name: str,
        vulnerability_type: str,
        passed: bool,
        classification: str,
        confidence: float
    ):
        """Track security test results."""
        if not _opik_enabled:
            return
        
        try:
            opik_client = self._get_opik()
            if opik_client:
                trace = opik_client.trace(
                    name="security_test",
                    metadata={
                        "test_name": test_name,
                        "vulnerability_type": vulnerability_type,
                        "passed": passed,
                        "classification": classification,
                        "confidence": confidence
                    }
                )
                trace.end()
        except Exception as e:
            logger.debug("Failed to track security test", error=str(e))
    
    def flush(self):
        """Flush any pending traces to Opik."""
        if self._opik_instance:
            try:
                self._opik_instance.flush()
            except Exception as e:
                logger.debug("Failed to flush Opik", error=str(e))
    
    def _categorize_error(self, error: str) -> str:
        """Categorize error without revealing sensitive details."""
        if not error:
            return None
        
        error_lower = error.lower()
        
        if "rate" in error_lower or "quota" in error_lower or "429" in error_lower:
            return "RATE_LIMIT"
        elif "auth" in error_lower or "credential" in error_lower or "401" in error_lower:
            return "AUTH_ERROR"
        elif "timeout" in error_lower:
            return "TIMEOUT"
        elif "connection" in error_lower or "network" in error_lower:
            return "NETWORK_ERROR"
        elif "json" in error_lower or "parse" in error_lower:
            return "PARSE_ERROR"
        elif "validation" in error_lower:
            return "VALIDATION_ERROR"
        else:
            return "UNKNOWN_ERROR"


# Global tracker instance
_tracker: Optional[OpikTracker] = None


def get_tracker() -> OpikTracker:
    """Get or create the global Opik tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = OpikTracker()
    return _tracker

