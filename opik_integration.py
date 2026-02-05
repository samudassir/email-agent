"""
Opik observability integration for Email Agent.

This module provides LLM observability without sending any PII to external services.
All data is scrubbed before being sent to Opik.
"""

import datetime
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


class LLMSpanWrapper:
    """Wrapper for LLM call spans to allow updating with response data."""
    
    def __init__(self, trace, input_data: dict, scrubber, model: str = ""):
        self.trace = trace
        self.input_data = input_data
        self.scrubber = scrubber
        self.model = model
        self.span = None
        self.start_time = time.time()
        
        # Create the span immediately so it captures the correct start time
        if self.trace:
            try:
                self.span = self.trace.span(name="llm_call")
            except Exception as e:
                logger.debug("Failed to create LLM span", error=str(e))
        
    def update_with_response(
        self,
        response_text: Optional[str],
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
        usage: Optional[dict] = None
    ):
        """
        Update the span with response data and end it.
        
        Args:
            response_text: Raw LLM response text (will be scrubbed)
            latency_ms: API call latency in ms
            success: Whether the call succeeded
            error: Error message if failed
            usage: Token usage dict with keys: prompt_tokens, completion_tokens, total_tokens
        """
        if not self.span:
            return
        
        try:
            # Scrub any PII from response (LLM might echo names/emails)
            safe_response = None
            if response_text and success:
                safe_response = self.scrubber.scrub_text(response_text)
            
            safe_output = {
                "response": safe_response if safe_response else None,
                "response_length": len(response_text) if response_text else 0,
                "success": success,
            }
            
            # Categorize error without revealing sensitive details
            error_type = None
            if error:
                error_lower = error.lower()
                if "rate" in error_lower or "quota" in error_lower or "429" in error_lower:
                    error_type = "RATE_LIMIT"
                elif "auth" in error_lower or "credential" in error_lower or "401" in error_lower:
                    error_type = "AUTH_ERROR"
                elif "timeout" in error_lower:
                    error_type = "TIMEOUT"
                elif "connection" in error_lower or "network" in error_lower:
                    error_type = "NETWORK_ERROR"
                elif "json" in error_lower or "parse" in error_lower:
                    error_type = "PARSE_ERROR"
                else:
                    error_type = "UNKNOWN_ERROR"
            
            safe_metadata = {
                "latency_ms": round(latency_ms, 2),
                "has_error": error is not None,
                "error_type": error_type,
            }
            
            # Add token counts to metadata for visibility
            if usage:
                safe_metadata["prompt_tokens"] = usage.get("prompt_tokens", 0)
                safe_metadata["completion_tokens"] = usage.get("completion_tokens", 0)
                safe_metadata["total_tokens"] = usage.get("total_tokens", 0)
            
            # End the span with all data at once, including usage tracking
            self.span.end(
                input=self.input_data,
                output=safe_output,
                metadata=safe_metadata,
                usage=usage,
                model=self.model if self.model else None,
                provider="google_ai" if self.model else None,
            )
        except Exception as e:
            logger.debug("Failed to update LLM span", error=str(e))
            # Still try to end the span even if data building fails
            try:
                self.span.end()
            except Exception:
                pass
    
    def log_feedback_score(self, name: str, value: float, reason: Optional[str] = None):
        """
        Log an evaluation feedback score on this LLM call span.
        
        Can be called after update_with_response() — feedback scores are
        sent as separate messages and don't require the span to be open.
        
        Args:
            name: Metric name (e.g. "json_parse_success")
            value: Score value (typically 0.0-1.0)
            reason: Optional explanation
        """
        if not self.span:
            return
        try:
            self.span.log_feedback_score(name=name, value=value, reason=reason)
        except Exception as e:
            logger.debug("Failed to log feedback score on LLM span", error=str(e))


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
        
        opik_client = self._get_opik()
        if not opik_client:
            yield None
            return
        
        trace = None
        try:
            trace = opik_client.trace(name=session_name)
            self._current_trace = trace
            start_time = time.time()
            
            yield trace
            
        except Exception as e:
            logger.debug("Opik trace session error", error=str(e))
            raise  # Re-raise so the caller sees the original exception
        finally:
            if trace:
                try:
                    duration_ms = (time.time() - start_time) * 1000
                    trace.update(metadata={"duration_ms": duration_ms})
                    trace.end()
                except Exception:
                    pass
            self._current_trace = None
    
    @contextmanager
    def span_llm_call(
        self,
        model: str,
        system_prompt: str,
        email_count: int,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        """
        Create a span for an LLM call within the current trace.
        
        Usage:
            with tracker.span_llm_call(model="gemini-2.0-flash", ...) as span:
                response = client.generate(...)
                span.update_with_response(response_text=response.text, ...)
        """
        if not _opik_enabled or not self._current_trace:
            yield None
            return
        
        # System prompt is safe to send (it's our instructions, not PII)
        # But we truncate if very long
        safe_system_prompt = system_prompt[:2000] if len(system_prompt) > 2000 else system_prompt
        
        safe_input = {
            "model": model,
            "system_prompt": safe_system_prompt,
            "email_count": email_count,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Create a span wrapper that can be updated
        span_wrapper = LLMSpanWrapper(
            trace=self._current_trace,
            input_data=safe_input,
            scrubber=self.scrubber,
            model=model
        )
        
        try:
            yield span_wrapper
        except Exception as e:
            # Record the error in the span, then re-raise
            try:
                span_wrapper.update_with_response(
                    response_text=None,
                    latency_ms=(time.time() - span_wrapper.start_time) * 1000,
                    success=False,
                    error=str(e)
                )
            except Exception:
                # Last resort: just end the span
                if span_wrapper.span:
                    try:
                        span_wrapper.span.end()
                    except Exception:
                        pass
            raise  # Re-raise so the caller sees the original exception
    
    def _end_span(
        self,
        name: str,
        safe_input: dict,
        safe_output: dict,
        safe_metadata: dict,
        latency_ms: float = 0,
        usage: Optional[dict] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """
        Create a span on the current trace and end it immediately with data.
        If no active trace, creates a standalone trace instead.
        
        Uses latency_ms to backdate the span's start_time so the duration
        reflects the actual operation time, not just the tracking call.
        
        Args:
            usage: Token usage dict with keys: prompt_tokens, completion_tokens, total_tokens
            model: LLM model name (e.g. "gemini-2.0-flash")
            provider: LLM provider (e.g. "google_ai")
        """
        opik_client = self._get_opik()
        if not opik_client:
            return
        
        # Compute real start_time from latency so span duration is accurate
        now = datetime.datetime.now(datetime.timezone.utc)
        start_time = now - datetime.timedelta(milliseconds=latency_ms) if latency_ms > 0 else None
        
        if self._current_trace:
            span = self._current_trace.span(name=name, start_time=start_time)
            span.end(
                input=safe_input,
                output=safe_output,
                metadata=safe_metadata,
                usage=usage,
                model=model,
                provider=provider,
            )
        else:
            trace = opik_client.trace(
                name=name,
                input=safe_input,
                output=safe_output,
                metadata=safe_metadata
            )
            trace.end()
    
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
            
            self._end_span("classify_email", safe_input, safe_output, safe_data, latency_ms=latency_ms)
                
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
            
            self._end_span("classify_batch", safe_input, safe_output, safe_data, latency_ms=total_latency_ms)
                
        except Exception as e:
            logger.info("Failed to track batch classification", error=str(e))
    
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
            
            self._end_span("session_complete", safe_input, safe_output, safe_data, latency_ms=total_latency_ms)
            # Flush to ensure all traces are sent
            self.flush()
            
        except Exception as e:
            logger.info("Failed to track session completion", error=str(e))
    
    def track_email_fetch(
        self,
        email_count: int,
        max_results: int,
        older_than: Optional[str],
        latency_ms: float,
        error: Optional[str] = None
    ):
        """
        Track email fetching operation as a span.
        
        Args:
            email_count: Number of emails fetched
            max_results: Maximum results requested
            older_than: Age filter applied (if any)
            latency_ms: Time taken to fetch emails
            error: Error message if failed
        """
        if not _opik_enabled:
            return
        
        try:
            safe_input = {
                "max_results": max_results,
                "older_than": older_than if older_than else "none",
            }
            
            safe_output = {
                "email_count": email_count,
                "success": error is None,
            }
            
            safe_metadata = {
                "latency_ms": round(latency_ms, 2),
                "has_error": error is not None,
                "error_type": self._categorize_error(error) if error else None,
            }
            
            self._end_span("fetch_emails", safe_input, safe_output, safe_metadata, latency_ms=latency_ms)
                    
        except Exception as e:
            logger.debug("Failed to track email fetch", error=str(e))
    
    def track_email_action(
        self,
        action: str,
        success: bool,
        latency_ms: float,
        dry_run: bool,
        email: Any = None,
        importance: str = "",
        confidence: float = 0.0,
        category: str = "",
        error: Optional[str] = None
    ):
        """
        Track an email action (trash, keep, label, etc.) as a span.
        
        Args:
            action: Action type (trash, keep, label, etc.)
            success: Whether the action succeeded
            latency_ms: Time taken for the action
            dry_run: Whether this is a dry run
            email: Email object (PII will be scrubbed)
            importance: Classification importance level
            confidence: Classification confidence score
            category: Email category
            error: Error message if failed
        """
        if not _opik_enabled:
            return
        
        try:
            safe_input = {
                "action": action,
                "dry_run": dry_run,
            }
            
            # Add scrubbed email context if available
            if email:
                email_meta = self.scrubber.scrub_email(email)
                safe_input["sender_domain"] = email_meta.sender_domain
                safe_input["subject_length"] = email_meta.subject_length
            
            safe_output = {
                "success": success,
                "importance": importance,
                "confidence": confidence,
                "category": category,
            }
            
            safe_metadata = {
                "latency_ms": round(latency_ms, 2),
                "has_error": error is not None,
                "error_type": self._categorize_error(error) if error else None,
            }
            
            self._end_span(f"email_action_{action}", safe_input, safe_output, safe_metadata, latency_ms=latency_ms)
                    
        except Exception as e:
            logger.debug("Failed to track email action", error=str(e))
    
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
    
    # Patterns that indicate potential prompt injection
    INJECTION_PATTERNS = [
        "ignore previous",
        "ignore all",
        "disregard instructions",
        "new instructions",
        "you are now",
        "act as",
        "pretend you",
        "system override",
        "admin mode",
        "classify as important",
        "mark as important",
        "do not trash",
        "don't trash",
        "[system]",
        "[admin]",
    ]
    
    def detect_suspicious_content(self, email: Any) -> dict:
        """
        Detect potential prompt injection or suspicious patterns in email.
        
        Returns dict with:
            - is_suspicious: bool
            - patterns_found: list of matched patterns
            - risk_level: "none", "low", "medium", "high"
        """
        subject = getattr(email, 'subject', '') or ''
        body = getattr(email, 'body_preview', '') or ''
        content = f"{subject} {body}".lower()
        
        patterns_found = []
        for pattern in self.INJECTION_PATTERNS:
            if pattern in content:
                patterns_found.append(pattern)
        
        # Determine risk level
        if len(patterns_found) >= 3:
            risk_level = "high"
        elif len(patterns_found) >= 2:
            risk_level = "medium"
        elif len(patterns_found) >= 1:
            risk_level = "low"
        else:
            risk_level = "none"
        
        return {
            "is_suspicious": len(patterns_found) > 0,
            "patterns_found": patterns_found,
            "risk_level": risk_level,
            "pattern_count": len(patterns_found)
        }
    
    def track_suspicious_activity(
        self,
        email: Any,
        classification_result: str,
        confidence: float,
        suspicious_info: dict
    ):
        """
        Track suspicious activity detected in an email.
        
        This helps identify potential prompt injection attempts.
        """
        if not _opik_enabled or not suspicious_info.get("is_suspicious"):
            return
        
        try:
            email_meta = self.scrubber.scrub_email(email)
            
            safe_input = {
                "sender_domain": email_meta.sender_domain,
                "subject_length": email_meta.subject_length,
                "body_length": email_meta.body_length,
            }
            
            safe_output = {
                "risk_level": suspicious_info["risk_level"],
                "pattern_count": suspicious_info["pattern_count"],
                "patterns_found": suspicious_info["patterns_found"],
                "classification_result": classification_result,
                "confidence": confidence,
                "possible_injection": suspicious_info["risk_level"] in ["medium", "high"],
            }
            
            opik_client = self._get_opik()
            if opik_client:
                trace = opik_client.trace(
                    name="suspicious_activity",
                    input=safe_input,
                    output=safe_output,
                    tags=["security", f"risk_{suspicious_info['risk_level']}"],
                    metadata={
                        "alert_type": "potential_prompt_injection",
                        "risk_level": suspicious_info["risk_level"],
                    }
                )
                trace.end()
                
            logger.warning(
                "Suspicious content detected",
                risk_level=suspicious_info["risk_level"],
                patterns=suspicious_info["patterns_found"],
                domain=email_meta.sender_domain
            )
                
        except Exception as e:
            logger.debug("Failed to track suspicious activity", error=str(e))
    
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
    
    # ── Online Evaluations ──────────────────────────────────────────────
    
    def evaluate_llm_output(
        self,
        llm_span: Optional["LLMSpanWrapper"],
        json_parse_ok: bool,
        expected_count: int,
        actual_count: int,
    ):
        """
        Score an LLM call span based on output quality.
        
        Logged as feedback scores visible in the Opik dashboard:
          - json_parse_success: 1.0 if LLM returned valid JSON, 0.0 otherwise
          - response_completeness: fraction of expected items returned (batch)
        
        Args:
            llm_span: The LLMSpanWrapper to score
            json_parse_ok: Whether the response parsed as valid JSON
            expected_count: Number of classifications expected (batch size)
            actual_count: Number of classifications actually returned
        """
        if not llm_span or not _opik_enabled:
            return
        
        try:
            llm_span.log_feedback_score(
                name="json_parse_success",
                value=1.0 if json_parse_ok else 0.0,
                reason="LLM response parsed as valid JSON" if json_parse_ok else "LLM response failed JSON parsing",
            )
            
            if expected_count > 0:
                completeness = min(actual_count / expected_count, 1.0)
                llm_span.log_feedback_score(
                    name="response_completeness",
                    value=round(completeness, 3),
                    reason=f"{actual_count}/{expected_count} classifications returned",
                )
        except Exception as e:
            logger.debug("Failed to evaluate LLM output", error=str(e))
    
    def evaluate_session(
        self,
        emails_processed: int,
        errors_count: int,
        avg_confidence: float,
        classifications: list,
    ):
        """
        Score the current trace with session-level quality metrics.
        
        Logged as feedback scores on the trace, visible in the Opik dashboard:
          - classification_confidence: average confidence across all classifications
          - error_rate: fraction of emails that errored (lower = better, inverted to 1 - rate)
          - low_confidence_ratio: fraction with confidence < 0.5 (inverted: 1 - ratio)
          - session_quality: composite weighted score
        
        Args:
            emails_processed: Total emails processed
            errors_count: Number of errors
            avg_confidence: Average confidence score
            classifications: List of ClassificationResult objects
        """
        if not _opik_enabled or not self._current_trace:
            return
        
        trace = self._current_trace
        
        try:
            # -- classification_confidence --
            trace.log_feedback_score(
                name="classification_confidence",
                value=round(avg_confidence, 3),
                reason=f"Average confidence across {emails_processed} emails",
            )
            
            # -- error_rate (inverted: 1.0 = no errors) --
            if emails_processed > 0:
                error_rate = errors_count / emails_processed
                trace.log_feedback_score(
                    name="error_free_rate",
                    value=round(1.0 - error_rate, 3),
                    reason=f"{errors_count} errors out of {emails_processed} emails",
                )
            
            # -- low_confidence_ratio (inverted: 1.0 = all high confidence) --
            if classifications:
                low_conf_count = sum(
                    1 for c in classifications if c.confidence < 0.5
                )
                low_conf_ratio = low_conf_count / len(classifications)
                trace.log_feedback_score(
                    name="high_confidence_rate",
                    value=round(1.0 - low_conf_ratio, 3),
                    reason=f"{low_conf_count}/{len(classifications)} classifications below 0.5 confidence",
                )
                
                # -- uncertain_ratio (inverted: 1.0 = no uncertain) --
                uncertain_count = sum(
                    1 for c in classifications
                    if (c.importance.value if hasattr(c.importance, 'value') else str(c.importance)) == "uncertain"
                )
                trace.log_feedback_score(
                    name="decisive_rate",
                    value=round(1.0 - (uncertain_count / len(classifications)), 3),
                    reason=f"{uncertain_count}/{len(classifications)} classified as uncertain",
                )
            
            # -- session_quality: composite score --
            if emails_processed > 0 and classifications:
                error_rate = errors_count / emails_processed
                low_conf_ratio = sum(1 for c in classifications if c.confidence < 0.5) / len(classifications)
                uncertain_ratio = sum(
                    1 for c in classifications
                    if (c.importance.value if hasattr(c.importance, 'value') else str(c.importance)) == "uncertain"
                ) / len(classifications)
                
                quality = (
                    0.40 * avg_confidence +           # 40% weight: confidence
                    0.25 * (1.0 - error_rate) +       # 25% weight: no errors
                    0.20 * (1.0 - low_conf_ratio) +   # 20% weight: high confidence
                    0.15 * (1.0 - uncertain_ratio)     # 15% weight: decisive
                )
                trace.log_feedback_score(
                    name="session_quality",
                    value=round(quality, 3),
                    reason=f"Composite: conf={avg_confidence:.2f}, errors={errors_count}, low_conf={low_conf_ratio:.2f}, uncertain={uncertain_ratio:.2f}",
                )
                
        except Exception as e:
            logger.debug("Failed to evaluate session", error=str(e))
    
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

