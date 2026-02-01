"""
PII Scrubber - Removes personally identifiable information before sending to observability tools.

This module ensures no sensitive data (emails, names, phone numbers, etc.) is sent to 
external services like Opik while still allowing useful metrics and patterns to be tracked.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class ScrubbedEmailMetadata:
    """Safe metadata extracted from an email - contains no PII."""
    sender_domain: str  # Domain only, e.g., "company.com"
    sender_hash: str  # Hashed sender for unique tracking without revealing identity
    has_recipient: bool  # Whether recipient exists (not the actual value)
    subject_length: int  # Length of subject (not content)
    body_length: int  # Length of body (not content)
    date_hour: int  # Hour of day (0-23)
    date_weekday: int  # Day of week (0-6)
    label_count: int  # Number of labels
    is_unread: bool


@dataclass
class ScrubbedClassificationMetadata:
    """Safe classification data - contains no PII."""
    email_id_hash: str  # Hashed email ID
    importance: str  # important/not_important/uncertain
    confidence: float
    category: str
    suggested_action: str
    is_whitelisted: bool
    # Reasoning is NOT included - may contain PII echoed by LLM


@dataclass
class ScrubbedSessionMetadata:
    """Safe session/batch data - aggregate metrics only."""
    batch_size: int
    emails_processed: int
    emails_whitelisted: int
    emails_important: int
    emails_not_important: int
    emails_uncertain: int
    emails_trashed: int
    emails_kept: int
    errors_count: int
    total_latency_ms: float
    avg_confidence: float
    dry_run: bool


class PIIScrubber:
    """Scrubs PII from data before sending to external observability services."""
    
    # Patterns that indicate PII
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    PHONE_PATTERN = re.compile(r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
    SSN_PATTERN = re.compile(r'\d{3}-\d{2}-\d{4}')
    
    # Common name patterns (simplified)
    NAME_PREFIXES = ['mr', 'mrs', 'ms', 'dr', 'prof']
    
    def __init__(self, salt: str = "email-agent-v1"):
        """
        Initialize scrubber with a salt for hashing.
        
        Args:
            salt: Salt for hashing - use consistent salt to track unique senders
                  across sessions without revealing identity
        """
        self.salt = salt
    
    def _hash_value(self, value: str) -> str:
        """Create a one-way hash of a value for tracking without revealing identity."""
        salted = f"{self.salt}:{value.lower()}"
        return hashlib.sha256(salted.encode()).hexdigest()[:16]
    
    def _extract_domain(self, email: str) -> str:
        """Extract domain from email address."""
        if "@" in email:
            return email.split("@")[-1].lower()
        return "unknown"
    
    def scrub_email(self, email: Any) -> ScrubbedEmailMetadata:
        """
        Extract safe metadata from an email object.
        
        Args:
            email: Email object with sender_email, recipient, subject, body_preview, etc.
            
        Returns:
            ScrubbedEmailMetadata with no PII
        """
        return ScrubbedEmailMetadata(
            sender_domain=self._extract_domain(getattr(email, 'sender_email', '')),
            sender_hash=self._hash_value(getattr(email, 'sender_email', '')),
            has_recipient=bool(getattr(email, 'recipient', '')),
            subject_length=len(getattr(email, 'subject', '') or ''),
            body_length=len(getattr(email, 'body_preview', '') or ''),
            date_hour=getattr(email, 'date', None).hour if hasattr(email, 'date') and email.date else 0,
            date_weekday=getattr(email, 'date', None).weekday() if hasattr(email, 'date') and email.date else 0,
            label_count=len(getattr(email, 'labels', []) or []),
            is_unread=getattr(email, 'is_unread', True)
        )
    
    def scrub_classification(
        self, 
        email_id: str,
        importance: str,
        confidence: float,
        category: str,
        suggested_action: str,
        is_whitelisted: bool = False
    ) -> ScrubbedClassificationMetadata:
        """
        Create safe classification metadata.
        
        Args:
            email_id: Original email ID (will be hashed)
            importance: Classification result
            confidence: Confidence score
            category: Email category
            suggested_action: Suggested action
            is_whitelisted: Whether sender was whitelisted
            
        Returns:
            ScrubbedClassificationMetadata with no PII
        """
        return ScrubbedClassificationMetadata(
            email_id_hash=self._hash_value(email_id),
            importance=importance,
            confidence=confidence,
            category=category,
            suggested_action=suggested_action,
            is_whitelisted=is_whitelisted
        )
    
    def scrub_session(
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
    ) -> ScrubbedSessionMetadata:
        """
        Create safe session metadata from aggregate stats.
        
        Returns:
            ScrubbedSessionMetadata with aggregate metrics only
        """
        return ScrubbedSessionMetadata(
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
    
    def scrub_text(self, text: str) -> str:
        """
        Remove PII from arbitrary text (use sparingly - prefer not sending text at all).
        
        Args:
            text: Text that may contain PII
            
        Returns:
            Text with PII replaced by placeholders
        """
        if not text:
            return ""
        
        result = text
        
        # Replace emails
        result = self.EMAIL_PATTERN.sub('[EMAIL]', result)
        
        # Replace phone numbers
        result = self.PHONE_PATTERN.sub('[PHONE]', result)
        
        # Replace SSNs
        result = self.SSN_PATTERN.sub('[SSN]', result)
        
        return result
    
    def to_safe_dict(self, metadata: Any) -> dict:
        """Convert a scrubbed metadata object to a dictionary for logging."""
        if hasattr(metadata, '__dataclass_fields__'):
            return {k: getattr(metadata, k) for k in metadata.__dataclass_fields__}
        return {}


# Global scrubber instance
_scrubber: Optional[PIIScrubber] = None


def get_scrubber(salt: str = "email-agent-v1") -> PIIScrubber:
    """Get or create the global PII scrubber instance."""
    global _scrubber
    if _scrubber is None:
        _scrubber = PIIScrubber(salt=salt)
    return _scrubber

