"""
Context management for email classification.

Tracks:
1. Sender history - how emails from each sender have been classified
2. User corrections - when users restore trashed emails (learning signal)
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import structlog

logger = structlog.get_logger()


@dataclass
class SenderStats:
    """Classification statistics for a sender."""
    domain: str
    important_count: int = 0
    not_important_count: int = 0
    uncertain_count: int = 0
    corrections: int = 0  # Times user restored a trashed email from this sender
    last_seen: Optional[str] = None
    
    @property
    def total(self) -> int:
        return self.important_count + self.not_important_count + self.uncertain_count
    
    @property
    def not_important_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.not_important_count / self.total
    
    @property
    def important_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.important_count / self.total
    
    @property
    def has_corrections(self) -> bool:
        return self.corrections > 0
    
    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "important_count": self.important_count,
            "not_important_count": self.not_important_count,
            "uncertain_count": self.uncertain_count,
            "corrections": self.corrections,
            "last_seen": self.last_seen
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SenderStats":
        return cls(
            domain=data.get("domain", ""),
            important_count=data.get("important_count", 0),
            not_important_count=data.get("not_important_count", 0),
            uncertain_count=data.get("uncertain_count", 0),
            corrections=data.get("corrections", 0),
            last_seen=data.get("last_seen")
        )


@dataclass 
class CorrectionRecord:
    """Record of a user correction (restored email)."""
    email_id: str
    sender_email: str
    sender_domain: str
    subject: str
    original_classification: str
    corrected_at: str
    
    def to_dict(self) -> dict:
        return {
            "email_id": self.email_id,
            "sender_email": self.sender_email,
            "sender_domain": self.sender_domain,
            "subject": self.subject,
            "original_classification": self.original_classification,
            "corrected_at": self.corrected_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CorrectionRecord":
        return cls(**data)


class ContextStore:
    """
    Persistent context store for email classification learning.
    
    Stores:
    - Sender statistics (classification history per domain)
    - User corrections (restored emails = learning signal)
    """
    
    def __init__(self, store_file: str = "context_store.json"):
        self.store_file = store_file
        self.sender_stats: dict[str, SenderStats] = {}
        self.corrections: list[CorrectionRecord] = []
        self._load()
    
    def _load(self):
        """Load context from file."""
        if os.path.exists(self.store_file):
            try:
                with open(self.store_file, "r") as f:
                    data = json.load(f)
                
                # Load sender stats
                for domain, stats in data.get("sender_stats", {}).items():
                    self.sender_stats[domain] = SenderStats.from_dict(stats)
                
                # Load corrections
                for correction in data.get("corrections", []):
                    self.corrections.append(CorrectionRecord.from_dict(correction))
                
                logger.info("Context loaded", 
                           senders=len(self.sender_stats), 
                           corrections=len(self.corrections))
            except Exception as e:
                logger.warning("Failed to load context", error=str(e))
    
    def _save(self):
        """Save context to file."""
        try:
            data = {
                "sender_stats": {
                    domain: stats.to_dict() 
                    for domain, stats in self.sender_stats.items()
                },
                "corrections": [c.to_dict() for c in self.corrections],
                "last_updated": datetime.now().isoformat()
            }
            with open(self.store_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save context", error=str(e))
    
    def _get_domain(self, email_address: str) -> str:
        """Extract domain from email address."""
        if "@" in email_address:
            return email_address.split("@")[-1].lower()
        return email_address.lower()
    
    def get_sender_stats(self, sender_email: str) -> Optional[SenderStats]:
        """Get classification history for a sender's domain."""
        domain = self._get_domain(sender_email)
        return self.sender_stats.get(domain)
    
    def record_classification(self, sender_email: str, classification: str):
        """Record a classification for learning."""
        domain = self._get_domain(sender_email)
        
        if domain not in self.sender_stats:
            self.sender_stats[domain] = SenderStats(domain=domain)
        
        stats = self.sender_stats[domain]
        stats.last_seen = datetime.now().isoformat()
        
        if classification == "important":
            stats.important_count += 1
        elif classification == "not_important":
            stats.not_important_count += 1
        else:
            stats.uncertain_count += 1
        
        self._save()
        logger.debug("Recorded classification", domain=domain, classification=classification)
    
    def record_correction(self, email_id: str, sender_email: str, subject: str, 
                         original_classification: str):
        """
        Record when user restores a trashed email (correction/learning signal).
        This indicates the classification was wrong.
        """
        domain = self._get_domain(sender_email)
        
        # Update sender stats
        if domain not in self.sender_stats:
            self.sender_stats[domain] = SenderStats(domain=domain)
        
        self.sender_stats[domain].corrections += 1
        
        # If we wrongly classified as not_important, decrement that and add to important
        if original_classification == "not_important":
            stats = self.sender_stats[domain]
            if stats.not_important_count > 0:
                stats.not_important_count -= 1
            stats.important_count += 1
        
        # Record the correction
        correction = CorrectionRecord(
            email_id=email_id,
            sender_email=sender_email,
            sender_domain=domain,
            subject=subject,
            original_classification=original_classification,
            corrected_at=datetime.now().isoformat()
        )
        self.corrections.append(correction)
        
        # Keep only last 100 corrections
        if len(self.corrections) > 100:
            self.corrections = self.corrections[-100:]
        
        self._save()
        logger.info("Recorded correction", 
                   domain=domain, 
                   subject=subject[:50],
                   total_corrections=self.sender_stats[domain].corrections)
    
    def get_context_for_sender(self, sender_email: str) -> Optional[str]:
        """
        Get context string to include in LLM prompt for a sender.
        Returns None if no history exists.
        """
        stats = self.get_sender_stats(sender_email)
        if not stats or stats.total == 0:
            return None
        
        domain = self._get_domain(sender_email)
        
        context_parts = [f"Historical data for sender domain '{domain}':"]
        context_parts.append(f"- {stats.total} emails classified previously")
        context_parts.append(f"- {stats.important_count} important ({stats.important_rate:.0%})")
        context_parts.append(f"- {stats.not_important_count} not important ({stats.not_important_rate:.0%})")
        
        if stats.has_corrections:
            context_parts.append(f"- ⚠️ User CORRECTED {stats.corrections} classification(s) from this sender")
            context_parts.append("  (User restored emails that were trashed - be more careful!)")
        
        # Find recent corrections for this domain
        recent_corrections = [
            c for c in self.corrections[-10:] 
            if c.sender_domain == domain
        ]
        if recent_corrections:
            context_parts.append("- Recent corrections:")
            for c in recent_corrections[-3:]:
                context_parts.append(f"  • '{c.subject[:40]}...' was wrongly classified as {c.original_classification}")
        
        return "\n".join(context_parts)
    
    def get_correction_patterns(self) -> list[str]:
        """Get patterns from corrections to include in system prompt."""
        if not self.corrections:
            return []
        
        # Group corrections by domain
        domain_corrections = {}
        for c in self.corrections:
            if c.sender_domain not in domain_corrections:
                domain_corrections[c.sender_domain] = []
            domain_corrections[c.sender_domain].append(c)
        
        patterns = []
        for domain, corrections in domain_corrections.items():
            if len(corrections) >= 2:
                patterns.append(
                    f"- Emails from '{domain}' have been incorrectly trashed {len(corrections)} times. "
                    f"Be more conservative with this sender."
                )
        
        return patterns
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        total_classifications = sum(s.total for s in self.sender_stats.values())
        total_corrections = len(self.corrections)
        domains_with_corrections = len([s for s in self.sender_stats.values() if s.has_corrections])
        
        return {
            "total_senders_tracked": len(self.sender_stats),
            "total_classifications": total_classifications,
            "total_corrections": total_corrections,
            "domains_with_corrections": domains_with_corrections,
            "correction_rate": total_corrections / total_classifications if total_classifications > 0 else 0
        }

