"""
Email classifier using LLM to determine importance.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import structlog
from google import genai
from google.genai import types

from config import Settings
from gmail_client import Email

logger = structlog.get_logger()


class ImportanceLevel(Enum):
    """Email importance classification."""
    IMPORTANT = "important"
    NOT_IMPORTANT = "not_important"
    UNCERTAIN = "uncertain"


@dataclass
class ClassificationResult:
    """Result of email classification."""
    email_id: str
    importance: ImportanceLevel
    confidence: float
    reasoning: str
    suggested_action: str
    category: str  # e.g., "newsletter", "work", "personal", "promotional"


class EmailClassifier:
    """LLM-based email classifier."""
    
    SYSTEM_PROMPT = """You are an intelligent email assistant that classifies emails by importance.

Your task is to analyze an email and determine if it's IMPORTANT or NOT IMPORTANT based on the user's criteria.

You must respond with a JSON object containing:
{{
    "importance": "important" | "not_important" | "uncertain",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation for the classification",
    "category": "work" | "personal" | "newsletter" | "promotional" | "notification" | "spam" | "financial" | "social" | "other",
    "suggested_action": "keep" | "trash" | "review"
}}

Guidelines:
- Be conservative with "not_important" - only classify as such if you're confident
- If uncertain, use "uncertain" with lower confidence
- Consider the sender, subject, and content together
- Personalized messages are usually important
- Automated/bulk messages are usually not important
- When in doubt, err on the side of "important"

IMPORTANT CRITERIA:
{important_criteria}

NOT IMPORTANT CRITERIA:
{not_important_criteria}"""

    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Configure Gemini client
        self.client = genai.Client(api_key=settings.gemini_api_key)
        
        # Build system prompt with criteria
        self.system_prompt = self.SYSTEM_PROMPT.format(
            important_criteria=settings.important_criteria,
            not_important_criteria=settings.not_important_criteria
        )
    
    def _is_whitelisted(self, email: Email) -> bool:
        """Check if sender is whitelisted (always important)."""
        sender_email = email.sender_email.lower()
        sender_domain = sender_email.split("@")[-1] if "@" in sender_email else ""
        
        # Check domain whitelist
        for domain in self.settings.get_whitelist_domains():
            if sender_domain == domain.lower():
                logger.debug("Sender whitelisted by domain", domain=domain)
                return True
        
        # Check sender whitelist
        for whitelisted in self.settings.get_whitelist_senders():
            if sender_email == whitelisted.lower():
                logger.debug("Sender whitelisted", sender=whitelisted)
                return True
        
        return False
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse LLM JSON response."""
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response", error=str(e))
            return {
                "importance": "uncertain",
                "confidence": 0.3,
                "reasoning": "Failed to parse classification response",
                "category": "other",
                "suggested_action": "review"
            }
    
    def classify(self, email: Email) -> ClassificationResult:
        """Classify a single email."""
        logger.info("Classifying email", subject=email.subject[:50], sender=email.sender_email)
        
        # Check whitelist first
        if self._is_whitelisted(email):
            return ClassificationResult(
                email_id=email.id,
                importance=ImportanceLevel.IMPORTANT,
                confidence=1.0,
                reasoning="Sender is on whitelist",
                suggested_action="keep",
                category="whitelisted"
            )
        
        # Build email summary for LLM
        email_summary = f"""
Email Details:
- From: {email.sender} <{email.sender_email}>
- To: {email.recipient}
- Subject: {email.subject}
- Date: {email.date.strftime('%Y-%m-%d %H:%M')}
- Preview: {email.snippet}

Body Preview:
{email.body_preview[:1000] if email.body_preview else "(No body content)"}
"""
        
        try:
            # Combine system prompt and user prompt for Gemini
            full_prompt = f"{self.system_prompt}\n\nPlease classify this email:\n\n{email_summary}"
            
            response = self.client.models.generate_content(
                model=self.settings.classifier_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for consistent classification
                    max_output_tokens=300
                )
            )
            
            result = self._parse_response(response.text)
            
            # Map importance string to enum
            importance_map = {
                "important": ImportanceLevel.IMPORTANT,
                "not_important": ImportanceLevel.NOT_IMPORTANT,
                "uncertain": ImportanceLevel.UNCERTAIN
            }
            importance = importance_map.get(result.get("importance", "uncertain"), ImportanceLevel.UNCERTAIN)
            
            return ClassificationResult(
                email_id=email.id,
                importance=importance,
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "No reasoning provided"),
                suggested_action=result.get("suggested_action", "review"),
                category=result.get("category", "other")
            )
            
        except Exception as e:
            logger.error("Classification failed", error=str(e))
            return ClassificationResult(
                email_id=email.id,
                importance=ImportanceLevel.UNCERTAIN,
                confidence=0.0,
                reasoning=f"Classification error: {str(e)}",
                suggested_action="review",
                category="error"
            )
    
    def classify_batch(self, emails: list[Email]) -> list[ClassificationResult]:
        """Classify multiple emails."""
        results = []
        for email in emails:
            result = self.classify(email)
            results.append(result)
        return results

