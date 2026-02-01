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
    
    def classify_batch(self, emails: list[Email], batch_size: int = 10) -> list[ClassificationResult]:
        """
        Classify multiple emails efficiently using batched LLM calls.
        Sends up to batch_size emails in a single API call.
        """
        results = []
        
        # Separate whitelisted emails (no API call needed)
        to_classify = []
        for email in emails:
            if self._is_whitelisted(email):
                results.append(ClassificationResult(
                    email_id=email.id,
                    importance=ImportanceLevel.IMPORTANT,
                    confidence=1.0,
                    reasoning="Sender is on whitelist",
                    suggested_action="keep",
                    category="whitelisted"
                ))
            else:
                to_classify.append(email)
        
        # Process non-whitelisted emails in batches
        for i in range(0, len(to_classify), batch_size):
            batch = to_classify[i:i + batch_size]
            batch_results = self._classify_batch_internal(batch)
            results.extend(batch_results)
        
        # Sort results to match original email order
        email_order = {email.id: idx for idx, email in enumerate(emails)}
        results.sort(key=lambda r: email_order.get(r.email_id, 999))
        
        return results
    
    def _classify_batch_internal(self, emails: list[Email]) -> list[ClassificationResult]:
        """Classify a batch of emails in a single API call."""
        if not emails:
            return []
        
        logger.info("Classifying batch", count=len(emails))
        
        # Build batch email summary
        email_summaries = []
        for idx, email in enumerate(emails, 1):
            summary = f"""
--- EMAIL {idx} ---
ID: {email.id}
From: {email.sender} <{email.sender_email}>
Subject: {email.subject}
Date: {email.date.strftime('%Y-%m-%d %H:%M')}
Preview: {email.snippet}
Body: {email.body_preview[:500] if email.body_preview else "(No body)"}
"""
            email_summaries.append(summary)
        
        batch_prompt = f"""{self.system_prompt}

Classify ALL {len(emails)} emails below. Return a JSON array with one object per email.

Each object must have:
- "email_id": the ID from the email
- "importance": "important" | "not_important" | "uncertain"
- "confidence": 0.0-1.0
- "reasoning": brief explanation
- "category": "work" | "personal" | "newsletter" | "promotional" | "notification" | "spam" | "financial" | "social" | "other"
- "suggested_action": "keep" | "trash" | "review"

{"".join(email_summaries)}

Respond with ONLY a JSON array:
[{{"email_id": "...", "importance": "...", ...}}, ...]"""

        try:
            response = self.client.models.generate_content(
                model=self.settings.classifier_model,
                contents=batch_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2000
                )
            )
            
            return self._parse_batch_response(response.text, emails)
            
        except Exception as e:
            error_str = str(e)
            logger.error("Batch classification failed", error=error_str)
            
            # Don't fallback on quota errors
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                logger.warning("Quota exhausted - cannot classify")
                return [
                    ClassificationResult(
                        email_id=email.id,
                        importance=ImportanceLevel.UNCERTAIN,
                        confidence=0.0,
                        reasoning="API quota exhausted",
                        suggested_action="review",
                        category="quota_error"
                    )
                    for email in emails
                ]
            
            # Fallback to individual for other errors
            logger.info("Falling back to individual classification")
            return [self.classify(email) for email in emails]
    
    def _parse_batch_response(self, response_text: str, emails: list[Email]) -> list[ClassificationResult]:
        """Parse batch LLM response."""
        try:
            text = response_text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            results_data = json.loads(text.strip())
            
            if not isinstance(results_data, list):
                raise ValueError("Expected JSON array")
            
            results_by_id = {r.get("email_id"): r for r in results_data}
            
            results = []
            importance_map = {
                "important": ImportanceLevel.IMPORTANT,
                "not_important": ImportanceLevel.NOT_IMPORTANT,
                "uncertain": ImportanceLevel.UNCERTAIN
            }
            
            for idx, email in enumerate(emails):
                result_data = results_by_id.get(email.id)
                if not result_data and idx < len(results_data):
                    result_data = results_data[idx]
                
                if result_data:
                    importance = importance_map.get(
                        result_data.get("importance", "uncertain"), 
                        ImportanceLevel.UNCERTAIN
                    )
                    results.append(ClassificationResult(
                        email_id=email.id,
                        importance=importance,
                        confidence=result_data.get("confidence", 0.5),
                        reasoning=result_data.get("reasoning", "No reasoning"),
                        suggested_action=result_data.get("suggested_action", "review"),
                        category=result_data.get("category", "other")
                    ))
                else:
                    results.append(ClassificationResult(
                        email_id=email.id,
                        importance=ImportanceLevel.UNCERTAIN,
                        confidence=0.0,
                        reasoning="Not in batch response",
                        suggested_action="review",
                        category="error"
                    ))
            
            logger.info("Batch classification complete", total=len(emails), classified=len(results))
            return results
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse batch response", error=str(e))
            return [self.classify(email) for email in emails]

