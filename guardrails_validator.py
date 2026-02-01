"""
Guardrails for validating email classification outputs.

Ensures LLM outputs are:
1. Valid JSON
2. Have required fields
3. Have valid field values
4. Meet confidence thresholds
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

import structlog

logger = structlog.get_logger()


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class ValidatorResult:
    """Result of a validation check."""
    def __init__(self, passed: bool, message: str = "", fixed_value: any = None):
        self.passed = passed
        self.message = message
        self.fixed_value = fixed_value


@dataclass
class GuardrailConfig:
    """Configuration for guardrails."""
    # Valid values for fields
    valid_importance: list[str] = None
    valid_categories: list[str] = None
    valid_actions: list[str] = None
    
    # Thresholds
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    
    # Behavior
    auto_fix: bool = True  # Attempt to fix invalid values
    max_retries: int = 2   # Retries on validation failure
    
    def __post_init__(self):
        if self.valid_importance is None:
            self.valid_importance = ["important", "not_important", "uncertain"]
        if self.valid_categories is None:
            self.valid_categories = [
                "work", "personal", "newsletter", "promotional", 
                "notification", "spam", "financial", "social", "other"
            ]
        if self.valid_actions is None:
            self.valid_actions = ["keep", "trash", "review"]


class OutputGuardrails:
    """
    Guardrails for validating and fixing LLM classification outputs.
    
    Usage:
        guardrails = OutputGuardrails()
        validated = guardrails.validate(llm_output)
    """
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self.validators: list[Callable] = [
            self._validate_json,
            self._validate_required_fields,
            self._validate_importance,
            self._validate_confidence,
            self._validate_category,
            self._validate_action,
            self._validate_reasoning,
        ]
    
    def validate(self, raw_output: str) -> dict:
        """
        Validate and optionally fix LLM output.
        
        Args:
            raw_output: Raw string output from LLM
            
        Returns:
            Validated and potentially fixed dict
            
        Raises:
            ValidationError: If validation fails and cannot be fixed
        """
        logger.debug("Validating LLM output", length=len(raw_output))
        
        # Track validation results
        results = []
        current_value = raw_output
        
        for validator in self.validators:
            result = validator(current_value)
            results.append((validator.__name__, result))
            
            if not result.passed:
                if self.config.auto_fix and result.fixed_value is not None:
                    logger.info("Auto-fixed validation error", 
                               validator=validator.__name__,
                               message=result.message)
                    current_value = result.fixed_value
                else:
                    logger.warning("Validation failed", 
                                  validator=validator.__name__,
                                  message=result.message)
                    raise ValidationError(f"{validator.__name__}: {result.message}")
            else:
                current_value = result.fixed_value or current_value
        
        # Ensure final output is dict
        if isinstance(current_value, str):
            current_value = json.loads(current_value)
        
        logger.debug("Validation passed", 
                    importance=current_value.get("importance"),
                    confidence=current_value.get("confidence"))
        
        return current_value
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might have markdown or other content."""
        # Try to find JSON in markdown code blocks
        if "```json" in text:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        if "```" in text:
            match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Try to find JSON object directly
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        
        return text.strip()
    
    def _validate_json(self, value: any) -> ValidatorResult:
        """Ensure output is valid JSON."""
        if isinstance(value, dict):
            return ValidatorResult(True, fixed_value=value)
        
        if not isinstance(value, str):
            return ValidatorResult(False, "Expected string or dict")
        
        # Try to extract and parse JSON
        extracted = self._extract_json(value)
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                return ValidatorResult(True, fixed_value=parsed)
            return ValidatorResult(False, "JSON is not an object")
        except json.JSONDecodeError as e:
            return ValidatorResult(False, f"Invalid JSON: {str(e)}")
    
    def _validate_required_fields(self, value: any) -> ValidatorResult:
        """Ensure all required fields are present."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except:
                return ValidatorResult(False, "Cannot parse JSON")
        
        required = ["importance", "confidence", "reasoning", "category", "suggested_action"]
        missing = [f for f in required if f not in value]
        
        if missing:
            # Try to add defaults for missing fields
            if self.config.auto_fix:
                fixed = value.copy()
                defaults = {
                    "importance": "uncertain",
                    "confidence": 0.5,
                    "reasoning": "No reasoning provided",
                    "category": "other",
                    "suggested_action": "review"
                }
                for field in missing:
                    fixed[field] = defaults[field]
                return ValidatorResult(True, f"Added defaults for: {missing}", fixed)
            
            return ValidatorResult(False, f"Missing required fields: {missing}")
        
        return ValidatorResult(True, fixed_value=value)
    
    def _validate_importance(self, value: any) -> ValidatorResult:
        """Ensure importance is valid."""
        if isinstance(value, str):
            value = json.loads(value)
        
        importance = value.get("importance", "").lower().strip()
        
        if importance in self.config.valid_importance:
            value["importance"] = importance
            return ValidatorResult(True, fixed_value=value)
        
        # Try to fix common mistakes
        if self.config.auto_fix:
            fixes = {
                "not important": "not_important",
                "notimportant": "not_important",
                "unimportant": "not_important",
                "low": "not_important",
                "high": "important",
                "medium": "uncertain",
                "unknown": "uncertain",
            }
            if importance in fixes:
                value["importance"] = fixes[importance]
                return ValidatorResult(True, f"Fixed '{importance}' to '{fixes[importance]}'", value)
            
            # Default to uncertain if unknown
            value["importance"] = "uncertain"
            return ValidatorResult(True, f"Unknown importance '{importance}', defaulting to 'uncertain'", value)
        
        return ValidatorResult(False, f"Invalid importance: {importance}")
    
    def _validate_confidence(self, value: any) -> ValidatorResult:
        """Ensure confidence is valid number between 0 and 1."""
        if isinstance(value, str):
            value = json.loads(value)
        
        confidence = value.get("confidence")
        
        try:
            conf_float = float(confidence)
            
            # Fix out-of-range values
            if conf_float < self.config.min_confidence:
                if self.config.auto_fix:
                    value["confidence"] = self.config.min_confidence
                    return ValidatorResult(True, f"Clamped confidence to {self.config.min_confidence}", value)
                return ValidatorResult(False, f"Confidence {conf_float} below minimum {self.config.min_confidence}")
            
            if conf_float > self.config.max_confidence:
                if self.config.auto_fix:
                    value["confidence"] = self.config.max_confidence
                    return ValidatorResult(True, f"Clamped confidence to {self.config.max_confidence}", value)
                return ValidatorResult(False, f"Confidence {conf_float} above maximum {self.config.max_confidence}")
            
            # Convert percentage to decimal if needed
            if conf_float > 1.0:
                if self.config.auto_fix:
                    value["confidence"] = conf_float / 100.0
                    return ValidatorResult(True, f"Converted percentage {conf_float} to {value['confidence']}", value)
            
            value["confidence"] = conf_float
            return ValidatorResult(True, fixed_value=value)
            
        except (TypeError, ValueError):
            if self.config.auto_fix:
                value["confidence"] = 0.5
                return ValidatorResult(True, "Invalid confidence, defaulting to 0.5", value)
            return ValidatorResult(False, f"Invalid confidence value: {confidence}")
    
    def _validate_category(self, value: any) -> ValidatorResult:
        """Ensure category is valid."""
        if isinstance(value, str):
            value = json.loads(value)
        
        category = value.get("category", "").lower().strip()
        
        if category in self.config.valid_categories:
            value["category"] = category
            return ValidatorResult(True, fixed_value=value)
        
        if self.config.auto_fix:
            # Map common variations
            category_map = {
                "marketing": "promotional",
                "promo": "promotional",
                "job": "notification",
                "jobs": "notification",
                "news": "newsletter",
                "alert": "notification",
                "security": "notification",
                "bill": "financial",
                "payment": "financial",
            }
            if category in category_map:
                value["category"] = category_map[category]
                return ValidatorResult(True, f"Mapped category to '{value['category']}'", value)
            
            value["category"] = "other"
            return ValidatorResult(True, f"Unknown category '{category}', defaulting to 'other'", value)
        
        return ValidatorResult(False, f"Invalid category: {category}")
    
    def _validate_action(self, value: any) -> ValidatorResult:
        """Ensure suggested_action is valid."""
        if isinstance(value, str):
            value = json.loads(value)
        
        action = value.get("suggested_action", "").lower().strip()
        
        if action in self.config.valid_actions:
            value["suggested_action"] = action
            return ValidatorResult(True, fixed_value=value)
        
        if self.config.auto_fix:
            # Map common variations
            action_map = {
                "delete": "trash",
                "remove": "trash",
                "archive": "keep",
                "save": "keep",
                "skip": "review",
                "check": "review",
            }
            if action in action_map:
                value["suggested_action"] = action_map[action]
                return ValidatorResult(True, f"Mapped action to '{value['suggested_action']}'", value)
            
            value["suggested_action"] = "review"
            return ValidatorResult(True, f"Unknown action '{action}', defaulting to 'review'", value)
        
        return ValidatorResult(False, f"Invalid action: {action}")
    
    def _validate_reasoning(self, value: any) -> ValidatorResult:
        """Ensure reasoning is present and not empty."""
        if isinstance(value, str):
            value = json.loads(value)
        
        reasoning = value.get("reasoning", "")
        
        if reasoning and len(str(reasoning).strip()) > 0:
            return ValidatorResult(True, fixed_value=value)
        
        if self.config.auto_fix:
            value["reasoning"] = "No reasoning provided"
            return ValidatorResult(True, "Added default reasoning", value)
        
        return ValidatorResult(False, "Reasoning is empty")


def create_guardrails(
    auto_fix: bool = True,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0
) -> OutputGuardrails:
    """Factory function to create guardrails with custom config."""
    config = GuardrailConfig(
        auto_fix=auto_fix,
        min_confidence=min_confidence,
        max_confidence=max_confidence
    )
    return OutputGuardrails(config)

