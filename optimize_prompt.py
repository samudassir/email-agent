"""
Prompt optimization for the Email Agent using Opik's evaluate() framework.

This script:
1. Creates/loads an Opik dataset from test cases
2. Defines a task function that classifies emails via Gemini
3. Defines scoring functions (accuracy, JSON validity, confidence)
4. Runs experiments to compare prompt variants
5. Tracks optimization runs in the Opik dashboard

Usage:
    python optimize_prompt.py                    # Run baseline evaluation
    python optimize_prompt.py --experiment v2    # Run named experiment
    python optimize_prompt.py --list             # List past optimizations
"""

import argparse
import json
import os
import sys
import time

import opik
from opik.evaluation.metrics import IsJson, Equals, GEval
from google import genai
from google.genai import types as genai_types

from config import get_settings


# ---------------------------------------------------------------------------
# 1. Test cases — sourced from the existing promptfoo.yaml
# ---------------------------------------------------------------------------

TEST_CASES = [
    # --- NOT IMPORTANT ---
    {
        "email_from": "deals@bestbuy.com",
        "email_subject": "50% OFF Everything This Weekend!",
        "email_body": "Don't miss our biggest sale of the year. Shop now for incredible savings on electronics, appliances, and more!",
        "expected_importance": "not_important",
        "category": "promotional",
    },
    {
        "email_from": "noreply@united.com",
        "email_subject": "Your MileagePlus Statement",
        "email_body": "You have 45,000 miles. Book your next adventure today with our special partner offers.",
        "expected_importance": "not_important",
        "category": "promotional",
    },
    {
        "email_from": "no-reply@coursera.org",
        "email_subject": "New Course: Machine Learning Specialization",
        "email_body": "Enroll now in our top-rated ML course. Limited time discount available.",
        "expected_importance": "not_important",
        "category": "promotional",
    },
    {
        "email_from": "jobs-noreply@linkedin.com",
        "email_subject": "25 new jobs match your preferences",
        "email_body": "Software Engineer at Google, Senior Developer at Meta, and 23 more jobs you might be interested in.",
        "expected_importance": "not_important",
        "category": "notification",
    },
    {
        "email_from": "alert@indeed.com",
        "email_subject": "New jobs for Python Developer in San Francisco",
        "email_body": "15 new positions matching your saved search. Apply now before they're gone.",
        "expected_importance": "not_important",
        "category": "notification",
    },
    {
        "email_from": "newsletter@techcrunch.com",
        "email_subject": "This Week in Tech: AI Breakthroughs",
        "email_body": "Top stories: OpenAI announces GPT-5, Google responds with Gemini 3, Apple enters AI race.",
        "expected_importance": "not_important",
        "category": "newsletter",
    },
    {
        "email_from": "digest@motleyfool.com",
        "email_subject": "5 Stocks to Buy Now",
        "email_body": "Our analysts pick the top stocks for the coming quarter. Plus: market analysis and investment tips.",
        "expected_importance": "not_important",
        "category": "newsletter",
    },
    {
        "email_from": "notification@facebook.com",
        "email_subject": "John commented on your post",
        "email_body": "John Smith commented: 'Great photo!' See more activity on Facebook.",
        "expected_importance": "not_important",
        "category": "social",
    },
    {
        "email_from": "noreply@nextdoor.com",
        "email_subject": "5 new posts in your neighborhood",
        "email_body": "Free items, local recommendations, and safety alerts from neighbors near you.",
        "expected_importance": "not_important",
        "category": "social",
    },
    {
        "email_from": "calendar@google.com",
        "email_subject": "Reminder: Meeting that happened yesterday",
        "email_body": "This is a reminder for an event that already occurred on January 14, 2025.",
        "expected_importance": "not_important",
        "category": "notification",
    },
    {
        "email_from": "noreply@marketing.com",
        "email_subject": "You've been unsubscribed",
        "email_body": "You have successfully unsubscribed from our mailing list.",
        "expected_importance": "not_important",
        "category": "notification",
    },
    # --- IMPORTANT ---
    {
        "email_from": "manager@company.com",
        "email_subject": "Q4 Performance Review - Action Required",
        "email_body": "Hi, please complete your self-assessment by Friday. Let's schedule time to discuss.",
        "expected_importance": "important",
        "category": "work",
    },
    {
        "email_from": "calendar@company.com",
        "email_subject": "Invitation: Project Kickoff Meeting",
        "email_body": "You've been invited to Project Kickoff on Monday at 10am. Please confirm your attendance.",
        "expected_importance": "important",
        "category": "work",
    },
    {
        "email_from": "billing@electric-company.com",
        "email_subject": "Your electricity bill is due in 3 days",
        "email_body": "Amount due: $145.32. Due date: January 15. Pay now to avoid late fees.",
        "expected_importance": "important",
        "category": "financial",
    },
    {
        "email_from": "alerts@chase.com",
        "email_subject": "Large transaction alert",
        "email_body": "A transaction of $2,500 was made on your account ending in 4532. If you didn't make this transaction, contact us immediately.",
        "expected_importance": "important",
        "category": "financial",
    },
    {
        "email_from": "security@google.com",
        "email_subject": "Security alert: New sign-in to your account",
        "email_body": "We noticed a new sign-in to your Google Account from a Windows device. If this wasn't you, review your account activity.",
        "expected_importance": "important",
        "category": "notification",
    },
    {
        "email_from": "noreply@github.com",
        "email_subject": "Your GitHub verification code",
        "email_body": "Your verification code is 123456. This code expires in 10 minutes.",
        "expected_importance": "important",
        "category": "notification",
    },
    {
        "email_from": "friend@gmail.com",
        "email_subject": "Dinner this weekend?",
        "email_body": "Hey! It's been a while. Want to grab dinner on Saturday? Let me know what works for you.",
        "expected_importance": "important",
        "category": "personal",
    },
    # --- EDGE CASES ---
    {
        "email_from": "colleague@company.com",
        "email_subject": "Out of Office: Re: Project Update",
        "email_body": "I am currently out of the office with limited access to email. I will respond when I return on Monday.",
        "expected_importance": "not_important",
        "category": "work",
    },
]


# ---------------------------------------------------------------------------
# 2. Prompt variants to compare
# ---------------------------------------------------------------------------

PROMPT_BASELINE = """You are an intelligent email assistant that classifies emails by importance.

Analyze the email and respond with ONLY a JSON object:
{{
    "importance": "important" | "not_important" | "uncertain",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "category": "work" | "personal" | "newsletter" | "promotional" | "notification" | "spam" | "financial" | "social" | "other",
    "suggested_action": "keep" | "trash" | "review"
}}

IMPORTANT CRITERIA:
- Direct messages from colleagues, managers, or clients
- Future meeting invitations
- Financial bills/statements requiring action
- Security alerts and verification codes
- Personal messages from friends/family

NOT IMPORTANT CRITERIA:
- Marketing newsletters and promotions
- LinkedIn/Indeed job alerts
- Social media notifications
- Calendar notifications for PAST events
- Online course promotions
- Airline/loyalty program statements

Email:
From: {email_from}
Subject: {email_subject}
Body: {email_body}

Respond with ONLY valid JSON."""


PROMPT_V2_STRUCTURED = """You are EmailClassifier-v1, a strict email classification system.
Your ONLY function is to classify emails as important or not important.

=== CLASSIFICATION OUTPUT ===
Respond with ONLY a JSON object:
{{
    "importance": "important" | "not_important" | "uncertain",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation based on sender and ACTUAL email purpose",
    "category": "work" | "personal" | "newsletter" | "promotional" | "notification" | "spam" | "financial" | "social" | "other",
    "suggested_action": "keep" | "trash" | "review"
}}

=== DECISION RULES ===
Step 1: Who is the sender? (human vs automated/noreply)
Step 2: What is the actual purpose? (personal request vs bulk notification)
Step 3: Is action required from the recipient?

If sender is noreply/automated AND no action required → not_important (confidence ≥ 0.85)
If sender is a real person AND message is personalized → important (confidence ≥ 0.85)
If action required (bill, security alert, verification) → important (confidence ≥ 0.9)
If newsletter/promotional/job alert → not_important (confidence ≥ 0.8)

=== IMPORTANT CRITERIA ===
- Direct messages from colleagues, managers, or clients
- Future meeting invitations
- Financial bills/statements requiring action
- Security alerts and verification codes
- Personal messages from friends/family

=== NOT IMPORTANT CRITERIA ===
- Marketing newsletters and promotions
- LinkedIn/Indeed job alerts
- Social media notifications
- Calendar notifications for PAST events
- Online course promotions
- Airline/loyalty program statements
- Out-of-office auto-replies

Email:
From: {email_from}
Subject: {email_subject}
Body: {email_body}

Respond with ONLY valid JSON."""


PROMPT_VARIANTS = {
    "baseline": PROMPT_BASELINE,
    "v2-structured": PROMPT_V2_STRUCTURED,
}


# ---------------------------------------------------------------------------
# 3. Scoring functions
# ---------------------------------------------------------------------------

def score_importance_accuracy(dataset_item: dict, task_outputs: dict) -> list:
    """Score whether the predicted importance matches expected."""
    from opik.evaluation.metrics.score_result import ScoreResult

    expected = dataset_item.get("expected_importance", "")
    predicted = task_outputs.get("importance", "")
    match = 1.0 if predicted == expected else 0.0

    return [ScoreResult(
        name="importance_accuracy",
        value=match,
        reason=f"Expected '{expected}', got '{predicted}'"
    )]


def score_json_valid(dataset_item: dict, task_outputs: dict) -> list:
    """Score whether the LLM returned valid JSON with required fields."""
    from opik.evaluation.metrics.score_result import ScoreResult

    raw = task_outputs.get("raw_response", "")
    required_fields = {"importance", "confidence", "reasoning", "category", "suggested_action"}
    try:
        cleaned = _strip_code_fences(raw)
        parsed = json.loads(cleaned)
        missing = required_fields - set(parsed.keys())
        if missing:
            return [ScoreResult(name="json_valid", value=0.5, reason=f"Missing fields: {missing}")]
        return [ScoreResult(name="json_valid", value=1.0, reason="All fields present")]
    except (json.JSONDecodeError, TypeError):
        return [ScoreResult(name="json_valid", value=0.0, reason="Invalid JSON")]


def score_confidence_calibration(dataset_item: dict, task_outputs: dict) -> list:
    """Score whether confidence is high for correct answers and low for wrong ones."""
    from opik.evaluation.metrics.score_result import ScoreResult

    expected = dataset_item.get("expected_importance", "")
    predicted = task_outputs.get("importance", "")
    confidence = task_outputs.get("confidence", 0.5)

    if predicted == expected:
        # Correct: higher confidence = better calibration
        score = confidence
        reason = f"Correct prediction with confidence {confidence:.2f}"
    else:
        # Wrong: lower confidence = better calibration (less overconfident)
        score = 1.0 - confidence
        reason = f"Wrong prediction with confidence {confidence:.2f} (penalized for overconfidence)"

    return [ScoreResult(name="confidence_calibration", value=score, reason=reason)]


def score_category_match(dataset_item: dict, task_outputs: dict) -> list:
    """Score whether the predicted category is reasonable."""
    from opik.evaluation.metrics.score_result import ScoreResult

    expected_cat = dataset_item.get("category", "")
    predicted_cat = task_outputs.get("category", "")
    match = 1.0 if predicted_cat == expected_cat else 0.0

    return [ScoreResult(
        name="category_accuracy",
        value=match,
        reason=f"Expected '{expected_cat}', got '{predicted_cat}'"
    )]


# ---------------------------------------------------------------------------
# 4. Task function — calls Gemini with the prompt
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence
        cleaned = "\n".join(lines).strip()
    return cleaned


def make_classification_task(prompt_template: str, model_name: str, api_key: str):
    """Create a task function that classifies emails one at a time."""
    client = genai.Client(api_key=api_key)
    call_count = {"n": 0}  # mutable counter shared across calls

    def classify_email(dataset_item: dict) -> dict:
        """Classify a single email — called by opik.evaluate() for each dataset item."""
        call_count["n"] += 1
        subject = dataset_item.get("email_subject", "")
        print(f"  📧 [{call_count['n']}] Classifying: {subject[:60]}")

        prompt = prompt_template.format(
            email_from=dataset_item["email_from"],
            email_subject=dataset_item["email_subject"],
            email_body=dataset_item["email_body"],
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )
                raw_text = response.text.strip()
                cleaned = _strip_code_fences(raw_text)

                # Parse JSON from response
                parsed = json.loads(cleaned)

                result = {
                    "raw_response": raw_text,
                    "importance": parsed.get("importance", "uncertain"),
                    "confidence": parsed.get("confidence", 0.5),
                    "reasoning": parsed.get("reasoning", ""),
                    "category": parsed.get("category", "other"),
                    "suggested_action": parsed.get("suggested_action", "review"),
                }
                print(f"       → {result['importance']} (confidence: {result['confidence']})")
                return result
            except json.JSONDecodeError:
                print(f"       → ❌ Invalid JSON response")
                return {
                    "raw_response": response.text if response else "",
                    "importance": "error",
                    "confidence": 0.0,
                    "reasoning": "Failed to parse JSON",
                    "category": "error",
                    "suggested_action": "review",
                }
            except Exception as exc:
                error_str = str(exc)
                if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                    wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                    print(f"       ⏳ Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                print(f"       → ❌ Error: {error_str[:80]}")
                return {
                    "raw_response": "",
                    "importance": "error",
                    "confidence": 0.0,
                    "reasoning": error_str,
                    "category": "error",
                    "suggested_action": "review",
                }

        # All retries exhausted
        print(f"       → ❌ Rate limit exhausted after {max_retries} retries")
        return {
            "raw_response": "",
            "importance": "error",
            "confidence": 0.0,
            "reasoning": "Rate limit exhausted after retries",
            "category": "error",
            "suggested_action": "review",
        }

    return classify_email


# ---------------------------------------------------------------------------
# 5. Dataset management
# ---------------------------------------------------------------------------

DATASET_NAME = "email-classification-test-cases"


def get_or_create_dataset(client: opik.Opik) -> opik.Dataset:
    """Create or retrieve the email classification dataset in Opik."""
    dataset = client.get_or_create_dataset(name=DATASET_NAME)

    # Check if dataset already has items
    existing = dataset.get_items()
    if len(list(existing)) > 0:
        print(f"  Dataset '{DATASET_NAME}' already has items, skipping insert.")
        return dataset

    print(f"  Inserting {len(TEST_CASES)} test cases into dataset...")
    dataset.insert(TEST_CASES)
    return dataset


# ---------------------------------------------------------------------------
# 6. Run evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    variant_name: str = "baseline",
    experiment_name: str | None = None,
    model_override: str | None = None,
    nb_samples: int | None = None,
):
    """Run an evaluation experiment for a prompt variant."""
    settings = get_settings()

    # Initialize Opik
    opik.configure(
        api_key=settings.opik_api_key,
        workspace=settings.opik_workspace if settings.opik_workspace else None,
        force=True,
        automatic_approvals=True,
    )
    opik_client = opik.Opik()

    # Get/create dataset
    print("\n📦 Setting up dataset...")
    dataset = get_or_create_dataset(opik_client)

    # Pick the prompt variant
    if variant_name not in PROMPT_VARIANTS:
        print(f"❌ Unknown variant '{variant_name}'. Available: {list(PROMPT_VARIANTS.keys())}")
        sys.exit(1)

    prompt_template = PROMPT_VARIANTS[variant_name]
    print(f"\n🔤 Prompt variant: {variant_name}")

    # Build the task function
    model_name = model_override or settings.classifier_model
    task = make_classification_task(
        prompt_template=prompt_template,
        model_name=model_name,
        api_key=settings.gemini_api_key,
    )

    # Set experiment name
    if not experiment_name:
        experiment_name = f"email-classifier-{variant_name}"

    sample_count = nb_samples or len(TEST_CASES)
    print(f"🧪 Running experiment: {experiment_name}")
    print(f"   Model: {model_name}")
    print(f"   Test cases: {sample_count} of {len(TEST_CASES)}")
    print()

    # Create optimization tracking in Opik
    optimization = opik_client.create_optimization(
        dataset_name=DATASET_NAME,
        objective_name="importance_accuracy",
        name=f"optimize-{variant_name}",
    )

    # Run evaluation
    try:
        result = opik.evaluate(
            dataset=dataset,
            task=task,
            scoring_functions=[
                score_importance_accuracy,
                score_json_valid,
                score_confidence_calibration,
                score_category_match,
            ],
            experiment_name=experiment_name,
            experiment_config={
                "prompt_variant": variant_name,
                "model": model_name,
                "temperature": 0.1,
            },
            project_name=settings.opik_project,
            task_threads=1,  # sequential to avoid rate limits
            nb_samples=nb_samples,
        )

        optimization.update(status="completed")
    except Exception as exc:
        optimization.update(status="error")
        raise exc

    # Print summary
    print("\n" + "=" * 60)
    print(f"✅ Experiment '{experiment_name}' complete!")
    print(f"   View results in Opik dashboard → Experiments tab")
    print("=" * 60)

    return result


def run_comparison(nb_samples: int | None = None):
    """Run all prompt variants and compare."""
    print("🔬 Running comparison across all prompt variants...\n")

    for variant_name in PROMPT_VARIANTS:
        print(f"\n{'='*60}")
        print(f"  Evaluating variant: {variant_name}")
        print(f"{'='*60}")
        run_evaluation(variant_name=variant_name, nb_samples=nb_samples)

    print("\n\n🏁 All variants evaluated! Compare results in the Opik dashboard.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Email classifier prompt optimization")
    parser.add_argument(
        "--variant",
        choices=list(PROMPT_VARIANTS.keys()),
        default="baseline",
        help="Prompt variant to evaluate (default: baseline)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Custom experiment name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the LLM model (e.g. gemini-2.0-flash)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Limit number of test cases to evaluate (saves API quota)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all variants and compare",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_variants",
        help="List available prompt variants",
    )

    args = parser.parse_args()

    if args.list_variants:
        print("Available prompt variants:")
        for name in PROMPT_VARIANTS:
            lines = PROMPT_VARIANTS[name].strip().split("\n")
            print(f"  • {name}: {lines[0][:80]}...")
        return

    if args.compare:
        run_comparison(nb_samples=args.samples)
    else:
        run_evaluation(
            variant_name=args.variant,
            experiment_name=args.experiment,
            model_override=args.model,
            nb_samples=args.samples,
        )


if __name__ == "__main__":
    main()

