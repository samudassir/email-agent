#!/usr/bin/env python
"""
Run all evaluations for the email classifier.

Usage:
    python run_evals.py              # Run all evaluations
    python run_evals.py --security   # Run security tests only
    python run_evals.py --promptfoo  # Show promptfoo instructions
    python run_evals.py --guardrails # Test guardrails
"""

import argparse
import sys
import json
from typing import Optional


def run_security_tests():
    """Run Garak-style security tests."""
    print("\n🔐 Running Security Tests (Garak-style)")
    print("="*50)
    
    from evals.security_tests import SecurityTestSuite
    
    suite = SecurityTestSuite()
    results = suite.run_all_tests()
    
    return results["failed"] == 0


def test_guardrails():
    """Test the guardrails validator."""
    print("\n🛡️ Testing Guardrails Validator")
    print("="*50)
    
    from guardrails_validator import OutputGuardrails, ValidationError
    
    guardrails = OutputGuardrails()
    
    test_cases = [
        {
            "name": "Valid JSON",
            "input": '{"importance": "important", "confidence": 0.9, "reasoning": "Work email", "category": "work", "suggested_action": "keep"}',
            "should_pass": True
        },
        {
            "name": "JSON in markdown",
            "input": '```json\n{"importance": "not_important", "confidence": 0.8, "reasoning": "Spam", "category": "promotional", "suggested_action": "trash"}\n```',
            "should_pass": True
        },
        {
            "name": "Invalid importance (auto-fix)",
            "input": '{"importance": "not important", "confidence": 0.9, "reasoning": "Test", "category": "other", "suggested_action": "review"}',
            "should_pass": True  # Should auto-fix
        },
        {
            "name": "Missing fields (auto-fix)",
            "input": '{"importance": "important"}',
            "should_pass": True  # Should add defaults
        },
        {
            "name": "Confidence as percentage (auto-fix)",
            "input": '{"importance": "uncertain", "confidence": 85, "reasoning": "Test", "category": "other", "suggested_action": "review"}',
            "should_pass": True  # Should convert 85 to 0.85
        },
        {
            "name": "Invalid JSON",
            "input": 'not json at all',
            "should_pass": False
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        print(f"\n📋 Test: {test['name']}")
        try:
            result = guardrails.validate(test["input"])
            if test["should_pass"]:
                print(f"   ✅ Passed - Validated successfully")
                print(f"      importance: {result.get('importance')}")
                print(f"      confidence: {result.get('confidence')}")
                passed += 1
            else:
                print(f"   ❌ Failed - Should have raised ValidationError")
                failed += 1
        except ValidationError as e:
            if not test["should_pass"]:
                print(f"   ✅ Passed - Correctly rejected: {str(e)[:50]}")
                passed += 1
            else:
                print(f"   ❌ Failed - Unexpected error: {str(e)[:50]}")
                failed += 1
        except Exception as e:
            print(f"   ❌ Failed - Unexpected exception: {str(e)[:50]}")
            failed += 1
    
    print(f"\n📊 Guardrails Test Results: {passed}/{passed+failed} passed")
    return failed == 0


def show_promptfoo_instructions():
    """Show instructions for running promptfoo."""
    print("\n📊 Promptfoo - Classification Accuracy Testing")
    print("="*50)
    print("""
Promptfoo tests the accuracy of your classification prompt against
a set of test cases with expected outcomes.

Setup:
    npm install -g promptfoo
    
    # Or use npx (no install needed)
    
Run Tests:
    cd evals
    npx promptfoo eval
    
View Results:
    npx promptfoo view
    
Configuration:
    evals/promptfoo.yaml      - Test configuration
    evals/prompt_template.txt - Prompt template
    
Test Categories:
    - Promotional emails (should be not_important)
    - Job alerts (should be not_important)
    - Newsletters (should be not_important)
    - Social notifications (should be not_important)
    - Work emails (should be important)
    - Financial alerts (should be important)
    - Security alerts (should be important)
    - Personal messages (should be important)
    - Edge cases (various)
    
Note: Requires GOOGLE_API_KEY environment variable for Gemini
""")


def main():
    parser = argparse.ArgumentParser(description="Run email classifier evaluations")
    parser.add_argument("--security", action="store_true", help="Run security tests only")
    parser.add_argument("--guardrails", action="store_true", help="Test guardrails validator")
    parser.add_argument("--promptfoo", action="store_true", help="Show promptfoo instructions")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any([args.security, args.guardrails, args.promptfoo, args.all]):
        args.all = True
    
    all_passed = True
    
    if args.promptfoo or args.all:
        show_promptfoo_instructions()
    
    if args.guardrails or args.all:
        if not test_guardrails():
            all_passed = False
    
    if args.security or args.all:
        if not run_security_tests():
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All evaluations passed!")
    else:
        print("❌ Some evaluations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

