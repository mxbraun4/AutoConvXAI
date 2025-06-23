#!/usr/bin/env python3
"""Test Query Patterns for Intent Recognition and Action Routing

This test script validates the natural language understanding capabilities
of the multi-agent system, particularly focusing on:

1. Context bleeding issues (ensuring overall queries reset context)
2. Intent classification accuracy for feature importance queries
3. Proper action routing for various query types

Test Categories:
    - Context Management: Tests for proper context reset behavior
    - Intent Classification: Tests for accurate intent recognition
    - Action Routing: Tests for correct action generation
    - Edge Cases: Tests for handling ambiguous or complex queries

Author: PhD Thesis Research - Intent Recognition System Validation
"""

import json
from typing import Dict, List, Tuple, Any

# Test data: (query, expected_intent, expected_action, description)
CONTEXT_BLEEDING_TESTS = [
    # These tests verify that "overall" keywords properly reset context
    ("how accurate is the model overall?", "performance", "score overall accuracy", 
     "Overall accuracy should reset context from previous filtering"),
    ("what is the total accuracy?", "performance", "score total accuracy", 
     "Total accuracy query should use full dataset"),
    ("how accurate is the model globally?", "performance", "score global accuracy", 
     "Global accuracy should ignore filtered context"),
    ("what is the entire model performance?", "performance", "score accuracy", 
     "Entire model performance should reset to full dataset"),
]

FEATURE_IMPORTANCE_TESTS = [
    # These tests verify proper intent classification for feature importance
    ("what are the most relevant features", "importance", "important all", 
     "Most relevant features should map to importance intent"),
    ("what features are most important", "importance", "important all", 
     "Important features query should map to importance intent"),
    ("which features matter most", "importance", "important all", 
     "Features that matter should map to importance intent"),
    ("show me the key features", "importance", "important all", 
     "Key features should map to importance intent"),
    ("what are the significant features", "importance", "important all", 
     "Significant features should map to importance intent"),
    ("feature ranking", "importance", "important all", 
     "Feature ranking should map to importance intent"),
    ("top 3 most important features", "importance", "important topk 3", 
     "Top N features should map to topk importance"),
]

ACTION_ROUTING_TESTS = [
    # These tests verify correct action generation
    ("predict for instance with id 2", "predict", "filter id 2 predict", 
     "Instance-specific prediction should filter first"),
    ("explain patient 5", "explain", "filter id 5 explain", 
     "Patient explanation should filter by ID first"),
    ("what would you predict for age > 50", "predict", "filter age greater 50 predict", 
     "Conditional prediction should use appropriate filtering"),
    ("show me accuracy", "performance", "score accuracy", 
     "Simple accuracy query should use score action"),
]

EDGE_CASE_TESTS = [
    # These tests handle complex or ambiguous queries
    ("what is your prediction for person 5", "predict", "filter id 5 predict", 
     "Natural language ID reference should be handled correctly"),
    ("why is prediction for person with id 200 set to 1", "explain", "filter id 200 explain", 
     "Complex explanation query should extract ID correctly"),
    ("predict for age > 50 and pregnant = no", "predict", "filter age greater 50 filter pregnancies equal 0 predict", 
     "Multiple conditions should generate multiple filter commands"),
]

ALL_TEST_CASES = [
    ("Context Bleeding Tests", CONTEXT_BLEEDING_TESTS),
    ("Feature Importance Tests", FEATURE_IMPORTANCE_TESTS), 
    ("Action Routing Tests", ACTION_ROUTING_TESTS),
    ("Edge Case Tests", EDGE_CASE_TESTS),
]


def validate_intent_classification(query: str, expected_intent: str, 
                                 extracted_intent: str) -> Tuple[bool, str]:
    """
    Validate Intent Classification Results
    
    Compares expected intent with extracted intent and provides detailed
    feedback on classification accuracy.
    
    Args:
        query: Original user query
        expected_intent: Expected intent classification
        extracted_intent: Actually extracted intent
        
    Returns:
        Tuple of (is_correct, feedback_message)
    """
    is_correct = expected_intent == extracted_intent
    
    if is_correct:
        feedback = f"✅ Correct intent classification: '{expected_intent}'"
    else:
        feedback = f"❌ Intent mismatch: expected '{expected_intent}', got '{extracted_intent}'"
        
    return is_correct, feedback


def validate_action_generation(query: str, expected_action: str, 
                             generated_action: str) -> Tuple[bool, str]:
    """
    Validate Action Generation Results
    
    Compares expected action with generated action, allowing for some
    flexibility in action formatting while maintaining semantic equivalence.
    
    Args:
        query: Original user query
        expected_action: Expected action command
        generated_action: Actually generated action
        
    Returns:
        Tuple of (is_correct, feedback_message)
    """
    # Normalize actions for comparison (remove extra whitespace, standardize format)
    def normalize_action(action: str) -> str:
        return ' '.join(action.lower().split())
    
    expected_norm = normalize_action(expected_action)
    generated_norm = normalize_action(generated_action)
    
    is_correct = expected_norm == generated_norm
    
    if is_correct:
        feedback = f"✅ Correct action generation: '{expected_action}'"
    else:
        feedback = f"❌ Action mismatch: expected '{expected_action}', got '{generated_action}'"
        
    return is_correct, feedback


def run_single_test(query: str, expected_intent: str, expected_action: str, 
                   description: str, test_function=None) -> Dict[str, Any]:
    """
    Run a Single Test Case
    
    Executes a single test case through the multi-agent system and validates
    the results against expected outcomes.
    
    Args:
        query: User query to test
        expected_intent: Expected intent classification
        expected_action: Expected action generation
        description: Test case description
        test_function: Optional function to actually run the test (for integration)
        
    Returns:
        Test result dictionary with validation details
    """
    print(f"\n🔍 Testing: {description}")
    print(f"Query: '{query}'")
    
    # For now, this is a structure for the test - in real implementation,
    # you would call your AutoGen decoder here
    if test_function:
        try:
            result = test_function(query)
            extracted_intent = result.get('intent', 'unknown')
            generated_action = result.get('action', 'unknown')
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return {
                'query': query,
                'expected_intent': expected_intent,
                'expected_action': expected_action,
                'description': description,
                'success': False,
                'error': str(e)
            }
    else:
        # Mock results for structure validation
        print("📝 Mock test execution (replace with actual AutoGen call)")
        extracted_intent = expected_intent  # Mock: assume correct
        generated_action = expected_action  # Mock: assume correct
    
    # Validate results
    intent_correct, intent_feedback = validate_intent_classification(
        query, expected_intent, extracted_intent)
    action_correct, action_feedback = validate_action_generation(
        query, expected_action, generated_action)
    
    print(f"Intent: {intent_feedback}")
    print(f"Action: {action_feedback}")
    
    overall_success = intent_correct and action_correct
    status_icon = "✅" if overall_success else "❌"
    print(f"{status_icon} Overall: {'PASS' if overall_success else 'FAIL'}")
    
    return {
        'query': query,
        'expected_intent': expected_intent,
        'expected_action': expected_action,
        'extracted_intent': extracted_intent,
        'generated_action': generated_action,
        'description': description,
        'intent_correct': intent_correct,
        'action_correct': action_correct,
        'overall_success': overall_success,
        'intent_feedback': intent_feedback,
        'action_feedback': action_feedback
    }


def run_test_suite(test_function=None) -> Dict[str, Any]:
    """
    Run Complete Test Suite
    
    Executes all test categories and provides comprehensive validation
    of the multi-agent natural language understanding system.
    
    Args:
        test_function: Optional function to execute tests (for integration)
        
    Returns:
        Complete test results with summary statistics
    """
    print("🚀 Starting Multi-Agent NLU Test Suite")
    print("=" * 50)
    
    all_results = []
    category_summaries = {}
    
    for category_name, test_cases in ALL_TEST_CASES:
        print(f"\n📂 Category: {category_name}")
        print("-" * 30)
        
        category_results = []
        for query, expected_intent, expected_action, description in test_cases:
            result = run_single_test(query, expected_intent, expected_action, 
                                   description, test_function)
            category_results.append(result)
            all_results.append(result)
        
        # Calculate category statistics
        total_tests = len(category_results)
        passed_tests = sum(1 for r in category_results if r['overall_success'])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        category_summaries[category_name] = {
            'total': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'pass_rate': pass_rate
        }
        
        print(f"\n📊 {category_name} Summary: {passed_tests}/{total_tests} passed ({pass_rate:.1f}%)")
    
    # Overall summary
    total_all = len(all_results)
    passed_all = sum(1 for r in all_results if r['overall_success'])
    overall_pass_rate = (passed_all / total_all) * 100 if total_all > 0 else 0
    
    print("\n" + "=" * 50)
    print(f"🏆 OVERALL RESULTS: {passed_all}/{total_all} passed ({overall_pass_rate:.1f}%)")
    print("=" * 50)
    
    return {
        'all_results': all_results,
        'category_summaries': category_summaries,
        'overall_stats': {
            'total': total_all,
            'passed': passed_all,
            'failed': total_all - passed_all,
            'pass_rate': overall_pass_rate
        }
    }


if __name__ == "__main__":
    """
    Main Test Execution
    
    This script can be run standalone to validate the test structure,
    or integrated with the actual AutoGen system for real validation.
    
    For integration, replace the mock test_function with actual
    AutoGen decoder calls.
    """
    print("🧪 Multi-Agent NLU Validation Test Suite")
    print("PhD Thesis Research - Intent Recognition & Action Routing")
    print()
    
    # Run the test suite
    results = run_test_suite()
    
    # Optionally save results to file for analysis
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to test_results.json")
    print("🎯 Use these tests to validate fixes for context bleeding and intent classification issues!") 