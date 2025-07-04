#!/usr/bin/env python3
"""
Run JSON format tests for AutoGen decoder evaluation.

Usage:
    python run_json_tests.py quick    # Run 20 test cases
    python run_json_tests.py full     # Run all 191 test cases
    python run_json_tests.py [N]      # Run N test cases
"""

import sys
import os
import json
import time
from datetime import datetime

# Add the parsing_accuracy directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'parsing_accuracy'))

try:
    from autogen_evaluator import AutoGenEvaluator
except ImportError:
    print("Error: Could not import autogen_evaluator")
    print("Make sure parsing_accuracy/autogen_evaluator.py exists")
    sys.exit(1)


def print_header():
    """Print a nice header for the test run."""
    print("\n" + "="*70)
    print("AUTOGEN DECODER EVALUATION TEST")
    print("="*70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Key: {'Set' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print("="*70 + "\n")


def print_results(report):
    """Print evaluation results in a nice format."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nTotal Cases Evaluated: {report['total_cases']}")
    print(f"Intent Accuracy: {report['intent_accuracy']:.2%}")
    print(f"Entity Accuracy: {report['entity_accuracy']:.2%}")
    print(f"Overall Accuracy: {report['overall_accuracy']:.2%}")
    print(f"Error Rate: {report['error_rate']:.2%}")
    
    print("\nIntent Breakdown:")
    print("-" * 40)
    for intent, data in sorted(report['intent_breakdown'].items()):
        accuracy = data['accuracy']
        correct = data['correct']
        total = data['total']
        print(f"  {intent:<15} {correct:>3}/{total:<3} ({accuracy:.2%})")
    
    # Show intent mismatches
    if report.get('intent_mismatches'):
        print("\nIntent Mismatches:")
        print("-" * 40)
        for i, mismatch in enumerate(report['intent_mismatches'], 1):
            print(f"\n{i}. Question: {mismatch['question'][:60]}...")
            print(f"   Expected: {mismatch['expected_intent']}")
            print(f"   Actual: {mismatch['actual_intent']}")
    
    # Show entity mismatches  
    if report.get('entity_mismatches'):
        print("\nEntity Mismatches:")
        print("-" * 40)
        for i, mismatch in enumerate(report['entity_mismatches'], 1):
            print(f"\n{i}. Question: {mismatch['question'][:60]}...")
            print(f"   Intent: {mismatch['expected_intent']} (correct)")
            print(f"   Expected entities: {mismatch['expected_entities']}")
            print(f"   Actual entities: {mismatch['actual_entities']}")
    
    # Show backwards compatibility if old format exists
    if report.get('sample_mismatches') and not report.get('intent_mismatches') and not report.get('entity_mismatches'):
        print("\nSample Mismatches:")
        print("-" * 40)
        for i, mismatch in enumerate(report['sample_mismatches'], 1):
            print(f"\n{i}. Question: {mismatch['question'][:60]}...")
            print(f"   Expected: {mismatch['expected_intent']}")
            print(f"   Actual: {mismatch['actual_intent']}")
    
    print("\n" + "="*70)
    
    # Comparison with old system
    old_accuracy = 0.76
    current_accuracy = report['overall_accuracy']
    
    if current_accuracy > old_accuracy:
        print(f"✅ IMPROVEMENT: {current_accuracy:.2%} vs old system's {old_accuracy:.2%} (+{(current_accuracy - old_accuracy):.2%})")
    elif current_accuracy == old_accuracy:
        print(f"➖ MATCHED: {current_accuracy:.2%} equals old system's {old_accuracy:.2%}")
    else:
        print(f"❌ BELOW TARGET: {current_accuracy:.2%} vs old system's {old_accuracy:.2%} ({(current_accuracy - old_accuracy):.2%})")
    
    print("="*70 + "\n")


def main():
    """Main test runner."""
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Usage: docker run -e OPENAI_API_KEY=your-key ttm-gpt4-test python run_json_tests.py quick")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == 'quick':
            max_cases = 20
        elif arg == 'full':
            max_cases = 191
        else:
            try:
                max_cases = int(arg)
            except ValueError:
                print(f"Invalid argument: {arg}")
                print("Use 'quick' for 20 cases, 'full' for all cases, or a number")
                sys.exit(1)
    else:
        max_cases = 20  # Default to quick mode
    
    print_header()
    
    # Initialize evaluator
    try:
        test_cases_file = 'parsing_accuracy/converted_test_cases.json'
        
        if not os.path.exists(test_cases_file):
            print(f"Error: Test cases file not found: {test_cases_file}")
            print("Make sure converted_test_cases.json exists in parsing_accuracy/")
            sys.exit(1)
        
        print(f"Loading test cases from: {test_cases_file}")
        evaluator = AutoGenEvaluator(test_cases_file)
        
        # Load test cases to show total available
        with open(test_cases_file, 'r') as f:
            total_available = len(json.load(f))
        
        print(f"Total test cases available: {total_available}")
        print(f"Running evaluation on: {min(max_cases, total_available)} cases")
        print("\nStarting evaluation...")
        print("-" * 40)
        
        # Run evaluation
        start_time = time.time()
        results = evaluator.evaluate_subset(max_cases=max_cases)
        elapsed_time = time.time() - start_time
        
        # Generate report
        report = evaluator.generate_report(results)
        
        # Print results
        print_results(report)
        
        print(f"Evaluation completed in {elapsed_time:.2f} seconds")
        print(f"Average time per case: {elapsed_time/len(results):.2f} seconds")
        
        # Save detailed results
        output_file = 'parsing_accuracy/evaluation_results.json'
        evaluator.save_results(results, report, output_file)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Exit with appropriate code
        if report['overall_accuracy'] >= 0.76:
            print("\n✅ Test PASSED - Meets or exceeds old system accuracy")
            sys.exit(0)
        else:
            print("\n❌ Test FAILED - Below old system accuracy")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()