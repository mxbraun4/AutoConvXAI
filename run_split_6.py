#!/usr/bin/env python3
"""
Evaluate split 6 independently.
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsing_accuracy.autogen_evaluator import AutoGenEvaluator

def main():
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    
    # Setup
    split_file = 'splits/split_6.json'
    results_dir = 'results_split_6'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting evaluation of split 6...")
    print(f"Results will be saved to {results_dir}/")
    
    start_time = time.time()
    
    # Initialize evaluator
    evaluator = AutoGenEvaluator(split_file, api_key, delay_between_calls=0.5)
    
    # Run evaluation on all cases in this split (single batch)
    test_cases = evaluator.test_cases
    print(f"Evaluating {len(test_cases)} test cases in split 6")
    
    # Evaluate all cases in one batch
    results = evaluator.evaluate_parallel(test_cases, max_workers=1, batch_refresh=True)
    
    # Generate report
    report = evaluator.generate_report(results)
    
    # Calculate time
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print(f"SPLIT 6 EVALUATION RESULTS")
    print("="*50)
    print(f"Total Cases: {report['total_cases']}")
    print(f"Action Accuracy: {report['action_accuracy']:.2%}")
    print(f"Entity Accuracy: {report['entity_accuracy']:.2%}")
    print(f"Overall Accuracy: {report['overall_accuracy']:.2%}")
    print(f"Error Rate: {report['error_rate']:.2%}")
    print(f"Time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
    
    print("\nAction Breakdown:")
    for intent, data in report['action_breakdown'].items():
        print(f"  {intent}: {data['correct']}/{data['total']} ({data['accuracy']:.2%})")
    
    # Save results
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    evaluator.save_results(results, report, results_file)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
