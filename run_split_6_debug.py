#!/usr/bin/env python3
"""
Evaluate split 6 with debug logging enabled.
"""

import os
import sys
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable DEBUG logging for AutoGen decoder
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autogen_split6_debug.log'),
        logging.StreamHandler()
    ]
)

# Set specific logger for autogen decoder to DEBUG
logging.getLogger('explain.decoders.autogen_decoder').setLevel(logging.DEBUG)

from parsing_accuracy.autogen_evaluator import AutoGenEvaluator

def main():
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    
    # Setup
    split_file = 'splits/split_6.json'
    results_dir = 'results_split_6_debug'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting evaluation of split 6 with DEBUG logging...")
    print(f"Results will be saved to {results_dir}/")
    print(f"Debug logs will be saved to autogen_split6_debug.log")
    
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
    
    # Save results
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    evaluator.save_results(results, report, results_file)
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n{'='*50}")
    print(f"SPLIT 6 EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total Cases: {report['total_cases']}")
    print(f"Action Accuracy: {report['action_accuracy']*100:.2f}%")
    print(f"Entity Accuracy: {report['entity_accuracy']*100:.2f}%")
    print(f"Overall Accuracy: {report['overall_accuracy']*100:.2f}%")
    print(f"Error Rate: {report['error_rate']*100:.2f}%")
    print(f"Time: {int(elapsed_time)} seconds ({minutes}.{seconds//6} minutes)")
    
    # Print action breakdown
    if 'action_breakdown' in report:
        print("\nAction Breakdown:")
        for action, stats in report['action_breakdown'].items():
            print(f"  {action}: {stats['correct']}/{stats['total']} ({stats['accuracy']*100:.2f}%)")
    
    print(f"\nResults saved to {results_dir}/evaluation_results.json")
    print(f"Debug logs saved to autogen_split6_debug.log")

if __name__ == "__main__":
    main()