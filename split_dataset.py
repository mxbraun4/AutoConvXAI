#!/usr/bin/env python3
"""
Split the test dataset into separate files for independent evaluation.

This allows running each split independently to avoid any cumulative issues.
"""

import json
import os
from typing import List, Dict, Any

def split_dataset(input_file: str, batch_size: int = 20):
    """Split the dataset into separate files."""
    
    # Load the original test cases
    with open(input_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    total_cases = len(test_cases)
    num_splits = (total_cases + batch_size - 1) // batch_size
    
    print(f"Splitting {total_cases} test cases into {num_splits} files of {batch_size} cases each")
    
    # Create splits directory
    splits_dir = 'splits'
    os.makedirs(splits_dir, exist_ok=True)
    
    # Split the data
    for split_num in range(num_splits):
        start_idx = split_num * batch_size
        end_idx = min(start_idx + batch_size, total_cases)
        split_cases = test_cases[start_idx:end_idx]
        
        # Create filename
        split_file = os.path.join(splits_dir, f'split_{split_num + 1}.json')
        
        # Save split
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_cases, f, indent=2, ensure_ascii=False)
        
        print(f"Split {split_num + 1}: {len(split_cases)} cases saved to {split_file}")
    
    # Create evaluation scripts for each split
    create_evaluation_scripts(num_splits)
    
    print(f"\nDataset split complete!")
    print(f"Run individual splits with: python3 run_split_X.py (where X is 1-{num_splits})")

def create_evaluation_scripts(num_splits: int):
    """Create individual evaluation scripts for each split."""
    
    for split_num in range(1, num_splits + 1):
        script_content = f'''#!/usr/bin/env python3
"""
Evaluate split {split_num} independently.
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
    split_file = 'splits/split_{split_num}.json'
    results_dir = 'results_split_{split_num}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting evaluation of split {split_num}...")
    print(f"Results will be saved to {{results_dir}}/")
    
    start_time = time.time()
    
    # Initialize evaluator
    evaluator = AutoGenEvaluator(split_file, api_key, delay_between_calls=0.5)
    
    # Run evaluation on all cases in this split (single batch)
    test_cases = evaluator.test_cases
    print(f"Evaluating {{len(test_cases)}} test cases in split {split_num}")
    
    # Evaluate all cases in one batch
    results = evaluator.evaluate_parallel(test_cases, max_workers=1, batch_refresh=True)
    
    # Generate report
    report = evaluator.generate_report(results)
    
    # Calculate time
    total_time = time.time() - start_time
    
    # Print summary
    print("\\n" + "="*50)
    print(f"SPLIT {split_num} EVALUATION RESULTS")
    print("="*50)
    print(f"Total Cases: {{report['total_cases']}}")
    print(f"Intent Accuracy: {{report['intent_accuracy']:.2%}}")
    print(f"Entity Accuracy: {{report['entity_accuracy']:.2%}}")
    print(f"Overall Accuracy: {{report['overall_accuracy']:.2%}}")
    print(f"Error Rate: {{report['error_rate']:.2%}}")
    print(f"Time: {{total_time:.0f}} seconds ({{total_time/60:.1f}} minutes)")
    
    print("\\nIntent Breakdown:")
    for intent, data in report['intent_breakdown'].items():
        print(f"  {{intent}}: {{data['correct']}}/{{data['total']}} ({{data['accuracy']:.2%}})")
    
    # Save results
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    evaluator.save_results(results, report, results_file)
    print(f"\\nResults saved to {{results_file}}")

if __name__ == "__main__":
    main()
'''
        
        script_file = f'run_split_{split_num}.py'
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_file, 0o755)

def main():
    """Main function to split the dataset."""
    input_file = 'parsing_accuracy/converted_test_cases.json'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return
    
    split_dataset(input_file, batch_size=20)

if __name__ == "__main__":
    main()