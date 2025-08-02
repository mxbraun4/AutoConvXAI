#!/usr/bin/env python3
"""
Subprocess-based batch evaluator to prevent resource accumulation.

This script runs each batch in a completely isolated subprocess to ensure
no state accumulation between batches.
"""

import os
import sys
import json
import time
import subprocess
from typing import List, Dict, Any

def run_batch_in_subprocess(batch_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single batch in an isolated subprocess."""
    
    # Create a temporary script that will run the batch
    script_content = f'''
import sys
import os
import json
sys.path.insert(0, '{os.getcwd()}')

from parsing_accuracy.autogen_evaluator import AutoGenEvaluator

# Get batch data from command line argument
batch_data = {repr(batch_data)}

# Initialize fresh evaluator in this subprocess
evaluator = AutoGenEvaluator(
    test_cases_file=batch_data["test_cases_file"], 
    api_key=batch_data["api_key"], 
    delay_between_calls=batch_data["delay_between_calls"]
)

# Run the batch
batch_cases = batch_data["test_cases"]
results = evaluator.evaluate_parallel(batch_cases, max_workers=1, batch_refresh=True)

# Generate report for this batch
batch_report = evaluator.generate_report(results)

# Return results
import pickle
with open(batch_data["result_file"], "wb") as f:
    pickle.dump({{"results": results, "report": batch_report}}, f)

print(f"Batch complete: {{len([r for r in results if r.overall_match])}}/{{len(results)}} matches")
'''
    
    # Write the script to a temporary file
    script_file = f"temp_batch_{batch_data['batch_num']}.py"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    try:
        # Run the batch in a subprocess
        result = subprocess.run([
            sys.executable, script_file
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode != 0:
            print(f"Batch {batch_data['batch_num']} subprocess failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        # Load results from the result file
        import pickle
        with open(batch_data["result_file"], "rb") as f:
            batch_results = pickle.load(f)
        
        # Clean up temp files
        os.remove(script_file)
        os.remove(batch_data["result_file"])
        
        return {"success": True, "data": batch_results}
        
    except subprocess.TimeoutExpired:
        print(f"Batch {batch_data['batch_num']} timed out")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"Batch {batch_data['batch_num']} failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        # Clean up temp files
        if os.path.exists(script_file):
            os.remove(script_file)

def run_subprocess_evaluation():
    """Run evaluation using subprocess isolation for each batch."""
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    # Load test cases
    test_cases_file = 'parsing_accuracy/converted_test_cases.json'
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # Configuration
    batch_size = 20
    total_cases = len(test_cases)
    num_batches = (total_cases + batch_size - 1) // batch_size
    
    print(f"Starting subprocess evaluation on all {total_cases} test cases...")
    print(f"Processing in {num_batches} batches of {batch_size} cases each")
    print("Each batch runs in complete isolation to prevent resource accumulation")
    
    # Create results directory
    results_dir = 'eval_subprocess_batches'
    os.makedirs(results_dir, exist_ok=True)
    
    start_time = time.time()
    all_results = []
    
    # Process each batch in subprocess
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_cases)
        batch_cases = test_cases[batch_start:batch_end]
        
        print(f"\\nBatch {batch_num + 1}/{num_batches}: Tests {batch_start + 1}-{batch_end}")
        print("-" * 60)
        
        # Prepare batch data for subprocess
        batch_data = {
            "batch_num": batch_num + 1,
            "test_cases": batch_cases,
            "test_cases_file": test_cases_file,
            "api_key": api_key,
            "delay_between_calls": 2.0,
            "result_file": f"temp_results_{batch_num}.pkl"
        }
        
        # Run batch in subprocess
        batch_result = run_batch_in_subprocess(batch_data)
        
        if batch_result["success"]:
            batch_results = batch_result["data"]["results"]
            all_results.extend(batch_results)
            
            # Save intermediate results
            intermediate_file = os.path.join(results_dir, f'evaluation_results_batch_{batch_num + 1}.json')
            from parsing_accuracy.autogen_evaluator import AutoGenEvaluator
            temp_evaluator = AutoGenEvaluator(test_cases_file, api_key)
            temp_report = temp_evaluator.generate_report(all_results)
            temp_evaluator.save_results(all_results, temp_report, intermediate_file)
            
            # Show batch summary
            batch_matches = sum(1 for r in batch_results if r.overall_match)
            print(f"Batch {batch_num + 1} complete: {batch_matches}/{len(batch_results)} matches")
        else:
            print(f"Batch {batch_num + 1} failed: {batch_result['error']}")
        
        # Break between batches
        if batch_num < num_batches - 1:
            print("Waiting 10 seconds before next batch...")
            time.sleep(10)
    
    # Generate final report
    temp_evaluator = AutoGenEvaluator(test_cases_file, api_key)
    final_report = temp_evaluator.generate_report(all_results)
    
    # Calculate time
    total_time = time.time() - start_time
    
    # Print summary
    print("\\n" + "="*60)
    print("SUBPROCESS EVALUATION RESULTS")
    print("="*60)
    print(f"Total Cases Evaluated: {final_report['total_cases']}")
    print(f"Intent Accuracy: {final_report['intent_accuracy']:.2%}")
    print(f"Entity Accuracy: {final_report['entity_accuracy']:.2%}")
    print(f"Overall Accuracy: {final_report['overall_accuracy']:.2%}")
    print(f"Error Rate: {final_report['error_rate']:.2%}")
    print(f"Total Time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
    
    # Save final results
    final_file = os.path.join(results_dir, 'evaluation_results_final.json')
    temp_evaluator.save_results(all_results, final_report, final_file)
    print(f"\\nResults saved to {final_file}")

if __name__ == "__main__":
    run_subprocess_evaluation()