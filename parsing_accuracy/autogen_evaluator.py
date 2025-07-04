"""
AutoGen Evaluation System

This module evaluates the AutoGen decoder's performance against the converted test cases
from the old talktomodel system. It compares the AutoGen output with expected JSON format.
"""

import json
import logging
import os
import sys
import time
import asyncio
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# Add the parent directory to sys.path to import the AutoGenDecoder
sys.path.append('/app')

try:
    from explain.decoders.autogen_decoder import AutoGenDecoder
except ImportError as e:
    print(f"Could not import AutoGenDecoder: {e}")
    print("This evaluator requires the AutoGenDecoder to be available.")
    sys.exit(1)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    question: str
    expected_json: Dict[str, Any]
    autogen_output: Dict[str, Any]
    intent_match: bool
    entities_match: bool
    overall_match: bool
    error_message: str = ""


class MockConversation:
    """Mock conversation object for testing AutoGen decoder."""
    
    def __init__(self):
        self.dataset_info = {
            'size': 768,
            'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
            'target_classes': [0, 1]
        }
        self.temp_dataset = None
        self.parse_operation = None
        self.last_parse_string = []
        self.class_names = ['No Diabetes', 'Diabetes']
    
    def get_var(self, var_name: str):
        """Mock get_var method - returns None to prevent context injection during testing."""
        # Return None for dataset and model to prevent "RECENT RESULTS CONTEXT" 
        # injection that contaminates test cases and makes everything appear as 'followup'
        return None


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self):
        self.contents = {
            'X': [[1, 2, 3, 4, 5, 6, 7, 8]] * 768,  # Mock 768 samples
            'y': [0, 1] * 384,  # Mock labels
            'cat': [],  # No categorical features in diabetes dataset
            'numeric': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        }


class MockModel:
    """Mock model for testing."""
    
    def predict(self, X):
        """Mock predict method."""
        return [0, 1] * (len(X) // 2)


class AutoGenEvaluator:
    """Evaluator for AutoGen decoder performance."""
    
    def __init__(self, test_cases_file: str, api_key: str = None, delay_between_calls: float = 0.5):
        """Initialize evaluator with test cases and API key."""
        self.test_cases_file = test_cases_file
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.delay_between_calls = delay_between_calls
        self.rate_limit_lock = threading.Lock()
        self.last_api_call_time = 0
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Load test cases
        with open(test_cases_file, 'r', encoding='utf-8') as f:
            self.test_cases = json.load(f)
        
        # Initialize AutoGen decoder
        self.decoder = AutoGenDecoder(api_key=self.api_key, model="gpt-4o-mini")
        
        # Initialize mock conversation
        self.conversation = MockConversation()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def evaluate_intent_match(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Check if intents match."""
        expected_intent = expected.get('intent', '')
        
        # Check multiple possible locations for intent in AutoGen output
        actual_intent = None
        if 'intent_response' in actual and 'intent' in actual['intent_response']:
            actual_intent = actual['intent_response']['intent']
        elif 'final_action' in actual:
            actual_intent = actual['final_action']
        elif 'intent' in actual:
            actual_intent = actual['intent']
        
        return expected_intent == actual_intent
    
    def evaluate_entities_match(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Check if entities match (with some flexibility)."""
        expected_entities = expected.get('entities', {})
        
        # Get entities from AutoGen output
        actual_entities = {}
        if 'intent_response' in actual and 'entities' in actual['intent_response']:
            actual_entities = actual['intent_response']['entities']
        elif 'command_structure' in actual:
            actual_entities = actual['command_structure']
        elif 'entities' in actual:
            actual_entities = actual['entities']
        
        # Check key entity fields
        for key in ['patient_id', 'features', 'operators', 'values', 'topk']:
            if key in expected_entities:
                if key not in actual_entities:
                    return False
                
                expected_val = expected_entities[key]
                actual_val = actual_entities[key]
                
                # Allow some flexibility in comparison
                if isinstance(expected_val, list) and isinstance(actual_val, list):
                    if len(expected_val) != len(actual_val):
                        return False
                    for exp, act in zip(expected_val, actual_val):
                        if str(exp).lower() != str(act).lower():
                            return False
                elif str(expected_val).lower() != str(actual_val).lower():
                    return False
        
        return True
    
    def evaluate_single_case(self, test_case: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single test case with fresh AutoGen decoder to prevent context accumulation."""
        question = test_case['question']
        expected_json = test_case['expected_json']
        
        try:
            # Apply rate limiting BEFORE creating decoder
            with self.rate_limit_lock:
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call_time
                if time_since_last_call < self.delay_between_calls:
                    time.sleep(self.delay_between_calls - time_since_last_call)
                self.last_api_call_time = time.time()
            
            # Create fresh AutoGen decoder for each test case to prevent context accumulation
            # Use unique API key instance to ensure no connection pooling
            import uuid
            unique_key = self.api_key  # Can't modify key but ensure fresh instance
            fresh_decoder = AutoGenDecoder(api_key=unique_key, model="gpt-4o-mini")
            fresh_conversation = MockConversation()
            
            # Get AutoGen output with fresh instances
            autogen_output = fresh_decoder.complete_sync(question, fresh_conversation)
            
            # Explicitly delete decoder to ensure cleanup
            del fresh_decoder
            
            # Check matches
            intent_match = self.evaluate_intent_match(expected_json, autogen_output)
            entities_match = self.evaluate_entities_match(expected_json, autogen_output)
            overall_match = intent_match and entities_match
            
            return EvaluationResult(
                question=question,
                expected_json=expected_json,
                autogen_output=autogen_output,
                intent_match=intent_match,
                entities_match=entities_match,
                overall_match=overall_match
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating question: {question[:50]}... Error: {e}")
            return EvaluationResult(
                question=question,
                expected_json=expected_json,
                autogen_output={},
                intent_match=False,
                entities_match=False,
                overall_match=False,
                error_message=str(e)
            )
    

    
    def evaluate_subset(self, max_cases: int = 50) -> List[EvaluationResult]:
        """Evaluate a subset of test cases."""
        self.logger.info(f"Evaluating {min(max_cases, len(self.test_cases))} test cases...")
        
        results = []
        for i, test_case in enumerate(self.test_cases[:max_cases]):
            self.logger.info(f"Evaluating case {i+1}/{min(max_cases, len(self.test_cases))}")
            result = self.evaluate_single_case(test_case)
            results.append(result)
            
            # Log progress
            if result.overall_match:
                self.logger.info("✓ Match")
            else:
                self.logger.info(f"✗ No match - Intent: {result.intent_match}, Entities: {result.entities_match}")
        
        return results
    
    def evaluate_parallel(self, test_cases: List[Dict[str, Any]], max_workers: int = 1) -> List[EvaluationResult]:
        """Evaluate test cases in parallel with controlled concurrency."""
        # CHANGED: Set max_workers to 1 to force sequential execution
        # This ensures no parallel state contamination
        results = [None] * len(test_cases)
        completed = 0
        total = len(test_cases)
        start_time = time.time()
        
        def evaluate_with_index(index_and_case):
            index, test_case = index_and_case
            result = self.evaluate_single_case(test_case)
            return index, result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for i, test_case in enumerate(test_cases):
                future = executor.submit(evaluate_with_index, (i, test_case))
                futures.append(future)
            
            # Process results as they complete
            for future in futures:
                index, result = future.result()
                results[index] = result
                completed += 1
                
                # Calculate and display progress
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                
                status = "✓" if result.overall_match else "✗"
                self.logger.info(f"{status} Case {completed}/{total} - ETA: {eta:.0f}s")
        
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate evaluation report."""
        total_cases = len(results)
        intent_matches = sum(1 for r in results if r.intent_match)
        entity_matches = sum(1 for r in results if r.entities_match)
        overall_matches = sum(1 for r in results if r.overall_match)
        errors = sum(1 for r in results if r.error_message)
        
        # Intent distribution analysis
        intent_accuracy = {}
        for result in results:
            expected_intent = result.expected_json.get('intent', 'unknown')
            if expected_intent not in intent_accuracy:
                intent_accuracy[expected_intent] = {'total': 0, 'correct': 0}
            intent_accuracy[expected_intent]['total'] += 1
            if result.intent_match:
                intent_accuracy[expected_intent]['correct'] += 1
        
        # Calculate accuracy per intent
        for intent in intent_accuracy:
            acc_data = intent_accuracy[intent]
            acc_data['accuracy'] = acc_data['correct'] / acc_data['total'] if acc_data['total'] > 0 else 0
        
        report = {
            'total_cases': total_cases,
            'intent_accuracy': intent_matches / total_cases if total_cases > 0 else 0,
            'entity_accuracy': entity_matches / total_cases if total_cases > 0 else 0,
            'overall_accuracy': overall_matches / total_cases if total_cases > 0 else 0,
            'error_rate': errors / total_cases if total_cases > 0 else 0,
            'intent_breakdown': intent_accuracy,
            'intent_mismatches': [],
            'entity_mismatches': []
        }
        
        # Separate intent and entity mismatches
        intent_mismatches = [r for r in results if not r.intent_match and not r.error_message][:5]
        entity_mismatches = [r for r in results if r.intent_match and not r.entities_match and not r.error_message][:5]
        
        # Add intent mismatches
        for mismatch in intent_mismatches:
            report['intent_mismatches'].append({
                'question': mismatch.question,
                'expected_intent': mismatch.expected_json.get('intent'),
                'actual_intent': mismatch.autogen_output.get('final_action') or 
                              mismatch.autogen_output.get('intent_response', {}).get('intent'),
                'expected_entities': mismatch.expected_json.get('entities', {}),
                'actual_entities': mismatch.autogen_output.get('command_structure', {})
            })
        
        # Add entity mismatches
        for mismatch in entity_mismatches:
            report['entity_mismatches'].append({
                'question': mismatch.question,
                'expected_intent': mismatch.expected_json.get('intent'),
                'actual_intent': mismatch.autogen_output.get('final_action') or 
                              mismatch.autogen_output.get('intent_response', {}).get('intent'),
                'expected_entities': mismatch.expected_json.get('entities', {}),
                'actual_entities': mismatch.autogen_output.get('command_structure', {})
            })
        
        return report
    
    def save_results(self, results: List[EvaluationResult], report: Dict[str, Any], 
                    output_file: str = None) -> None:
        """Save evaluation results to file."""
        if output_file is None:
            output_file = 'evaluation_results.json'  # Write to current directory instead
        
        output_data = {
            'report': report,
            'detailed_results': []
        }
        
        for result in results:
            output_data['detailed_results'].append({
                'question': result.question,
                'expected': result.expected_json,
                'actual': result.autogen_output,
                'intent_match': result.intent_match,
                'entities_match': result.entities_match,
                'overall_match': result.overall_match,
                'error': result.error_message
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")



def evaluate_all_cases():
    """Run evaluation on all test cases with batch processing and parallel execution."""
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run the evaluation.")
        return
    
    # Initialize evaluator with smart rate limiting
    test_cases_file = '/app/parsing_accuracy/converted_test_cases.json'
    evaluator = AutoGenEvaluator(test_cases_file, api_key, delay_between_calls=0.5)
    
    total_cases = len(evaluator.test_cases)
    batch_size = 50
    num_batches = (total_cases + batch_size - 1) // batch_size
    
    print(f"Starting AutoGen evaluation on all {total_cases} test cases...")
    print(f"Processing in {num_batches} batches of {batch_size} cases each")
    print("Using sequential processing with 0.5s rate limiting to prevent context contamination")
    
    # Estimate time
    estimated_time = total_cases * 0.5  # 0.5s per call, sequential
    print(f"Estimated time: {estimated_time:.0f} seconds ({estimated_time/60:.1f} minutes)\n")
    
    start_time = time.time()
    all_results = []
    
    # Process in batches
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_cases)
        batch_cases = evaluator.test_cases[batch_start:batch_end]
        
        print(f"\nBatch {batch_num + 1}/{num_batches}: Tests {batch_start + 1}-{batch_end}")
        print("-" * 50)
        
        # Evaluate batch in parallel (now sequential with max_workers=1)
        batch_results = evaluator.evaluate_parallel(batch_cases, max_workers=1)
        all_results.extend(batch_results)
        
        # Save intermediate results
        intermediate_file = f'evaluation_results_batch_{batch_num + 1}.json'
        temp_report = evaluator.generate_report(all_results)
        evaluator.save_results(all_results, temp_report, intermediate_file)
        
        # Show batch summary
        batch_matches = sum(1 for r in batch_results if r.overall_match)
        print(f"Batch {batch_num + 1} complete: {batch_matches}/{len(batch_results)} matches")
    
    # Generate final report
    report = evaluator.generate_report(all_results)
    
    # Calculate actual time taken
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print("AUTOGEN EVALUATION RESULTS")
    print("="*50)
    print(f"Total Cases Evaluated: {report['total_cases']}")
    print(f"Intent Accuracy: {report['intent_accuracy']:.2%}")
    print(f"Entity Accuracy: {report['entity_accuracy']:.2%}")
    print(f"Overall Accuracy: {report['overall_accuracy']:.2%}")
    print(f"Error Rate: {report['error_rate']:.2%}")
    print(f"Total Time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average Time per Test: {total_time/total_cases:.2f} seconds")
    
    print("\nIntent Breakdown:")
    for intent, data in report['intent_breakdown'].items():
        print(f"  {intent}: {data['correct']}/{data['total']} ({data['accuracy']:.2%})")
    
    if report['intent_mismatches']:
        print("\nTop 10 Intent Mismatches:")
        for i, mismatch in enumerate(report['intent_mismatches'][:10], 1):
            print(f"{i}. Expected: {mismatch['expected_intent']} | Actual: {mismatch['actual_intent']}")
            print(f"   Question: {mismatch['question'][:60]}...")
    
    # Save final results
    evaluator.save_results(all_results, report, 'evaluation_results_final.json')
    
    print(f"\nDetailed results saved to evaluation_results_final.json")
    print(f"Intermediate batch results saved as evaluation_results_batch_*.json")

def main():
    """Main evaluation function - backward compatibility."""
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run the evaluation.")
        return
    
    # Initialize evaluator
    test_cases_file = '/app/parsing_accuracy/converted_test_cases.json'
    evaluator = AutoGenEvaluator(test_cases_file, api_key)
    
    # Run evaluation on subset (for speed)
    print("Starting AutoGen evaluation...")
    results = evaluator.evaluate_subset(max_cases=20)  # Start with 20 cases
    
    # Generate report
    report = evaluator.generate_report(results)
    
    # Print summary
    print("\n" + "="*50)
    print("AUTOGEN EVALUATION RESULTS")
    print("="*50)
    print(f"Total Cases Evaluated: {report['total_cases']}")
    print(f"Intent Accuracy: {report['intent_accuracy']:.2%}")
    print(f"Entity Accuracy: {report['entity_accuracy']:.2%}")
    print(f"Overall Accuracy: {report['overall_accuracy']:.2%}")
    print(f"Error Rate: {report['error_rate']:.2%}")
    
    print("\nIntent Breakdown:")
    for intent, data in report['intent_breakdown'].items():
        print(f"  {intent}: {data['correct']}/{data['total']} ({data['accuracy']:.2%})")
    
    if report['intent_mismatches']:
        print("\nIntent Mismatches:")
        for i, mismatch in enumerate(report['intent_mismatches'], 1):
            print(f"{i}. Expected: {mismatch['expected_intent']} | Actual: {mismatch['actual_intent']}")
            print(f"   Question: {mismatch['question'][:60]}...")
    
    # Save results
    evaluator.save_results(results, report)
    
    print(f"\nDetailed results saved to evaluation_results.json")


if __name__ == "__main__":
    main()  # Test with 20 cases to see improvement from corrected labels