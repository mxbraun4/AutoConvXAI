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
    from nlu.autogen_decoder import AutoGenDecoder
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
    action_match: bool
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
        # Initialize mock dataset and model for proper context
        self._mock_dataset = MockDataset()
        self._mock_model = MockModel()
    
    def get_var(self, var_name: str):
        """Mock get_var method - returns mock objects to provide proper context for entity extraction."""
        # Return mock dataset to provide feature context for entity extraction
        # This helps agents understand available features and validate entities
        if var_name == 'dataset':
            return self._mock_dataset
        elif var_name == 'model':
            # Return None for model to prevent followup detection from prediction comparison
            # We only need dataset context for entity extraction
            return None
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
        
        # Initialize shared AutoGen decoder for batch processing
        self.decoder = None
        self._create_fresh_decoder()
        
        # Set up clean logging - suppress AutoGen framework noise
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Create custom logger for evaluation results only
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Suppress specific noisy loggers
        logging.getLogger('autogen_agentchat').setLevel(logging.WARNING)
        logging.getLogger('autogen_ext').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
    
    def _create_fresh_decoder(self):
        """Create a fresh decoder instance by completely replacing the old one."""
        # Simply replace the decoder without trying to clean up the old one
        # Let Python's garbage collector handle the cleanup naturally
        if self.decoder:
            try:
                import gc
                # Just delete the reference and force garbage collection
                del self.decoder
                gc.collect()
            except Exception as e:
                self.logger.warning(f"Decoder cleanup error: {e}")
        
        # Create completely new decoder instance
        self.decoder = AutoGenDecoder(api_key=self.api_key, model="gpt-4o-mini")
    
    def evaluate_action_match(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Check if actions match."""
        expected_action = expected.get('action', '')
        
        # Check multiple possible locations for action in AutoGen output
        actual_action = None
        if 'action_response' in actual and 'action' in actual['action_response']:
            actual_action = actual['action_response']['action']
        elif 'final_action' in actual:
            actual_action = actual['final_action']
        elif 'action' in actual:
            actual_action = actual['action']
        
        return expected_action == actual_action
    
    def evaluate_entities_match(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Check if entities match (with proper handling of template structure)."""
        expected_entities = expected.get('entities', {})
        
        # Get entities from AutoGen output
        actual_entities = {}
        if 'action_response' in actual and 'entities' in actual['action_response']:
            actual_entities = actual['action_response']['entities']
        elif 'command_structure' in actual:
            actual_entities = actual['command_structure']
        elif 'entities' in actual:
            actual_entities = actual['entities']
        
        # FIXED: Only check entities that are expected to have non-null values
        # This handles the template structure properly where agents output all fields
        for key, expected_val in expected_entities.items():
            # Skip null expected values - they're not requirements
            if expected_val is None:
                continue
                
            # Check if the key exists in actual entities
            if key not in actual_entities:
                return False
                
            actual_val = actual_entities[key]
            
            # Skip null actual values if we expect something concrete
            if actual_val is None and expected_val is not None:
                return False
            
            # Compare values with flexibility
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
        """Evaluate a single test case using shared decoder with fresh conversation context."""
        question = test_case['question']
        expected_json = test_case['expected_json']
        
        try:
            # Apply rate limiting
            with self.rate_limit_lock:
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call_time
                if time_since_last_call < self.delay_between_calls:
                    time.sleep(self.delay_between_calls - time_since_last_call)
                self.last_api_call_time = time.time()
            
            # Use shared decoder with fresh conversation context
            fresh_conversation = MockConversation()
            
            # Get AutoGen output using shared decoder
            autogen_output = self.decoder.complete_sync(question, fresh_conversation)
            
            # Check matches
            action_match = self.evaluate_action_match(expected_json, autogen_output)
            entities_match = self.evaluate_entities_match(expected_json, autogen_output)
            overall_match = action_match and entities_match
            
            return EvaluationResult(
                question=question,
                expected_json=expected_json,
                autogen_output=autogen_output,
                action_match=action_match,
                entities_match=entities_match,
                overall_match=overall_match
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating question: {question[:50]}... Error: {e}")
            return EvaluationResult(
                question=question,
                expected_json=expected_json,
                autogen_output={},
                action_match=False,
                entities_match=False,
                overall_match=False,
                error_message=str(e)
            )
    

    
    def evaluate_subset(self, max_cases: int = 50) -> List[EvaluationResult]:
        """Evaluate a subset of test cases."""
        self.logger.info(f"Evaluating {min(max_cases, len(self.test_cases))} test cases...")
        
        results = []
        for i, test_case in enumerate(self.test_cases[:max_cases]):
            result = self.evaluate_single_case(test_case)
            results.append(result)
            
            # Clean progress logging
            if result.overall_match:
                self.logger.info(f"Case {i+1}: ✓ Match")
            else:
                self.logger.info(f"Case {i+1}: ✗ No match (Action: {result.action_match}, Entities: {result.entities_match})")
        
        return results
    
    def evaluate_parallel(self, test_cases: List[Dict[str, Any]], max_workers: int = 1, batch_refresh: bool = True) -> List[EvaluationResult]:
        """Evaluate test cases using shared decoder with optional batch refresh."""
        # Create fresh decoder at start of batch if requested
        if batch_refresh:
            self._create_fresh_decoder()
        
        # Sequential execution to avoid any state contamination
        results = []
        completed = 0
        total = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            result = self.evaluate_single_case(test_case)
            results.append(result)
            completed += 1
            
            # Clean progress logging
            status = "✓" if result.overall_match else "✗"
            self.logger.info(f"Case {completed}/{total}: {status}")
        
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate evaluation report."""
        total_cases = len(results)
        action_matches = sum(1 for r in results if r.action_match)
        entity_matches = sum(1 for r in results if r.entities_match)
        overall_matches = sum(1 for r in results if r.overall_match)
        errors = sum(1 for r in results if r.error_message)
        
        # Action distribution analysis
        action_accuracy = {}
        for result in results:
            expected_action = result.expected_json.get('action', 'unknown')
            if expected_action not in action_accuracy:
                action_accuracy[expected_action] = {'total': 0, 'correct': 0}
            action_accuracy[expected_action]['total'] += 1
            if result.action_match:
                action_accuracy[expected_action]['correct'] += 1
        
        # Calculate accuracy per action
        for action in action_accuracy:
            acc_data = action_accuracy[action]
            acc_data['accuracy'] = acc_data['correct'] / acc_data['total'] if acc_data['total'] > 0 else 0
        
        report = {
            'total_cases': total_cases,
            'action_accuracy': action_matches / total_cases if total_cases > 0 else 0,
            'entity_accuracy': entity_matches / total_cases if total_cases > 0 else 0,
            'overall_accuracy': overall_matches / total_cases if total_cases > 0 else 0,
            'error_rate': errors / total_cases if total_cases > 0 else 0,
            'action_breakdown': action_accuracy,
            'action_mismatches': [],
            'entity_mismatches': []
        }
        
        # Separate action and entity mismatches
        action_mismatches = [r for r in results if not r.action_match and not r.error_message][:5]
        entity_mismatches = [r for r in results if r.action_match and not r.entities_match and not r.error_message][:5]
        
        # Add action mismatches
        for mismatch in action_mismatches:
            report['action_mismatches'].append({
                'question': mismatch.question,
                'expected_action': mismatch.expected_json.get('action'),
                'actual_action': mismatch.autogen_output.get('final_action') or 
                              mismatch.autogen_output.get('action_response', {}).get('action'),
                'expected_entities': mismatch.expected_json.get('entities', {}),
                'actual_entities': mismatch.autogen_output.get('command_structure', {})
            })
        
        # Add entity mismatches
        for mismatch in entity_mismatches:
            report['entity_mismatches'].append({
                'question': mismatch.question,
                'expected_action': mismatch.expected_json.get('action'),
                'actual_action': mismatch.autogen_output.get('final_action') or 
                              mismatch.autogen_output.get('action_response', {}).get('action'),
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
                'action_match': result.action_match,
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
    test_cases_file = '/app/evaluation/parsing_accuracy/converted_test_cases.json'
    evaluator = AutoGenEvaluator(test_cases_file, api_key, delay_between_calls=0.1)
    
    total_cases = len(evaluator.test_cases)
    batch_size = 20
    num_batches = (total_cases + batch_size - 1) // batch_size
    
    print(f"Starting AutoGen evaluation on all {total_cases} test cases...")
    print(f"Processing in {num_batches} batches of {batch_size} cases each")
    print("Using sequential processing with 0.1s rate limiting and 10s breaks between batches")
    
    # Estimate time including batch breaks
    test_time = total_cases * 0.1  # 0.1s per call, sequential
    break_time = (num_batches - 1) * 10  # 10s break between batches
    estimated_time = test_time + break_time
    print(f"Estimated time: {estimated_time:.0f} seconds ({estimated_time/60:.1f} minutes)\n")
    
    # Create results directory
    results_dir = 'eval_20_batches'
    os.makedirs(results_dir, exist_ok=True)
    
    start_time = time.time()
    all_results = []
    
    # Process in batches
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_cases)
        batch_cases = evaluator.test_cases[batch_start:batch_end]
        
        print(f"\nBatch {batch_num + 1}/{num_batches}: Tests {batch_start + 1}-{batch_end}")
        print("-" * 50)
        
        # Evaluate batch with fresh decoder at start of each batch
        batch_results = evaluator.evaluate_parallel(batch_cases, max_workers=1, batch_refresh=True)
        all_results.extend(batch_results)
        
        # Save intermediate results
        intermediate_file = os.path.join(results_dir, f'evaluation_results_batch_{batch_num + 1}.json')
        temp_report = evaluator.generate_report(all_results)
        evaluator.save_results(all_results, temp_report, intermediate_file)
        
        # Show batch summary
        batch_matches = sum(1 for r in batch_results if r.overall_match)
        print(f"Batch {batch_num + 1} complete: {batch_matches}/{len(batch_results)} matches")
        
        # Add 10-second break between batches (except after the last batch)
        if batch_num < num_batches - 1:
            print("Waiting 10 seconds before next batch to allow rate limits to reset...")
            time.sleep(10)
    
    # Generate final report
    report = evaluator.generate_report(all_results)
    
    # Calculate actual time taken
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print("AUTOGEN EVALUATION RESULTS")
    print("="*50)
    print(f"Total Cases Evaluated: {report['total_cases']}")
    print(f"Action Accuracy: {report['action_accuracy']:.2%}")
    print(f"Entity Accuracy: {report['entity_accuracy']:.2%}")
    print(f"Overall Accuracy: {report['overall_accuracy']:.2%}")
    print(f"Error Rate: {report['error_rate']:.2%}")
    print(f"Total Time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average Time per Test: {total_time/total_cases:.2f} seconds")
    
    print("\nAction Breakdown:")
    for action, data in report['action_breakdown'].items():
        print(f"  {action}: {data['correct']}/{data['total']} ({data['accuracy']:.2%})")
    
    if report['action_mismatches']:
        print("\nTop 10 Action Mismatches:")
        for i, mismatch in enumerate(report['action_mismatches'][:10], 1):
            print(f"{i}. Expected: {mismatch['expected_action']} | Actual: {mismatch['actual_action']}")
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
    test_cases_file = '/app/evaluation/parsing_accuracy/converted_test_cases.json'
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
    print(f"Action Accuracy: {report['action_accuracy']:.2%}")
    print(f"Entity Accuracy: {report['entity_accuracy']:.2%}")
    print(f"Overall Accuracy: {report['overall_accuracy']:.2%}")
    print(f"Error Rate: {report['error_rate']:.2%}")
    
    print("\nAction Breakdown:")
    for action, data in report['action_breakdown'].items():
        print(f"  {action}: {data['correct']}/{data['total']} ({data['accuracy']:.2%})")
    
    if report['action_mismatches']:
        print("\nAction Mismatches:")
        for i, mismatch in enumerate(report['action_mismatches'], 1):
            print(f"{i}. Expected: {mismatch['expected_action']} | Actual: {mismatch['actual_action']}")
            print(f"   Question: {mismatch['question'][:60]}...")
    
    # Save results
    evaluator.save_results(results, report)
    
    print(f"\nDetailed results saved to evaluation_results.json")


if __name__ == "__main__":
    main()  # Test with 20 cases to see improvement from corrected labels