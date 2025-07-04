"""
AutoGen Evaluation System

This module evaluates the AutoGen decoder's performance against the converted test cases
from the old talktomodel system. It compares the AutoGen output with expected JSON format.
"""

import json
import logging
import os
import sys
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add the parent directory to sys.path to import the AutoGenDecoder
sys.path.append('/mnt/e/Talktomodel_adjusted')

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
        """Mock get_var method."""
        if var_name == 'dataset':
            return MockDataset()
        elif var_name == 'model':
            return MockModel()
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
    
    def __init__(self, test_cases_file: str, api_key: str = None):
        """Initialize evaluator with test cases and API key."""
        self.test_cases_file = test_cases_file
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
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
        logging.basicConfig(level=logging.INFO)
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
        """Evaluate a single test case."""
        question = test_case['question']
        expected_json = test_case['expected_json']
        
        try:
            # Get AutoGen output
            autogen_output = self.decoder.complete_sync(question, self.conversation)
            
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
            output_file = '/mnt/e/Talktomodel_adjusted/parsing_accuracy/evaluation_results.json'
        
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


def main():
    """Main evaluation function."""
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run the evaluation.")
        return
    
    # Initialize evaluator
    test_cases_file = '/mnt/e/Talktomodel_adjusted/parsing_accuracy/converted_test_cases.json'
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
    
    if report['sample_mismatches']:
        print("\nSample Mismatches:")
        for i, mismatch in enumerate(report['sample_mismatches'], 1):
            print(f"{i}. Expected: {mismatch['expected_intent']} | Actual: {mismatch['actual_intent']}")
            print(f"   Question: {mismatch['question'][:60]}...")
    
    # Save results
    evaluator.save_results(results, report)
    
    print(f"\nDetailed results saved to evaluation_results.json")


if __name__ == "__main__":
    main()