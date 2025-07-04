"""
Test Suite Parser for Converting Old SQL-like Format to New JSON Format

This module parses the diabetes_test_suite.txt file and converts the old SQL-like
commands to the new JSON format used by the AutoGen system.
"""

import re
import json
from typing import List, Dict, Tuple, Any, Optional


class TestSuiteParser:
    """Parser for converting old test suite format to new JSON format."""
    
    def __init__(self, test_file_path: str):
        """Initialize parser with test file path."""
        self.test_file_path = test_file_path
        self.test_cases = []
        
    def parse_test_file(self) -> List[Dict[str, Any]]:
        """Parse the test suite file and extract question-answer pairs."""
        with open(self.test_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_question = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove line numbers (format: "  123→content")
            line = re.sub(r'^\s*\d+→', '', line)
            
            # Check if this is a command line (contains [E])
            if '[E]' in line:
                if current_question:
                    # This is the expected command for the current question
                    command = line.replace('[E]', '').strip()
                    self.test_cases.append({
                        'question': current_question,
                        'old_command': command,
                        'line_number': len(self.test_cases) + 1
                    })
                    current_question = None
            else:
                # This is a question
                current_question = line
        
        return self.test_cases
    
    def convert_to_json_format(self) -> List[Dict[str, Any]]:
        """Convert all test cases to new JSON format."""
        if not self.test_cases:
            self.parse_test_file()
        
        converted_cases = []
        for case in self.test_cases:
            json_format = self._convert_command_to_json(case['old_command'])
            converted_cases.append({
                'question': case['question'],
                'old_command': case['old_command'],
                'expected_json': json_format,
                'line_number': case['line_number']
            })
        
        return converted_cases
    
    def _convert_command_to_json(self, command: str) -> Dict[str, Any]:
        """Convert a single SQL-like command to JSON format."""
        # Clean the command
        command = command.strip()
        
        # Split command into parts
        parts = command.split(' and ')
        
        # Initialize JSON structure
        json_result = {
            "intent": "explain",  # default
            "entities": {}
        }
        
        # Determine primary intent based on command structure
        if any(part.startswith('filter ') for part in parts):
            # If there's filtering, determine if it's pure filter or filter+explain
            if any(part in ['explain features', 'explain cfe'] for part in parts):
                # Combined filter and explain - intent should be explain with filter entities
                json_result['intent'] = 'explain'
            else:
                # Pure filter
                json_result['intent'] = 'filter'
        elif any(part.startswith('important') for part in parts):
            json_result['intent'] = 'important'
        elif 'likelihood' in parts:
            json_result['intent'] = 'likelihood'
        elif 'explain cfe' in parts:
            json_result['intent'] = 'counterfactual'
        
        # Process each part
        for part in parts:
            part = part.strip()
            self._process_command_part(part, json_result)
        
        return json_result
    
    def _process_command_part(self, part: str, json_result: Dict[str, Any]) -> None:
        """Process a single part of the command."""
        part = part.strip()
        
        # Filter commands
        if part.startswith('filter '):
            filter_content = part[7:]  # Remove 'filter '
            
            # Check for ID filtering
            if filter_content.startswith('id '):
                patient_id = int(filter_content[3:])
                json_result['intent'] = 'filter'
                json_result['entities']['patient_id'] = patient_id
                
            # Check for feature filtering (e.g., "glucose greater than 120")
            elif ' greater than ' in filter_content or ' less than ' in filter_content:
                self._parse_feature_filter(filter_content, json_result)
                
        # Explain commands
        elif part == 'explain features':
            json_result['intent'] = 'explain'
            
        elif part == 'explain cfe':
            json_result['intent'] = 'counterfactual'
            
        # Important commands
        elif part.startswith('important'):
            json_result['intent'] = 'important'
            if 'topk' in part:
                # Extract number (e.g., "important topk 3")
                match = re.search(r'topk (\d+)', part)
                if match:
                    json_result['entities']['topk'] = int(match.group(1))
        
        # Likelihood commands
        elif part == 'likelihood':
            json_result['intent'] = 'likelihood'
    
    def _parse_feature_filter(self, filter_content: str, json_result: Dict[str, Any]) -> None:
        """Parse feature filtering expressions."""
        json_result['intent'] = 'filter'
        
        # Parse "feature operator value" pattern
        if ' greater than ' in filter_content:
            feature, value_str = filter_content.split(' greater than ')
            operator = '>'
        elif ' less than ' in filter_content:
            feature, value_str = filter_content.split(' less than ')
            operator = '<'
        else:
            return
        
        # Extract feature name and value
        feature = feature.strip()
        try:
            value = float(value_str.strip())
            if value.is_integer():
                value = int(value)
        except ValueError:
            value = value_str.strip()
        
        # Add to entities
        json_result['entities']['features'] = [feature]
        json_result['entities']['operators'] = [operator]
        json_result['entities']['values'] = [value]
    
    def save_converted_cases(self, output_file: str) -> None:
        """Save converted test cases to a JSON file."""
        converted_cases = self.convert_to_json_format()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_cases, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(converted_cases)} converted test cases to {output_file}")
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversion process."""
        if not self.test_cases:
            self.parse_test_file()
        
        converted_cases = self.convert_to_json_format()
        
        # Count intents
        intent_counts = {}
        for case in converted_cases:
            intent = case['expected_json']['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Count entities
        entity_types = set()
        for case in converted_cases:
            entities = case['expected_json']['entities']
            entity_types.update(entities.keys())
        
        return {
            'total_cases': len(converted_cases),
            'intent_distribution': intent_counts,
            'entity_types': sorted(list(entity_types)),
            'sample_cases': converted_cases[:5]  # First 5 cases for review
        }


def main():
    """Main function to demonstrate the parser."""
    parser = TestSuiteParser('/mnt/e/Talktomodel_adjusted/parsing_accuracy/diabetes_test_suite.txt')
    
    # Parse and convert
    stats = parser.get_conversion_stats()
    print("Conversion Statistics:")
    print(f"Total cases: {stats['total_cases']}")
    print(f"Intent distribution: {stats['intent_distribution']}")
    print(f"Entity types: {stats['entity_types']}")
    
    # Save converted cases
    parser.save_converted_cases('/mnt/e/Talktomodel_adjusted/parsing_accuracy/converted_test_cases.json')
    
    # Show sample cases
    print("\nSample converted cases:")
    for i, case in enumerate(stats['sample_cases'], 1):
        print(f"{i}. Question: {case['question']}")
        print(f"   Old: {case['old_command']}")
        print(f"   New: {case['expected_json']}")
        print()


if __name__ == "__main__":
    main()