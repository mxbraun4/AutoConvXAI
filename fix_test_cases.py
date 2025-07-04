#!/usr/bin/env python3
"""
Fix test cases by changing 'likelihood' intent to 'predict' to match the current system design.
"""

import json

def fix_test_cases():
    """Update test cases to change likelihood to predict."""
    
    # Load the test cases
    with open('parsing_accuracy/converted_test_cases.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # Count changes
    changes_made = 0
    
    # Update likelihood to predict
    for case in test_cases:
        if case['expected_json']['intent'] == 'likelihood':
            case['expected_json']['intent'] = 'predict'
            changes_made += 1
            print(f"Changed case {case['line_number']}: {case['question'][:60]}...")
    
    # Save the updated test cases
    with open('parsing_accuracy/converted_test_cases.json', 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    print(f"\nFixed {changes_made} test cases:")
    print(f"Changed 'likelihood' → 'predict' to match your system design")
    
    # Show the distribution after changes
    intent_counts = {}
    for case in test_cases:
        intent = case['expected_json']['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"\nUpdated intent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")
    
    print(f"\nTotal test cases: {len(test_cases)}")

if __name__ == "__main__":
    fix_test_cases()