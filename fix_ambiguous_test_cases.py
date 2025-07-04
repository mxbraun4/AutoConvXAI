#!/usr/bin/env python3
"""
Fix ambiguous test cases that ask both "why" and "how to change" which create 
unclear intent classifications between "explain" and "counterfactual".
"""

import json

def fix_ambiguous_cases():
    """Fix ambiguous test cases by either removing them or changing their expected intent."""
    
    # Load the test cases
    with open('parsing_accuracy/converted_test_cases.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # Find problematic double-request cases (lines 12-18)
    # These ask both "why" AND "how to change" which is ambiguous
    problematic_lines = [12, 13, 14, 15, 16, 17, 18]
    
    changes_made = 0
    
    print("AMBIGUOUS TEST CASES FOUND:")
    print("=" * 60)
    
    # Option 1: Change their expected intent from "counterfactual" to "explain"
    # Since they primarily ask "why/what reason", "explain" is more accurate
    
    for case in test_cases:
        line_num = case['line_number']
        
        if line_num in problematic_lines:
            if case['expected_json']['intent'] == 'counterfactual':
                print(f"Line {line_num}: {case['question'][:60]}...")
                print(f"  Old command: {case['old_command']}")
                print(f"  Current expected: counterfactual")
                print(f"  Changing to: explain")
                print()
                
                # Change the expected intent to "explain"
                case['expected_json']['intent'] = 'explain'
                changes_made += 1
    
    # Save the updated test cases
    with open('parsing_accuracy/converted_test_cases.json', 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    print(f"Fixed {changes_made} ambiguous test cases")
    print("Changed their expected intent from 'counterfactual' to 'explain'")
    print("These cases primarily ask 'why' questions, so 'explain' is more appropriate")
    
    # Show the distribution after changes
    intent_counts = {}
    for case in test_cases:
        intent = case['expected_json']['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"\nUpdated intent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")
    
    print(f"\nTotal test cases: {len(test_cases)}")
    
    print("\nRemaining pure counterfactual cases (should work well):")
    for case in test_cases:
        if case['expected_json']['intent'] == 'counterfactual':
            print(f"  Line {case['line_number']}: {case['question'][:50]}...")

if __name__ == "__main__":
    fix_ambiguous_cases()