#!/usr/bin/env python3
"""
Fix incorrect labels for statistical queries in the test dataset.
Changes 'explain' to 'statistic' for questions asking about mean/average.
"""

import json

# Load the test cases
with open('parsing_accuracy/converted_test_cases.json', 'r') as f:
    data = json.load(f)

# Track changes
changes_made = []

# Update the cases
for i, case in enumerate(data):
    question = case.get('question', '').lower()
    current_action = case.get('expected_json', {}).get('action')
    
    # Check if this is a statistical query incorrectly labeled as explain
    if current_action == 'explain' and ('average' in question or 'mean' in question):
        # Check if it's asking about statistical measures (not explaining methodology)
        if any(phrase in question for phrase in ['what age', 'what is the mean', "what's the mean", "what's the average"]):
            old_action = case['expected_json']['action']
            case['expected_json']['action'] = 'statistic'
            
            # Also ensure entities include the feature being asked about
            if 'age' in question.lower():
                case['expected_json']['entities'] = {'features': ['age']}
            
            changes_made.append({
                'line': case.get('line_number', i),
                'question': case['question'],
                'old_action': old_action,
                'new_action': 'statistic'
            })

# Save the updated test cases
with open('parsing_accuracy/converted_test_cases.json', 'w') as f:
    json.dump(data, f, indent=2)

# Print summary of changes
print(f"Fixed {len(changes_made)} test cases:")
for change in changes_made:
    print(f"  Line {change['line']}: '{change['question']}'")
    print(f"    Changed action from '{change['old_action']}' to '{change['new_action']}'")

print(f"\nUpdated test cases saved to parsing_accuracy/converted_test_cases.json")