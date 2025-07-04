"""
Evaluation Framework for Testing AutoGen vs Old System

This framework provides tools for comparing the new AutoGen-based intent parsing
with the old SQL-like system. Run this after installing AutoGen dependencies.

Dependencies needed:
pip install autogen-agentchat[openai]

Usage:
1. Set OPENAI_API_KEY environment variable
2. Run: python evaluation_framework.py
"""

import json
import os
from typing import Dict, List, Any


def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """Load converted test cases from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_test_distribution(test_cases: List[Dict[str, Any]]) -> None:
    """Analyze the distribution of intents in test cases."""
    intent_counts = {}
    entity_types = set()
    
    for case in test_cases:
        intent = case['expected_json']['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        entities = case['expected_json'].get('entities', {})
        entity_types.update(entities.keys())
    
    print("Test Case Distribution:")
    print(f"Total cases: {len(test_cases)}")
    print(f"Intent distribution: {intent_counts}")
    print(f"Entity types found: {sorted(list(entity_types))}")
    
    # Show sample cases for each intent
    print("\nSample cases by intent:")
    intent_samples = {}
    for case in test_cases:
        intent = case['expected_json']['intent']
        if intent not in intent_samples:
            intent_samples[intent] = []
        if len(intent_samples[intent]) < 2:  # Show max 2 samples per intent
            intent_samples[intent].append(case)
    
    for intent, samples in intent_samples.items():
        print(f"\n{intent.upper()}:")
        for sample in samples:
            print(f"  Q: {sample['question'][:60]}...")
            print(f"  Expected: {sample['expected_json']}")


def compare_with_autogen(question: str) -> Dict[str, Any]:
    """
    Template function for comparing with AutoGen.
    Replace this with actual AutoGen integration once dependencies are installed.
    """
    # This would be replaced with actual AutoGen call:
    # from explain.decoders.autogen_decoder import AutoGenDecoder
    # decoder = AutoGenDecoder(api_key=os.getenv('OPENAI_API_KEY'))
    # result = decoder.complete_sync(question, conversation)
    
    # For now, return a mock result
    return {
        "message": "AutoGen dependencies not installed",
        "instructions": [
            "Install AutoGen: pip install autogen-agentchat[openai]",
            "Set API key: export OPENAI_API_KEY=your_key_here",
            "Re-run this script to test actual performance"
        ]
    }


def manual_evaluation_sample() -> None:
    """Show a few test cases for manual evaluation."""
    test_cases = load_test_cases('/mnt/e/Talktomodel_adjusted/parsing_accuracy/converted_test_cases.json')
    
    print("\nManual Evaluation Sample (first 5 cases):")
    print("="*60)
    
    for i, case in enumerate(test_cases[:5], 1):
        print(f"\n{i}. Question: {case['question']}")
        print(f"   Old Command: {case['old_command']}")
        print(f"   Expected JSON: {case['expected_json']}")
        
        # Show what AutoGen would need to produce
        expected = case['expected_json']
        print(f"   AutoGen should produce:")
        print(f"     Intent: {expected['intent']}")
        if expected.get('entities'):
            print(f"     Entities: {expected['entities']}")
        
        print(f"   Test: Does AutoGen output match this expectation?")
        print("   " + "-"*50)


def create_evaluation_instructions() -> None:
    """Create instructions for running the full evaluation."""
    instructions = """
EVALUATION SETUP INSTRUCTIONS
=============================

1. INSTALL DEPENDENCIES:
   pip install autogen-agentchat[openai]

2. SET API KEY:
   export OPENAI_API_KEY=your_openai_api_key_here

3. RUN EVALUATION:
   python parsing_accuracy/autogen_evaluator.py

4. REVIEW RESULTS:
   - Check parsing_accuracy/evaluation_results.json
   - Compare intent accuracy with old system (was ~76%)
   - Analyze which types of queries work better/worse

5. EXPECTED METRICS TO TRACK:
   - Intent accuracy (% of intents correctly identified)
   - Entity accuracy (% of entities correctly extracted)
   - Overall accuracy (% of complete matches)
   - Performance by intent type (filter, explain, important, etc.)

6. COMPARISON WITH OLD SYSTEM:
   - Old system: ~76% accuracy
   - Goal: Match or exceed this with better entity extraction
   - Focus areas: Complex filtering, counterfactual queries

7. ANALYSIS AREAS:
   - Which question types are hardest to parse?
   - Are there patterns in misclassifications?
   - How does performance vary by intent type?
   - What improvements could be made to prompts?

The conversion has prepared 191 test cases covering:
- explain: 55 cases (filtered explanations)
- filter: 87 cases (data filtering)
- likelihood: 29 cases (prediction confidence)
- important: 9 cases (feature importance)
- counterfactual: 11 cases (what-if analysis)
"""
    
    with open('/mnt/e/Talktomodel_adjusted/parsing_accuracy/EVALUATION_INSTRUCTIONS.md', 'w') as f:
        f.write(instructions)
    
    print("Created EVALUATION_INSTRUCTIONS.md with setup details")


def main():
    """Main function to analyze converted test cases and prepare evaluation."""
    test_cases_file = '/mnt/e/Talktomodel_adjusted/parsing_accuracy/converted_test_cases.json'
    
    if not os.path.exists(test_cases_file):
        print(f"Error: {test_cases_file} not found")
        print("Run test_suite_parser.py first to generate converted test cases")
        return
    
    # Load and analyze test cases
    test_cases = load_test_cases(test_cases_file)
    analyze_test_distribution(test_cases)
    
    # Show manual evaluation sample
    manual_evaluation_sample()
    
    # Test AutoGen integration (will show mock result without dependencies)
    print("\nTesting AutoGen Integration:")
    result = compare_with_autogen("What reasoning do you typically use to determine if people have diabetes when glucose levels are over 120?")
    print(json.dumps(result, indent=2))
    
    # Create evaluation instructions
    create_evaluation_instructions()
    
    print("\n" + "="*60)
    print("SUMMARY: Test Conversion Complete!")
    print("="*60)
    print(f"✓ Converted {len(test_cases)} test cases from old format to JSON")
    print("✓ Created evaluation framework for AutoGen testing")
    print("✓ Generated setup instructions for full evaluation")
    print("\nNext Steps:")
    print("1. Install AutoGen dependencies (see EVALUATION_INSTRUCTIONS.md)")
    print("2. Set OpenAI API key")
    print("3. Run autogen_evaluator.py to test actual performance")
    print("4. Compare results with old system's ~76% accuracy")


if __name__ == "__main__":
    main()