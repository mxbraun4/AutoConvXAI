#!/usr/bin/env python3
"""Test script to see what validation agent is actually doing."""

import os
import sys
import json

# Add path for imports  
sys.path.append('/app')

from nlu.autogen_decoder import AutoGenDecoder
from evaluation.parsing_accuracy.autogen_evaluator import MockConversation

def test_validation_agent():
    """Test validation agent with a query that should trigger entity corrections."""
    
    # Test query with clear operator that should be converted
    test_query = "What method would you use to figure out if people with glucose levels over 120 have diabetes?"
    
    print(f"Testing query: {test_query}")
    print("=" * 80)
    
    # Initialize decoder
    api_key = os.getenv('OPENAI_API_KEY')
    decoder = AutoGenDecoder(api_key=api_key, model="gpt-4o-mini")
    
    # Create mock conversation
    conversation = MockConversation()
    
    # Process the query
    result = decoder.complete_sync(test_query, conversation)
    
    print("RESULT:")
    print(json.dumps(result, indent=2))
    print()
    
    # Check what each agent produced
    if 'action_response' in result:
        print("FIRST AGENT (ActionExtractor) entities:")
        print(json.dumps(result['action_response']['entities'], indent=2))
        print()
    
    if 'command_structure' in result:
        print("FINAL entities (after validation):")
        print(json.dumps(result['command_structure'], indent=2))
        print()
        
    # Check if validation occurred
    original_entities = result.get('action_response', {}).get('entities', {})
    final_entities = result.get('command_structure', {})
    
    print("VALIDATION ANALYSIS:")
    print(f"Original features: {original_entities.get('features')}")
    print(f"Final features: {final_entities.get('features')}")
    print(f"Original operators: {original_entities.get('operators')}")
    print(f"Final operators: {final_entities.get('operators')}")
    print(f"Original values: {original_entities.get('values')}")
    print(f"Final values: {final_entities.get('values')}")
    
    # Check if any changes occurred
    changes = []
    for key in ['features', 'operators', 'values', 'filter_type']:
        if original_entities.get(key) != final_entities.get(key):
            changes.append(f"{key}: {original_entities.get(key)} → {final_entities.get(key)}")
    
    if changes:
        print(f"\nVALIDATION CHANGES DETECTED:")
        for change in changes:
            print(f"  - {change}")
    else:
        print(f"\nNO VALIDATION CHANGES - This is the problem!")
        print("Expected: features=['Glucose'], operators=['>'], values=[120]")

if __name__ == "__main__":
    test_validation_agent()