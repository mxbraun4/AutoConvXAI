#!/usr/bin/env python3
"""
Simple test for critical thinking validation logic without AutoGen dependencies
"""
import json
import sys
import os

# Add the project root to path to import modules
sys.path.insert(0, '/mnt/e/TalkToModel')

def test_critical_thinking_logic():
    """Test the critical thinking validation logic"""
    
    # Import the class but don't instantiate it
    from explain.autogen_decoder import AutoGenDecoder
    
    # Test response classification logic directly
    print("=== Testing Response Classification Logic ===")
    
    # Create a dummy instance to access methods (this will fail but we can test the logic)
    try:
        # This will fail, but we want to test the methods
        decoder = AutoGenDecoder.__new__(AutoGenDecoder)
        
        # Test response classification
        test_responses = [
            {"intent": "performance", "entities": {}, "confidence": 0.9},
            {"validated_intent": "important", "entities": {}, "confidence": 0.85},
            {"action": "score", "entities": {}, "confidence": 0.9}
        ]
        
        for i, response in enumerate(test_responses):
            response_type = decoder._classify_response_type(response)
            print(f"Response {i+1}: Type = {response_type}")
            print(f"  Input: {response}")
            print()
        
        # Test complete response integration
        print("=== Testing Complete Response Integration ===")
        
        intent_response = {"intent": "performance", "entities": {"features": ["age"]}, "confidence": 0.9}
        intent_validation_response = {
            "validated_intent": "important", 
            "entities": {"features": ["age"]}, 
            "confidence": 0.85,
            "critical_analysis": "This could mean feature importance rather than performance",
            "alternative_interpretations": ["statistics", "interactions"]
        }
        action_response = {"action": "important", "entities": {"features": ["age"]}, "confidence": 0.9}
        
        complete_response = decoder._create_complete_response(
            intent_response, intent_validation_response, action_response
        )
        
        print("Complete Response Structure:")
        for key, value in complete_response.items():
            print(f"  {key}: {value}")
        
        print("\n=== Key Integration Points ===")
        print(f"Original Intent: {intent_response['intent']}")
        print(f"Validated Intent: {intent_validation_response['validated_intent']}")
        print(f"Final Action: {action_response['action']}")
        print(f"Used Intent for Reset Logic: {complete_response['intent_response']['intent']}")
        print(f"Critical Analysis: {intent_validation_response['critical_analysis']}")
        
        return True
        
    except Exception as e:
        print(f"Error in test: {e}")
        return False

if __name__ == "__main__":
    success = test_critical_thinking_logic()
    if success:
        print("\n✅ Critical thinking validation logic test completed!")
    else:
        print("\n❌ Test failed!")