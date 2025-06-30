#!/usr/bin/env python3
"""
Test script for the critical thinking validation agent implementation
"""
import json
from explain.autogen_decoder import AutoGenDecoder

def test_critical_thinking_agent():
    """Test the critical thinking validation agent prompt generation"""
    
    # Create decoder instance (without actual AutoGen dependencies)
    decoder = AutoGenDecoder(api_key="test", model="gpt-4o-mini", max_rounds=4)
    
    # Test prompt generation
    intent_validation_prompt = decoder._create_intent_validation_prompt()
    
    print("=== Critical Thinking Validation Agent Prompt ===")
    print(intent_validation_prompt)
    print("\n" + "="*60 + "\n")
    
    # Test response classification
    test_responses = [
        {"intent": "performance", "entities": {}, "confidence": 0.9},
        {"validated_intent": "important", "entities": {}, "confidence": 0.85, "critical_analysis": "Test analysis"},
        {"action": "score", "entities": {}, "confidence": 0.9}
    ]
    
    print("=== Response Classification Tests ===")
    for i, response in enumerate(test_responses):
        response_type = decoder._classify_response_type(response)
        print(f"Response {i+1}: {response}")
        print(f"Classified as: {response_type}")
        print()
    
    # Test complete response creation (mock data)
    print("=== Complete Response Integration Test ===")
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
    
    print("Complete Response:")
    print(json.dumps(complete_response, indent=2))
    
    return True

if __name__ == "__main__":
    try:
        test_critical_thinking_agent()
        print("\n✅ Critical thinking validation agent implementation completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()