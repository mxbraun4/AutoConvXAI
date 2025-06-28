#!/usr/bin/env python3
"""Debug script to test AutoGen query parsing."""

import os
import sys
import json

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

def test_autogen_parsing():
    """Test AutoGen parsing with the problematic queries."""
    
    # Mock conversation object for testing
    class MockConversation:
        def get_var(self, var_name):
            if var_name == 'dataset':
                return type('Dataset', (), {
                    'contents': {
                        'X': type('DataFrame', (), {
                            'columns': ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'diabetespedigreefunction', 'age']
                        })()
                    }
                })()
            return None
        
        @property
        def class_names(self):
            return ['No Diabetes', 'Diabetes']
        
        @property
        def temp_dataset(self):
            return None
    
    # Test queries that are failing
    test_queries = [
        "how accurate is the model on instances with age > 50",
        "what is the average age in the dataset?",
        "what is the model's accuracy for patients over 50?",
        "show me the mean age",
        "how well does the model perform on older patients?"
    ]
    
    print("🧪 Testing AutoGen query parsing\n")
    
    # Check if we can import AutoGen decoder
    try:
        from explain.autogen_decoder import AutoGenDecoder, AUTOGEN_AVAILABLE
        print(f"✅ AutoGen import successful (available: {AUTOGEN_AVAILABLE})")
        
        if not AUTOGEN_AVAILABLE:
            print("❌ AutoGen not available - stopping test")
            return
            
    except ImportError as e:
        print(f"❌ AutoGen import failed: {e}")
        return
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY found - skipping live tests")
        print("Set OPENAI_API_KEY environment variable to test AutoGen parsing")
        return
    
    try:
        # Initialize decoder
        decoder = AutoGenDecoder(api_key=api_key, model="gpt-4o", max_rounds=3)
        print("✅ AutoGen decoder initialized\n")
        
        # Test each query
        mock_conversation = MockConversation()
        
        for i, query in enumerate(test_queries, 1):
            print(f"🔍 Test {i}: {query}")
            try:
                response = decoder.complete_sync(query, mock_conversation)
                
                print(f"   Response type: {type(response)}")
                if isinstance(response, dict):
                    print(f"   Keys: {list(response.keys())}")
                    
                    if "generation" in response:
                        print(f"   ✅ Generated action: {response['generation']}")
                    elif "direct_response" in response:
                        print(f"   💬 Direct response: {response['direct_response'][:100]}...")
                    else:
                        print(f"   ⚠️ Unexpected response format: {response}")
                        
                    if "confidence" in response:
                        print(f"   📊 Confidence: {response['confidence']}")
                        
                else:
                    print(f"   ❌ Non-dict response: {response}")
                    
            except Exception as e:
                print(f"   💥 Error: {e}")
            
            print()
    
    except Exception as e:
        print(f"❌ Failed to initialize AutoGen decoder: {e}")

if __name__ == "__main__":
    test_autogen_parsing()