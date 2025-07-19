#!/usr/bin/env python3
"""Test script to verify consensus termination with detailed logging."""

import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to show ALL messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the decoder
from nlu.autogen_decoder import AutoGenDecoder

# Mock conversation for testing
class MockConversation:
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
        return None

# Test cases that should trigger consensus quickly
test_queries = [
    "Show me patients with age > 50",
    "What are the important features?",
    "Predict for patient 5"
]

print("Testing consensus termination mechanism...")
print("=" * 60)

decoder = AutoGenDecoder()
conversation = MockConversation()

for i, query in enumerate(test_queries, 1):
    print(f"\nTest {i}: {query}")
    print("-" * 40)
    
    try:
        result = decoder.decode_to_json(query, conversation)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 40)

print("\nTest complete! Check the logs above for consensus detection.")