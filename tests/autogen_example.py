#!/usr/bin/env python3
"""
AutoGen Example Usage

This script demonstrates how to use the AutoGen decoder for natural language understanding.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explain.autogen_decoder import AutoGenNaturalLanguageUnderstanding

def main():
    """Example usage of AutoGen decoder"""
    
    # Initialize the AutoGen decoder
    print("Initializing AutoGen Natural Language Understanding...")
    decoder = AutoGenNaturalLanguageUnderstanding()
    
    # Example queries
    queries = [
        "What is the most important feature?",
        "Show me the data summary",
        "Explain the prediction for the first sample"
    ]
    
    # Process each query
    for query in queries:
        print(f"\n--- Processing: '{query}' ---")
        try:
            result = decoder.decode_user_input(query, context={})
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()