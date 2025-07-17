#!/usr/bin/env python3
"""Run full AutoGen evaluation on all 191 test cases."""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the evaluation
from parsing_accuracy.autogen_evaluator import evaluate_all_cases

if __name__ == "__main__":
    print("Starting full evaluation on all 191 test cases...")
    print("This will take approximately 1.5-2 minutes...")
    evaluate_all_cases()