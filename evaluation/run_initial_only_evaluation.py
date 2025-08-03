#!/usr/bin/env python3
"""Run AutoGen evaluation on all test cases using ONLY initial extraction (first agent output)."""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the evaluation
from parsing_accuracy.autogen_evaluator import evaluate_all_cases_initial_only

if __name__ == "__main__":
    print("Starting initial extraction only evaluation on all test cases...")
    print("This evaluates ONLY the first agent's output before validation...")
    evaluate_all_cases_initial_only()