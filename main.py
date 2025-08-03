#!/usr/bin/env python3
"""
AutoConvXAI: Interactive AI explanations through multi-agent natural language processing.

Architecture: Natural Language → AutoGen → Actions → LLM Formatter
"""
import os
import logging
import pandas as pd
import pickle
from app import create_app
from app.conversation import Conversation
from app.core import SimpleActionDispatcher
from app.routes import init_routes
from nlu.autogen_decoder import AutoGenDecoder
from formatter.llm_formatter import LLMFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-4o-mini')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required!")

logger.info("Initializing 3-component architecture...")

# Component 1: Multi-agent intent parsing
decoder = AutoGenDecoder(api_key=OPENAI_API_KEY, model=GPT_MODEL, max_rounds=4)

# Load dataset and model
logger.info("Loading dataset and model...")
dataset = pd.read_csv('data/diabetes.csv')
with open('models/diabetes_model_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Component 2: Action dispatcher
action_dispatcher = SimpleActionDispatcher(dataset, model)

# Component 3: Response formatter
formatter = LLMFormatter(OPENAI_API_KEY, GPT_MODEL)

conversation = Conversation(dataset, model)
app = create_app()
init_routes(conversation, action_dispatcher, formatter, dataset)

logger.info(f"Ready! Dataset: {len(dataset)} instances, Model: {type(model).__name__}")

if __name__ == "__main__":
    print("\nAutoConvXAI - Interactive AI Explanations")
    print("Architecture: Natural Language → AutoGen → Actions → LLM Formatter")
    print("Web interface: http://localhost:5000")
    print("API endpoint: POST /query")
    app.run(host='0.0.0.0', port=5000, debug=True)