#!/usr/bin/env python3
"""
Simple TalkToModel: Natural Language → AutoGen → Actions → LLM Formatter
Clean 3-component architecture as originally envisioned
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-4o-mini')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required!")

# Initialize components
logger.info("🚀 Initializing simple 3-component architecture...")

# Component 1: AutoGen decoder for intent parsing
# max_rounds=4 enables 3-agent collaboration without overthinking (Intent → Action → Validation → Intent)
decoder = AutoGenDecoder(api_key=OPENAI_API_KEY, model=GPT_MODEL, max_rounds=4)

# Load dataset and model
logger.info("📊 Loading dataset and model...")
dataset = pd.read_csv('data/diabetes.csv')
with open('models/diabetes_model_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Component 2: Action dispatcher for explainability
action_dispatcher = SimpleActionDispatcher(dataset, model)

# Component 3: LLM formatter for natural language output
formatter = LLMFormatter(OPENAI_API_KEY, GPT_MODEL)

# Initialize conversation
conversation = Conversation(dataset, model)

# Create Flask app
app = create_app()

# Initialize routes with dependencies
init_routes(conversation, action_dispatcher, formatter, dataset)

logger.info(f"✅ Ready! Dataset: {len(dataset)} instances, Model: {type(model).__name__}")

if __name__ == "__main__":
    print("\n🤖 TalkToModel - Simple Architecture")
    print("Natural Language → AutoGen → Actions → LLM Formatter")
    print("Available at: http://localhost:5000")
    print("API: POST /query with JSON: {'query': 'your question'}")
    app.run(host='0.0.0.0', port=5000, debug=True)