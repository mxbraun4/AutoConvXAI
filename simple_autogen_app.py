#!/usr/bin/env python3
"""
Simple AutoGen-only Flask application for TalkToModel
"""
import os
import json
import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the AutoGen decoder
from explain.autogen_decoder import AutoGenDecoder

app = Flask(__name__)

# Get configuration from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-4o')

# Check API key
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required!")

# Initialize the AutoGen decoder
print("🚀 Initializing AutoGen decoder...")
decoder = AutoGenDecoder(
    api_key=OPENAI_API_KEY,
    model=GPT_MODEL,
    max_rounds=3
)
print("✅ AutoGen decoder ready!")

@app.route('/')
def home():
    """Simple home page."""
    return """
    <h1>AutoGen TalkToModel</h1>
    <p>Simple AutoGen-only interface for natural language queries.</p>
    <form action="/query" method="post">
        <textarea name="query" placeholder="Enter your query..." rows="4" cols="50"></textarea><br>
        <button type="submit">Submit Query</button>
    </form>
    """

@app.route('/query', methods=['POST'])
def process_query():
    """Process query using AutoGen decoder."""
    try:
        if request.method == "POST":
            # Get query from form or JSON
            if request.is_json:
                data = request.get_json()
                user_query = data.get("query", "")
            else:
                user_query = request.form.get("query", "")
            
            if not user_query:
                return jsonify({"error": "No query provided"}), 400
            
            logger.info(f"Processing query: {user_query}")
            
            # Process with AutoGen
            result = decoder.complete_sync(user_query, conversation=None)
            
            logger.info(f"Result: {result}")
            
            # Return result
            if request.is_json:
                return jsonify(result)
            else:
                return f"<h2>Query:</h2><p>{user_query}</p><h2>Result:</h2><pre>{json.dumps(result, indent=2)}</pre><br><a href='/'>Back</a>"
                
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_msg = f"Error: {str(e)}"
        if request.is_json:
            return jsonify({"error": error_msg}), 500
        else:
            return f"<h2>Error:</h2><p>{error_msg}</p><br><a href='/'>Back</a>"

if __name__ == "__main__":
    print("\n🤖 Simple AutoGen-only TalkToModel")
    print("Available at: http://localhost:5000")
    print("API endpoint: POST /query with JSON: {'query': 'your question'}")
    app.run(host='0.0.0.0', port=5000, debug=True)