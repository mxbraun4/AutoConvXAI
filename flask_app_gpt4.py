"""Clean Flask app using AutoGen Multi-Agent system for TalkToModel."""
import json
import logging
from logging.config import dictConfig
import os

from flask import Flask, render_template, request, Blueprint

from explain.enhanced_logic import EnhancedExplainBot


# Set up logging
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)
bp = Blueprint('host', __name__, template_folder='templates')

# Get configuration from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL = os.getenv('BASE_URL', '/')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-4o')

# Check API key before initializing
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required!")

# Initialize the enhanced bot with AutoGen multi-agent system
print(f"🚀 Initializing TalkToModel with AutoGen Multi-Agent System...")
BOT = EnhancedExplainBot(
    model_file_path="data/diabetes_model_logistic_regression.pkl",
    dataset_file_path="data/diabetes.csv", 
    background_dataset_file_path="data/diabetes.csv",
    dataset_index_column=None,
    target_variable_name="y",
    categorical_features=["pregnancies"],
    numerical_features=["glucose", "bloodpressure", "skinthickness", "insulin", "bmi", "diabetespedigreefunction", "age"],
    remove_underscores=True,
    name="diabetes",
    openai_api_key=OPENAI_API_KEY,
    gpt_model=GPT_MODEL
)
print(f"✅ AutoGen Multi-Agent TalkToModel ready!")


@bp.route('/')
def home():
    """Load the explanation interface."""
    app.logger.info("Loaded interface with AutoGen multi-agent backend")
    objective = BOT.conversation.describe.get_dataset_objective()
    return render_template("index.html", 
                         currentUserId="user", 
                         datasetObjective=objective)


@bp.route("/get_response", methods=['POST'])
def get_bot_response():
    """Get bot response using AutoGen multi-agent system."""
    if request.method == "POST":
        app.logger.info("Generating AutoGen multi-agent response...")
        try:
            data = json.loads(request.data)
            user_text = data["userInput"]
            
            # 🤖 MULTI-AGENT SYSTEM - Specialized agents collaborate for best results!
            response = BOT.update_state(user_text, BOT.conversation)
            
            app.logger.info("✅ AutoGen multi-agent response generated successfully")
            return response
            
        except Exception as e:
            app.logger.error(f"Error with AutoGen multi-agent system: {e}")
            return "I encountered an error. Please try rephrasing your question."


@bp.route("/sample_prompt", methods=["POST"])
def sample_prompt():
    """Generate sample prompts for users."""
    data = json.loads(request.data)
    action = data["action"]
    username = data.get("thisUserName", "user")
    
    # Import here to avoid circular imports
    from explain.sample_prompts_by_action import sample_prompt_for_action
    
    prompt = sample_prompt_for_action(action,
                                      BOT.conversation.prompts.filename_to_prompt_id,
                                      BOT.conversation.prompts.final_prompt_set,
                                      real_ids=BOT.conversation.get_training_data_ids())
    
    app.logger.info(f"Generated sample prompt for action '{action}' for user '{username}'")
    
    BOT.log({
        "username": username,
        "requested_action_generation": action,
        "generated_prompt": prompt,
        "system": "autogen_multi_agent"
    })
    
    return prompt


@bp.route("/log_feedback", methods=['POST'])
def log_feedback():
    """Log user feedback."""
    feedback = request.data.decode("utf-8")
    app.logger.info(feedback)
    
    # Parse feedback (same as original)
    split_feedback = feedback.split(" || ")
    if len(split_feedback) >= 3:
        message_id = split_feedback[0][len("MessageID: "):]
        feedback_text = split_feedback[1][len("Feedback: "):]
        username = split_feedback[2][len("Username: "):]
        
        BOT.log({
            "id": message_id,
            "feedback_text": feedback_text,
            "username": username,
            "system": "autogen_multi_agent"
        })
    
    return ""


# Register blueprint
app.register_blueprint(bp, url_prefix=BASE_URL)


# Example usage and comparison
if __name__ == "__main__":
    print("\n🤖 AutoGen Multi-Agent Query Examples:")
    print("Before (complex parsing): 'um, can you like explain why patient 5 got diabetes?'")
    print("After (Multi-Agent):      Intent Agent → Action Agent → Validator → Perfect understanding")
    print()
    print("Before (single model): One model handles everything")
    print("After (Multi-Agent):   Specialized agents collaborate for better accuracy")
    print()
    print("Query complexity: ANY → Multi-Agent team handles it all with specialized expertise ✨")
    print("🎯 Intent Extraction Agent: Understands what you want")
    print("⚡ Action Planning Agent: Plans the right action")
    print("✅ Validation Agent: Ensures correctness")
    print("🤝 Coordinator Agent: Brings it all together")
    
    # Run Flask app on all interfaces for Docker
    app.run(host='0.0.0.0', port=4455, debug=False) 