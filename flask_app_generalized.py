"""Flask app using Generalized Dynamic Code Generation System."""
import json
import logging
from logging.config import dictConfig
import os

from flask import Flask, render_template, request, Blueprint

from explain.generalized_logic import GeneralizedExplainBot

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

# Initialize the generalized bot with Dynamic Code Generation
print(f"🚀 Initializing TalkToModel with Generalized Dynamic Code Generation...")
print(f"⚡ ENABLING TRULY GENERALIZABLE QUERY PROCESSING!")
BOT = GeneralizedExplainBot(
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

print(f"✅ Generalized TalkToModel ready with Dynamic Code Generation!")

# Check for optional dependencies
try:
    from explain.explanation import DICE_ML_AVAILABLE
    if not DICE_ML_AVAILABLE:
        print(f"⚠️  dice_ml not available - counterfactual explanations disabled")
        print(f"   💡 For full features, build with: docker build -t ttm-gpt4-full --target full .")
except ImportError:
    pass

print(f"🎯 Generalized System Benefits:")
print(f"   🔧 Dynamic code generation for ANY query combination")
print(f"   🤖 AutoGen-powered intelligent extraction")
print(f"   ⚡ No predefined action strings - truly generalizable")
print(f"   🎛️ Supports unlimited filters, operators, and operations")
print(f"   🔀 Automatically optimized pandas/numpy execution")
print(f"   💻 Generated code is human-readable and debuggable")


@bp.route('/')
def home():
    """Load the explanation interface."""
    app.logger.info("Loaded interface with generalized dynamic code generation backend")
    objective = BOT.conversation.describe.get_dataset_objective()
    return render_template("index.html", 
                         currentUserId="user", 
                         datasetObjective=objective)


@bp.route("/get_response", methods=['POST'])
def get_bot_response():
    """Get bot response using Generalized Dynamic Code Generation."""
    if request.method == "POST":
        app.logger.info("Generating response with dynamic code generation...")
        try:
            data = json.loads(request.data)
            user_text = data["userInput"]
            
            # 🚀 GENERALIZED SYSTEM - Dynamic code generation for ANY query!
            response = BOT.update_state(user_text, BOT.conversation)
            
            app.logger.info("✅ Generalized response generated successfully")
            return response
            
        except Exception as e:
            app.logger.error(f"Error with generalized system: {e}")
            return "I encountered an error processing your question. Please try rephrasing it."


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
        "system": "generalized_dynamic_code_generation"
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
            "system": "generalized_dynamic_code_generation"
        })
    
    return ""


# Register blueprint
app.register_blueprint(bp, url_prefix=BASE_URL)


# Example usage and comparison
if __name__ == "__main__":
    print("\n🤖 Generalized Dynamic Code Generation Examples:")
    print("=" * 60)
    print("📊 ANY Statistical Query:")
    print("  'average age for diabetic patients' → Generates pandas code to filter and calculate")
    print("  'statistics for women with BMI > 30' → Dynamically creates filter + stats code")
    print()
    print("🎯 ANY Performance Query:")
    print("  'accuracy for patients over 50 with glucose > 140' → Multi-filter accuracy code")
    print("  'how well does model perform on young patients' → Age filter + performance code")
    print()
    print("🔮 ANY Prediction Query:")
    print("  'predict for patients with age > 45 and BMI > 28' → Multi-filter prediction code")
    print("  'what would model predict for pregnant women' → Pregnancy filter + prediction")
    print()
    print("🧠 Generated Code Example:")
    print("Instead of static 'filter age greater 50 score accuracy' strings...")
    print("We generate executable Python:")
    print("""
    def execute_query(dataset, model, explainer, conversation):
        filtered_dataset = dataset[dataset['age'] > 50]
        filtered_dataset = filtered_dataset[filtered_dataset['glucose'] > 140]
        accuracy = model.score(filtered_dataset.drop('y', axis=1), filtered_dataset['y'])
        return f'Accuracy (age>50, glucose>140): {accuracy:.2f}%'
    """)
    print()
    print("🎉 Benefits:")
    print("✅ Handles UNLIMITED filter combinations")
    print("✅ Supports ANY operation type")
    print("✅ Automatically optimized execution")
    print("✅ Human-readable generated code")
    print("✅ Easy to extend with new operations")
    print("✅ No hardcoded action mappings needed")
    
    # Run Flask app on all interfaces for Docker
    app.run(host='0.0.0.0', port=4455, debug=False)