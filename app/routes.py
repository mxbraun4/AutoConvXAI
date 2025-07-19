"""Flask routes for the TalkToModel application"""
import logging
import random
from flask import Blueprint, request, jsonify, render_template, Response
from app.conversation import Conversation
from app.core import SimpleActionDispatcher, process_user_query

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

# This will be initialized from main.py
conversation = None
action_dispatcher = None
formatter = None
dataset = None

def init_routes(conv, dispatcher, fmt, data):
    """Initialize route dependencies"""
    global conversation, action_dispatcher, formatter, dataset
    conversation = conv
    action_dispatcher = dispatcher
    formatter = fmt
    dataset = data

@main_bp.route('/')
def home():
    """Render the main conversational interface"""
    return render_template('index.html', 
                         datasetObjective="predict diabetes based on patient health metrics",
                         dataset_size=len(dataset),
                         action_count=len(action_dispatcher.actions))

@main_bp.route('/query', methods=['POST'])
def process_query():
    """Handle API queries - returns JSON response"""
    try:
        # Get query
        if request.is_json:
            user_query = request.get_json().get('query', '')
        else:
            user_query = request.form.get('query', '')
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        
        logger.info(f"API query: {user_query}")
        
        action_name, action_result, formatted_response = process_user_query(
            user_query, conversation, action_dispatcher, formatter
        )
        
        logger.info(f"API response generated using action: {action_name}")
        
        return jsonify({
            "response": formatted_response,
            "action_used": action_name,
            "raw_result": str(action_result)
        })
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/sample_prompt', methods=['POST'])
def sample_prompt():
    """Generate sample prompts for the suggestion buttons"""
    try:
        # Get action from JSON or form data
        data = request.get_json() if request.is_json else {}
        action = data.get('action', '') if data else request.form.get('action', 'important')
        
        # Simple sample prompts for different categories
        samples = {
            'data': [
                "Tell me about this dataset",
                "How many patients are in the dataset?",
                "What's the average age of patients?"
            ],
            'filter': [
                "Show me patients with age > 50",
                "Filter to patients with BMI > 30",
                "Show instances where glucose > 120"
            ],
            'important': [
                "What are the most important features for predicting diabetes?",
                "Which features have the strongest influence on the model?",
                "Show me feature importance rankings"
            ],
            'explain': [
                "Why did the model predict diabetes for patient 5?",
                "Explain the prediction for this patient",
                "What factors led to this prediction?"
            ],
            'predict': [
                "Predict diabetes for age=50, BMI=30, glucose=120",
                "What's the prediction for BMI=35, glucose=150?",
                "Make a prediction for these values"
            ],
            'likelihood': [
                "How confident is the model in this prediction?",
                "What's the prediction probability?",
                "Show me the likelihood scores"
            ],
            'change': [
                "What if the patient's BMI was 25 instead?",
                "Change glucose to 90 and show the prediction",
                "What would happen if age was 30?"
            ],
            'mistakes': [
                "What are the model's biggest mistakes?",
                "Show me cases where the model was wrong",
                "Which predictions were incorrect?"
            ],
            'interact': [
                "How do age and BMI interact with each other?",
                "What are the feature interaction effects?",
                "Which features work together?"
            ],
            'statistics': [
                "What are the glucose statistics?",
                "Show me BMI distribution",
                "Give me detailed stats for age"
            ],
            'statistic': [
                "Show me glucose statistics",
                "What's the mean and standard deviation for BMI?",
                "Give me statistical summary for age"
            ],
            'count': [
                "How many patients are in this dataset?",
                "Count the number of diabetes cases",
                "How many instances are there?"
            ],
            'define': [
                "What does BMI mean?",
                "Define glucose levels",
                "Explain what DiabetesPedigreeFunction is"
            ],
            'self': [
                "Tell me about yourself",
                "What can you help me with?",
                "Describe your capabilities"
            ],
            'show': [
                "Show me patient 10",
                "Display the first 5 patients",
                "Show me patients with diabetes"
            ],
            'function': [
                "What can you help me with?",
                "What are your capabilities?",
                "How can I explore this dataset?"
            ],
            'score': [
                "How well does the model perform?",
                "What's the model accuracy?",
                "Show me performance metrics"
            ],
            'mistake': [
                "What are the model's biggest mistakes?",
                "Show me cases where the model was wrong",
                "Which predictions were incorrect?"
            ],
            'cfe': [
                "Show counterfactuals for this patient",
                "What are the alternatives to flip this prediction?",
                "How can we change this prediction?",
                "What would make this patient less likely to have diabetes?",
                "Which features should be different?",
                "Show me scenarios that would change the outcome"
            ],
            'counterfactual': [
                "Show counterfactuals for patient 5",
                "Generate counterfactual explanations",
                "What changes would flip this prediction?",
                "Show me alternative scenarios"
            ],
            'labels': [
                "What do these labels mean?",
                "Explain the target variable",
                "What is the dataset trying to predict?"
            ],
            'label': [
                "Show me the actual labels for this data",
                "What are the ground truth outcomes?",
                "Display the target values for these patients"
            ],
            'description': [
                "Tell me about this dataset",
                "What data do we have?",
                "Describe the features"
            ],
            'followup': [
                "Tell me more about that",
                "Explain that better",
                "What else can you tell me?"
            ],
            'model': [
                "Tell me about the model",
                "What model are we using?",
                "Model information"
            ],
            'new_estimate': [
                "Predict for a new patient with BMI=35",
                "Make a prediction for new values",
                "What's the prediction for a 60 year old with glucose=150?"
            ],
            'whatif': [
                "What if BMI was 25 instead of 30?",
                "Change age to 40 and see what happens",
                "What would happen with different values?"
            ]
        }
        
        # Return a random sample from the selected category
        selected_samples = samples.get(action, samples['important'])
        return Response(random.choice(selected_samples), mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Error generating samples: {e}")
        return Response("Ask me about the diabetes dataset!", mimetype='text/plain')

@main_bp.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    """Handle chat messages - this is the main conversational endpoint"""
    try:
        # Get message from the chat interface - frontend sends JSON with userInput
        data = request.get_json() if request.is_json else {}
        user_query = data.get('userInput', '') if data else request.form.get('msg', '')
        
        if not user_query:
            return jsonify({"error": "No message provided"})
        
        logger.info(f"Chat query: {user_query}")
        
        action_name, action_result, formatted_response = process_user_query(
            user_query, conversation, action_dispatcher, formatter
        )
        
        logger.info(f"Chat response generated using action: {action_name}")
        
        # Return in the format expected by the frontend: "response<>log_info"
        log_info = f"Action: {action_name}"
        response_text = f"{formatted_response}<>{log_info}"
        return Response(response_text, mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_text = f"I encountered an error: {str(e)}<>Error"
        return Response(error_text, mimetype='text/plain')

@main_bp.route('/log_feedback', methods=['POST'])
def log_feedback():
    """Log user feedback (simplified implementation)"""
    try:
        feedback = request.form.get('feedback', '')
        logger.info(f"User feedback: {feedback}")
        return jsonify({"status": "logged"})
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({"status": "error"})