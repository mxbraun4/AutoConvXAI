#!/usr/bin/env python3
"""
Simple TalkToModel: Natural Language → AutoGen → Actions → LLM Formatter
Clean 3-component architecture as originally envisioned
"""
import os
import logging
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template, Response
from explain.autogen_decoder import AutoGenDecoder
from explain.actions.get_action_functions import get_all_action_functions_map
from explain.explanation import MegaExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-4o-mini')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required!")

class SimpleActionDispatcher:
    """Simple dispatcher that calls explainability actions based on AutoGen intent"""
    
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.actions = get_all_action_functions_map()
        logger.info(f"Loaded {len(self.actions)} explainability actions")
    
    def execute_action(self, action_name, action_args, conversation):
        """Execute a single explainability action"""
        if action_name not in self.actions:
            return f"Unknown action: {action_name}"
        
        try:
            action_func = self.actions[action_name]
            
            # Create proper context for the conversation object
            if not hasattr(conversation, 'rounding_precision'):
                conversation.rounding_precision = 2
            if not hasattr(conversation, 'default_metric'):
                conversation.default_metric = "accuracy"
            if not hasattr(conversation, 'describe'):
                describe_obj = type('Describe', (), {})()
                describe_obj.get_dataset_description = lambda: "diabetes prediction based on patient health metrics"
                describe_obj.get_eval_performance = lambda model, metric: ""
                conversation.describe = describe_obj
            if not hasattr(conversation, 'store_followup_desc'):
                conversation.store_followup_desc = lambda x: None
            
            # Call action with expected parameters (conversation, parse_text, i, **kwargs)
            # Most actions expect these 3 parameters plus optional kwargs for entities
            parse_text = action_args.get('parse_text', ['data'])  # Default to basic data operation
            i = action_args.get('i', 0)  # Default index
            
            # Create kwargs excluding positional parameters to avoid conflicts
            kwargs = {k: v for k, v in action_args.items() if k not in ['parse_text', 'i']}
            result = action_func(conversation, parse_text, i, **kwargs)
            return result
            
        except Exception as e:
            logger.error(f"Error executing {action_name}: {e}")
            return f"Error executing {action_name}: {str(e)}"

class LLMFormatter:
    """Formats action results into natural language using LLM"""
    
    def __init__(self, decoder):
        self.decoder = decoder
    
    def format_response(self, user_query, action_name, action_result):
        """Format raw action result into natural language"""
        format_prompt = f"""
        The user asked: "{user_query}"
        
        I executed the explainability action "{action_name}" which returned:
        {action_result}
        
        Please format this result into a clear, natural language response that:
        1. Directly answers the user's question
        2. Explains what analysis was performed
        3. Presents the results in an understandable way
        4. Is concise but informative
        
        Response:"""
        
        try:
            formatted = self.decoder.complete_sync(format_prompt, None)
            return formatted.get('direct_response', str(action_result))
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return str(action_result)

# Initialize components
logger.info("🚀 Initializing simple 3-component architecture...")

# Component 1: AutoGen decoder for intent parsing
decoder = AutoGenDecoder(api_key=OPENAI_API_KEY, model=GPT_MODEL, max_rounds=2)

# Load dataset and model
logger.info("📊 Loading dataset and model...")
dataset = pd.read_csv('data/diabetes.csv')
with open('data/diabetes_model_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Component 2: Action dispatcher for explainability
action_dispatcher = SimpleActionDispatcher(dataset, model)

# Component 3: LLM formatter for natural language output
formatter = LLMFormatter(decoder)

class DatasetManager:
    """Manages dataset structure, X/y splits, and data preparation"""
    
    def __init__(self, dataset):
        self.original_dataset = dataset
        self.X_data = None
        self.y_data = None
        self.full_data_with_target = None
        self.target_col = None
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        """Prepare dataset with proper X/y split"""
        # The diabetes dataset uses the target column name 'y'. Fallback to 'Outcome' for compatibility.
        self.target_col = 'y' if 'y' in self.original_dataset.columns else ('Outcome' if 'Outcome' in self.original_dataset.columns else None)

        if self.target_col:
            self.X_data = self.original_dataset.drop(self.target_col, axis=1)  # Features only for model prediction
            self.y_data = self.original_dataset[self.target_col]               # Target variable
            self.full_data_with_target = self.original_dataset            # Full dataset includes target
        else:
            # No explicit target column found – assume dataset contains only features
            self.X_data = self.original_dataset
            self.y_data = None
            self.full_data_with_target = self.original_dataset
    
    def get_dataset_contents(self):
        """Return dataset contents in the format expected by actions"""
        return {
            'X': self.X_data,  # Only features - this goes to the model (8 features)
            'y': self.y_data,  # Target variable separately
            'full_data': self.full_data_with_target,  # Full dataset with target for filtering
            'cat': [],  # Categorical features (diabetes dataset is all numeric)
            'numeric': list(self.X_data.columns),  # Feature names only (8 features)
            'ids_to_regenerate': []
        }

class VariableStore:
    """Manages stored variables used by actions"""
    
    def __init__(self):
        self.stored_vars = {}
    
    def add_var(self, name, contents, kind=None):
        """Store new variables"""
        var_obj = type('Variable', (), {'contents': contents})()
        self.stored_vars[name] = var_obj
    
    def get_var(self, name):
        """Retrieve stored variables by name"""
        return self.stored_vars.get(name)
    
    def get_all_vars(self):
        """Get all stored variables"""
        return self.stored_vars

class ConversationHistory:
    """Manages conversation history and followup descriptions"""
    
    def __init__(self):
        self.history = []
        self.followup = ""
    
    def add_turn(self, query, response):
        """Add a conversation turn"""
        self.history.append({'query': query, 'response': response})
    
    def store_followup_desc(self, desc):
        """Store followup description for later use"""
        self.followup = desc
    
    def get_followup_desc(self):
        """Retrieve followup description"""
        return self.followup

class ExplainerManager:
    """Manages LIME explainer setup and configuration"""
    
    def __init__(self, model, X_data):
        self.model = model
        self.X_data = X_data
        self.mega_explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Initialize mega_explainer for LIME explanations"""
        def prediction_function(x):
            """Wrapper for model prediction"""
            return self.model.predict_proba(x)
        
        import os
        cache_dir = os.path.join(os.getcwd(), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "mega-explainer-tabular.pkl")
        
        from explain.explanation import MegaExplainer
        self.mega_explainer = MegaExplainer(
            prediction_fn=prediction_function,
            data=self.X_data,  # Only feature columns, not target
            cat_features=[],  # All numeric features for diabetes dataset
            cache_location=cache_path,
            class_names=["No Diabetes", "Diabetes"],
            use_selection=False  # Use LIME directly as requested
        )
    
    def get_explainer(self):
        """Get the configured explainer"""
        return self.mega_explainer

class MetadataManager:
    """Manages feature definitions and class name mappings"""
    
    def __init__(self):
        self.feature_definitions = {}
        self.class_names = {0: "No Diabetes", 1: "Diabetes"}
        self.rounding_precision = 2
        self.default_metric = "accuracy"
        self.username = "user"
    
    def get_feature_definition(self, feature_name):
        """Get feature description"""
        return self.feature_definitions.get(feature_name, "")
    
    def get_class_name_from_label(self, label):
        """Convert label to class name"""
        return self.class_names.get(label, str(label))

class FilterStateManager:
    """Manages temporary dataset and parse operations for filtering"""
    
    def __init__(self, dataset_manager, variable_store):
        self.dataset_manager = dataset_manager
        self.variable_store = variable_store
        self.parse_operation = []
        self.last_parse_string = []
        self.temp_dataset = None
        # Don't reset immediately - wait for variables to be initialized
    
    def add_interpretable_parse_op(self, text):
        """Add interpretable operation description"""
        self.parse_operation.append(text)
    
    def build_temp_dataset(self, save=True):
        """Create temporary dataset for filtering operations"""
        # For now, just return the main dataset
        # In a full implementation, this would apply current filters
        return self.variable_store.get_var('dataset')
    
    def reset_temp_dataset(self):
        """Properly reset temp_dataset to full dataset with deep copy"""
        self._reset_temp_dataset()
        
        # Clear filter history
        self.parse_operation = []
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Reset temp_dataset to full dataset: {len(self.temp_dataset.contents['X'])} instances")
    
    def _reset_temp_dataset(self):
        """Internal method to reset temp dataset"""
        original_dataset = self.variable_store.get_var('dataset').contents
        
        # Create a deep copy of the dataset structure
        import copy
        reset_contents = {
            'X': original_dataset['X'].copy(),  # Deep copy of DataFrame
            'y': original_dataset['y'].copy() if original_dataset['y'] is not None else None,  # Deep copy of Series
            'full_data': original_dataset['full_data'].copy(),
            'cat': original_dataset['cat'].copy(),
            'numeric': original_dataset['numeric'].copy(),
            'ids_to_regenerate': original_dataset['ids_to_regenerate'].copy()
        }
        
        # Create new temp_dataset object with copied contents
        self.temp_dataset = type('Variable', (), {'contents': reset_contents})()

# Simple conversation context - now using composition
class SimpleConversation:
    def __init__(self, dataset, model):
        # Initialize all the specialized managers in correct order
        self.dataset_manager = DatasetManager(dataset)
        self.variable_store = VariableStore()
        self.history_manager = ConversationHistory()
        self.explainer_manager = ExplainerManager(model, self.dataset_manager.X_data)
        self.metadata_manager = MetadataManager()
        
        # Initialize variables that actions expect BEFORE FilterStateManager
        self._setup_variables(model)
        
        # Initialize FilterStateManager after variables are set up
        self.filter_manager = FilterStateManager(self.dataset_manager, self.variable_store)
        
        # Now initialize the temp dataset
        self.filter_manager._reset_temp_dataset()
        
        # Create a proper describe object with methods
        describe_obj = type('Describe', (), {})()
        describe_obj.get_dataset_description = lambda: "diabetes prediction based on patient health metrics"
        describe_obj.get_eval_performance = lambda model, metric: ""
        describe_obj.get_score_text = lambda y_true, y_pred, metric, precision, data_name: f"Model accuracy: {(y_true == y_pred).mean():.3f} on {data_name}"
        self.describe = describe_obj
    
    def _setup_variables(self, model):
        """Setup variables that actions expect"""
        dataset_contents = self.dataset_manager.get_dataset_contents()
        dataset_obj = type('Variable', (), {'contents': dataset_contents})()
        model_obj = type('Variable', (), {'contents': model})()
        mega_explainer_obj = type('Variable', (), {'contents': self.explainer_manager.get_explainer()})()
        
        self.variable_store.add_var('dataset', dataset_contents)
        self.variable_store.add_var('model', model)
        self.variable_store.add_var('mega_explainer', self.explainer_manager.get_explainer())
        
        # For backward compatibility, expose temp_dataset
        self.temp_dataset = dataset_obj
    
    # Delegate methods to appropriate managers
    def add_turn(self, query, response):
        return self.history_manager.add_turn(query, response)
    
    def get_var(self, name):
        return self.variable_store.get_var(name)
    
    def add_var(self, name, contents, kind=None):
        return self.variable_store.add_var(name, contents, kind)
    
    def store_followup_desc(self, desc):
        return self.history_manager.store_followup_desc(desc)
    
    def get_followup_desc(self):
        return self.history_manager.get_followup_desc()
    
    def add_interpretable_parse_op(self, text):
        return self.filter_manager.add_interpretable_parse_op(text)
    
    def get_class_name_from_label(self, label):
        return self.metadata_manager.get_class_name_from_label(label)
    
    def get_feature_definition(self, feature_name):
        return self.metadata_manager.get_feature_definition(feature_name)
    
    def build_temp_dataset(self, save=True):
        return self.filter_manager.build_temp_dataset(save)
    
    def reset_temp_dataset(self):
        return self.filter_manager.reset_temp_dataset()
    
    # Expose metadata attributes for backward compatibility
    @property
    def history(self):
        return self.history_manager.history
    
    @property
    def stored_vars(self):
        return self.variable_store.get_all_vars()
    
    @property
    def parse_operation(self):
        return self.filter_manager.parse_operation
    
    @property
    def last_parse_string(self):
        return self.filter_manager.last_parse_string
    
    @property
    def rounding_precision(self):
        return self.metadata_manager.rounding_precision
    
    @property
    def default_metric(self):
        return self.metadata_manager.default_metric
    
    @property
    def class_names(self):
        return self.metadata_manager.class_names
    
    @property
    def feature_definitions(self):
        return self.metadata_manager.feature_definitions
    
    @property
    def username(self):
        return self.metadata_manager.username
    
    @property
    def followup(self):
        return self.history_manager.followup

conversation = SimpleConversation(dataset, model)

logger.info(f"✅ Ready! Dataset: {len(dataset)} instances, Model: {type(model).__name__}")

@app.route('/')
def home():
    """Render the main conversational interface"""
    return render_template('index.html', 
                         datasetObjective="predict diabetes based on patient health metrics",
                         dataset_size=len(dataset),
                         action_count=len(action_dispatcher.actions))


@app.route('/sample_prompt', methods=['POST'])
def sample_prompt():
    """Generate sample prompts for the suggestion buttons"""
    try:
        # Get action from JSON or form data
        data = request.get_json() if request.is_json else {}
        action = data.get('action', '') if data else request.form.get('action', 'important')
        
        # Simple sample prompts for different categories
        samples = {
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
                "What would happen if age was 40?",
                "Make a prediction for these values"
            ],
            'likelihood': [
                "How confident is the model in this prediction?",
                "What's the prediction probability?",
                "Show me the likelihood scores"
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
            'whatif': [
                "What if the patient's BMI was 25 instead?",
                "How would the prediction change if age was 30?",
                "What happens if glucose level is 90?"
            ],
            'mistake': [
                "What are the model's biggest mistakes?",
                "Show me cases where the model was wrong",
                "Which predictions were incorrect?"
            ],
            'cfe': [
                "How can we change this prediction?",
                "What would make this patient less likely to have diabetes?",
                "Which features should be different?"
            ],
            'labels': [
                "What do these labels mean?",
                "Explain the target variable",
                "What is the dataset trying to predict?"
            ],
            'interactions': [
                "How do features interact with each other?",
                "What are the feature interaction effects?",
                "Which features work together?"
            ],
            'description': [
                "Tell me about this dataset",
                "What data do we have?",
                "Describe the features"
            ]
        }
        
        # Return a random sample from the selected category
        selected_samples = samples.get(action, samples['important'])
        import random
        return Response(random.choice(selected_samples), mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Error generating samples: {e}")
        return Response("Ask me about the diabetes dataset!", mimetype='text/plain')

@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    """Handle chat messages - this is the main conversational endpoint"""
    try:
        # Get message from the chat interface - frontend sends JSON with userInput
        data = request.get_json() if request.is_json else {}
        user_query = data.get('userInput', '') if data else request.form.get('msg', '')
        
        if not user_query:
            return jsonify({"error": "No message provided"})
        
        logger.info(f"Chat query: {user_query}")
        
        # Reset dataset state to prevent persistent filtering between queries
        conversation.reset_temp_dataset()  # Proper deep copy reset
        
        # Use the same processing pipeline as /query
        autogen_response = decoder.complete_sync(user_query, conversation)
        
        # Extract action from AutoGen response and map intent to proper action
        intent = autogen_response.get('intent', 'data')
        
        # Map intents to proper action names (based on action_functions.py mapping)
        intent_to_action = {
            'data': 'data',
            'performance': 'score', 
            'predict': 'predict',
            'explain': 'explain',
            'important': 'important',
            'filter': 'filter',
            'casual': 'function'
        }
        
        # NO FALLBACKS - Fail fast if intent is not recognized
        if intent not in intent_to_action:
            raise ValueError(f"Unknown intent '{intent}' - AutoGen must produce valid intents only")
        
        action_name = intent_to_action[intent]
        
        # Extract entities and pass them as action arguments
        entities = autogen_response.get('entities', {})
        # Build parse_text from user_query tokens for better context parsing
        user_tokens = user_query.lower().split()
        action_args = {
            'features': entities.get('features', []),
            'operators': entities.get('operators', []),
            'values': entities.get('values', []),
            'patient_id': entities.get('patient_id'),
            'filter_type': entities.get('filter_type'),  # Add filter_type for clean filtering
            'prediction_values': entities.get('prediction_values', []),  # For prediction filtering
            'label_values': entities.get('label_values', []),  # For label filtering
            'parse_text': user_tokens  # Provide full token list for actions that rely on parse_text
        }
        
        # CLEAN ARCHITECTURE: Auto-apply ID filtering when patient_id is present
        # This maintains generalizability by using the existing filter system with AutoGen entities
        if entities.get('patient_id') is not None:
            patient_id = entities.get('patient_id')
            # Apply ID filter using clean AutoGen entities
            filter_args = {
                'features': ['id'],  # Use 'id' as the feature name
                'operators': ['='],  # Use equals operator
                'values': [patient_id],  # The patient ID value
                'filter_type': 'feature'  # This is feature-based filtering
            }
            filter_result = action_dispatcher.execute_action('filter', filter_args, conversation)
            logger.info(f"Auto-applied ID filter for patient {patient_id}: {filter_result}")
        
        logger.info(f"AutoGen identified action: {action_name} with entities: {entities}")
        
        action_result = action_dispatcher.execute_action(action_name, action_args, conversation)
        formatted_response = formatter.format_response(user_query, action_name, action_result)
        
        conversation.add_turn(user_query, formatted_response)
        
        logger.info(f"Chat response generated using action: {action_name}")
        
        # Return in the format expected by the frontend: "response<>log_info"
        log_info = f"Action: {action_name}"
        response_text = f"{formatted_response}<>{log_info}"
        return Response(response_text, mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_text = f"I encountered an error: {str(e)}<>Error"
        return Response(error_text, mimetype='text/plain')

@app.route('/log_feedback', methods=['POST'])
def log_feedback():
    """Log user feedback (simplified implementation)"""
    try:
        feedback = request.form.get('feedback', '')
        logger.info(f"User feedback: {feedback}")
        return jsonify({"status": "logged"})
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({"status": "error"})

if __name__ == "__main__":
    print("\n🤖 TalkToModel - Simple Architecture")
    print("Natural Language → AutoGen → Actions → LLM Formatter")
    print("Available at: http://localhost:5000")
    print("API: POST /query with JSON: {'query': 'your question'}")
    app.run(host='0.0.0.0', port=5000, debug=True)