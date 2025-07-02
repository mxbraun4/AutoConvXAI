#!/usr/bin/env python3
"""
Simple TalkToModel: Natural Language → AutoGen → Actions → LLM Formatter
Clean 3-component architecture as originally envisioned
"""
import os
import logging
import pandas as pd
import pickle
import openai
from flask import Flask, request, jsonify, render_template, Response
from explain.decoders.autogen_decoder import AutoGenDecoder
from explain.actions.get_action_functions import get_all_action_functions_map
from explain.core.explanation import MegaExplainer

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
                describe_obj.get_dataset_objective = lambda: "predict diabetes risk in patients"
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
    """Intelligent LLM-based formatter that creates question-tailored responses"""
    
    def __init__(self, api_key, model='gpt-4o-mini'):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def format_response(self, user_query, action_name, action_result, autogen_intent, conversation=None):
        """Format action result into natural, question-tailored language using AutoGen-detected intent"""
        
        # Extract context about the conversation
        context_info = self._build_context(conversation)
        
        # Create intelligent formatting prompt based on action type and AutoGen intent
        format_prompt = self._create_intelligent_prompt(
            user_query, action_name, action_result, autogen_intent, context_info, conversation
        )
        
        try:
            # Use direct OpenAI API call for formatting
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data scientist assistant. Provide clear, concise responses based only on the provided data. Be conversational but brief (2-3 sentences max). Never speculate or add information not in the data."},
                    {"role": "user", "content": format_prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            formatted = response.choices[0].message.content.strip()
                
            # Fallback to action result if formatting fails
            if not formatted or len(formatted.strip()) < 10:
                logger.warning(f"LLM formatter returned short response: {formatted}")
                return self._create_fallback_response(action_result)
                
            return formatted
            
        except Exception as e:
            logger.error(f"Error in LLM formatting: {e}")
            return self._create_fallback_response(action_result)
    
    def _build_context(self, conversation):
        """Extract relevant context from conversation"""
        if not conversation:
            return {}
            
        context = {}
        
        # Dataset context
        try:
            dataset = conversation.get_var('dataset')
            if dataset:
                context['dataset_size'] = len(dataset.contents.get('X', []))
                context['features'] = list(dataset.contents.get('X', {}).columns) if hasattr(dataset.contents.get('X', {}), 'columns') else []
        except:
            pass
            
        # NEW: Enhanced filtering context using filter state
        try:
            if hasattr(conversation, 'get_filter_context_for_response'):
                filter_context = conversation.get_filter_context_for_response()
                context['filter_context'] = filter_context
                
                if filter_context['type'] == 'query_filter':
                    context['communication_style'] = 'query_applied_filter'
                    context['filter_description'] = filter_context['description']
                    context['original_size'] = filter_context['original_size']
                    context['current_size'] = filter_context['filtered_size']
                elif filter_context['type'] == 'inherited_filter':
                    context['communication_style'] = 'inherited_filter'
                    context['filter_description'] = filter_context['description']
                    context['original_size'] = filter_context['original_size']
                    context['current_size'] = filter_context['filtered_size']
                else:
                    context['communication_style'] = 'no_filter'
                    context['original_size'] = filter_context['original_size']
                    context['current_size'] = filter_context['filtered_size']
            else:
                # Fallback to old logic
                if hasattr(conversation, 'temp_dataset') and conversation.temp_dataset:
                    temp_size = len(conversation.temp_dataset.contents.get('X', []))
                    full_size = context.get('dataset_size', temp_size)
                    if temp_size < full_size:
                        context['filtered'] = True
                        context['filtered_size'] = temp_size
                        context['filter_description'] = getattr(conversation, 'parse_operation', [])
                    else:
                        context['filtered'] = False
                        context['using_full_dataset'] = True
                else:
                    context['filtered'] = False
                    context['using_full_dataset'] = True
        except:
            pass
            
        # Model context
        try:
            if hasattr(conversation, 'class_names'):
                context['class_names'] = conversation.class_names
        except:
            pass
            
        return context
    
    def _create_intelligent_prompt(self, user_query, action_name, action_result, autogen_intent, context_info, conversation=None):
        """Create an intelligent prompt tailored to the specific action and AutoGen-detected intent"""
        
        # Build context string with clear filtering explanation
        context_str = ""
        
        # NEW: Use enhanced filter context for clearer communication
        communication_style = context_info.get('communication_style', 'unknown')
        
        if communication_style == 'query_applied_filter':
            # This query applied a new filter - communicate as "X out of Y total"
            original_size = context_info['original_size']
            current_size = context_info['current_size']
            filter_desc = context_info['filter_description']
            context_str += f"QUERY FILTER APPLIED: User asked about {filter_desc}. Found {current_size} instances out of {original_size} total that match this criteria. "
            context_str += f"COMMUNICATION STYLE: Say 'Of the {original_size} total instances, {current_size} have {filter_desc}' NOT 'In the filtered dataset of {current_size} instances'. "
            
        elif communication_style == 'inherited_filter':
            # Data was already filtered - communicate about the subset
            original_size = context_info['original_size']
            current_size = context_info['current_size']
            filter_desc = context_info['filter_description']
            context_str += f"INHERITED FILTER: Dataset was already filtered to {current_size} instances (from {original_size} total) where {filter_desc}. "
            context_str += f"COMMUNICATION STYLE: Say 'In the filtered subset of {current_size} instances' or 'Among the {current_size} instances where {filter_desc}'. "
            
        elif communication_style == 'no_filter':
            # No filtering applied
            total_size = context_info['current_size']
            context_str += f"NO FILTER: Using complete dataset ({total_size} instances). "
            context_str += f"COMMUNICATION STYLE: Say 'Of the {total_size} total instances' or 'In the complete dataset'. "
            
        else:
            # Fallback to old logic
            if context_info.get('filtered'):
                filter_size = context_info['filtered_size']
                total_size = context_info.get('dataset_size', filter_size)
                
                # Determine what kind of filtering happened by looking at the user query and results
                if filter_size == total_size:
                    # Full dataset was used after reset
                    context_str += f"Dataset: Complete dataset ({total_size} instances). "
                else:
                    # Data was filtered
                    context_str += f"Dataset: Filtered to {filter_size} instances from {total_size} total. "
                    
                    # Add percentage clarification if needed
                    if hasattr(action_result, '__getitem__') and isinstance(action_result, (dict, tuple)):
                        result_data = action_result[0] if isinstance(action_result, tuple) else action_result
                        if isinstance(result_data, dict):
                            request_type = result_data.get('request_type')
                            if request_type == 'class_distribution':
                                context_str += f"Percentages refer to the {filter_size} filtered instances. "
            elif context_info.get('using_full_dataset'):
                total_size = context_info['dataset_size']
                context_str += f"Dataset: Complete dataset ({total_size} instances). "
                
        if context_info.get('class_names'):
            class_list = ', '.join(context_info['class_names'].values())
            context_str += f"The model predicts: {class_list}. "
        
        # Add recent conversation context for comparative questions
        if conversation and hasattr(conversation, 'conversation_turns') and len(conversation.conversation_turns) > 0:
            recent_turn = conversation.conversation_turns[-1]
            if recent_turn.get('action_name') == 'score':
                context_str += f"Previous result: {recent_turn.get('response', '')} "
        
        # Create tailored prompt based on action and intent
        base_prompt = f"""
You are an expert data scientist explaining machine learning results to users. 

USER QUESTION: "{user_query}"
ACTION PERFORMED: {action_name}
AUTOGEN INTENT: {autogen_intent}
CONTEXT: {context_str}

RAW RESULTS:
{action_result}

TASK: Transform these raw results into a brief, natural response (2-3 sentences max) that:

1. DIRECTLY ANSWERS the user's specific question with factual data only
2. Uses plain language (avoid technical jargon)
3. States only what is shown in the raw results
4. Never speculates, assumes, or adds context not in the data
5. Be conversational but concise
6. CRITICAL: If data is filtered, be clear about what subset is being analyzed
7. CRITICAL: For percentages, always specify they are percentages of the filtered subset, not the original dataset

"""
        
        # Add intent-specific instructions based on AutoGen intent
        if autogen_intent == 'count' or autogen_intent == 'data':
            base_prompt += """
COUNT/DATA INTENT: The user wants to know HOW MANY. Be direct and clear:
- ALWAYS use simple language: "There are X instances with [condition]" or "Of the Y total instances, X have [condition]"
- NEVER say "In the filtered dataset" - this confuses users
- Just state the facts: numbers and what they represent
- Skip technical implementation details about filtering
- Be concise - one clear sentence with the count
"""
        elif autogen_intent == 'casual':
            base_prompt += """
CASUAL INTENT: The user is making conversational comments or observations. Respond naturally:
- Acknowledge their observation appropriately
- Provide relevant context if helpful
- Keep responses brief and conversational
- Don't introduce new statistics unless directly relevant to their comment
"""
        elif autogen_intent == 'explain':
            base_prompt += """
EXPLAIN INTENT: The user wants to understand WHY something happened. Focus on:
- Clear causal explanations
- The most influential factors
- Use phrases like "because", "due to", "the main reason"
- Make it feel like you're walking them through the reasoning
"""
        elif autogen_intent == 'predict':
            base_prompt += """
PREDICT INTENT: The user wants to know WHAT WOULD HAPPEN. Focus on:
- Clear prediction outcomes
- Confidence levels
- What factors drive the prediction
- Use phrases like "would result in", "likely to", "expected outcome"
"""
        elif autogen_intent == 'important':
            base_prompt += """
IMPORTANCE INTENT: The user wants to know WHAT MATTERS MOST. Focus on:
- Ranking and prioritization
- Relative importance
- Impact on outcomes
- Use phrases like "most critical", "key factors", "strongest influence"
"""
        elif autogen_intent == 'performance':
            base_prompt += """
PERFORMANCE INTENT: The user wants to know HOW WELL the model works. Focus on:
- Accuracy and reliability
- Strengths and limitations
- Practical implications
- Use phrases like "performs well", "accurate in", "reliable for"
"""
        elif autogen_intent == 'whatif':
            base_prompt += """
WHAT-IF INTENT: The user wants to explore scenarios. Focus on:
- Clear before/after comparisons
- Impact of changes on predictions
- Practical implications of modifications
- Use phrases like "if you changed", "would result in", "the impact would be"
"""
        elif autogen_intent == 'counterfactual':
            base_prompt += """
COUNTERFACTUAL INTENT: The user wants alternative scenarios to change the outcome. Focus on:
- What specific changes would flip the prediction
- Clear feature modifications needed
- Practical actionable changes
- Alternative pathways to different outcomes
- Use phrases like "to change the outcome", "alternatively", "if instead", "would need to modify"
"""
        elif autogen_intent == 'mistakes':
            base_prompt += """
MISTAKES INTENT: The user wants to understand model errors. Focus on:
- Specific cases where model was wrong
- Patterns in mistakes
- Why errors occurred
- Use phrases like "incorrectly predicted", "failed to recognize", "got wrong because"
"""
        elif autogen_intent == 'confidence':
            base_prompt += """
CONFIDENCE INTENT: The user wants certainty information. Focus on:
- Probability scores and what they mean
- Model certainty levels
- Reliability of predictions
- Use phrases like "confident that", "probability of", "certain about"
"""
        elif autogen_intent == 'interactions':
            base_prompt += """
INTERACTIONS INTENT: The user wants to understand feature relationships. Focus on:
- How features work together
- Combined effects
- Synergistic relationships
- Use phrases like "work together", "combined effect", "interact to"
"""
        elif autogen_intent == 'statistics':
            base_prompt += """
STATISTICS INTENT: The user wants numerical summaries. Focus on:
- Clear statistical descriptions
- Meaningful comparisons
- Distribution characteristics
- Use phrases like "on average", "typically", "ranges from"
"""
        elif autogen_intent == 'define':
            base_prompt += """
DEFINE INTENT: The user wants explanations of terms. Focus on:
- Clear, simple definitions
- Practical context
- Why it matters for health/diabetes
- Use phrases like "refers to", "means", "is important because"
"""
        elif autogen_intent == 'about':
            base_prompt += """
ABOUT INTENT: The user wants system information. Focus on:
- Capabilities and features
- What you can help with
- Encouraging exploration
- Use phrases like "I can help you", "my capabilities include", "you can ask me"
"""
        
        base_prompt += """
RESPONSE (2-3 sentences max, factual only, no speculation):"""
        
        return base_prompt
    
    def _create_fallback_response(self, action_result):
        """Create a simple fallback when LLM formatting fails"""
        if isinstance(action_result, tuple) and len(action_result) >= 1:
            result_data = action_result[0]
        else:
            result_data = action_result
            
        if isinstance(result_data, dict):
            # Handle structured data from converted actions
            result_type = result_data.get('type', 'unknown')
            
            if result_type == 'performance_score':
                return f"The model achieves {result_data['score_percentage']}% accuracy on {result_data['instances_evaluated']} instances."
            elif result_type == 'single_prediction':
                conf = f" with {result_data['confidence']}% confidence" if result_data.get('confidence') else ""
                return f"The model predicts: {result_data['prediction_class']}{conf}"
            elif result_type == 'data_summary':
                return f"Dataset contains {result_data['dataset_size']} instances with features: {', '.join(result_data['features'][:3])}..."
            elif result_type == 'feature_statistics':
                stats = result_data['statistics']
                if stats.get('type') == 'numerical':
                    return f"{result_data['feature_name']} statistics: mean={stats['mean']}, std={stats['std']}"
                else:
                    return f"{result_data['feature_name']} distribution: {stats.get('distribution', {})}"
            elif result_type == 'single_explanation':
                conf = f" with {result_data['confidence']}% confidence" if result_data.get('confidence') else ""
                return f"Instance {result_data['instance_id']} prediction: {result_data['prediction_class']}{conf}. Top features: {', '.join([f['feature'] for f in result_data['feature_importance'][:3]])}"
            elif result_type == 'multiple_explanations':
                summary = f"Analyzed {result_data['total_instances']} instances. "
                pred_summary = result_data['prediction_summary']
                for class_name, stats in pred_summary.items():
                    summary += f"{class_name}: {stats['count']} ({stats['percentage']}%) "
                return summary
            elif result_type == 'what_if_change':
                return f"Changed {result_data['feature_name']} by {result_data['operation']} {result_data['value']} for {result_data['instances_affected']} instance(s)"
            elif result_type == 'counterfactual_explanation':
                summary = f"Generated {result_data['total_counterfactuals']} counterfactual scenarios for instance {result_data['instance_id']}. "
                summary += f"Original prediction: {result_data['original_prediction_class']}. "
                if result_data.get('summary'):
                    summary += f"Key changes: {result_data['summary']}"
                return summary
            elif result_type == 'error':
                return f"Error: {result_data['message']}"
            else:
                return f"Result: {str(result_data)}"
        else:
            return str(result_data)

# Initialize components
logger.info("🚀 Initializing simple 3-component architecture...")

# Component 1: AutoGen decoder for intent parsing
# max_rounds=4 enables 3-agent collaboration without overthinking (Intent → Action → Validation → Intent)
decoder = AutoGenDecoder(api_key=OPENAI_API_KEY, model=GPT_MODEL, max_rounds=4)

# Load dataset and model
logger.info("📊 Loading dataset and model...")
dataset = pd.read_csv('data/diabetes.csv')
with open('data/diabetes_model_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Component 2: Action dispatcher for explainability
action_dispatcher = SimpleActionDispatcher(dataset, model)

# Component 3: LLM formatter for natural language output
formatter = LLMFormatter(OPENAI_API_KEY, GPT_MODEL)

class Conversation:
    """Conversational context with memory for multi-turn interactions"""
    
    def __init__(self, dataset, model):
        # Dataset preparation (consolidated from DatasetManager)
        self.target_col = 'y' if 'y' in dataset.columns else ('Outcome' if 'Outcome' in dataset.columns else None)
        if self.target_col:
            self.X_data = dataset.drop(self.target_col, axis=1)
            self.y_data = dataset[self.target_col]
            self.full_data = dataset
        else:
            self.X_data = dataset
            self.y_data = None
            self.full_data = dataset
        
        # Conversational memory system
        self.history = []
        self.followup = ""
        self.parse_operation = []
        self.last_parse_string = []
        
        # NEW: Enhanced filter state tracking
        self.last_action = None
        self.last_action_args = None
        self.last_filter_applied = None
        self.last_result = None
        self.conversation_turns = []
        
        # NEW: Filter state management for clearer communication
        self.filter_state = {
            'has_inherited_filter': False,  # Was data filtered from previous operations?
            'current_filter_description': '',  # Human-readable description of current filter
            'query_applied_filter': False,  # Did this query apply a new filter?
            'query_filter_description': '',  # Description of filter applied by current query
            'original_size': len(dataset),
            'current_size': len(dataset)
        }
        
        # Configuration (consolidated from MetadataManager)
        self.rounding_precision = 2
        self.default_metric = "accuracy" 
        self.class_names = {0: "No Diabetes", 1: "Diabetes"}
        self.feature_definitions = {}
        self.username = "user"
        
        # Variables (consolidated from VariableStore)
        self.stored_vars = self._setup_variables(model)
        
        # Setup explainer (consolidated from ExplainerManager)
        self._setup_explainer(model)
        
        # Initialize temp dataset
        self.temp_dataset = self._create_temp_dataset()
        
        # Create describe object for backward compatibility
        describe_obj = type('Describe', (), {})()
        describe_obj.get_dataset_description = lambda: "diabetes prediction based on patient health metrics"
        describe_obj.get_eval_performance = lambda model, metric: ""
        describe_obj.get_score_text = lambda y_true, y_pred, metric, precision, data_name: f"Model accuracy: {(y_true == y_pred).mean():.3f} on {data_name}"
        self.describe = describe_obj
    
    def _setup_variables(self, model):
        """Setup variables that actions expect"""
        dataset_contents = {
            'X': self.X_data,
            'y': self.y_data,
            'full_data': self.full_data,
            'cat': [],  # All numeric features for diabetes dataset
            'numeric': list(self.X_data.columns),
            'ids_to_regenerate': []
        }
        
        # Create prediction probability function for interaction analysis
        def prediction_probability_function(x, *args, **kwargs):
            return model.predict_proba(x)
        
        return {
            'dataset': type('Variable', (), {'contents': dataset_contents})(),
            'model': type('Variable', (), {'contents': model})(),
            'model_prob_predict': type('Variable', (), {'contents': prediction_probability_function})(),
            'mega_explainer': None  # Will be set after explainer setup
        }
    
    def _setup_explainer(self, model):
        """Initialize LIME explainer and TabularDice explainer"""
        def prediction_function(x):
            return model.predict_proba(x)
        
        import os
        cache_dir = os.path.join(os.getcwd(), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "mega-explainer-tabular.pkl")
        
        from explain.core.explanation import MegaExplainer
        mega_explainer = MegaExplainer(
            prediction_fn=prediction_function,
            data=self.X_data,
            cat_features=[],
            cache_location=cache_path,
            class_names=["No Diabetes", "Diabetes"],
            use_selection=True  # Enable SHAP for better explanations
        )
        
        self.stored_vars['mega_explainer'] = type('Variable', (), {'contents': mega_explainer})()
        
        # Initialize TabularDice for counterfactual explanations
        from explain.core.explanation import TabularDice
        dice_cache_path = os.path.join(cache_dir, "dice-tabular.pkl")
        tabular_dice = TabularDice(
            model=model,
            data=self.X_data,
            num_features=list(self.X_data.columns),
            num_cfes_per_instance=10,
            num_in_short_summary=3,
            desired_class="opposite",
            cache_location=dice_cache_path,
            class_names={0: "No Diabetes", 1: "Diabetes"}
        )
        
        self.stored_vars['tabular_dice'] = type('Variable', (), {'contents': tabular_dice})()
    
    def _create_temp_dataset(self):
        """Create temp dataset object"""
        return type('Variable', (), {'contents': self.stored_vars['dataset'].contents.copy()})()
    
    def reset_temp_dataset(self):
        """Reset temp dataset to full dataset"""
        import copy
        original_contents = self.stored_vars['dataset'].contents
        
        reset_contents = {
            'X': original_contents['X'].copy(),
            'y': original_contents['y'].copy() if original_contents['y'] is not None else None,
            'full_data': original_contents['full_data'].copy(),
            'cat': original_contents['cat'].copy(),
            'numeric': original_contents['numeric'].copy(),
            'ids_to_regenerate': original_contents['ids_to_regenerate'].copy()
        }
        
        self.temp_dataset = type('Variable', (), {'contents': reset_contents})()
        self.parse_operation = []
        
        # NEW: Reset filter state
        self.filter_state.update({
            'has_inherited_filter': False,
            'current_filter_description': '',
            'query_applied_filter': False,
            'query_filter_description': '',
            'current_size': self.filter_state['original_size']
        })
        
        logger.info(f"Reset temp_dataset to full dataset: {len(self.temp_dataset.contents['X'])} instances")
    
    def mark_query_filter_applied(self, filter_description, resulting_size):
        """Mark that this query applied a new filter"""
        self.filter_state.update({
            'query_applied_filter': True,
            'query_filter_description': filter_description,
            'current_size': resulting_size
        })
    
    def mark_inherited_filter(self, filter_description, current_size):
        """Mark that data was already filtered from previous operations"""
        self.filter_state.update({
            'has_inherited_filter': True,
            'current_filter_description': filter_description,
            'query_applied_filter': False,
            'query_filter_description': '',
            'current_size': current_size
        })
    
    def get_filter_context_for_response(self):
        """Get filter context for response formatting"""
        if self.filter_state['query_applied_filter']:
            return {
                'type': 'query_filter',
                'description': self.filter_state['query_filter_description'],
                'original_size': self.filter_state['original_size'],
                'filtered_size': self.filter_state['current_size']
            }
        elif self.filter_state['has_inherited_filter']:
            return {
                'type': 'inherited_filter',
                'description': self.filter_state['current_filter_description'],
                'original_size': self.filter_state['original_size'],
                'filtered_size': self.filter_state['current_size']
            }
        else:
            return {
                'type': 'no_filter',
                'description': '',
                'original_size': self.filter_state['original_size'],
                'filtered_size': self.filter_state['current_size']
            }
    
    # Enhanced interface methods with conversational memory
    def add_turn(self, query, response, action_name=None, action_args=None, action_result=None):
        turn_data = {
            'query': query, 
            'response': response,
            'action_name': action_name,
            'action_args': action_args,
            'timestamp': len(self.conversation_turns)
        }
        self.history.append({'query': query, 'response': response})
        self.conversation_turns.append(turn_data)
        
        # Update last operation tracking
        if action_name:
            self.last_action = action_name
            self.last_action_args = action_args
            self.last_result = action_result
            
            # Track filter operations specifically
            if action_name == 'filter':
                self.last_filter_applied = action_args
    
    def get_var(self, name):
        return self.stored_vars.get(name)
    
    def add_var(self, name, contents, kind=None):
        var_obj = type('Variable', (), {'contents': contents})()
        self.stored_vars[name] = var_obj
    
    def store_followup_desc(self, desc):
        self.followup = desc
    
    def get_followup_desc(self):
        return self.followup
    
    def add_interpretable_parse_op(self, text):
        self.parse_operation.append(text)
    
    def get_class_name_from_label(self, label):
        return self.class_names.get(label, str(label))
    
    def get_feature_definition(self, feature_name):
        return self.feature_definitions.get(feature_name, "")
    
    def build_temp_dataset(self, save=True):
        return self.get_var('dataset')

conversation = Conversation(dataset, model)

logger.info(f"✅ Ready! Dataset: {len(dataset)} instances, Model: {type(model).__name__}")

def _safe_model_predict(model, data):
    """Utility function to predict with sklearn compatibility (avoids feature name warnings)."""
    try:
        # Try with DataFrame first (preserves feature names)
        return model.predict(data)
    except Exception:
        # Fallback to numpy array if DataFrame fails
        return model.predict(data.values)

def _should_reset_filter_context(intent, entities, user_query, conversation):
    """Determine if we should reset the filter context for a new query."""
    
    # Never reset for these intents that explicitly want to work with current context
    context_dependent_intents = {
        'followup'
    }
    if intent in context_dependent_intents:
        return False
    
    # Reset if there's no current filtering (nothing to lose)
    if not hasattr(conversation, 'temp_dataset') or conversation.temp_dataset is None:
        return False
    
    current_size = len(conversation.temp_dataset.contents['X'])
    full_size = len(conversation.get_var('dataset').contents['X'])
    
    # No filter applied, nothing to reset
    if current_size == full_size:
        return False
    
    # Check for explicit patient ID references - reset if it's a DIFFERENT patient
    patient_id = entities.get('patient_id')
    if patient_id is not None:
        # If we have a current filter and it's filtering by a different patient ID, reset
        current_filter_ops = getattr(conversation, 'parse_operation', [])
        current_patient_match = None
        for op in current_filter_ops:
            if 'id equal to' in str(op):
                # Extract the current patient ID from the filter description
                try:
                    current_patient_match = int(str(op).split('id equal to ')[1])
                    break
                except:
                    pass
        
        # Reset if asking about a different patient than currently filtered
        if current_patient_match is not None and current_patient_match != patient_id:
            return True
        
        return False  # Keep context when asking about the same patient
    
    # Check for context-preserving keywords in the query
    context_keywords = ['same', 'this', 'that', 'here', 'current', 'filtered', 'than', 'compared', 'versus', 'vs', 'better', 'worse', 'less', 'more']
    query_lower = user_query.lower()
    if any(keyword in query_lower for keyword in context_keywords):
        return False
    
    # Reset for new analysis intents when we have existing filters
    new_analysis_intents = {
        'performance', 'data', 'statistics', 'count', 'important', 'mistakes', 
        'confidence', 'interactions', 'predict', 'whatif'
    }
    
    # Reset if it's a new analysis intent with new filtering criteria
    if intent in new_analysis_intents:
        # Check if this query introduces its own filtering
        has_new_filters = (
            entities.get('features') or 
            entities.get('filter_type') or 
            entities.get('operators') or
            entities.get('prediction_values') or
            entities.get('label_values')
        )
        
        # Reset if we have new filtering criteria or if it's a general question
        return has_new_filters or intent in {'performance', 'data', 'count', 'mistakes'}
    
    return False

def process_user_query(user_query):
    """Shared processing logic for both query routes"""
    # NO MORE AUTO-RESET - Let conversational context persist!
    # Users can explicitly say "reset" or "start fresh" if they want to clear context
    
    # Simple 3-Component Pipeline
    
    # Component 1: AutoGen parses intent
    autogen_response = decoder.complete_sync(user_query, conversation)
    
    # Extract action and entities from AutoGen response
    final_action = autogen_response.get('final_action', 'data')
    entities = autogen_response.get('command_structure', {})
    
    # Extract intent from the nested intent_response for filter reset logic
    intent_response = autogen_response.get('intent_response', {})
    intent = intent_response.get('intent', 'data')
    
    # CRITICAL THINKING FILTER RESET: Check if validation agent determined this query needs full dataset
    validation_response = autogen_response.get('validation_response', {})
    requires_full_dataset = validation_response.get('requires_full_dataset', False)
    
    if requires_full_dataset:
        conversation.reset_temp_dataset()
        logger.info(f"Critical thinking agent determined query requires full dataset - reset filter for: {intent}")
    else:
        # SMART FILTER RESET: Reset filters when starting a new analysis that doesn't reference previous context
        should_reset_filter = _should_reset_filter_context(intent, entities, user_query, conversation)
        if should_reset_filter:
            conversation.reset_temp_dataset()
            logger.info(f"Auto-reset filter context for new analysis: {intent}")
    
    # Component 2: Action dispatcher executes explainability functions  
    # Apply filtering if AutoGen detected filtering entities
    # EXCEPTION: Don't apply filters for performance queries - they need full dataset
    filter_result = None
    if (entities.get('filter_type') or entities.get('features') or entities.get('patient_id') is not None) and final_action != 'score':
        # Auto-apply filtering based on AutoGen entities
        try:
            filter_result = action_dispatcher.execute_action('filter', entities, conversation)
            if filter_result:
                logger.info(f"Auto-applied {entities.get('filter_type', 'feature')} filter: {filter_result}")
        except Exception as e:
            logger.error(f"Auto-filter failed: {e}")
            # Reset on filter failure
            conversation.reset_temp_dataset()
    
    # Execute the main action (but skip filter since it was already applied)
    if final_action != 'filter':
        action_result = action_dispatcher.execute_action(final_action, entities, conversation)
    else:
        action_result = filter_result
    
    # Component 3: Format with LLM (pass AutoGen intent and conversation for context)
    logger.info(f"Action result before formatting: {action_result}")
    formatted_response = formatter.format_response(user_query, final_action, action_result, intent, conversation)
    
    # Add to conversation history with full context
    conversation.add_turn(user_query, formatted_response, final_action, entities, action_result)
    
    return final_action, action_result, formatted_response

@app.route('/')
def home():
    """Render the main conversational interface"""
    return render_template('index.html', 
                         datasetObjective="predict diabetes based on patient health metrics",
                         dataset_size=len(dataset),
                         action_count=len(action_dispatcher.actions))

@app.route('/query', methods=['POST'])
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
        
        action_name, action_result, formatted_response = process_user_query(user_query)
        
        logger.info(f"API response generated using action: {action_name}")
        
        return jsonify({
            "response": formatted_response,
            "action_used": action_name,
            "raw_result": str(action_result)
        })
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500


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
            'whatif': [
                "What if the patient's BMI was 25 instead?",
                "Change glucose to 90 and show the prediction",
                "What would happen if age was 30?"
            ],
            'mistakes': [
                "What are the model's biggest mistakes?",
                "Show me cases where the model was wrong",
                "Which predictions were incorrect?"
            ],
            'interactions': [
                "How do age and BMI interact with each other?",
                "What are the feature interaction effects?",
                "Which features work together?"
            ],
            'statistics': [
                "What are the glucose statistics?",
                "Show me BMI distribution",
                "Give me detailed stats for age"
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
            'about': [
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
            'alternatives': [
                "What are the alternatives to this outcome?",
                "Show alternative scenarios",
                "What other possibilities exist?",
                "Give me alternative explanations"
            ],
            'scenarios': [
                "Show different scenarios",
                "What scenarios would change the outcome?",
                "Generate alternative scenarios",
                "Explore different possibilities"
            ],
            'labels': [
                "What do these labels mean?",
                "Explain the target variable",
                "What is the dataset trying to predict?"
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
            'predictionfilter': [
                "Show cases where model predicted diabetes",
                "Filter to model predictions = 1",
                "Where did the model predict positive?"
            ],
            'labelfilter': [
                "Show actual diabetic patients",
                "Filter to ground truth = 1",
                "Actual positive cases only"
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
        
        action_name, action_result, formatted_response = process_user_query(user_query)
        
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