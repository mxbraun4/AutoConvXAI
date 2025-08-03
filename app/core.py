"""Core business logic for TalkToModel"""
import logging
from explainability.actions.get_action_functions import get_all_action_functions_map

logger = logging.getLogger(__name__)

class SimpleActionDispatcher:
    """Dispatches explainability actions based on parsed user intents.
    
    This class acts as the central coordinator between AutoGen intent parsing
    and explainability action execution, managing the available actions and
    routing requests to appropriate handlers.
    """
    
    def __init__(self, dataset, model):
        """Initialize dispatcher with dataset and model.
        
        Args:
            dataset: The dataset to operate on
            model: The trained ML model for predictions and explanations
        """
        self.dataset = dataset
        self.model = model
        self.actions = get_all_action_functions_map()
        logger.info(f"Loaded {len(self.actions)} explainability actions")
    
    def execute_action(self, action_name, action_args, conversation):
        """Execute a specific explainability action with given arguments.
        
        Args:
            action_name (str): Name of the action to execute (e.g., 'important', 'predict')
            action_args (dict): Arguments extracted from user query by AutoGen
            conversation: Conversation object maintaining context and state
            
        Returns:
            Result of the action execution or error message
        """
        if action_name not in self.actions:
            return f"Unknown action: {action_name}"
        
        try:
            action_func = self.actions[action_name]
            
            # Ensure conversation object has required attributes for action execution
            if not hasattr(conversation, 'rounding_precision'):
                conversation.rounding_precision = 2
            if not hasattr(conversation, 'default_metric'):
                conversation.default_metric = "accuracy"
            if not hasattr(conversation, 'class_names'):
                conversation.class_names = {0: "No Diabetes", 1: "Diabetes"}
            if not hasattr(conversation, 'describe'):
                describe_obj = type('Describe', (), {})()
                describe_obj.get_dataset_description = lambda: "diabetes prediction based on patient health metrics"
                describe_obj.get_dataset_objective = lambda: "predict diabetes risk in patients"
                describe_obj.get_eval_performance = lambda model, metric: ""
                conversation.describe = describe_obj
            if not hasattr(conversation, 'store_followup_desc'):
                conversation.store_followup_desc = lambda x: None
            
            # Call action function with standardized parameters
            # Actions expect: conversation, parse_text, index, and entity kwargs
            parse_text = action_args.get('parse_text', ['filter'])  # Default to basic data operation
            i = action_args.get('i', 0)  # Default index
            
            # Prepare keyword arguments, excluding positional parameters
            kwargs = {k: v for k, v in action_args.items() if k not in ['parse_text', 'i']}
            result = action_func(conversation, parse_text, i, **kwargs)
            return result
            
        except Exception as e:
            logger.error(f"Error executing {action_name}: {e}")
            return f"Error executing {action_name}: {str(e)}"


def _safe_model_predict(model, data):
    """Safely make predictions handling both DataFrame and numpy array inputs.
    
    Args:
        model: Trained sklearn model
        data: Input data (DataFrame or numpy array)
        
    Returns:
        Model predictions
    """
    try:
        # Try with DataFrame first (preserves feature names)
        return model.predict(data)
    except Exception:
        # Fallback to numpy array if DataFrame fails
        return model.predict(data.values)

def _should_reset_filter_context(action_name, entities, user_query, conversation):
    """Determine whether to reset dataset filters for a new query.
    
    This function implements smart context management by analyzing:
    - Action type and whether it needs full dataset
    - Patient ID references (reset if different patient)
    - Context keywords indicating user wants to maintain current filter
    - New filtering criteria that would conflict with existing filters
    
    Args:
        action_name (str): The action being requested
        entities (dict): Extracted entities from user query
        user_query (str): Original user query text
        conversation: Current conversation state
        
    Returns:
        bool: True if filters should be reset, False to maintain context
    """
    
    # Context-dependent actions that should maintain current filtering
    context_dependent_actions = {
        'followup'
    }
    if action_name in context_dependent_actions:
        return False
    
    # No existing filter to reset - safe to proceed
    if not hasattr(conversation, 'temp_dataset') or conversation.temp_dataset is None:
        return False
    
    current_size = len(conversation.temp_dataset.contents['X'])
    full_size = len(conversation.get_var('dataset').contents['X'])
    
    # No filter applied, nothing to reset
    if current_size == full_size:
        return False
    
    # Handle patient-specific queries - reset only if different patient
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
    
    # Look for keywords indicating user wants to maintain current context
    context_keywords = ['same', 'this', 'that', 'here', 'current', 'filtered', 'than', 'compared', 'versus', 'vs', 'better', 'worse', 'less', 'more']
    query_lower = user_query.lower()
    if any(keyword in query_lower for keyword in context_keywords):
        return False
    
    # Actions that typically start fresh analysis
    new_analysis_actions = {
        'score', 'data', 'statistic', 'important', 'mistakes', 
        'interact', 'predict', 'whatif', 'filter'
    }
    
    # Reset if it's a new analysis action with new filtering criteria
    if action_name in new_analysis_actions:
        # Check if this query introduces its own filtering
        has_new_filters = (
            entities.get('features') or 
            entities.get('filter_type') or 
            entities.get('operators') or
            entities.get('prediction_values') or
            entities.get('label_values')
        )
        
        # Reset if we have new filtering criteria or if it's a general question
        return has_new_filters or action_name in {'score', 'data', 'mistakes'}
    
    return False

def process_user_query(user_query, conversation, action_dispatcher, formatter):
    """Main pipeline for processing user queries through the 3-component architecture.
    
    This function orchestrates the complete query processing flow:
    1. AutoGen multi-agent system parses user intent and extracts entities
    2. Smart filter context management decides whether to maintain or reset data filters
    3. Action dispatcher executes appropriate explainability actions
    4. LLM formatter converts technical results to natural language
    5. Conversation history is updated with context
    
    Args:
        user_query (str): The user's natural language question
        conversation: Conversation object maintaining state and context
        action_dispatcher (SimpleActionDispatcher): Action execution coordinator
        formatter: LLM-based response formatter
        
    Returns:
        tuple: (final_action, action_result, formatted_response)
    """
    from nlu.autogen_decoder import AutoGenDecoder
    import os
    
    # Initialize or reuse AutoGen decoder for intent parsing
    if not hasattr(conversation, '_decoder'):
        api_key = os.getenv('OPENAI_API_KEY')
        model = os.getenv('GPT_MODEL', 'gpt-4o-mini')
        conversation._decoder = AutoGenDecoder(api_key=api_key, model=model, max_rounds=4)
    
    decoder = conversation._decoder
    
    # COMPONENT 1: Multi-agent intent parsing
    autogen_response = decoder.complete_sync(user_query, conversation)
    
    # Extract structured intent and entities from multi-agent response
    final_action = autogen_response.get('final_action', 'data')
    entities = autogen_response.get('command_structure', {})
    
    # Extract intent from the nested intent_response for filter reset logic
    intent_response = autogen_response.get('intent_response', {})
    
    
    # SMART FILTERING: Validation agent determines if full dataset is needed
    validation_response = autogen_response.get('validation_response', {})
    requires_full_dataset = validation_response.get('requires_full_dataset', False)
    
    if requires_full_dataset:
        conversation.reset_temp_dataset()
        logger.info(f"Critical thinking agent determined query requires full dataset - reset filter for: {final_action}")
    else:
        # CONTEXT MANAGEMENT: Reset filters for new analysis not referencing previous context
        should_reset_filter = _should_reset_filter_context(final_action, entities, user_query, conversation)
        if should_reset_filter:
            conversation.reset_temp_dataset()
            logger.info(f"Auto-reset filter context for new analysis: {final_action}")
    
    # COMPONENT 2: Action execution with intelligent filtering
    # Auto-apply filters based on extracted entities, with exceptions for:
    # - Performance queries (need full dataset)
    # - Prediction queries (create new instances)
    # - What-if scenarios (modify current context)
    # - Definition queries (general concepts)
    # - Model queries (independent of patient data)
    filter_result = None
    should_filter = (entities.get('filter_type') or entities.get('features') or entities.get('patient_id') is not None)
    skip_filtering = final_action in ['score', 'predict', 'new_estimate', 'change', 'define', 'model', 'whatif', 'interact']
    
    if should_filter and not skip_filtering:
        # Apply detected filters before main action execution
        try:
            filter_result = action_dispatcher.execute_action('filter', entities, conversation)
            if filter_result:
                logger.info(f"Auto-applied {entities.get('filter_type', 'feature')} filter: {filter_result}")
        except Exception as e:
            logger.error(f"Auto-filter failed: {e}")
            # Reset on filter failure
            conversation.reset_temp_dataset()
    
    # Execute primary action (skip if filter was the main action)
    if final_action != 'filter':
        action_result = action_dispatcher.execute_action(final_action, entities, conversation)
    else:
        action_result = filter_result
    
    # COMPONENT 3: Natural language response formatting
    logger.info(f"Action result before formatting: {action_result}")
    formatted_response = formatter.format_response(user_query, final_action, action_result, conversation)
    
    # Update conversation history with complete turn context
    conversation.add_turn(user_query, formatted_response, final_action, entities, action_result)
    
    return final_action, action_result, formatted_response