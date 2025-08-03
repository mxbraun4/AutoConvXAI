"""Core business logic for TalkToModel"""
import logging
from explainability.actions.get_action_functions import get_all_action_functions_map

logger = logging.getLogger(__name__)

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
            
            # Call action with expected parameters (conversation, parse_text, i, **kwargs)
            # Most actions expect these 3 parameters plus optional kwargs for entities
            parse_text = action_args.get('parse_text', ['filter'])  # Default to basic data operation
            i = action_args.get('i', 0)  # Default index
            
            # Create kwargs excluding positional parameters to avoid conflicts
            kwargs = {k: v for k, v in action_args.items() if k not in ['parse_text', 'i']}
            result = action_func(conversation, parse_text, i, **kwargs)
            return result
            
        except Exception as e:
            logger.error(f"Error executing {action_name}: {e}")
            return f"Error executing {action_name}: {str(e)}"


def _safe_model_predict(model, data):
    """Utility function to predict with sklearn compatibility (avoids feature name warnings)."""
    try:
        # Try with DataFrame first (preserves feature names)
        return model.predict(data)
    except Exception:
        # Fallback to numpy array if DataFrame fails
        return model.predict(data.values)

def _should_reset_filter_context(action_name, entities, user_query, conversation):
    """Determine if we should reset the filter context for a new query."""
    
    # Never reset for these actions that explicitly want to work with current context
    context_dependent_actions = {
        'followup'
    }
    if action_name in context_dependent_actions:
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
    
    # Reset for new analysis actions when we have existing filters
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
    """Shared processing logic for both query routes"""
    from nlu.autogen_decoder import AutoGenDecoder
    import os
    
    # Get decoder from conversation or create new one
    if not hasattr(conversation, '_decoder'):
        api_key = os.getenv('OPENAI_API_KEY')
        model = os.getenv('GPT_MODEL', 'gpt-4o-mini')
        conversation._decoder = AutoGenDecoder(api_key=api_key, model=model, max_rounds=4)
    
    decoder = conversation._decoder
    
    # Component 1: AutoGen parses intent
    autogen_response = decoder.complete_sync(user_query, conversation)
    
    # Extract action and entities from AutoGen response
    final_action = autogen_response.get('final_action', 'data')
    entities = autogen_response.get('command_structure', {})
    
    # Extract intent from the nested intent_response for filter reset logic
    intent_response = autogen_response.get('intent_response', {})
    
    
    # CRITICAL THINKING FILTER RESET: Check if validation agent determined this query needs full dataset
    validation_response = autogen_response.get('validation_response', {})
    requires_full_dataset = validation_response.get('requires_full_dataset', False)
    
    if requires_full_dataset:
        conversation.reset_temp_dataset()
        logger.info(f"Critical thinking agent determined query requires full dataset - reset filter for: {final_action}")
    else:
        # SMART FILTER RESET: Reset filters when starting a new analysis that doesn't reference previous context
        should_reset_filter = _should_reset_filter_context(final_action, entities, user_query, conversation)
        if should_reset_filter:
            conversation.reset_temp_dataset()
            logger.info(f"Auto-reset filter context for new analysis: {final_action}")
    
    # Component 2: Action dispatcher executes explainability functions  
    # Apply filtering if AutoGen detected filtering entities
    # EXCEPTIONS: Don't apply filters for these actions:
    # - 'score': performance queries need full dataset
    # - 'predict': prediction queries with '=' operators should create new instances, not filter
    # - 'change': what-if scenarios should modify current context, not filter to existing data
    # - 'define': definition queries explain general concepts, not specific instances
    # - 'model': model information queries are independent of patient data
    # - 'new_estimate': new instance predictions should create hypothetical instances, not filter existing data
    filter_result = None
    should_filter = (entities.get('filter_type') or entities.get('features') or entities.get('patient_id') is not None)
    skip_filtering = final_action in ['score', 'predict', 'new_estimate', 'change', 'define', 'model', 'whatif', 'interact']
    
    if should_filter and not skip_filtering:
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
    
    # Component 3: Format with LLM (pass action and conversation for context)
    logger.info(f"Action result before formatting: {action_result}")
    formatted_response = formatter.format_response(user_query, final_action, action_result, conversation)
    
    # Add to conversation history with full context
    conversation.add_turn(user_query, formatted_response, final_action, entities, action_result)
    
    return final_action, action_result, formatted_response