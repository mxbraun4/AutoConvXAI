"""LLM-powered response formatter for natural language responses.

Uses OpenAI GPT models to convert structured action results into
conversational responses tailored to user queries.
"""
import logging
import openai

logger = logging.getLogger(__name__)

class LLMFormatter:
    """LLM-powered formatter for context-aware conversational responses.
    
    Uses OpenAI API to generate natural language responses from structured
    action results, with conversation context and query-specific tailoring.
    """
    
    # LLM generation parameters
    MAX_TOKENS = 150  # Keep responses concise
    TEMPERATURE = 0.3  # Low temperature for consistent, factual responses
    CONCEPTUAL_ACTIONS = {'predict', 'whatif', 'define', 'model', 'self', 'new_estimate'}  # Actions that don't need data context
    
    def __init__(self, api_key, model='gpt-4o-mini'):
        """Initialize LLM formatter with OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini for cost efficiency)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def format_response(self, user_query, action_name, action_result, conversation=None):
        """Convert structured action result to natural language response.
        
        Args:
            user_query: Original user question
            action_name: Type of action performed
            action_result: Structured result data
            conversation: Context for filter state and history
            
        Returns:
            str: Natural language response
        """
        
        format_prompt = self._create_intelligent_prompt(user_query, action_name, action_result, conversation)
        
        try:
            # Use direct OpenAI API call for formatting
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data scientist assistant. Provide clear, concise responses based only on the provided data. Be conversational but brief (2-3 sentences max)."},
                    {"role": "user", "content": format_prompt}
                ],
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE
            )
            
            formatted = response.choices[0].message.content.strip()
            
            # Log short responses for debugging but don't fallback
            if not formatted or len(formatted.strip()) < 10:
                logger.warning(f"LLM formatter returned short response: {formatted}")
                
            return formatted
            
        except Exception as e:
            logger.error(f"Error in LLM formatting: {e}")
            # Re-raise the error instead of using fallback
            raise
    
    
    def _build_context_string(self, action_name, conversation):
        """Build context string directly"""
        
        # Skip context for conceptual queries
        if action_name in self.CONCEPTUAL_ACTIONS or not conversation:
            return ""
            
        context_str = ""
        
        # Basic filter info
        try:
            if hasattr(conversation, 'get_filter_context_for_response'):
                filter_context = conversation.get_filter_context_for_response()
                if filter_context['filtered_size'] != filter_context['original_size']:
                    context_str = f"Dataset filtered to {filter_context['filtered_size']} of {filter_context['original_size']} instances. "
        except:
            pass
        
        # Previous result context for comparative analysis
        try:
            if hasattr(conversation, 'last_result') and conversation.last_result:
                if hasattr(conversation, 'last_action') and conversation.last_action:
                    context_str += f"Previous action: {conversation.last_action}. "
                    
                # Add previous query for context continuity
                if hasattr(conversation, 'history') and conversation.history:
                    last_query = conversation.history[-1].get('query', '')
                    if last_query:
                        context_str += f"Previous query: \"{last_query}\". "
                    
                # Add summary of previous result based on type
                prev_result = conversation.last_result
                if isinstance(prev_result, dict):
                    result_type = prev_result.get('type', 'unknown')
                    if result_type == 'filter_results':
                        count = prev_result.get('count', 0)
                        context_str += f"Previous result: Found {count} instances. "
                    elif result_type == 'prediction':
                        pred_value = prev_result.get('prediction', 'unknown')
                        context_str += f"Previous result: Prediction was {pred_value}. "
                    elif result_type == 'feature_statistics':
                        feature = prev_result.get('feature_name', 'unknown')
                        context_str += f"Previous result: Statistics for {feature}. "
                    elif result_type == 'data_summary':
                        dataset_size = prev_result.get('dataset_size', 0)
                        context_str += f"Previous result: Dataset analysis of {dataset_size} instances. "
                    else:
                        context_str += f"Previous result: {result_type} analysis. "
        except:
            pass
        
        return context_str
    
    def _create_intelligent_prompt(self, user_query, action_name, action_result, conversation):
        """Create an intelligent prompt tailored to the specific action"""
        
        # Handle followup action specially
        if action_name == 'followup':
            return self._create_followup_prompt(user_query, action_result)
        
        context_str = self._build_context_string(action_name, conversation)
        
        # Create streamlined prompt focused on specifics
        base_prompt = f"""USER QUESTION: "{user_query}"
ACTION: {action_name}
CONTEXT: {context_str}

RAW RESULTS:
{action_result}

"""
        
        # Add comparative instructions when previous results exist
        has_previous_results = conversation and hasattr(conversation, 'last_result') and conversation.last_result
        if has_previous_results:
            base_prompt += "INSTRUCTION: If user asks comparative questions (e.g., 'compare to before', 'how does this differ'), reference the previous result provided in CONTEXT. "
        
        # Only add special instructions for actions that truly need them
        if action_name == 'whatif':
            base_prompt += "FOCUS: Compare BEFORE/AFTER predictions clearly, state if prediction CHANGED or STAYED SAME"
        
        return base_prompt
    
    def _create_followup_prompt(self, user_query, action_result):
        """Create a simplified prompt for follow-up questions."""
        
        return f"""USER QUESTION: "{user_query}"
SYSTEM RESPONSE: {action_result}

Convert the system response to a conversational answer."""