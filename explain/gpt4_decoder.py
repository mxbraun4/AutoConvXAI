"""GPT-4 Minimal Function Calling Decoder for TalkToModel.

This module uses minimal function calling - just one function that outputs action syntax directly.
No complex mappings, no translation layer, maximum simplicity.
"""
import json
import os
from typing import Dict, List, Any, Optional
import openai


class GPT4Decoder:
    """Minimal GPT-4 function calling decoder."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 max_retries: int = 2):
        """Initialize GPT-4 decoder.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: GPT-4 model to use
            max_retries: Number of retries on errors
        """
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model
        self.max_retries = max_retries
        
        # Define the single function for action execution
        self.functions = self._define_minimal_functions()
    
    def _validate_and_fix_action(self, action_syntax: str) -> str:
        """Validate and fix common action syntax errors."""
        if not action_syntax or not isinstance(action_syntax, str):
            return "explain"
        
        action_syntax = action_syntax.strip().lower()
        parts = action_syntax.split()
        
        if not parts:
            return "explain"
        
        # Fix common errors
        if parts[0] == "important":
            # "important" alone should become "important all"
            if len(parts) == 1:
                return "important all"
            # Validate important parameters
            elif len(parts) >= 2:
                if parts[1] in ["all", "topk"]:
                    if parts[1] == "topk" and len(parts) >= 3:
                        try:
                            int(parts[2])  # Validate number
                            return " ".join(parts[:3])
                        except ValueError:
                            return "important all"
                    return " ".join(parts[:2])
                else:
                    # Assume it's a feature name
                    return " ".join(parts[:2])
        
        elif parts[0] == "filter":
            # Remove compound actions like "filter age greater 50 predict"
            # Keep only the filter part
            if len(parts) >= 4:
                # Handle two-word operators like "greater equal" or "less equal"
                if len(parts) >= 5 and parts[2] in ["greater", "less"] and parts[3] == "equal":
                    # filter feature greater equal value [predict]
                    if len(parts) >= 6 and parts[-1] == "predict":
                        return " ".join(parts[:-1])  # Remove predict
                    return " ".join(parts[:5])  # Keep filter feature greater equal value
                else:
                    # Standard filter: filter feature operator value
                    if parts[-1] == "predict":
                        # Remove the "predict" part
                        return " ".join(parts[:-1])
                    # Keep first 4 parts: filter feature operator value
                    return " ".join(parts[:4])
        
        elif parts[0] in ["predict", "explain", "show"]:
            # These should be fine as-is or with one parameter
            return " ".join(parts[:2])
        
        elif parts[0] in ["score", "mistake", "data"]:
            # These can be standalone or with one parameter
            return " ".join(parts[:2])
        
        # Validate against known actions
        valid_actions = ["filter", "predict", "explain", "important", "score", "show", "change", "mistake", "data"]
        if parts[0] not in valid_actions:
            return "explain"  # Safe fallback
        
        return action_syntax

    def _define_minimal_functions(self) -> List[Dict]:
        """Define minimal function set - just one function that outputs action syntax."""
        return [
            {
                "name": "execute_action",
                "description": "Execute a model analysis action using the action syntax",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action command using syntax like: 'filter age greater 50 predict', 'explain 5', 'score accuracy', etc."
                        },
                        "reasoning": {
                            "type": "string", 
                            "description": "Brief explanation of why this action was chosen"
                        }
                    },
                    "required": ["action"]
                }
            }
        ]
    
    def _build_context_prompt(self, user_query: str, conversation) -> str:
        """Build context prompt with action syntax documentation."""
        # Get dataset info for context
        dataset = conversation.get_var('dataset')
        features_info = ""
        if dataset:
            cat_features = dataset.contents.get('cat', [])
            num_features = dataset.contents.get('numeric', [])
            target_classes = conversation.class_names
            
            features_info = f"""
Dataset Context:
- Features: {', '.join(cat_features + num_features)}
- Target classes: {', '.join(map(str, target_classes))}
- Dataset size: {len(dataset.contents.get('X', []))} rows
"""

        prompt = f"""{features_info}

You help users understand a machine learning model. When they ask for analysis, call the execute_action function.

ACTION SYNTAX REFERENCE:
- filter {'{feature}'} {'{operator}'} {'{value}'} - Filter data (operators: greater, less, greaterequal, lessequal, equal)
- predict - Show predictions for current data
- predict {'{id}'} - Show prediction for specific ID  
- explain {'{id}'} - Explain prediction for ID
- explain {'{id}'} lime - LIME explanation 
- explain {'{id}'} shap - SHAP explanation
- important all - Show all feature importance rankings
- important topk {'{number}'} - Show top N features
- important {'{feature}'} - Show importance of specific feature
- score {'{metric}'} - Model performance (accuracy, precision, recall, f1, roc, default)
- show {'{id}'} - Show data for ID
- change {'{feature}'} {'{operation}'} {'{value}'} - What-if analysis (operations: set, increase, decrease)
- mistake - Show model errors
- data - Show data summary

IMPORTANT RULES:
1. Use ONE action at a time - do not combine actions like "filter age greater 50 predict"
2. For filtering then predicting, use just "filter age greater 50" 
3. For feature importance, always specify "all", "topk N", or a specific feature name
4. Each action must be complete and standalone

EXAMPLES:
"Show predictions for patients over 50" → "filter age greater 50"
"What features are most important?" → "important all"
"Top 3 most important features?" → "important topk 3"
"Why did patient 5 get diabetes?" → "explain 5"
"How accurate is the model?" → "score accuracy"
"Show patient 3" → "show 3"

User Query: {user_query}"""

        return prompt

    def complete(self, user_query: str, conversation, grammar: str = None) -> Dict[str, Any]:
        """Generate completion using minimal GPT-4 function calling.
        
        Args:
            user_query: User's natural language query
            conversation: Conversation object with context
            grammar: Ignored (kept for compatibility)
            
        Returns:
            Dict with action information compatible with existing system
        """
        # Check for casual conversation
        casual_indicators = [
            "hello", "hi", "thanks", "thank you", "bye", "goodbye", 
            "how are you", "what's up", "great", "awesome", "cool"
        ]
        
        if any(indicator in user_query.lower() for indicator in casual_indicators):
            # Handle as conversation without function calling
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for a diabetes prediction model. Respond conversationally to greetings and casual remarks."},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.7
                )
                
                conversational_response = response.choices[0].message.content.strip()
                return {
                    "generation": None,
                    "direct_response": conversational_response,
                    "confidence": 0.9,
                    "method": "gpt4_conversation"
                }
            except Exception as e:
                return {
                    "generation": None,
                    "direct_response": "Hello! I'm here to help you understand the diabetes prediction model. What would you like to know?",
                    "confidence": 0.8,
                    "method": "gpt4_fallback"
                }
        
        # Use function calling for analysis queries
        prompt = self._build_context_prompt(user_query, conversation)
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at understanding machine learning queries. For analysis requests, call the execute_action function with the appropriate action syntax. For casual conversation, respond normally."},
                        {"role": "user", "content": prompt}
                    ],
                    functions=self.functions,
                    function_call="auto",
                    temperature=0.2
                )
                
                message = response.choices[0].message
                
                if message.function_call:
                    # GPT-4 called the function - extract the action
                    function_args = json.loads(message.function_call.arguments)
                    action_syntax = function_args.get("action", "explain")
                    
                    # Validate and fix common action syntax errors
                    original_action = action_syntax
                    action_syntax = self._validate_and_fix_action(action_syntax)
                    
                    # Log if action was modified
                    if original_action != action_syntax:
                        print(f"GPT4Decoder: Fixed action '{original_action}' → '{action_syntax}'")
                    
                    return {
                        "generation": f"parsed: {action_syntax}[e]",
                        "confidence": 0.95,
                        "method": "gpt4_minimal_function_calling",
                        "reasoning": function_args.get("reasoning", ""),
                        "raw_action": action_syntax
                    }
                else:
                    # GPT-4 responded conversationally
                    conversational_response = message.content.strip()
                    return {
                        "generation": None,
                        "direct_response": conversational_response,
                        "confidence": 0.9,
                        "method": "gpt4_conversation"
                    }
                    
            except Exception as e:
                if attempt < self.max_retries:
                    continue
                else:
                    # Return safe fallback
                    return {
                        "generation": f"parsed: explain[e]",
                        "error": str(e),
                        "confidence": 0.1,
                        "method": "gpt4_error"
                    }


# Integration functions for compatibility
def create_gpt4_decoder(**kwargs) -> GPT4Decoder:
    """Factory function to create GPT-4 decoder."""
    return GPT4Decoder(**kwargs)


def get_gpt4_predict_func(api_key: str = None, model: str = "gpt-4"):
    """Get prediction function compatible with existing decoder interface."""
    decoder = GPT4Decoder(api_key=api_key, model=model)
    
    def predict_func(prompt: str, grammar: str = None, conversation=None):
        # Extract actual user query from prompt if needed
        if "Query:" in prompt:
            user_query = prompt.split("Query:")[-1].strip()
        else:
            user_query = prompt
            
        return decoder.complete(user_query, conversation, grammar)
    
    return predict_func 