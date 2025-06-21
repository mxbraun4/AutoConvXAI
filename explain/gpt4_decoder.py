"""GPT-4 Function Calling Decoder for TalkToModel.

This module replaces the complex grammar-guided parsing with clean GPT-4 function calling.
Much simpler and more accurate than the current MP+/T5 approach.
"""
import json
import os
from typing import Dict, List, Any, Optional
import openai


class GPT4Decoder:
    """Clean GPT-4 function calling decoder."""
    
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
        
        # Define the available functions for the model
        self.functions = self._define_explanation_functions()
    
    def _define_explanation_functions(self) -> List[Dict]:
        """Define all available explanation functions for GPT-4."""
        return [
            {
                "name": "explain_prediction",
                "description": "Explain why the model made a specific prediction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_point_id": {"type": "integer", "description": "ID of data point to explain"},
                        "explanation_type": {
                            "type": "string", 
                            "enum": ["lime", "shap", "counterfactual"],
                            "description": "Type of explanation to generate"
                        }
                    },
                    "required": ["data_point_id"]
                }
            },
            {
                "name": "filter_data",
                "description": "Filter the dataset based on feature conditions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "conditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "feature": {"type": "string"},
                                    "operator": {"type": "string", "enum": ["=", ">", "<", ">=", "<=", "!="]},
                                    "value": {"type": ["string", "number", "boolean"]}
                                },
                                "required": ["feature", "operator", "value"]
                            }
                        },
                        "logic": {"type": "string", "enum": ["and", "or"], "default": "and"}
                    },
                    "required": ["conditions"]
                }
            },
            {
                "name": "show_feature_importance",
                "description": "Show which features are most important for predictions",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "num_features": {"type": "integer", "default": 5, "description": "Number of top features to show"},
                        "for_class": {"type": ["string", "integer"], "description": "Specific class to analyze"}
                    }
                }
            },
            {
                "name": "predict_outcome",
                "description": "Make a prediction for a data point or hypothetical scenario", 
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_point_id": {"type": "integer", "description": "Existing data point ID"},
                        "feature_values": {
                            "type": "object",
                            "description": "Feature values for prediction (for what-if scenarios)"
                        }
                    }
                }
            },
            {
                "name": "show_data_summary",
                "description": "Show summary statistics or data overview",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "features": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Specific features to summarize"
                        },
                        "summary_type": {
                            "type": "string",
                            "enum": ["basic", "detailed", "distribution"],
                            "default": "basic"
                        }
                    }
                }
            },
            {
                "name": "what_if_analysis",
                "description": "Analyze how changing feature values affects predictions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "base_data_point_id": {"type": "integer", "description": "Starting data point"},
                        "changes": {
                            "type": "object",
                            "description": "Feature changes to apply"
                        }
                    },
                    "required": ["base_data_point_id", "changes"]
                }
            },
            {
                "name": "find_similar_cases",
                "description": "Find data points similar to a given case",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reference_id": {"type": "integer", "description": "Reference data point ID"},
                        "num_similar": {"type": "integer", "default": 5}
                    },
                    "required": ["reference_id"]
                }
            },
            {
                "name": "analyze_mistakes", 
                "description": "Show cases where the model made incorrect predictions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "error_type": {
                            "type": "string",
                            "enum": ["false_positive", "false_negative", "all"],
                            "default": "all"
                        },
                        "num_examples": {"type": "integer", "default": 5}
                    }
                }
            }
        ]
    
    def _build_context_prompt(self, user_query: str, conversation) -> str:
        """Build context-aware prompt for GPT-4."""
        # Get dataset info
        dataset = conversation.get_var('dataset')
        features_info = ""
        if dataset:
            cat_features = dataset.contents.get('cat', [])
            num_features = dataset.contents.get('numeric', [])
            target_classes = conversation.class_names
            
            features_info = f"""
Dataset Context:
- Categorical features: {', '.join(cat_features)}
- Numerical features: {', '.join(num_features)} 
- Target classes: {', '.join(map(str, target_classes))}
- Dataset size: {len(dataset.contents.get('X', []))} rows
"""

        prompt = f"""{features_info}

User Query: {user_query}

You are helping a user understand a machine learning model's predictions through conversation.
Analyze the user's query and call the appropriate function(s) to help them.

Guidelines:
- For questions about "why" or "how", use explain_prediction
- For "show me cases where..." use filter_data  
- For "what if" scenarios, use what_if_analysis
- For "most important features", use show_feature_importance
- For prediction requests, use predict_outcome
- For finding errors, use analyze_mistakes

Choose the most appropriate function based on the user's intent."""

        return prompt

    def complete(self, user_query: str, conversation, grammar: str = None) -> Dict[str, Any]:
        """Generate completion using GPT-4 function calling.
        
        Args:
            user_query: User's natural language query
            conversation: Conversation object with context
            grammar: Ignored (kept for compatibility)
            
        Returns:
            Dict with function call information compatible with existing system
        """
        prompt = self._build_context_prompt(user_query, conversation)
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": """You are an AI assistant helping users understand a diabetes prediction machine learning model. 

You can either:
1. Chat normally for greetings, general questions, or conversational queries
2. Call specific functions when users want to analyze the model/data

Only call functions when the user specifically wants to:
- Understand feature importance
- Analyze predictions or explanations  
- Filter or explore the data
- Run what-if scenarios
- Find model mistakes

For simple greetings or casual conversation, just respond naturally without calling any functions."""},
                        {"role": "user", "content": prompt}
                    ],
                    functions=self.functions,
                    function_call="auto",
                    temperature=0.3  # Allow some creativity for conversations
                )
                
                message = response.choices[0].message
                
                if message.function_call:
                    # User wants to analyze the model - convert to format expected by existing system
                    function_name = message.function_call.name
                    function_args = json.loads(message.function_call.arguments)
                    
                    # Map GPT-4 functions to existing action format
                    action_mapping = {
                        "explain_prediction": "explain",
                        "filter_data": "filter", 
                        "show_feature_importance": "important",
                        "predict_outcome": "predict",
                        "show_data_summary": "data",
                        "what_if_analysis": "change",
                        "find_similar_cases": "show",
                        "analyze_mistakes": "mistake"
                    }
                    
                    mapped_action = action_mapping.get(function_name, function_name)
                    
                    # Convert function arguments to grammar-like format for existing actions
                    parsed_text = self._convert_to_action_format(mapped_action, function_args)
                    
                    return {
                        "generation": f"parsed: {parsed_text}[e]",
                        "function_call": {
                            "name": function_name,
                            "arguments": function_args
                        },
                        "confidence": 0.95,  # High confidence for GPT-4
                        "method": "gpt4_function_calling"
                    }
                else:
                    # User is just chatting - return their conversational response directly
                    conversational_response = message.content.strip()
                    return {
                        "generation": None,  # No parsing needed
                        "direct_response": conversational_response,
                        "confidence": 0.9,
                        "method": "gpt4_conversation"
                    }
                    
            except Exception as e:
                if attempt < self.max_retries:
                    continue
                else:
                    # Return error information
                    return {
                        "generation": f"parsed: explain[e]",  # Safe fallback
                        "error": str(e),
                        "confidence": 0.1,
                        "method": "gpt4_error"
                    }
    
    def _convert_to_action_format(self, action: str, args: Dict) -> str:
        """Convert GPT-4 function arguments to existing action format."""
        
        if action == "filter" and "conditions" in args:
            # Convert filter conditions to existing format
            conditions = args["conditions"]
            filter_parts = []
            for condition in conditions:
                feature = condition["feature"]
                operator = condition["operator"] 
                value = condition["value"]
                
                # Map operators to existing format
                op_mapping = {"=": "", ">": "greater", "<": "less", ">=": "greaterequal", "<=": "lessequal"}
                op_text = op_mapping.get(operator, "")
                
                if op_text:
                    filter_parts.append(f"filter {feature} {op_text} {value}")
                else:
                    filter_parts.append(f"filter {feature} {value}")
            
            return " ".join(filter_parts)
        
        elif action == "explain" and "data_point_id" in args:
            return f"explain {args['data_point_id']}"
        
        elif action == "important":
            # Default to showing top 5 most important features if no specific feature requested
            if "num_features" in args:
                return f"important topk {args['num_features']}"
            else:
                return "important all"  # Show all features by default
        
        elif action == "predict" and "data_point_id" in args:
            return f"predict {args['data_point_id']}"
        
        elif action == "change" and "changes" in args:
            changes = args["changes"]
            change_parts = []
            for feature, value in changes.items():
                change_parts.append(f"change {feature} {value}")
            return " ".join(change_parts)
        
        else:
            # Generic fallback
            return action


# Integration function to replace existing decoder
def create_gpt4_decoder(**kwargs) -> GPT4Decoder:
    """Factory function to create GPT-4 decoder."""
    return GPT4Decoder(**kwargs)


# Compatibility function for existing codebase
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