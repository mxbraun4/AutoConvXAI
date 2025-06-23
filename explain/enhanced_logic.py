"""Enhanced ExplainBot with clean GPT-4 function calling.

This replaces the complex grammar-guided parsing with simple, accurate GPT-4 function calls.
"""
import pickle
from random import seed as py_random_seed
import secrets
import os

import numpy as np
from flask import Flask

from explain.action import run_action
from explain.conversation import Conversation
from explain.explanation import MegaExplainer, TabularDice
from explain.utils import read_and_format_data
from explain.write_to_log import log_dialogue_input

app = Flask(__name__)


def load_sklearn_model(filepath):
    """Loads a sklearn model."""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


class EnhancedExplainBot:
    """Enhanced ExplainBot using GPT-4 function calling instead of complex parsing."""

    def __init__(self,
                 model_file_path: str,
                 dataset_file_path: str,
                 background_dataset_file_path: str,
                 dataset_index_column: int,
                 target_variable_name: str,
                 categorical_features: list[str],
                 numerical_features: list[str],
                 remove_underscores: bool,
                 name: str,
                 openai_api_key: str = None,
                 gpt_model: str = "gpt-4o",
                 seed: int = 0,
                 feature_definitions: dict = None,
                 preload_explanations: bool = False,
                 use_generalized_actions: bool = False):
        """Initialize enhanced bot with AutoGen multi-agent system.

        Args:
            ... (same as original ExplainBot) ...
            openai_api_key: OpenAI API key for GPT-4 
            gpt_model: GPT-4 model variant to use for all agents
            use_generalized_actions: If True, uses generalized tool-augmented agents
                                   instead of hardcoded action mappings
        """
        # Set seeds
        np.random.seed(seed)
        py_random_seed(seed)

        self.bot_name = name
        self.use_generalized_actions = use_generalized_actions

        # Initialize AutoGen multi-agent decoder
        app.logger.info(f"Loading AutoGen multi-agent decoder ({gpt_model})...")
        
        # Check for API key
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass openai_api_key parameter.")
        
        try:
            from explain.autogen_decoder import AutoGenDecoder
            self.decoder = AutoGenDecoder(api_key=api_key, model=gpt_model)
            
            if self.use_generalized_actions:
                app.logger.info("✅ AutoGen multi-agent decoder ready with GENERALIZED TOOL-AUGMENTED ACTIONS!")
            else:
                app.logger.info("✅ AutoGen multi-agent decoder ready with traditional actions!")
                
        except ImportError as e:
            error_msg = f"AutoGen packages not available: {e}"
            app.logger.error(error_msg)
            raise ImportError(f"{error_msg}\nPlease install with: pip install autogen-agentchat>=0.4.0 autogen-core>=0.4.0 autogen-ext>=0.4.0")
        except Exception as e:
            app.logger.error(f"Failed to initialize AutoGen decoder: {e}")
            raise

        # Set up conversation (same as before)
        self.conversation = Conversation(eval_file_path=dataset_file_path,
                                         feature_definitions=feature_definitions)
        
        # Make bot available to conversation for on-demand loading
        self.conversation.bot = self
        
        # Initialize dataset description
        from explain.dataset_description import DatasetDescription
        self.conversation.describe = DatasetDescription()

        # Load model and dataset (unchanged)
        self.load_model(model_file_path)
        self.load_dataset(dataset_file_path,
                          dataset_index_column,
                          target_variable_name,
                          categorical_features,
                          numerical_features,
                          remove_underscores)

        background_dataset = self.load_dataset_background(background_dataset_file_path,
                                                          dataset_index_column,
                                                          target_variable_name,
                                                          categorical_features,
                                                          numerical_features,
                                                          remove_underscores)

        # Store background dataset for later use
        self.background_dataset = background_dataset
        self.explanations_loaded = False
        self.preload_explanations = preload_explanations
        
        # Always set up lazy explainer (instantaneous, no computation)
        self.load_explanations(background_dataset=background_dataset)
        app.logger.info("⚡ Using lazy explanation loading for instant startup!")

    def load_model(self, filepath: str):
        """Load model (unchanged from original)."""
        model = load_sklearn_model(filepath)
        self.conversation.add_var('model', model, 'model')
        
        # Also store prediction functions
        predict_f = model.predict
        predict_proba_f = getattr(model, 'predict_proba', None)
        
        self.conversation.add_var('model_predict', predict_f, 'function')
        if predict_proba_f:
            self.conversation.add_var('model_prob_predict', predict_proba_f, 'function')

    def load_dataset(self, filepath: str, index_col: int, target_var_name: str,
                     cat_features: list[str], num_features: list[str], 
                     remove_underscores: bool):
        """Load dataset (unchanged from original)."""
        
        # read_and_format_data returns a tuple: (dataset, y_values, cat_features, num_features)
        dataset, y_values, cat_features, num_features = read_and_format_data(filepath, 
                                                                             index_col,
                                                                             target_var_name, 
                                                                             cat_features,
                                                                             num_features,
                                                                             remove_underscores)
        
        # Create dataset dictionary in the expected format
        dataset_dict = {
            'X': dataset,
            'y': y_values,
            'cat': cat_features,
            'numeric': num_features,
            'ids_to_regenerate': []
        }
        
        self.conversation.add_var('dataset', dataset_dict, 'dataset')
        
        # Store class names for conversation
        unique_targets = y_values.unique()
        self.conversation.class_names = [str(x) for x in unique_targets]

    def load_dataset_background(self, filepath: str, index_col: int, target_var_name: str,
                               cat_features: list[str], num_features: list[str], 
                               remove_underscores: bool):
        """Load background dataset for explanations."""
        # read_and_format_data returns a tuple, we need to convert it to the expected format
        dataset, y_values, cat_features, num_features = read_and_format_data(filepath, index_col, target_var_name, 
                                                                             cat_features, num_features, remove_underscores)
        
        # Return in the expected format for MegaExplainer
        return {
            'X': dataset,
            'y': y_values,
            'cat': cat_features,
            'numeric': num_features
        }

    def load_explanations(self, background_dataset):
        """Load explanations using lazy loading for better performance."""
        app.logger.info("Setting up lazy explanation loading...")

        pred_f = self.conversation.get_var('model_prob_predict').contents
        model = self.conversation.get_var('model').contents
        data = self.conversation.get_var('dataset').contents['X']
        categorical_f = self.conversation.get_var('dataset').contents['cat']
        numeric_f = self.conversation.get_var('dataset').contents['numeric']

        # Use LazyMegaExplainer instead of MegaExplainer
        from explain.lazy_mega_explainer import LazyMegaExplainer
        
        lazy_explainer = LazyMegaExplainer(
            prediction_fn=pred_f,
            data=background_dataset['X'],
            cat_features=categorical_f,
            class_names=self.conversation.class_names
        )
        
        # Load counterfactual explanations (keep these as-is for now)
        tabular_dice = TabularDice(model=model,
                                   data=data,
                                   num_features=numeric_f,
                                   class_names=self.conversation.class_names)
        # Don't pre-compute all counterfactuals either
        # tabular_dice.get_explanations(ids=list(data.index), data=data)

        # Add to conversation
        self.conversation.add_var('mega_explainer', lazy_explainer, 'explanation')
        self.conversation.add_var('tabular_dice', tabular_dice, 'explanation')
        self.explanations_loaded = True
        
        app.logger.info("✅ Lazy explanation loading configured (explanations computed on-demand)")

    def ensure_explanations_loaded(self):
        """Load explanations on-demand if not already loaded."""
        if not self.explanations_loaded:
            app.logger.info("Loading explanations on-demand...")
            self.load_explanations(self.background_dataset)
    
    def load_explanations_for_instance(self, instance_id: int):
        """Load explanations for a specific instance only (more efficient)."""
        if not hasattr(self, 'lite_explainer'):
            # Create a lightweight explainer that only computes when needed
            pred_f = self.conversation.get_var('model_prob_predict').contents
            model = self.conversation.get_var('model').contents
            data = self.conversation.get_var('dataset').contents['X']
            categorical_f = self.conversation.get_var('dataset').contents['cat']
            numeric_f = self.conversation.get_var('dataset').contents['numeric']
            
            # Create explainer but don't pre-compute all explanations
            self.lite_explainer = MegaExplainer(prediction_fn=pred_f,
                                               data=self.background_dataset['X'],
                                               cat_features=categorical_f,
                                               class_names=self.conversation.class_names)
            
            # Add to conversation for compatibility
            self.conversation.add_var('mega_explainer', self.lite_explainer, 'explanation')
        
        # Only compute explanation for the specific instance
        if instance_id in self.conversation.get_var('dataset').contents['X'].index:
            instance_data = self.conversation.get_var('dataset').contents['X'].loc[[instance_id]]
            self.lite_explainer.get_explanations(ids=[instance_id], data=instance_data)

    @staticmethod
    def gen_almost_surely_unique_id(n_bytes: int = 30):
        """Generate unique ID."""
        return secrets.token_hex(n_bytes)

    @staticmethod
    def log(logging_input: dict):
        """Log conversation."""
        log_dialogue_input(logging_input)

    def update_state(self, text: str, user_session_conversation: Conversation):
        """Enhanced update_state using AutoGen multi-agent system.
        
        The multi-agent system provides specialized agents for better accuracy and debugging.
        """
        try:
            app.logger.info(f"User query: {text}")
            
            # Use AutoGen multi-agent system
            completion = self.decoder.complete_sync(text, user_session_conversation)
            app.logger.info(f"AutoGen completion: {completion}")
            
            # Check if we need to reset context based on agent analysis
            if completion.get("intent_response") and completion["intent_response"].get("entities", {}).get("context_reset", False):
                app.logger.info("Context reset detected - rebuilding temp dataset")
                user_session_conversation.build_temp_dataset()
            
            # Handle both conversational responses and action commands consistently
            if completion.get("direct_response"):
                # Agents decided to respond conversationally (no function call needed)
                app.logger.info(f"Conversational response: {completion['direct_response']}")
                return completion["direct_response"]
                
            elif completion.get("generation"):
                # Agents decided to call a function - run through action system
                generation = completion["generation"]
                app.logger.info(f"Generated action: {generation}")
                
                # More robust parsing - handle both formats
                parsed_text = None
                if "parsed:" in generation:
                    # Extract action from "parsed: action[e]" format
                    parsed_parts = generation.split("parsed:")
                    if len(parsed_parts) > 1:
                        action_part = parsed_parts[-1].strip()
                        # Remove [e] suffix if present
                        if "[e]" in action_part:
                            parsed_text = action_part.split("[e]")[0].strip()
                        else:
                            parsed_text = action_part.strip()
                else:
                    # Handle direct action format
                    parsed_text = generation.strip()
                
                # Ensure we have a valid action
                if not parsed_text or len(parsed_text.strip()) == 0:
                    parsed_text = "explain"  # Safe fallback
                
                app.logger.info(f"Parsed action: {parsed_text}")
                
                # Route to generalized or traditional action system
                if self.use_generalized_actions:
                    response = self._handle_generalized_action(text, completion, user_session_conversation)
                else:
                    response = self._handle_traditional_action(parsed_text, user_session_conversation)
                
                # Add helpful metadata for debugging
                method = completion.get("method", "autogen_multi_agent")
                confidence = completion.get("confidence", 0.0)
                action_type = "generalized" if self.use_generalized_actions else "traditional"
                app.logger.info(f"Action '{parsed_text}' executed via {method} ({action_type} mode, confidence: {confidence:.2f})")
                
                # Log agent reasoning if available (AutoGen feature)
                if "agent_reasoning" in completion:
                    app.logger.info(f"Agent reasoning: {completion['agent_reasoning']}")
                
                return response
            else:
                app.logger.warning("No generation or direct response in completion")
                return "I'm sorry, I couldn't understand your request. Could you please rephrase it?"
                
        except Exception as e:
            app.logger.error(f"Error in AutoGen multi-agent processing: {e}")
            import traceback
            full_traceback = traceback.format_exc()
            app.logger.error(f"Full traceback: {full_traceback}")
            
            # Try to provide more helpful error information
            error_context = f"Query: '{text}'"
            app.logger.error(f"Error context: {error_context}")
            
            # Attempt a simple fallback for common patterns
            try:
                app.logger.info("Attempting fallback processing...")
                fallback_result = self._simple_fallback_processing(text, user_session_conversation)
                if fallback_result:
                    app.logger.info("Fallback processing succeeded")
                    return fallback_result
            except Exception as fallback_error:
                app.logger.error(f"Fallback processing also failed: {fallback_error}")
            
            return f"I encountered an error processing your request. Please try rephrasing your question. (Error: {str(e)})"

    def _handle_generalized_action(self, text: str, completion: dict, conversation: Conversation) -> str:
        """Handle action using the new generalized tool-augmented system."""
        try:
            # Import generalized agents - this should ALWAYS work when generalized actions are enabled
            from explain.generalized_agents import generalized_action_dispatcher
            
            # Extract intent and entities from AutoGen completion
            intent_response = completion.get("intent_response", {})
            intent = intent_response.get("intent", "unknown")
            entities = intent_response.get("entities", {})
            
            app.logger.info(f"Generalized dispatcher: intent='{intent}', entities={entities}")
            
            # Use generalized action dispatcher with original user query
            response, status = generalized_action_dispatcher(text, intent, entities, conversation)
            
            if status == 0:
                app.logger.warning(f"Generalized action failed: {response}")
                # Fallback to traditional system if generalized fails
                app.logger.info("Falling back to traditional action system...")
                # Parse the generation properly before passing to traditional system
                generation = completion.get("generation", "explain")
                parsed_fallback = self._parse_generation(generation)
                return self._handle_traditional_action(parsed_fallback, conversation)
            
            return response
            
        except ImportError as e:
            app.logger.error(f"CRITICAL: Cannot import generalized agents when use_generalized_actions=True: {e}")
            raise ImportError(f"Generalized agents are required but cannot be imported: {e}")
        except Exception as e:
            app.logger.error(f"Error in generalized action handling: {e}")
            # Fallback to traditional system
            app.logger.info("Error in generalized system, falling back to traditional...")
            generation = completion.get("generation", "explain")
            parsed_fallback = self._parse_generation(generation)
            return self._handle_traditional_action(parsed_fallback, conversation)

    def _handle_traditional_action(self, parsed_text: str, conversation: Conversation) -> str:
        """Handle action using the traditional hardcoded action system."""
        try:
            # Use smart dispatcher for more intelligent action handling
            try:
                from explain.smart_action_dispatcher import get_smart_dispatcher
                from explain.actions.get_action_functions import get_all_action_functions_map
                
                # Get API key
                api_key = self.decoder.api_key if hasattr(self.decoder, 'api_key') else os.getenv('OPENAI_API_KEY')
                
                # Get available features
                available_features = []
                if hasattr(conversation, 'dataset') and conversation.dataset:
                    dataset = conversation.dataset.contents
                    if 'numeric' in dataset:
                        available_features.extend(dataset['numeric'])
                    if 'cat' in dataset:
                        available_features.extend(dataset['cat'])
                
                # Use smart dispatcher
                dispatcher = get_smart_dispatcher(api_key)
                response, status = dispatcher.dispatch(
                    parsed_text,
                    conversation,
                    get_all_action_functions_map(),
                    available_features
                )
                
                if status == 0:
                    app.logger.warning(f"Traditional action failed: {response}")
                
                return response
                
            except ImportError:
                # Fallback to original action system if smart dispatcher not available
                app.logger.warning("Smart dispatcher not available, using legacy action system")
                return run_action(conversation, None, parsed_text)
            except Exception as e:
                app.logger.error(f"Error in smart dispatcher: {e}")
                # Fallback to original action system
                return run_action(conversation, None, parsed_text)
                
        except Exception as e:
            app.logger.error(f"Error in traditional action handling: {e}")
            return f"Error processing action: {str(e)}"

    def _parse_generation(self, generation: str) -> str:
        """Parse generation text and extract clean action command."""
        # More robust parsing - handle both formats
        parsed_text = None
        if "parsed:" in generation:
            # Extract action from "parsed: action[e]" format
            parsed_parts = generation.split("parsed:")
            if len(parsed_parts) > 1:
                action_part = parsed_parts[-1].strip()
                # Remove [e] suffix if present
                if "[e]" in action_part:
                    parsed_text = action_part.split("[e]")[0].strip()
                else:
                    parsed_text = action_part.strip()
        else:
            # Handle direct action format
            parsed_text = generation.strip()
        
        # Ensure we have a valid action
        if not parsed_text or len(parsed_text.strip()) == 0:
            parsed_text = "explain"  # Safe fallback
        
        return parsed_text

    def _simple_fallback_processing(self, text: str, conversation) -> str:
        """Simple fallback processing for common patterns when AutoGen fails."""
        text_lower = text.lower()
        
        # Pattern 1: Feature importance
        if any(word in text_lower for word in ['important', 'importance', 'significant', 'key']) and 'feature' in text_lower:
            app.logger.info("Fallback: Detected feature importance query")
            from explain.action import run_action
            return run_action(conversation, None, "important all")
            
        # Pattern 2: Explanation requests with ID
        import re
        explain_match = re.search(r'(?:why|explain).*?(?:person|patient|instance|id)\s*(?:with\s+id\s+)?(\d+)', text_lower)
        if explain_match:
            patient_id = explain_match.group(1)
            app.logger.info(f"Fallback: Detected explanation request for ID {patient_id}")
            from explain.action import run_action
            return run_action(conversation, None, f"filter id {patient_id} explain")
            
        # Pattern 3: Pregnancy predictions
        if 'predict' in text_lower and ('pregnant' in text_lower or 'pregnancy' in text_lower):
            if any(word in text_lower for word in ['no', 'not', 'false', 'never']):
                app.logger.info("Fallback: Detected non-pregnant prediction query")
                from explain.action import run_action
                return run_action(conversation, None, "filter pregnancies equal 0 predict")
            else:
                app.logger.info("Fallback: Detected pregnant prediction query")
                from explain.action import run_action
                return run_action(conversation, None, "filter pregnancies greater 0 predict")
        
        # Pattern 4: Simple prediction requests
        if 'predict' in text_lower:
            predict_match = re.search(r'(?:predict|prediction).*?(?:person|patient|instance|id)\s*(\d+)', text_lower)
            if predict_match:
                patient_id = predict_match.group(1)
                app.logger.info(f"Fallback: Detected prediction request for ID {patient_id}")
                from explain.action import run_action
                return run_action(conversation, None, f"filter id {patient_id} predict")
        
        return None


# Example configuration update
def update_config_for_autogen():
    """Show how to update gin config for AutoGen multi-agent system."""
    
    config_update = """
# AutoGen Multi-Agent Configuration
EnhancedExplainBot.openai_api_key = None  # Will use OPENAI_API_KEY env var
EnhancedExplainBot.gpt_model = "gpt-4o"   # Model used by all agents
EnhancedExplainBot.preload_explanations = False  # Skip slow explanation pre-computation

# NEW: Generalized Action System (EXPERIMENTAL)
# Set to True to use tool-augmented agents instead of hardcoded actions
EnhancedExplainBot.use_generalized_actions = False  # Set to True to enable

# GENERALIZED MODE BENEFITS:
# ✅ Automatically adapts to any dataset structure
# ✅ Reduces hardcoded rules and mappings  
# ✅ Uses dynamic pandas/sklearn code generation
# ✅ More flexible and extensible
# ✅ Better error handling with automatic fallbacks
#
# TRADITIONAL MODE BENEFITS:
# ✅ Well-tested with existing datasets
# ✅ Optimized for diabetes prediction use case
# ✅ Fast execution for known action patterns

# For faster startup, explanations are loaded on-demand
# Set to True if you want all explanations pre-computed (slower startup, faster explanation queries)

# Remove all the T5/MP+ complexity - no longer needed with multi-agent approach:
# ExplainBot.parsing_model_name = "ucinlp/diabetes-t5-small"  # No longer needed
# ExplainBot.t5_config = None  # No longer needed
# ExplainBot.use_guided_decoding = True  # No longer needed
"""
    
    print("📝 Add this to your gin config for AutoGen multi-agent:")
    print(config_update)


