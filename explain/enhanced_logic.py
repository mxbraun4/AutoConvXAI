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
                 feature_definitions: dict = None):
        """Initialize enhanced bot with AutoGen multi-agent system.

        Args:
            ... (same as original ExplainBot) ...
            openai_api_key: OpenAI API key for GPT-4 
            gpt_model: GPT-4 model variant to use for all agents
        """
        # Set seeds
        np.random.seed(seed)
        py_random_seed(seed)

        self.bot_name = name

        # Initialize AutoGen multi-agent decoder
        app.logger.info(f"Loading AutoGen multi-agent decoder ({gpt_model})...")
        
        # Check for API key
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass openai_api_key parameter.")
        
        try:
            from explain.autogen_decoder import AutoGenDecoder
            self.decoder = AutoGenDecoder(api_key=api_key, model=gpt_model)
            app.logger.info("✅ AutoGen multi-agent decoder ready!")
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

        # Load explanations (unchanged)
        self.load_explanations(background_dataset=background_dataset)

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
        """Load explanations (unchanged from original)."""
        app.logger.info("Loading explanations...")

        pred_f = self.conversation.get_var('model_prob_predict').contents
        model = self.conversation.get_var('model').contents
        data = self.conversation.get_var('dataset').contents['X']
        categorical_f = self.conversation.get_var('dataset').contents['cat']
        numeric_f = self.conversation.get_var('dataset').contents['numeric']

        # Load LIME/SHAP explanations
        # MegaExplainer expects the 'X' DataFrame from background_dataset
        mega_explainer = MegaExplainer(prediction_fn=pred_f,
                                       data=background_dataset['X'],
                                       cat_features=categorical_f,
                                       class_names=self.conversation.class_names)
        mega_explainer.get_explanations(ids=list(data.index), data=data)
        
        # Load counterfactual explanations
        tabular_dice = TabularDice(model=model,
                                   data=data,
                                   num_features=numeric_f,
                                   class_names=self.conversation.class_names)
        tabular_dice.get_explanations(ids=list(data.index), data=data)

        # Add to conversation
        self.conversation.add_var('mega_explainer', mega_explainer, 'explanation')
        self.conversation.add_var('tabular_dice', tabular_dice, 'explanation')

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
                
                # Use smart dispatcher for more intelligent action handling
                try:
                    from explain.smart_action_dispatcher import smart_dispatcher
                    from explain.actions.get_action_functions import get_all_action_functions_map
                    
                    # Create a module-like object with the actions dictionary
                    class ActionModule:
                        actions = get_all_action_functions_map()
                    
                    response, status = smart_dispatcher.parse_and_execute(
                        parsed_text, 
                        user_session_conversation, 
                        ActionModule()
                    )
                except ImportError:
                    # Fallback to original action system if smart dispatcher not available
                    app.logger.warning("Smart dispatcher not available, using legacy action system")
                    response = run_action(user_session_conversation, None, parsed_text)
                
                # Add helpful metadata for debugging
                method = completion.get("method", "autogen_multi_agent")
                confidence = completion.get("confidence", 0.0)
                app.logger.info(f"Action '{parsed_text}' executed via {method} (confidence: {confidence:.2f})")
                
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
            app.logger.error(f"Full traceback: {traceback.format_exc()}")
            return "I encountered an error processing your request. Please try again."


# Example configuration update
def update_config_for_autogen():
    """Show how to update gin config for AutoGen multi-agent system."""
    
    config_update = """
# AutoGen Multi-Agent Configuration
EnhancedExplainBot.openai_api_key = None  # Will use OPENAI_API_KEY env var
EnhancedExplainBot.gpt_model = "gpt-4o"   # Model used by all agents

# Remove all the T5/MP+ complexity - no longer needed with multi-agent approach:
# ExplainBot.parsing_model_name = "ucinlp/diabetes-t5-small"  # No longer needed
# ExplainBot.t5_config = None  # No longer needed
# ExplainBot.use_guided_decoding = True  # No longer needed
"""
    
    print("📝 Add this to your gin config for AutoGen multi-agent:")
    print(config_update)


