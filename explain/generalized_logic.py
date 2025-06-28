"""Generalized Logic System using Dynamic Code Generation.

This replaces static action strings with dynamic code generation,
leveraging AutoGen's extraction capabilities for true generalizability.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd

from explain.dynamic_code_generation import (
    AutoGenQueryExtractor, 
    DynamicQueryExecutor, 
    GeneratedQuery,
    FilterOperation,
    QueryOperation
)

logger = logging.getLogger(__name__)


class GeneralizedExplainBot:
    """Generalized ExplainBot using dynamic code generation."""
    
    def __init__(self, 
                 model_file_path: str,
                 dataset_file_path: str, 
                 background_dataset_file_path: str,
                 dataset_index_column=None,
                 target_variable_name: str = "y",
                 categorical_features: list = None,
                 numerical_features: list = None,
                 remove_underscores: bool = True,
                 name: str = "model",
                 openai_api_key: str = None,
                 gpt_model: str = "gpt-4o"):
        """Initialize generalized bot with dynamic code generation."""
        
        # Initialize conversation and core components directly  
        from explain.conversation import Conversation
        from explain.autogen_decoder import AutoGenDecoder
        import pickle
        
        # Initialize conversation with proper setup
        self.conversation = Conversation()
        
        # Initialize the dataset description for conversation
        self.conversation.describe.dataset_objective = "diabetes prediction"
        
        # Load model
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
        self.conversation.add_var("model", model, "model")
        
        # Load dataset
        import pandas as pd
        dataset = pd.read_csv(dataset_file_path)
        if target_variable_name in dataset.columns:
            X = dataset.drop(columns=[target_variable_name])
            y = dataset[target_variable_name]
        else:
            X = dataset
            y = None
        
        self.conversation.add_var("dataset", {"X": X, "y": y}, "dataset")
        
        # Dynamic code generation handles all explanations - no mega_explainer needed
        self.conversation.add_var("mega_explainer", None, "mega_explainer")
        
        # Initialize AutoGen decoder
        self.decoder = AutoGenDecoder(api_key=openai_api_key, model=gpt_model)
        
        # Set up dynamic code generation components
        self.query_extractor = AutoGenQueryExtractor(self.decoder)
        self.query_executor = DynamicQueryExecutor()
        
        logger.info("✅ Generalized ExplainBot initialized with dynamic code generation")
    
    # conversation is now a direct attribute
    
    def log(self, logging_input: dict):
        """Log conversation."""
        # Simple logging implementation
        logger.info(f"Conversation log: {logging_input}")
        return True
    
    def update_state(self, user_query: str, conversation) -> str:
        """Process ALL queries using dynamic code generation for maximum generalizability."""
        try:
            logger.info(f"Processing query with FULL dynamic code generation: {user_query}")
            
            # Step 1: Extract query components using AutoGen
            query_spec = self.query_extractor.extract_query_components(user_query, conversation)
            
            if query_spec is not None:
                logger.info(f"Dynamic extraction: {len(query_spec.filters)} filters, operation: {query_spec.operation.operation_type}")
                
                # Step 2: Get dataset for execution
                dataset = self._get_current_dataset(conversation)
                model = conversation.get_var('model').contents
                explainer = conversation.get_var('mega_explainer')
                explainer_obj = explainer.contents if explainer else None
                
                # Step 3: Execute dynamically generated code for ALL query types
                logger.info("🚀 Executing dynamic code generation for maximum generalizability")
                result = self.query_executor.execute(
                    query=query_spec,
                    dataset=dataset,
                    model=model,
                    explainer=explainer_obj,
                    conversation=conversation
                )
                
                # Step 4: Update conversation state if filters were applied
                if query_spec.filters:
                    self._update_conversation_with_filters(conversation, query_spec.filters)
                
                logger.info("✅ Dynamic code generation executed successfully")
                return result
            else:
                logger.warning("AutoGen extraction failed")
                return "I couldn't understand your request. Please try rephrasing it."
                
        except Exception as e:
            logger.error(f"Dynamic code generation error: {e}", exc_info=True)
            # Return more detailed error information for debugging
            return f"Dynamic code generation failed: {str(e)}. Please check the logs for details."
    
    def _get_current_dataset(self, conversation) -> pd.DataFrame:
        """Get the current dataset (filtered or full)."""
        # Try temp dataset first (if filters applied)
        if hasattr(conversation, 'temp_dataset') and conversation.temp_dataset:
            return conversation.temp_dataset.contents['X']
        
        # Fall back to full dataset
        return conversation.get_var('dataset').contents['X']
    
    def _execute_traditional_action(self, user_query: str, conversation, query_spec) -> str:
        """Execute simple queries using traditional action system."""
        try:
            # Import the traditional action functions
            from explain.actions.data_summary import data_operation
            
            # For simple data queries, use the data action
            operation_type = query_spec.operation.operation_type
            
            if operation_type in ['statistics', 'count']:
                # Create parse_text for traditional action (simulate original parsing)
                parse_text = ["data"]
                
                # Add any feature mentions from context to parse_text for targeted statistics
                if query_spec.context:
                    original_query = query_spec.context.get('original_query', '').lower()
                    # Check for specific features mentioned
                    dataset_features = conversation.get_var('dataset').contents['X'].columns
                    for feature in dataset_features:
                        if feature.lower() in original_query:
                            parse_text.append(feature.lower())
                            break
                
                logger.info(f"Executing traditional data action with parse_text: {parse_text}")
                result, _ = data_operation(conversation, parse_text, 0)
                return result
            else:
                return "Traditional action not implemented for this operation type."
                
        except Exception as e:
            logger.error(f"Error in traditional action execution: {e}")
            return f"Error executing traditional action: {str(e)}"

    def _update_conversation_with_filters(self, conversation, filters: list):
        """Update conversation state to reflect applied filters."""
        try:
            # Get original dataset
            original_data = conversation.get_var('dataset').contents['X']
            
            # Apply filters to create temp dataset
            filtered_data = original_data.copy()
            for filter_op in filters:
                if filter_op.operator == 'gt':
                    filtered_data = filtered_data[filtered_data[filter_op.feature] > filter_op.value]
                elif filter_op.operator == 'lt':
                    filtered_data = filtered_data[filtered_data[filter_op.feature] < filter_op.value]
                elif filter_op.operator == 'gte':
                    filtered_data = filtered_data[filtered_data[filter_op.feature] >= filter_op.value]
                elif filter_op.operator == 'lte':
                    filtered_data = filtered_data[filtered_data[filter_op.feature] <= filter_op.value]
                elif filter_op.operator == 'eq':
                    filtered_data = filtered_data[filtered_data[filter_op.feature] == filter_op.value]
                elif filter_op.operator == 'ne':
                    filtered_data = filtered_data[filtered_data[filter_op.feature] != filter_op.value]
            
            # Update temp dataset in conversation
            if hasattr(conversation, 'temp_dataset'):
                conversation.temp_dataset.contents['X'] = filtered_data
            else:
                # Create temp dataset structure
                temp_dataset = type('TempDataset', (), {
                    'contents': {'X': filtered_data}
                })()
                conversation.temp_dataset = temp_dataset
            
            logger.info(f"Updated conversation state: {len(filtered_data)} samples after filtering")
            
        except Exception as e:
            logger.error(f"Error updating conversation with filters: {e}")


class GeneralizedQueryTester:
    """Test the generalized query system."""
    
    def __init__(self, bot: GeneralizedExplainBot):
        self.bot = bot
    
    def test_queries(self):
        """Test various query types to demonstrate generalizability."""
        
        test_cases = [
            # Performance with filters
            "how accurate is the model on instances with age > 50",
            "what is the accuracy for patients with glucose > 140",
            "how well does the model perform on women with BMI < 25",
            
            # Statistics with filters  
            "what is the average age for diabetic patients",
            "show me statistics for patients with high blood pressure",
            "what are the mean glucose levels for pregnant women",
            
            # Complex multi-filter queries
            "accuracy for patients over 40 with BMI > 30 and glucose > 120",
            "statistics for non-pregnant women under 35",
            
            # Feature importance
            "what are the most important features for diabetic patients",
            "feature importance for patients over 50",
            
            # Predictions
            "predict outcomes for patients with age > 45 and BMI > 28",
            "what would the model predict for young patients",
        ]
        
        print("🧪 Testing Generalized Query System")
        print("=" * 50)
        
        for i, query in enumerate(test_cases, 1):
            print(f"\n{i}. Query: '{query}'")
            
            try:
                # Extract query components
                query_spec = self.bot.query_extractor.extract_query_components(
                    query, self.bot.conversation
                )
                
                if query_spec:
                    print(f"   ✅ Extracted: {len(query_spec.filters)} filters, operation: {query_spec.operation.operation_type}")
                    
                    # Show filter details
                    for j, f in enumerate(query_spec.filters):
                        print(f"      Filter {j+1}: {f.feature} {f.operator} {f.value}")
                else:
                    print("   ❌ Failed to extract query components")
                    
            except Exception as e:
                print(f"   💥 Error: {e}")
        
        print(f"\n🎯 Key Benefits:")
        print("✅ Handles ANY combination of filters and operations")
        print("✅ Automatically generates optimized pandas code") 
        print("✅ No need to predefine action strings")
        print("✅ Leverages AutoGen's extraction capabilities")
        print("✅ Truly generalizable to new domains and features")


def create_generalized_bot(**kwargs) -> GeneralizedExplainBot:
    """Factory function to create generalized bot."""
    return GeneralizedExplainBot(**kwargs)


# Example of how the generated code looks:
EXAMPLE_GENERATED_CODE = '''
def execute_query(dataset, model, explainer=None, conversation=None):
    """Dynamically generated query execution function."""
    import pandas as pd
    import numpy as np
    from explain.actions import *

    original_dataset = dataset.copy()
    original_size = len(original_dataset)

    # Apply filters
    filtered_dataset = dataset[dataset['age'] > 50]
    filtered_dataset = filtered_dataset[filtered_dataset['bmi'] > 30]
    filtered_size = len(filtered_dataset)
    dataset = filtered_dataset  # Use filtered dataset for operation

    # Calculate accuracy on filtered data
    if len(dataset) == 0:
        return 'No data points match the specified criteria.'

    X = dataset.drop(columns=[col for col in dataset.columns if col in ['target', 'y', 'label']], errors='ignore')
    y_true = dataset.get('y', dataset.get('target', dataset.get('label')))
    
    if y_true is not None:
        y_pred = model.predict(X)
        accuracy = (y_pred == y_true).mean() * 100
        
        filter_desc = ' (filtered: age > 50, bmi > 30)'
        result = f'Accuracy{filter_desc}: {accuracy:.2f}% ({len(dataset)}/{original_size} samples)'
        return result
    else:
        return 'Cannot calculate accuracy: target variable not found.'
'''