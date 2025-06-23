"""
Generalized Multi-Agent System for Data Exploration

This module implements a more flexible, tool-augmented approach to data exploration
that reduces hardcoded rules and enables dynamic discovery of dataset capabilities.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import ast
import inspect
from dataclasses import dataclass
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    VISUALIZATION_AVAILABLE = False

from io import StringIO
import sys
import traceback

logger = logging.getLogger(__name__)


@dataclass
class DatasetSchema:
    """Dynamic schema discovery for datasets."""
    features: List[str]
    feature_types: Dict[str, str]  # feature_name -> 'numeric'/'categorical'/'datetime'
    feature_ranges: Dict[str, Tuple[Any, Any]]  # min, max for numeric
    feature_categories: Dict[str, List[Any]]  # unique values for categorical
    target_column: Optional[str]
    target_classes: List[Any]
    size: int
    
    @classmethod
    def discover(cls, df: pd.DataFrame, target_col: str = None) -> 'DatasetSchema':
        """Automatically discover dataset schema."""
        features = list(df.columns)
        if target_col and target_col in features:
            features.remove(target_col)
        
        feature_types = {}
        feature_ranges = {}
        feature_categories = {}
        
        for col in features:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_types[col] = 'numeric'
                feature_ranges[col] = (df[col].min(), df[col].max())
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_types[col] = 'datetime'
                feature_ranges[col] = (df[col].min(), df[col].max())
            else:
                feature_types[col] = 'categorical'
                feature_categories[col] = list(df[col].unique())
        
        target_classes = []
        if target_col and target_col in df.columns:
            target_classes = list(df[target_col].unique())
        
        return cls(
            features=features,
            feature_types=feature_types,
            feature_ranges=feature_ranges,
            feature_categories=feature_categories,
            target_column=target_col,
            target_classes=target_classes,
            size=len(df)
        )
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """Get comprehensive info about a feature."""
        if feature_name not in self.features:
            return None
        
        info = {
            'name': feature_name,
            'type': self.feature_types[feature_name],
            'exists': True
        }
        
        if feature_name in self.feature_ranges:
            info['range'] = self.feature_ranges[feature_name]
        if feature_name in self.feature_categories:
            info['categories'] = self.feature_categories[feature_name]
            
        return info
    
    def suggest_operations(self, user_intent: str, entities: Dict[str, Any] = None) -> List[str]:
        """Suggest relevant operations based on schema, user intent, and entities."""
        operations = []
        entities = entities or {}
        intent_lower = user_intent.lower()
        
        # Feature comparison queries (NEW - handles "Is glucose more important than age")
        comparison_patterns = [
            'more important than', 'compared to', 'versus', 'vs', 
            'better predictor than', 'stronger than', 'compare'
        ]
        if any(pattern in intent_lower for pattern in comparison_patterns):
            operations.append('feature_importance')  # Single operation type
        
        # Error analysis queries (ENHANCED - handles "typical errors")
        if any(word in intent_lower for word in ['error', 'mistake', 'incorrect', 'wrong', 'misclassified']):
            if 'typical' in intent_lower or 'pattern' in intent_lower:
                operations.append('error_analysis')  # Single operation type
            else:
                operations.append('error_analysis')  # Single operation type
        
        # Complex what-if with group filtering (ENHANCED)
        what_if_patterns = ['what if', 'change', 'increase', 'decrease', 'set', 'likelihood']
        if any(pattern in intent_lower for pattern in what_if_patterns):
            # Check if it involves group analysis (men, women, older than, etc.)
            group_patterns = ['men', 'women', 'older', 'younger', 'patients with', 'instances with']
            if any(pattern in intent_lower for pattern in group_patterns):
                operations.extend(['filtering', 'prediction'])  # Only essential operations
            else:
                operations.append('prediction')  # Single operation type
        
        # Check if we need filtering based on actual conditions (more precise)
        needs_filtering = (
            # Explicit filtering keywords
            'filter' in intent_lower or 'subset' in intent_lower or
            # ID-based queries
            entities.get('patient_id') is not None or
            # Feature-based conditions with actual operators/values
            (entities.get('features') and entities.get('operators') and entities.get('values')) or
            # Instance-specific queries
            any(phrase in intent_lower for phrase in ['instance', 'patient', 'data point', 'person with', 'people with']) or
            # Context reset is False (continuing from previous filter)
            (entities.get('context_reset') is False and any(entities.get(key) for key in ['features', 'operators', 'values']))
        )
        
        if needs_filtering:
            operations.append('filtering')  # Single operation type
        
        # Performance/accuracy queries
        if any(word in intent_lower for word in ['accurate', 'accuracy', 'performance', 'how good', 'how well']):
            operations.append('model_evaluation')  # Single operation type
        
        # Prediction queries
        elif any(word in intent_lower for word in ['predict', 'prediction', 'classification', 'likelihood']):
            operations.append('prediction')  # Single operation type
        
        # Feature importance queries  
        if any(word in intent_lower for word in ['important', 'importance', 'feature', 'significance', 'influential']):
            operations.append('feature_importance')  # Single operation type
        
        # Visualization queries
        if any(word in intent_lower for word in ['plot', 'visualize', 'chart', 'graph']):
            operations.append('visualization')  # Single operation type
        
        # Data summary queries (more specific patterns)
        if any(phrase in intent_lower for phrase in ['average', 'mean', 'summary', 'describe', 'statistics', 'dataset summary', 'data summary', 'how many', 'total']):
            operations.append('data_summary')  # Single operation type
        
        # Remove duplicates while preserving order
        unique_operations = []
        seen = set()
        for op in operations:
            if op not in seen:
                unique_operations.append(op)
                seen.add(op)
        
        return unique_operations


class DataOperationTool:
    """Base class for data operation tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def can_handle(self, operation_type: str, schema: DatasetSchema) -> bool:
        """Check if this tool can handle the requested operation."""
        raise NotImplementedError
    
    def generate_code(self, intent: str, entities: Dict, schema: DatasetSchema) -> str:
        """Generate Python code for the operation."""
        raise NotImplementedError
    
    def execute(self, code: str, context: Dict[str, Any]) -> Tuple[Any, str]:
        """Execute the generated code safely."""
        try:
            # Capture stdout for text output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Execute code
            exec(code, context)
            
            # Get any printed output
            output_text = captured_output.getvalue()
            sys.stdout = old_stdout
            
            # Look for result variable
            result = context.get('result', output_text if output_text else 'Operation completed')
            
            return result, ""
        except Exception as e:
            sys.stdout = old_stdout
            error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return None, error_msg


class FilteringTool(DataOperationTool):
    """Tool for data filtering operations."""
    
    def __init__(self):
        super().__init__(
            name="filtering",
            description="Filter datasets based on conditions"
        )
    
    def can_handle(self, operation_type: str, schema: DatasetSchema) -> bool:
        return operation_type in ['filtering', 'data_selection', 'subset']
    
    def generate_code(self, intent: str, entities: Dict, schema: DatasetSchema) -> str:
        """Generate pandas filtering code."""
        conditions = []
        
        # Handle special case: patient_id filtering (ID-based filtering)
        patient_id = entities.get('patient_id')
        if patient_id is not None:
            code = f"""
# Filter to specific instance by ID (using DataFrame index)
try:
    if {patient_id} in df.index:
        filtered_df = df.loc[[{patient_id}]]
        result = f"Filtered to instance {patient_id} (1 row out of {{len(df)}} total)"
        print(f"Applied ID filter: instance {patient_id}")
        print(f"Resulting dataset shape: {{filtered_df.shape}}")
        
        # Store filtered dataset for subsequent operations
        df = filtered_df
    else:
        result = f"Instance {patient_id} not found in dataset"
        print(f"Error: Instance {patient_id} not in index {{list(df.index[:10])}}...")
        df = df.iloc[:0]  # Empty dataset
except Exception as e:
    result = f"Error filtering by ID {patient_id}: {{str(e)}}"
    print(result)
"""
            return code
        
        # Extract filtering conditions from entities
        features = entities.get('features', [])
        operators = entities.get('operators', [])
        values = entities.get('values', [])
        
        for i, feature in enumerate(features):
            if i < len(operators) and i < len(values):
                operator = operators[i]
                value = values[i]
                
                # Special handling for 'id' feature - use DataFrame index
                if feature == 'id':
                    code = f"""
# Filter by ID using DataFrame index
try:
    if {value} in df.index:
        filtered_df = df.loc[[{value}]]
        result = f"Filtered to instance {value} (1 row out of {{len(df)}} total)"
        print(f"Applied ID filter: instance {value}")
        
        # Store filtered dataset for subsequent operations
        df = filtered_df
    else:
        result = f"Instance {value} not found in dataset"
        df = df.iloc[:0]  # Empty dataset
except Exception as e:
    result = f"Error filtering by ID {value}: {{str(e)}}"
    print(result)
"""
                    return code
                
                # Map operators to pandas operations
                op_map = {
                    '>': '>',
                    'greater': '>',
                    '<': '<', 
                    'less': '<',
                    '>=': '>=',
                    'greaterequal': '>=',
                    '<=': '<=',
                    'lessequal': '<=',
                    '==': '==',
                    'equal': '=='
                }
                
                pandas_op = op_map.get(operator, '==')
                
                # Handle different value types
                if isinstance(value, str) and not value.isdigit():
                    condition = f"(df['{feature}'] {pandas_op} '{value}')"
                else:
                    condition = f"(df['{feature}'] {pandas_op} {value})"
                
                conditions.append(condition)
        
        if conditions:
            filter_expr = " & ".join(conditions)
            code = f"""
# Filter dataset based on conditions
filtered_df = df[{filter_expr}]
result = f"Filtered dataset to {{len(filtered_df)}} rows out of {{len(df)}} total rows"
print(f"Applied filter: {filter_expr}")
print(f"Resulting dataset shape: {{filtered_df.shape}}")

# Store filtered dataset for subsequent operations
df = filtered_df
"""
        else:
            # No filtering conditions - this tool should not have been selected
            # Return a no-op that doesn't interfere with subsequent operations
            code = """
# No filtering required - operating on full dataset
result = ""  # Empty result to avoid confusion
"""
        
        return code


class ModelAnalysisTool(DataOperationTool):
    """Tool for model analysis and evaluation."""
    
    def __init__(self):
        super().__init__(
            name="model_analysis", 
            description="Analyze model performance and predictions"
        )
    
    def can_handle(self, operation_type: str, schema: DatasetSchema) -> bool:
        return operation_type in ['prediction', 'model_evaluation', 'performance']
    
    def generate_code(self, intent: str, entities: Dict, schema: DatasetSchema) -> str:
        """Generate model analysis code."""
        if 'accuracy' in intent.lower() or 'performance' in intent.lower():
            code = """
# Evaluate model performance
X = df[feature_columns]
y_true = df[target_column] if target_column in df.columns else None

if model is not None and y_true is not None:
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    
    result = f"Model accuracy on {len(X)} instances: {accuracy:.1%}"
    print(result)
    
    # Additional metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    print(f"\\nDetailed classification report:")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{class_name}: precision={metrics.get('precision', 0):.3f}, recall={metrics.get('recall', 0):.3f}")
else:
    result = "Model or target data not available for evaluation"
"""
        elif 'predict' in intent.lower():
            code = """
# Make predictions
X = df[feature_columns]

if model is not None:
    predictions = model.predict(X)
    
    if len(predictions) == 1:
        pred_class = predictions[0]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            prob_text = ", ".join([f"{class_names[i] if class_names else i}: {prob:.3f}" for i, prob in enumerate(probabilities)])
            result = f"Prediction: {class_names[pred_class] if class_names else pred_class} (probabilities: {prob_text})"
        else:
            result = f"Prediction: {class_names[pred_class] if class_names else pred_class}"
    else:
        unique_preds, counts = np.unique(predictions, return_counts=True)
        pred_summary = []
        for pred, count in zip(unique_preds, counts):
            percentage = (count / len(predictions)) * 100
            pred_name = class_names[pred] if class_names else pred
            pred_summary.append(f"{pred_name}: {count} instances ({percentage:.1f}%)")
        
        result = f"Predictions for {len(predictions)} instances:\\n" + "\\n".join(pred_summary)
    
    print(result)
else:
    result = "Model not available for prediction"
"""
        else:
            code = """
result = "Model analysis requested but specific operation not clear"
"""
        
        return code


class FeatureAnalysisTool(DataOperationTool):
    """Tool for feature importance and analysis."""
    
    def __init__(self):
        super().__init__(
            name="feature_analysis",
            description="Analyze feature importance and relationships"
        )
    
    def can_handle(self, operation_type: str, schema: DatasetSchema) -> bool:
        return operation_type in ['feature_importance', 'feature_analysis']
    
    def generate_code(self, intent: str, entities: Dict, schema: DatasetSchema) -> str:
        """Generate feature analysis code."""
        code = """
# Feature importance analysis
X = df[feature_columns]
y = df[target_column] if target_column in df.columns else None

if model is not None and y is not None:
    try:
        # Try to get feature importance from model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = list(zip(feature_columns, importances))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            result = "Feature Importance (from model):\\n"
            for i, (feature, importance) in enumerate(feature_importance[:10]):  # Top 10
                result += f"{i+1}. {feature}: {importance:.4f}\\n"
                
        else:
            # Use permutation importance
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            feature_importance = list(zip(feature_columns, perm_importance.importances_mean))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            result = "Feature Importance (permutation-based):\\n"
            for i, (feature, importance) in enumerate(feature_importance[:10]):  # Top 10  
                result += f"{i+1}. {feature}: {importance:.4f}\\n"
        
        print(result)
        
    except Exception as e:
        result = f"Could not compute feature importance: {str(e)}"
        print(result)
else:
    result = "Model or target data not available for feature analysis"
    print(result)
"""
        return code


class DataSummaryTool(DataOperationTool):
    """Tool for data summarization and statistics."""
    
    def __init__(self):
        super().__init__(
            name="data_summary",
            description="Generate data summaries and statistics"
        )
    
    def can_handle(self, operation_type: str, schema: DatasetSchema) -> bool:
        return operation_type in ['descriptive_statistics', 'data_summary', 'summary']
    
    def generate_code(self, intent: str, entities: Dict, schema: DatasetSchema) -> str:
        """Generate data summary code."""
        # Check for specific feature queries like "average age", "mean glucose", etc.
        intent_lower = intent.lower()
        
        # Handle specific feature statistics requests
        if any(word in intent_lower for word in ['average', 'mean']) and 'age' in intent_lower:
            code = """
# Handle specific age statistics request
if 'age' in df.columns:
    avg_age = round(df['age'].mean(), 2)
    std_age = round(df['age'].std(), 2)
    min_age = round(df['age'].min(), 2)
    max_age = round(df['age'].max(), 2)
    
    dataset_size = len(df)
    
    result = f"For the {dataset_size} patients in the dataset:\\n\\n"
    result += f"**Age Statistics:**\\n"
    result += f"• Average age: **{avg_age} years**\\n"
    result += f"• Standard deviation: {std_age} years\\n"
    result += f"• Age range: {min_age} to {max_age} years\\n"
    
    print(result)
else:
    result = "Age information is not available in this dataset."
    print(result)
"""
            return code
        
        # Check for other specific feature statistics
        for feature_name in schema.features:
            if (any(word in intent_lower for word in ['average', 'mean']) and 
                feature_name.lower() in intent_lower):
                code = f"""
# Handle specific {feature_name} statistics request
if '{feature_name}' in df.columns:
    avg_val = round(df['{feature_name}'].mean(), 2)
    std_val = round(df['{feature_name}'].std(), 2)
    min_val = round(df['{feature_name}'].min(), 2)
    max_val = round(df['{feature_name}'].max(), 2)
    
    dataset_size = len(df)
    
    result = f"For the {{dataset_size}} patients in the dataset:\\n\\n"
    result += f"**{feature_name.title()} Statistics:**\\n"
    result += f"• Average: **{{avg_val}}**\\n"
    result += f"• Standard deviation: {{std_val}}\\n"
    result += f"• Range: {{min_val}} to {{max_val}}\\n"
    
    print(result)
else:
    result = "{feature_name} information is not available in this dataset."
    print(result)
"""
                return code
        
        # Default general data summary
        code = """
# General data summary and statistics
result = f"Dataset Summary:\\n"
result += f"- Total records: {len(df)}\\n"
result += f"- Features: {len(df.columns)}\\n"

if target_column and target_column in df.columns:
    target_counts = df[target_column].value_counts()
    result += f"- Target classes: {list(target_counts.index)}\\n"
    for class_val, count in target_counts.items():
        percentage = (count / len(df)) * 100
        class_name = class_names[class_val] if class_names and class_val < len(class_names) else class_val
        result += f"  - {class_name}: {count} ({percentage:.1f}%)\\n"

result += f"\\nFeature Summary:\\n"
for col in df.columns:
    if col != target_column:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            result += f"- {col}: mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]\\n"
        else:
            unique_count = df[col].nunique()
            most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
            result += f"- {col}: {unique_count} unique values, most common='{most_common}'\\n"

print(result)
"""
        return code


class GeneralizedActionPlanner:
    """
    Generalized action planner that uses tool-augmented agents
    instead of hardcoded action mappings.
    """
    
    def __init__(self):
        self.tools = [
            FilteringTool(),
            ModelAnalysisTool(), 
            FeatureAnalysisTool(),
            DataSummaryTool()
        ]
        self.schema = None
    
    def initialize_schema(self, df: pd.DataFrame, target_col: str = None) -> DatasetSchema:
        """Initialize dataset schema for dynamic discovery."""
        self.schema = DatasetSchema.discover(df, target_col)
        logger.info(f"Discovered schema: {len(self.schema.features)} features, {self.schema.size} records")
        return self.schema
    
    def plan_operations(self, intent: str, entities: Dict[str, Any]) -> List[Tuple[DataOperationTool, str]]:
        """Plan operations based on intent and available tools."""
        if not self.schema:
            raise ValueError("Schema not initialized. Call initialize_schema() first.")
        
        # Discover relevant operations (pass entities for better detection)
        suggested_ops = self.schema.suggest_operations(intent, entities)
        
        # Find tools that can handle these operations
        selected_tools = []
        tool_names_used = set()  # Avoid duplicate tools
        
        for op_type in suggested_ops:
            for tool in self.tools:
                if tool.can_handle(op_type, self.schema) and tool.name not in tool_names_used:
                    # Special check for filtering tool - only include if actual conditions exist
                    if tool.name == "filtering":
                        has_conditions = (
                            entities.get('patient_id') is not None or
                            entities.get('features') or
                            (entities.get('context_reset') is False and entities.get('features'))
                        )
                        if not has_conditions:
                            continue  # Skip filtering tool if no conditions
                    
                    code = tool.generate_code(intent, entities, self.schema)
                    selected_tools.append((tool, code))
                    tool_names_used.add(tool.name)
                    break  # Use first matching tool for each operation type
        
        return selected_tools
    
    def execute_plan(self, tool_code_pairs: List[Tuple[DataOperationTool, str]], 
                    context: Dict[str, Any]) -> List[Tuple[Any, str]]:
        """Execute the planned operations."""
        results = []
        
        for tool, code in tool_code_pairs:
            logger.info(f"Executing {tool.name} operation")
            logger.debug(f"Generated code:\n{code}")
            
            result, error = tool.execute(code, context)
            results.append((result, error))
            
            if error:
                logger.error(f"Error in {tool.name}: {error}")
                break  # Stop on first error
        
        return results


def create_generalized_context(conversation) -> Dict[str, Any]:
    """Create execution context for generalized operations."""
    try:
        # Get data and model from conversation
        dataset = conversation.get_var('dataset').contents
        model = conversation.get_var('model').contents
        
        df = dataset['X'].copy()
        target_col = None
        
        # Add target column if available
        if 'y' in dataset and len(dataset['y']) > 0:
            target_col = 'target'
            df[target_col] = dataset['y']
        
        # Get feature information
        feature_columns = [col for col in df.columns if col != target_col]
        class_names = getattr(conversation, 'class_names', None)
        
        context = {
            'df': df,
            'model': model,
            'feature_columns': feature_columns,
            'target_column': target_col,
            'class_names': class_names,
            'pd': pd,
            'np': np,
            'accuracy_score': accuracy_score,
            'classification_report': classification_report,
            'permutation_importance': permutation_importance
        }
        
        return context
        
    except Exception as e:
        logger.error(f"Error creating context: {e}")
        return {}


# Integration with existing system
def generalized_action_dispatcher(intent: str, entities: Dict[str, Any], conversation) -> Tuple[str, int]:
    """
    Main entry point for generalized action dispatching.
    
    This replaces the hardcoded action mapping with dynamic tool selection.
    """
    try:
        logger.info(f"Generalized dispatcher called: intent='{intent}', entities={entities}")
        
        # Create execution context
        context = create_generalized_context(conversation)
        if not context:
            return "Could not initialize data context", 0
        
        # Initialize planner and schema
        planner = GeneralizedActionPlanner()
        schema = planner.initialize_schema(context['df'], context['target_column'])
        
        # Plan operations
        operations = planner.plan_operations(intent, entities)
        logger.info(f"Planned operations: {[(tool.name, 'code') for tool, code in operations]}")
        
        if not operations:
            return f"No suitable operations found for intent: {intent}", 0
        
        # Execute operations
        results = planner.execute_plan(operations, context)
        
        # Format results - filter out empty results
        success_results = []
        for result, error in results:
            if error:
                logger.error(f"Operation failed: {error}")
                return f"Operation failed: {error}", 0
            if result and str(result).strip():  # Only include non-empty results
                success_results.append(str(result))
        
        if success_results:
            final_result = "<br>".join(success_results)
            logger.info(f"Generalized dispatcher succeeded with {len(success_results)} results")
            return final_result, 1
        else:
            # If no results but no errors, operations completed successfully but had no output
            logger.info("Operations completed successfully with no output")
            return "Operations completed successfully", 1
            
    except Exception as e:
        logger.error(f"Error in generalized action dispatcher: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"System error: {str(e)}", 0 