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
    
    def suggest_operations(self, user_query: str, entities: Dict[str, Any] = None) -> List[str]:
        """Suggest relevant operations based on schema, user query, and entities using semantic analysis."""
        operations = []
        entities = entities or {}
        
        logger.info(f"suggest_operations: Processing query='{user_query}', entities={entities}")
        
        # SEMANTIC OPERATION MAPPING - Let each tool decide if it can handle the query
        # This is much more generalizable than pattern matching
        for tool in [ConversationalTool(), FilteringTool(), ModelAnalysisTool(), 
                     FeatureAnalysisTool(), DataSummaryTool(), ErrorAnalysisTool()]:
            
            # Let each tool assess its own relevance to the query
            relevance_score = self._assess_tool_relevance(tool, user_query, entities)
            
            if relevance_score > 0.5:  # Threshold for relevance
                operation_type = tool.name.replace('_tool', '').replace('tool', '')
                operations.append(operation_type)
                logger.info(f"suggest_operations: Added '{operation_type}' (relevance: {relevance_score:.2f})")
        
        # If no tools matched, fall back to data_summary as default
        if not operations:
            operations.append('data_summary')
            logger.info("suggest_operations: Defaulting to 'data_summary'")
        
        return operations
    
    def _assess_tool_relevance(self, tool, user_query: str, entities: Dict[str, Any]) -> float:
        """Let each tool assess its own relevance to the query (generalizable approach)."""
        query_lower = user_query.lower()
        
        # ConversationalTool - handles general knowledge, health concepts (NOT dataset analysis)
        if isinstance(tool, ConversationalTool):
            # Specific patterns for general health knowledge (not data analysis)
            pure_health_patterns = ['isnt', "isn't", 'is that', 'is it', 'should i', 'would you say', 'do you think']
            statistical_patterns = ['average', 'mean', 'value', 'count', 'summary', 'statistics', 'total']
            
            # Only route to conversational for pure health questions, NOT statistical analysis
            has_pure_health = any(pattern in query_lower for pattern in pure_health_patterns)
            has_statistical = any(pattern in query_lower for pattern in statistical_patterns)
            
            if has_pure_health and not has_statistical:
                return 0.8  # High relevance for pure health questions like "isn't BMI 32 high?"
            return 0.1
        
        # TraditionalXAITool - handles complex explanations, LIME, SHAP, counterfactuals
        elif isinstance(tool, TraditionalXAITool):
            xai_indicators = [
                'explain', 'explanation', 'why', 'how', 'lime', 'shap', 
                'counterfactual', 'what would need', 'different prediction',
                'interaction', 'interact', 'combined effect', 'what if', 'change'
            ]
            # Exclude simple statistical questions that generalized tools handle better
            simple_indicators = ['average', 'mean', 'count', 'summary', 'statistics']
            
            has_xai = any(x in query_lower for x in xai_indicators)
            has_simple = any(s in query_lower for s in simple_indicators)
            
            if has_xai and not has_simple:
                return 0.9  # Very high relevance for XAI queries
            return 0.1
        
        # FilteringTool - handles conditions, comparisons
        elif isinstance(tool, FilteringTool):
            filter_indicators = ['>', '<', '=', 'greater', 'less', 'equal', 'filter', 'where', 'with', 'given']
            return 0.7 if any(f in query_lower for f in filter_indicators) else 0.1
        
        # ModelAnalysisTool - handles predictions, accuracy, model evaluation
        elif isinstance(tool, ModelAnalysisTool):
            # Check for new instance patterns first (highest priority)
            import re
            new_instance_patterns = [
                r'new\s+instance', r'predict\s+for\s+a\s+new', r'predict\s+for\s+an\s+instance', 
                r'hypothetical\s+instance', r'what\s+would.*predict.*new'
            ]
            if any(re.search(pattern, query_lower) for pattern in new_instance_patterns):
                return 0.95  # Very high priority for new instance predictions
            
            model_indicators = ['predict', 'accuracy', 'performance', 'model', 'evaluation', 'score', 'instance', 'prediction for']
            return 0.8 if any(m in query_lower for m in model_indicators) else 0.1
        
        # FeatureAnalysisTool - handles importance, significance, influence
        elif isinstance(tool, FeatureAnalysisTool):
            feature_indicators = ['important', 'influence', 'significant', 'feature', 'factor', 'affect']
            return 0.7 if any(f in query_lower for f in feature_indicators) else 0.1
        
        # DataSummaryTool - handles statistics, summaries, distributions
        elif isinstance(tool, DataSummaryTool):
            summary_indicators = ['average', 'mean', 'summary', 'statistics', 'distribution', 'skewed', 'count', 'value for']
            return 0.8 if any(s in query_lower for s in summary_indicators) else 0.2
        
        # ErrorAnalysisTool - handles errors, mistakes, failures
        elif isinstance(tool, ErrorAnalysisTool):
            error_indicators = ['error', 'mistake', 'wrong', 'incorrect', 'fail', 'misclassified']
            return 0.7 if any(e in query_lower for e in error_indicators) else 0.1
        
        return 0.1  # Default low relevance


class DataOperationTool:
    """Base class for data operation tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def can_handle(self, operation_type: str, schema: DatasetSchema) -> bool:
        """Check if this tool can handle the requested operation."""
        raise NotImplementedError
    
    def generate_code(self, user_query: str, entities: Dict, schema: DatasetSchema) -> str:
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
    """Tool for filtering datasets based on conditions."""
    
    def __init__(self):
        super().__init__(
            name="filtering",
            description="Filter datasets based on conditions"
        )
    
    def can_handle(self, operation_type: str, schema: DatasetSchema) -> bool:
        return operation_type in ['filtering', 'data_selection', 'subset']
    
    def generate_code(self, user_query: str, entities: Dict, schema: DatasetSchema) -> str:
        """Generate pandas filtering code."""
        conditions = []
        
        # Handle special case: patient_id filtering (ID-based filtering)
        patient_id = entities.get('patient_id')
        if patient_id is not None:
            code = f"""
# Filter to specific instance by ID (handle both 0-based and 1-based indexing)
try:
    found_instance = False
    instance_id = {patient_id}
    
    # Try direct index lookup first
    if instance_id in df.index:
        filtered_df = df.loc[[instance_id]]
        result = f"Filtered to instance {patient_id} (1 row out of {{len(df)}} total)"
        found_instance = True
    # Try 0-based indexing (user says "instance 2" meaning index 1)
    elif instance_id - 1 in df.index:
        actual_id = instance_id - 1
        filtered_df = df.loc[[actual_id]]
        result = f"Filtered to instance {patient_id} (index {{actual_id}}) - 1 row out of {{len(df)}} total"
        found_instance = True
    # Try iloc-based lookup for positional indexing
    elif 0 <= instance_id < len(df):
        filtered_df = df.iloc[[instance_id]]
        result = f"Filtered to instance at position {patient_id} (1 row out of {{len(df)}} total)"
        found_instance = True
    # Try iloc with offset for 1-based user input
    elif 0 <= instance_id - 1 < len(df):
        filtered_df = df.iloc[[instance_id - 1]]
        result = f"Filtered to instance {patient_id} (position {{instance_id - 1}}) - 1 row out of {{len(df)}} total"
        found_instance = True
    
    if found_instance:
        print(f"Applied ID filter: instance {patient_id}")
        print(f"Resulting dataset shape: {{filtered_df.shape}}")
        df = filtered_df
    else:
        result = f"Instance {patient_id} not found in dataset (tried index {{instance_id}}, {{instance_id-1}}, and positional lookup)"
        print(f"Error: Instance {patient_id} not found. Dataset has {{len(df)}} rows with index range {{list(df.index[:5])}}...{{list(df.index[-2:])}}")
        df = df.iloc[:0]  # Empty dataset
        
except Exception as e:
    result = f"Error filtering by ID {patient_id}: {{str(e)}}"
    print(result)
    df = df.iloc[:0]  # Empty dataset
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
    
    def generate_code(self, user_query: str, entities: Dict, schema: DatasetSchema) -> str:
        """Generate model analysis code."""
        query_lower = user_query.lower()
        
        # PRIORITY 1: NEW INSTANCE PREDICTIONS (highest priority)
        import re
        new_instance_patterns = [
            r'new\s+instance', r'new\s+isntance', r'predict\s+for\s+a\s+new', r'predict\s+for\s+an\s+instance',
            r'what\s+would.*predict.*new', r'hypothetical\s+instance',
            r'predict.*new.*instance', r'new.*patient', r'hypothetical.*patient'
        ]
        has_new_instance = any(re.search(pattern, query_lower) for pattern in new_instance_patterns)
        has_values = ('=' in query_lower or 'given' in query_lower or 'with' in query_lower)
        
        if has_new_instance and has_values:
            code = f"""
# Handle new instance prediction with specific values
import re

# Extract feature-value pairs from the query  
query_text = "{user_query}"
pattern = r'(\\w+)\\s*=\\s*(\\d+(?:\\.\\d+)?)'
matches = re.findall(pattern, query_text)

if matches and model is not None:
    # Create new instance with provided values
    new_instance = {{}}
    for feature, value in matches:
        if feature.lower() in [col.lower() for col in feature_columns]:
            # Map to actual column name
            actual_col = [col for col in feature_columns if col.lower() == feature.lower()][0]
            new_instance[actual_col] = float(value)
    
    # Fill missing features with dataset averages (generalized approach)
    for col in feature_columns:
        if col not in new_instance:
            new_instance[col] = df[col].mean()
    
    # Create DataFrame for prediction
    X_new = pd.DataFrame([new_instance])[feature_columns]
    
    # Make prediction - FIXED to be more transparent
    prediction = model.predict(X_new)[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_new)[0]
        
        # Find the class with highest probability for verification
        predicted_class = np.argmax(probabilities)
        highest_prob = probabilities[predicted_class]
        
        # Format probabilities clearly
        prob_text = []
        for i, prob in enumerate(probabilities):
            prob_text.append(f"Class {{i}}: {{prob:.3f}}")
        
        result = f"Prediction: Class {{prediction}} (highest probability: {{highest_prob:.3f}})\\n"
        result += f"All probabilities: {{', '.join(prob_text)}}"
        
        # Add verification
        if prediction != predicted_class:
            result += f"\\nWARNING: Model predict() returned {{prediction}} but highest probability is for class {{predicted_class}}"
    else:
        result = f"Prediction for new instance: {{prediction}}"
        
    # Add input values to result
    input_summary = ", ".join([f"{{k}}={{v}}" for k, v in new_instance.items()])
    result += f"\\nInput values: {{input_summary}}"
else:
    result = "Could not extract feature values from query or model not available"
"""
            return code
        
        # PRIORITY 2: PERFORMANCE EVALUATION DETECTION
        performance_patterns = [
            'accuracy', 'accurate', 'performance', 'how good', 'how well', 
            'score', 'evaluate', 'evaluation', 'correct', 'error rate',
            'precision', 'recall', 'f1', 'metrics'
        ]
        
        if any(pattern in query_lower for pattern in performance_patterns):
            code = """
# Evaluate model performance (generalized)
X = df[feature_columns]
y_true = df[target_column] if target_column in df.columns else None

if model is not None and y_true is not None:
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Simple, clean output
    result = f"Model accuracy: {accuracy:.1%} ({len(X)} instances)"
else:
    result = "Model or target data not available for evaluation"
"""
            return code
        
        # PRIORITY 3: PREDICTION QUERIES
        elif 'predict' in query_lower:
            if 'all' in user_query.lower() and 'instances' in user_query.lower():
                # Handle overall prediction distribution queries
                code = """
# Make predictions for all instances and show distribution
X = df[feature_columns]

if model is not None:
    predictions = model.predict(X)
    
    # Calculate prediction distribution
    unique_preds, counts = np.unique(predictions, return_counts=True)
    total = len(predictions)
    
    pred_summary = []
    for pred, count in zip(unique_preds, counts):
        percentage = (count / total) * 100
        pred_name = class_names[pred] if class_names and pred < len(class_names) else str(pred)
        pred_summary.append(f"{pred}, {percentage:.3f}%")
    
    result = "For all the instances in the data, the model predicts:\\n" + "\\n".join(pred_summary)
else:
    result = "Model not available for prediction"
"""
            else:
                # Standard single prediction
                code = """
# Make predictions with improved error handling
X = df[feature_columns]

if model is not None and len(df) > 0:
    try:
        predictions = model.predict(X)
        
        if len(predictions) == 1:
            pred_class = predictions[0]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                if class_names and len(class_names) > max(range(len(probabilities))):
                    prob_text = ", ".join([f"{class_names[i]}: {prob:.3f}" for i, prob in enumerate(probabilities)])
                else:
                    prob_text = ", ".join([f"Class {i}: {prob:.3f}" for i, prob in enumerate(probabilities)])
                pred_name = class_names[pred_class] if class_names and pred_class < len(class_names) else f"Class {pred_class}"
                result = f"Prediction: {pred_name} (probabilities: {prob_text})<br><br>"
            else:
                pred_name = class_names[pred_class] if class_names and pred_class < len(class_names) else f"Class {pred_class}"
                result = f"Prediction: {pred_name}<br><br>"
        else:
            unique_preds, counts = np.unique(predictions, return_counts=True)
            pred_summary = []
            for pred, count in zip(unique_preds, counts):
                percentage = (count / len(predictions)) * 100
                pred_name = class_names[pred] if class_names and pred < len(class_names) else f"Class {pred}"
                pred_summary.append(f"{pred_name}: {count} instances ({percentage:.1f}%)")
            
            result = f"Predictions for {len(predictions)} instances:<br>" + "<br>".join(pred_summary) + "<br><br>"
    except Exception as e:
        result = f"Error making predictions: {str(e)}<br><br>"
elif len(df) == 0:
    result = "No data available for prediction (empty dataset)<br><br>"
else:
    result = "Model not available for prediction<br><br>"
"""
        else:
            code = """
result = "Model analysis requested but specific operation not clear<br><br>"
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
    
    def generate_code(self, user_query: str, entities: Dict, schema: DatasetSchema) -> str:
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
            
            result = "Feature Importance (from model):<br>"
            for i, (feature, importance) in enumerate(feature_importance[:10]):  # Top 10
                result += f"{i+1}. {feature}: {importance:.4f}<br>"
                
        else:
            # Use permutation importance
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            feature_importance = list(zip(feature_columns, perm_importance.importances_mean))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            result = "Feature Importance (permutation-based):<br>"
            for i, (feature, importance) in enumerate(feature_importance[:10]):  # Top 10  
                result += f"{i+1}. {feature}: {importance:.4f}<br>"
        
        result += "<br>"
        
    except Exception as e:
        result = f"Could not compute feature importance: {str(e)}<br><br>"
else:
    result = "Model or target data not available for feature analysis<br><br>"
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
        # Handle both 'data' and 'data_summary' operation types
        return operation_type in ['descriptive_statistics', 'data_summary', 'summary', 'data']
    
    def generate_code(self, user_query: str, entities: Dict, schema: DatasetSchema) -> str:
        """Generate data summary code."""
        # Check for specific feature queries like "average age", "mean glucose", etc.
        query_lower = user_query.lower()
        
        # Debug logging
        logger.info(f"DataSummaryTool: Processing query='{query_lower}', entities={entities}")
        
        # CHECK FOR COUNT QUERIES FIRST (special handling)
        count_patterns = ['how many', 'count', 'number of']
        if any(pattern in query_lower for pattern in count_patterns):
            logger.info("DataSummaryTool: Detected count query")
            code = """
# Handle count query (works with filtered data)
result = f"Total count: {len(df)} instances"
"""
            return code
        
        # FULLY GENERALIZED STATISTICAL OPERATION DETECTION
        # Define all possible statistical operations and their pandas equivalents
        stat_operations = {
            'mean': {'patterns': ['mean', 'average', 'avg'], 'pandas_method': 'mean()', 'description': 'Mean'},
            'median': {'patterns': ['median', 'middle'], 'pandas_method': 'median()', 'description': 'Median'},
            'std': {'patterns': ['standard deviation', 'std', 'deviation', 'stddev'], 'pandas_method': 'std()', 'description': 'Standard deviation'},
            'min': {'patterns': ['minimum', 'min', 'lowest', 'smallest'], 'pandas_method': 'min()', 'description': 'Minimum'},
            'max': {'patterns': ['maximum', 'max', 'highest', 'largest', 'biggest'], 'pandas_method': 'max()', 'description': 'Maximum'},
            'sum': {'patterns': ['sum', 'total'], 'pandas_method': 'sum()', 'description': 'Sum'},
            'var': {'patterns': ['variance', 'var'], 'pandas_method': 'var()', 'description': 'Variance'},
        }
        
        # Find which statistical operation is being requested
        detected_stat_op = None
        best_stat_score = 0
        
        for stat_name, stat_info in stat_operations.items():
            score = sum(1 for pattern in stat_info['patterns'] if pattern in query_lower)
            if score > best_stat_score:
                best_stat_score = score
                detected_stat_op = stat_name
        
        # Find which feature is being requested by matching against all available features
        requested_feature = None
        best_feature_score = 0
        
        for feature_name in schema.features:
            # Generate all possible ways this feature might be mentioned
            feature_patterns = [
                feature_name.lower(),
                feature_name.lower() + 's',  # plural form
                feature_name.lower().rstrip('s'),  # singular form if it ends with 's'
            ]
            
            # Check how many patterns match (for scoring)
            matches = sum(1 for pattern in feature_patterns if pattern in query_lower)
            
            if matches > best_feature_score:
                best_feature_score = matches
                requested_feature = feature_name
        
        # Generate code if both operation and feature are detected
        if detected_stat_op and requested_feature and best_stat_score > 0 and best_feature_score > 0:
            stat_info = stat_operations[detected_stat_op]
            logger.info(f"DataSummaryTool: Detected '{detected_stat_op}' operation for feature '{requested_feature}'")
            
            code = f"""
# Handle generalized statistical operation request
requested_feature = '{requested_feature}'
stat_operation = '{detected_stat_op}'

if requested_feature in df.columns:
    # Execute the detected statistical operation
    stat_value = round(df[requested_feature].{stat_info['pandas_method']}, 2)
    
    # Generate dynamic output based on detected operation and feature
    result = f"{stat_info['description']} value for {{requested_feature}}: {{stat_value}}"
else:
    result = f"{{requested_feature}} information is not available in this dataset."
"""
            return code
        
        # HANDLE CLASS/LABEL EXPLANATION QUERIES
        explanation_patterns = ['what does', 'what do', 'meaning of', 'means', 'represent']
        if any(pattern in query_lower for pattern in explanation_patterns):
            # Check if asking about target classes (0, 1, etc.)
            if any(num in query_lower for num in ['0', '1', '2', '3', 'class', 'label', 'target']):
                logger.info("DataSummaryTool: Detected class explanation query")
                code = """
# Handle class/label explanation query
if target_column and target_column in df.columns:
    unique_classes = df[target_column].unique()
    class_counts = df[target_column].value_counts()
    
    result = "Class meanings in the dataset:\\n"
    for class_val in sorted(unique_classes):
        count = class_counts[class_val]
        percentage = (count / len(df)) * 100
        if class_names and class_val < len(class_names):
            class_name = class_names[class_val]
            result += f"• {class_val} = {class_name} ({count} instances, {percentage:.1f}%)\\n"
        else:
            # Provide meaningful interpretation based on common patterns
            if class_val == 0:
                result += f"• {class_val} = Negative/No condition ({count} instances, {percentage:.1f}%)\\n"
            elif class_val == 1:
                result += f"• {class_val} = Positive/Has condition ({count} instances, {percentage:.1f}%)\\n"
            else:
                result += f"• {class_val} = Class {class_val} ({count} instances, {percentage:.1f}%)\\n"
else:
    result = "Target class information is not available in this dataset."
"""
                return code
        
        # Default general data summary
        logger.info("DataSummaryTool: Using default general data summary")
        code = """
# General data summary and statistics
result = f"Dataset Summary:<br>"
result += f"- Total records: {len(df)}<br>"
result += f"- Features: {len(df.columns)}<br>"

if target_column and target_column in df.columns:
    target_counts = df[target_column].value_counts()
    result += f"- Target classes: {list(target_counts.index)}<br>"
    for class_val, count in target_counts.items():
        percentage = (count / len(df)) * 100
        class_name = class_names[class_val] if class_names and class_val < len(class_names) else class_val
        result += f"  - {class_name}: {count} ({percentage:.1f}%)<br>"

result += f"<br>Feature Summary:<br>"
for col in df.columns:
    if col != target_column:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            result += f"- {col}: mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]<br>"
        else:
            unique_count = df[col].nunique()
            most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
            result += f"- {col}: {unique_count} unique values, most common='{most_common}'<br>"

result += "<br>"
"""
        return code


class ConversationalTool(DataOperationTool):
    """Tool for handling conversational and general knowledge queries."""
    
    def __init__(self):
        super().__init__(
            name="conversational",
            description="Handle general knowledge and conversational queries"
        )
    
    def can_handle(self, operation_type: str, schema: DatasetSchema) -> bool:
        return operation_type in ['conversational', 'general', 'knowledge']
    
    def generate_code(self, user_query: str, entities: Dict, schema: DatasetSchema) -> str:
        """Generate conversational response code."""
        query_lower = user_query.lower()
        
        # BMI-related queries - GENERALIZED to extract BMI values from context
        if 'bmi' in query_lower:
            if any(word in query_lower for word in ['high', 'rather high', 'too high']):
                code = '''
# Handle BMI knowledge query - generalized approach
import re

# Try to extract BMI value from query context or recent results
bmi_value = None

# Look for BMI value in current query
bmi_match = re.search(r'bmi[:\\s]*([0-9]+(?:\\.[0-9]+)?)', query_lower)
if bmi_match:
    bmi_value = float(bmi_match.group(1))

# If no value in query, try to get from dataset context (last computed average BMI)
if bmi_value is None and 'bmi' in df.columns:
    # Get the most recent BMI value if we're looking at filtered data
    if len(df) == 1:
        bmi_value = df['bmi'].iloc[0]
    else:
        bmi_value = df['bmi'].mean()

if bmi_value is not None:
    # Classify BMI
    if bmi_value < 18.5:
        category = "Underweight"
        health_impact = "may indicate malnutrition or other health issues"
    elif bmi_value < 25:
        category = "Normal weight"
        health_impact = "is associated with lower health risks"
    elif bmi_value < 30:
        category = "Overweight"
        health_impact = "may increase risk of diabetes, heart disease, and other health conditions"
    else:
        if bmi_value < 35:
            category = "Obese Class I"
        elif bmi_value < 40:
            category = "Obese Class II"
        else:
            category = "Obese Class III"
        health_impact = "is associated with significantly increased health risks including diabetes, heart disease, and other complications"
    
    result = f"""Yes, a BMI of {bmi_value:.1f} is considered {category.lower()}. Here's the BMI classification:

• Underweight: BMI < 18.5
• Normal weight: BMI 18.5-24.9
• Overweight: BMI 25-29.9
• Obese Class I: BMI 30-34.9
• Obese Class II: BMI 35-39.9
• Obese Class III: BMI ≥ 40

A BMI of {bmi_value:.1f} falls into the {category} category, which {health_impact}. This is why BMI is often a significant factor in diabetes risk prediction models."""
else:
    result = """BMI values above 25 are generally considered high. Here's the BMI classification:

• Normal weight: BMI 18.5-24.9
• Overweight: BMI 25-29.9
• Obese Class I: BMI 30-34.9
• Obese Class II: BMI 35-39.9
• Obese Class III: BMI ≥ 40

Higher BMI values are associated with increased health risks, which is why BMI is a key factor in diabetes prediction models."""
'''
            elif any(word in query_lower for word in ['normal', 'good', 'healthy']):
                code = '''
# Handle BMI normality query - generalized
import re

# Try to extract BMI value from context
bmi_value = None
bmi_match = re.search(r'bmi[:\\s]*([0-9]+(?:\\.[0-9]+)?)', query_lower)
if bmi_match:
    bmi_value = float(bmi_match.group(1))
elif 'bmi' in df.columns:
    if len(df) == 1:
        bmi_value = df['bmi'].iloc[0]
    else:
        bmi_value = df['bmi'].mean()

if bmi_value is not None:
    is_normal = 18.5 <= bmi_value < 25
    result = f"""A BMI of {bmi_value:.1f} is {"" if is_normal else "not "}considered normal/healthy. 

Normal/healthy BMI range is 18.5-24.9. {"This BMI falls within the healthy range." if is_normal else f"This BMI indicates {'underweight' if bmi_value < 18.5 else 'overweight/obese'} status that may increase health risks, particularly for conditions like diabetes."}"""
else:
    result = "Normal/healthy BMI range is 18.5-24.9. BMI values outside this range may indicate increased health risks, which is why it's an important feature in diabetes prediction models."
'''
            else:
                code = '''
result = "BMI (Body Mass Index) is calculated as weight in kg divided by height in meters squared. It's used to categorize weight status and assess health risks. What specific aspect of BMI would you like to know about?"
'''
        
        # Pregnancy-related queries
        elif 'pregnanc' in query_lower and not any(word in query_lower for word in ['dataset', 'data', 'model']):
            if any(word in query_lower for word in ['average', 'normal', 'typical', 'how many']):
                code = '''
result = """Average number of pregnancies varies by demographics and time period:

• Global average: Around 2.4 children per woman (2021)
• Developed countries: Often 1.5-2.0 children per woman
• Historical context: Women in past generations often had 4-6+ children
• Individual variation: Some women have 0 pregnancies, others 10+

The number of pregnancies can be a factor in diabetes risk assessment, particularly for gestational diabetes history and metabolic changes associated with pregnancy."""
'''
            else:
                code = '''
result = "Pregnancy-related health information can vary widely. What specific aspect would you like to know about - average pregnancy numbers, health impacts, or diabetes risk factors?"
'''
        
        # Age-related queries
        elif 'age' in query_lower and any(word in query_lower for word in ['old', 'young', 'normal']):
            code = '''
result = """Age is a significant factor in diabetes risk. Generally:

• Type 1 diabetes: Can occur at any age, often diagnosed in children/young adults
• Type 2 diabetes: Risk increases with age, especially after 45
• Gestational diabetes: Occurs during pregnancy

The diabetes dataset likely focuses on Type 2 diabetes risk, where older age is associated with higher risk due to factors like decreased insulin sensitivity and longer exposure to risk factors."""
'''
        
        # Glucose-related queries
        elif 'glucose' in query_lower:
            code = '''
result = """Blood glucose levels are crucial for diabetes diagnosis:

• Normal fasting glucose: < 100 mg/dL
• Prediabetes: 100-125 mg/dL  
• Diabetes: ≥ 126 mg/dL (fasting) or ≥ 200 mg/dL (random)

High glucose levels indicate the body's inability to properly regulate blood sugar, which is the defining characteristic of diabetes."""
'''
        
        # General health queries
        elif any(word in query_lower for word in ['healthy', 'normal', 'good', 'bad']):
            code = '''
result = "I'd be happy to help explain health-related concepts! Could you be more specific about what health metric or condition you're asking about? I can provide information about BMI, glucose levels, blood pressure, or other health indicators."
'''
        
        # Default conversational response
        else:
            code = '''
result = "I'm here to help with both diabetes risk analysis and general health questions. Feel free to ask about specific health metrics, or we can dive into the dataset analysis. What would you like to explore?"
'''
        
        return code


class ErrorAnalysisTool(DataOperationTool):
    """Tool for analyzing model errors and failure patterns."""
    
    def __init__(self):
        super().__init__(
            name="error_analysis",
            description="Analyze model errors and prediction failures"
        )
    
    def can_handle(self, operation_type: str, schema: DatasetSchema) -> bool:
        return operation_type in ['error_analysis', 'mistakes', 'errors']
    
    def generate_code(self, user_query: str, entities: Dict, schema: DatasetSchema) -> str:
        """Generate error analysis code."""
        code = """
# Analyze model errors and failure patterns
X = df[feature_columns]
y_true = df[target_column] if target_column in df.columns else None

if model is not None and y_true is not None:
    y_pred = model.predict(X)
    
    # Find incorrect predictions
    errors = y_true != y_pred
    error_count = errors.sum()
    total_count = len(y_true)
    error_rate = (error_count / total_count) * 100
    
    result = f"Model error analysis:\\n"
    result += f"• Total errors: {error_count} out of {total_count} ({error_rate:.1f}%)\\n"
    
    if error_count > 0:
        # Analyze error patterns
        error_df = df[errors]
        result += f"\\nError patterns:\\n"
        
        # Show statistics for errors
        for col in feature_columns[:3]:  # Top 3 features
            if col in error_df.columns:
                mean_errors = error_df[col].mean()
                mean_correct = df[~errors][col].mean()
                result += f"• {col}: errors avg={mean_errors:.1f}, correct avg={mean_correct:.1f}\\n"
    
    result += "\\nUse feature analysis tools to investigate specific error patterns."
else:
    result = "Model or target data not available for error analysis"
"""
        return code


class TraditionalXAITool(DataOperationTool):
    """Tool that routes to traditional XAI actions for specialized explanations."""
    
    def __init__(self):
        super().__init__(
            name="traditional_xai",
            description="Route to traditional XAI system for LIME, SHAP, counterfactuals, etc."
        )
    
    def can_handle(self, operation_type: str, schema: DatasetSchema) -> bool:
        return operation_type in ['explanation', 'lime', 'shap', 'counterfactual', 'interaction', 'what_if']
    
    def generate_code(self, user_query: str, entities: Dict, schema: DatasetSchema) -> str:
        """Route to traditional action system for complex XAI."""
        query_lower = user_query.lower()
        
        # Determine traditional action based on query
        if any(pattern in query_lower for pattern in ['lime', 'explain', 'explanation', 'why']):
            action = "explain"
        elif any(pattern in query_lower for pattern in ['shap']):
            action = "explain shap"
        elif any(pattern in query_lower for pattern in ['counterfactual', 'what would need', 'different prediction']):
            action = "counterfactual"
        elif any(pattern in query_lower for pattern in ['interaction', 'interact', 'combined effect']):
            action = "interaction"
        elif any(pattern in query_lower for pattern in ['what if', 'change', 'modify']):
            action = "whatif"
        else:
            action = "explain"  # Default to explanation
        
        code = f"""
# Route to traditional XAI system for sophisticated explanations
from explain.action import run_action

# Use traditional action system for complex XAI capabilities
traditional_action = "{action}"
try:
    response, status = run_action(traditional_action, df, conversation)
    if status == 1:
        result = response
    else:
        result = "XAI analysis failed. Please try a different approach."
except Exception as e:
    result = f"XAI analysis error: {{str(e)}}"
"""
        return code


class GeneralizedActionPlanner:
    """
    Hybrid action planner that uses generalized tools for simple operations
    and routes to traditional XAI system for complex explanations.
    """
    
    def __init__(self):
        self.tools = [
            ConversationalTool(),  # Natural conversation
            TraditionalXAITool(),  # Complex XAI routing
            FilteringTool(),       # Data filtering
            ModelAnalysisTool(),   # Predictions, accuracy
            FeatureAnalysisTool(), # Feature importance
            DataSummaryTool(),     # Statistics, summaries
            ErrorAnalysisTool()    # Error analysis
        ]
        self.schema = None
    
    def initialize_schema(self, df: pd.DataFrame, target_col: str = None) -> DatasetSchema:
        """Initialize dataset schema for dynamic discovery."""
        self.schema = DatasetSchema.discover(df, target_col)
        logger.info(f"Discovered schema: {len(self.schema.features)} features, {self.schema.size} records")
        return self.schema
    
    def plan_operations(self, user_query: str, entities: Dict[str, Any]) -> List[Tuple[DataOperationTool, str]]:
        """Plan operations based on user query and available tools."""
        if not self.schema:
            raise ValueError("Schema not initialized. Call initialize_schema() first.")
        
        logger.info(f"plan_operations: Planning for query='{user_query}', entities={entities}")
        
        # Discover relevant operations (pass entities for better detection)
        suggested_ops = self.schema.suggest_operations(user_query, entities)
        logger.info(f"plan_operations: Suggested operations: {suggested_ops}")
        
        # Find tools that can handle these operations
        selected_tools = []
        tool_names_used = set()  # Avoid duplicate tools
        
        for op_type in suggested_ops:
            logger.info(f"plan_operations: Looking for tool to handle operation '{op_type}'")
            for tool in self.tools:
                can_handle = tool.can_handle(op_type, self.schema)
                logger.info(f"plan_operations: Tool '{tool.name}' can_handle('{op_type}'): {can_handle}")
                
                if can_handle and tool.name not in tool_names_used:
                    # Special check for filtering tool - only include if actual conditions exist
                    if tool.name == "filtering":
                        has_conditions = (
                            entities.get('patient_id') is not None or
                            entities.get('features') or
                            (entities.get('context_reset') is False and entities.get('features'))
                        )
                        logger.info(f"plan_operations: Filtering tool conditions check: {has_conditions}")
                        if not has_conditions:
                            logger.info("plan_operations: Skipping filtering tool - no conditions")
                            continue  # Skip filtering tool if no conditions
                    
                    code = tool.generate_code(user_query, entities, self.schema)
                    selected_tools.append((tool, code))
                    tool_names_used.add(tool.name)
                    logger.info(f"plan_operations: Selected tool '{tool.name}' for operation '{op_type}'")
                    break  # Use first matching tool for each operation type
        
        logger.info(f"plan_operations: Final selected tools: {[tool.name for tool, code in selected_tools]}")
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
        # Ensure temp_dataset is initialized for compatibility with traditional actions
        if not hasattr(conversation, 'temp_dataset') or conversation.temp_dataset is None:
            logger.info("Building temp_dataset for generalized operations")
            conversation.build_temp_dataset()
        
        # Get data and model from conversation with null checks
        dataset_var = conversation.get_var('dataset')
        model_var = conversation.get_var('model')
        
        if dataset_var is None:
            raise ValueError("Dataset not found in conversation context")
        if model_var is None:
            raise ValueError("Model not found in conversation context")
            
        dataset = dataset_var.contents
        model = model_var.contents
        
        if dataset is None:
            raise ValueError("Dataset contents is None")
        if model is None:
            raise ValueError("Model contents is None")
        
        # Use temp_dataset if available (filtered data), otherwise use main dataset
        if hasattr(conversation, 'temp_dataset') and conversation.temp_dataset is not None:
            temp_data = conversation.temp_dataset.contents
            df = temp_data['X'].copy()
            logger.info(f"Using temp_dataset with {len(df)} rows (filtered data)")
        else:
            df = dataset['X'].copy()
            logger.info(f"Using main dataset with {len(df)} rows")
        
        target_col = None
        
        # Add target column if available
        if 'y' in dataset and len(dataset['y']) > 0:
            target_col = 'target'
            # Only add target for rows that exist in current df (important for filtered data)
            if len(dataset['y']) >= len(df):
                df[target_col] = dataset['y'][:len(df)]
            else:
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
def generalized_action_dispatcher(user_query: str, intent: str, entities: Dict[str, Any], conversation) -> Tuple[str, int]:
    """
    Main entry point for generalized action dispatching.
    
    This replaces the hardcoded action mapping with dynamic tool selection.
    """
    try:
        logger.info(f"generalized_action_dispatcher: Called with query='{user_query}', intent='{intent}', entities={entities}")
        
        # Create execution context
        context = create_generalized_context(conversation)
        if not context:
            logger.error("generalized_action_dispatcher: Could not initialize data context")
            return "Could not initialize data context", 0
        
        # Initialize planner and schema
        planner = GeneralizedActionPlanner()
        schema = planner.initialize_schema(context['df'], context['target_column'])
        
        # Plan operations using the original user query
        operations = planner.plan_operations(user_query, entities)
        logger.info(f"generalized_action_dispatcher: Planned {len(operations)} operations: {[(tool.name, 'code') for tool, code in operations]}")
        
        if not operations:
            logger.warning(f"generalized_action_dispatcher: No suitable operations found for query: {user_query}")
            return f"No suitable operations found for query: {user_query}", 0
        
        # Execute operations
        results = planner.execute_plan(operations, context)
        logger.info(f"generalized_action_dispatcher: Executed operations, got {len(results)} results")
        
        # Format results - filter out empty results
        success_results = []
        for i, (result, error) in enumerate(results):
            logger.info(f"generalized_action_dispatcher: Result {i}: result='{str(result)[:100] if result else 'None'}', error='{error}'")
            if error:
                logger.error(f"Operation failed: {error}")
                return f"Operation failed: {error}", 0
            if result and str(result).strip():  # Only include non-empty results
                success_results.append(str(result))
        
        if success_results:
            final_result = "<br>".join(success_results)
            logger.info(f"generalized_action_dispatcher: Success with {len(success_results)} results, final result length: {len(final_result)}")
            return final_result, 1
        else:
            # If no results but no errors, operations completed successfully but had no output
            logger.warning("generalized_action_dispatcher: Operations completed successfully but produced no output")
            return "Operations completed successfully", 1
            
    except Exception as e:
        logger.error(f"Error in generalized action dispatcher: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"System error: {str(e)}", 0 