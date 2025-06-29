"""Dynamic Code Generation Framework for AutoGen-powered ML Query Processing.

This framework uses AutoGen's extraction capabilities to generate executable Python code
for data filtering, model evaluation, and statistical analysis operations.
"""

import ast
import inspect
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


def find_column_name(feature_name: str, dataset_columns: List[str]) -> str:
    """Find the actual column name in the dataset using case-insensitive matching.
    
    This makes the system more generalizable across different naming conventions.
    
    Args:
        feature_name: The feature name to find (might be different case)
        dataset_columns: List of actual column names in the dataset
        
    Returns:
        The actual column name if found, otherwise returns the original feature_name
    """
    # First try exact match
    if feature_name in dataset_columns:
        return feature_name
    
    # Try case-insensitive match
    feature_lower = feature_name.lower()
    for col in dataset_columns:
        if col.lower() == feature_lower:
            return col
    
    # If no match found, return original (will likely cause an error, but preserves behavior)
    return feature_name


@dataclass
class FilterOperation:
    """Represents a single filter operation."""
    feature: str
    operator: str  # 'gt', 'lt', 'gte', 'lte', 'eq', 'ne', 'in', 'contains'
    value: Any
    
    def to_pandas_expression(self) -> str:
        """Convert to pandas boolean indexing expression."""
        # Handle index-based filtering (e.g., "instance 2", "row 5")
        if self.feature.lower() in ['index', 'instance', 'row', 'id']:
            if self.operator in ['=', '==', 'eq']:
                return f"dataset.index == {self.value}"
            elif self.operator in ['>', 'gt']:
                return f"dataset.index > {self.value}"
            elif self.operator in ['<', 'lt']:
                return f"dataset.index < {self.value}"
            elif self.operator in ['>=', 'gte']:
                return f"dataset.index >= {self.value}"
            elif self.operator in ['<=', 'lte']:
                return f"dataset.index <= {self.value}"
            else:
                return f"dataset.index == {self.value}"  # Default to equality
        
        # Note: Arithmetic operations (+, -) are handled separately in counterfactual analysis
        # They should not be used for filtering, only for value modifications
        
        # Map symbol operators to internal codes for regular features
        operator_map = {
            '>': 'gt',
            '<': 'lt', 
            '>=': 'gte',
            '<=': 'lte',
            '=': 'eq',
            '==': 'eq',
            '!=': 'ne',
            '<>': 'ne'
        }
        
        # Use mapped operator if available, otherwise use as-is
        op = operator_map.get(self.operator, self.operator)
        
        # Handle arithmetic operators that shouldn't be in filters
        if op in ['+', '-', 'add', 'subtract', 'increase', 'decrease']:
            raise ValueError(f"Arithmetic operator '{self.operator}' should not be used for filtering. Use counterfactual analysis instead.")
        
        # Use case-insensitive column matching for better generalizability
        actual_feature = f"find_column_name('{self.feature}', list(dataset.columns))"
        
        if op == 'gt':
            return f"dataset[{actual_feature}] > {self.value}"
        elif op == 'lt':
            return f"dataset[{actual_feature}] < {self.value}"
        elif op == 'gte':
            return f"dataset[{actual_feature}] >= {self.value}"
        elif op == 'lte':
            return f"dataset[{actual_feature}] <= {self.value}"
        elif op == 'eq':
            return f"dataset[{actual_feature}] == {self.value}"
        elif op == 'ne':
            return f"dataset[{actual_feature}] != {self.value}"
        elif op == 'in':
            return f"dataset[{actual_feature}].isin({self.value})"
        elif op == 'contains':
            return f"dataset[{actual_feature}].str.contains('{self.value}')"
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


@dataclass
class QueryOperation:
    """Represents the operation to perform after filtering."""
    operation_type: str  # 'accuracy', 'statistics', 'predict', 'explain', 'importance'
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class GeneratedQuery:
    """Complete query specification for code generation."""
    filters: List[FilterOperation]
    operation: QueryOperation
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class CodeGenerator:
    """Generates executable Python code from query specifications."""
    
    def __init__(self):
        self.available_operations = {
            'accuracy': self._generate_accuracy_code,
            'statistics': self._generate_statistics_code,
            'predict': self._generate_prediction_code,
            'explain': self._generate_explanation_code,
            'importance': self._generate_importance_code,
            'count': self._generate_count_code,
            'new_predict': self._generate_new_instance_prediction_code,
            'show_data': self._generate_show_data_code,
            'counterfactual': self._generate_counterfactual_code,
            'similarity': self._generate_similarity_code,
            'model_info': self._generate_model_info_code,
            'interaction': self._generate_interaction_code,
        }
    
    def generate(self, query: GeneratedQuery) -> str:
        """Generate executable Python code for the query."""
        code_parts = []
        
        # Function signature
        code_parts.append("def execute_query(dataset, model, explainer=None, conversation=None):")
        code_parts.append("    \"\"\"Dynamically generated query execution function.\"\"\"")
        code_parts.append("")
        
        # Store original dataset for context
        code_parts.append("    original_dataset = dataset.copy()")
        code_parts.append("    original_size = len(original_dataset)")
        code_parts.append("")
        
        # Apply filters
        if query.filters:
            code_parts.append("    # Apply filters")
            for i, filter_op in enumerate(query.filters):
                if i == 0:
                    code_parts.append(f"    filtered_dataset = dataset[{filter_op.to_pandas_expression()}]")
                else:
                    # Replace 'dataset' with 'filtered_dataset' for subsequent filters
                    expr = filter_op.to_pandas_expression().replace('dataset', 'filtered_dataset')
                    code_parts.append(f"    filtered_dataset = filtered_dataset[{expr}]")
            
            code_parts.append("    filtered_size = len(filtered_dataset)")
            code_parts.append("    dataset = filtered_dataset  # Use filtered dataset for operation")
            code_parts.append("")
        else:
            code_parts.append("    # No filters applied")
            code_parts.append("    filtered_size = len(dataset)")
            code_parts.append("")
        
        # Generate operation-specific code
        operation_generator = self.available_operations.get(query.operation.operation_type)
        if operation_generator:
            operation_code = operation_generator(query)
            code_parts.extend(operation_code)
        else:
            code_parts.append(f"    raise ValueError('Unknown operation: {query.operation.operation_type}')")
        
        return "\n".join(code_parts)
    
    def _generate_accuracy_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for accuracy evaluation."""
        return [
            "    # Calculate accuracy on filtered data",
            "    if len(dataset) == 0:",
            "        return 'No data points match the specified criteria.'",
            "",
            "    X = dataset.drop(columns=[col for col in dataset.columns if col in ['target', 'y', 'label']], errors='ignore')",
            "    y_true = dataset.get('y', dataset.get('target', dataset.get('label')))",
            "    ",
            "    if y_true is None:",
            "        # Try to get target from conversation context",
            "        if conversation and hasattr(conversation, 'get_var'):",
            "            target_data = conversation.get_var('dataset')",
            "            if target_data and 'y' in target_data.contents:",
            "                y_true = target_data.contents['y'].loc[dataset.index]",
            "        ",
            "    if y_true is not None:",
            "        y_pred = model.predict(X)",
            "        accuracy = (y_pred == y_true).mean() * 100",
            "        ",
            "        filter_desc = '' if len(query.filters) == 0 else f' (filtered: {_describe_filters(query.filters)})'",
            "        result = f'Accuracy{filter_desc}: {accuracy:.2f}% ({len(dataset)}/{original_size} samples)'",
            "        return result",
            "    else:",
            "        return 'Cannot calculate accuracy: target variable not found.'"
        ]
    
    def _generate_statistics_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for statistical analysis."""
        return [
            "    # Generate statistics for filtered data",
            "    if len(dataset) == 0:",
            "        return 'No data points match the specified criteria.'",
            "",
            "    stats_result = []",
            "    filter_desc = '' if len(query.filters) == 0 else f' (filtered by {_describe_filters(query.filters)})'",
            "",
            "    # Check if specific feature requested (e.g., 'average age')",
            "    query_context = str(query.context) if query.context else ''",
            "    all_text = query_context.lower()",
            "",
            "    # Look for specific feature names in query context (case-insensitive)",
            "    requested_feature = None",
            "    feature_names = list(dataset.columns)",
            "    for feature in feature_names:",
            "        if feature.lower() in all_text:",
            "            requested_feature = feature",
            "            break",
            "    ",
            "    # Also try AutoGen extracted features if available",
            "    if not requested_feature and hasattr(query, 'filters') and query.filters:",
            "        for filter_op in query.filters:",
            "            potential_feature = find_column_name(filter_op.feature, feature_names)",
            "            if potential_feature in feature_names:",
            "                requested_feature = potential_feature",
            "                break",
            "",
            "    if requested_feature:",
            "        # Generate focused statistics for requested feature",
            "        mean_val = dataset[requested_feature].mean()",
            "        std_val = dataset[requested_feature].std()", 
            "        min_val = dataset[requested_feature].min()",
            "        max_val = dataset[requested_feature].max()",
            "        ",
            "        stats_result.append(f'{requested_feature.title()} Statistics{filter_desc}:')",
            "        stats_result.append(f'• Average: {mean_val:.2f}')",
            "        stats_result.append(f'• Standard deviation: {std_val:.2f}')",
            "        stats_result.append(f'• Range: {min_val:.2f} to {max_val:.2f}')",
            "        stats_result.append(f'• Sample size: {len(dataset)} instances')",
            "    else:",
            "        # Generate comprehensive statistics for all features",
            "        stats_result.append(f'Dataset Statistics{filter_desc}:')",
            "        stats_result.append(f'Sample size: {len(dataset)} out of {original_size} total')",
            "        stats_result.append('')",
            "",
            "        # Calculate statistics for numeric columns",
            "        numeric_cols = dataset.select_dtypes(include=[np.number]).columns",
            "        for col in numeric_cols:",
            "            mean_val = dataset[col].mean()",
            "            std_val = dataset[col].std()",
            "            min_val = dataset[col].min()",
            "            max_val = dataset[col].max()",
            "            stats_result.append(f'{col.title()} Statistics:')",
            "            stats_result.append(f'  Average: {mean_val:.2f}')",
            "            stats_result.append(f'  Std Dev: {std_val:.2f}')",
            "            stats_result.append(f'  Range: {min_val:.2f} to {max_val:.2f}')",
            "            stats_result.append('')",
            "",
            "    return '\\n'.join(stats_result)"
        ]
    
    def _generate_prediction_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for prediction queries - handles filtering by prediction values."""
        return [
            "    # Separate prediction filters from feature filters",
            "    # Handle any variation of prediction feature names",
            "    prediction_keywords = ['prediction', 'predicted', 'pred', 'predicts', 'classify', 'classified']",
            "    prediction_filters = [f for f in query.filters if f.feature.lower() in prediction_keywords]",
            "    feature_filters = [f for f in query.filters if f.feature.lower() not in prediction_keywords]",
            "    ",
            "    # Start with full dataset",
            "    working_dataset = dataset.copy()",
            "    ",
            "    # Apply feature-based filters first (if any)",
            "    if feature_filters:",
            "        for filter_op in feature_filters:",
            "            actual_feature = find_column_name(filter_op.feature, list(working_dataset.columns))",
            "            if actual_feature in working_dataset.columns:",
            "                if filter_op.operator == 'eq':",
            "                    working_dataset = working_dataset[working_dataset[actual_feature] == filter_op.value]",
            "                elif filter_op.operator == 'gt':",
            "                    working_dataset = working_dataset[working_dataset[actual_feature] > filter_op.value]",
            "                elif filter_op.operator == 'lt':",
            "                    working_dataset = working_dataset[working_dataset[actual_feature] < filter_op.value]",
            "    ",
            "    if len(working_dataset) == 0:",
            "        return 'No data points match the specified criteria.'",
            "    ",
            "    # ALWAYS generate predictions first (required for prediction filtering)",
            "    X = working_dataset.drop(columns=[col for col in working_dataset.columns if col in ['target', 'y', 'label']], errors='ignore')",
            "    predictions = model.predict(X)",
            "    probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None",
            "    ",
            "    # NOW apply prediction-based filtering (after predictions exist)",
            "    if prediction_filters:",
            "        prediction_mask = None",
            "        for filter_op in prediction_filters:",
            "            if filter_op.operator == 'eq':",
            "                mask = predictions == filter_op.value",
            "            elif filter_op.operator == 'gt':",
            "                mask = predictions > filter_op.value",
            "            elif filter_op.operator == 'lt':",
            "                mask = predictions < filter_op.value",
            "            else:",
            "                continue",
            "            ",
            "            if prediction_mask is None:",
            "                prediction_mask = mask",
            "            else:",
            "                prediction_mask = prediction_mask & mask",
            "        ",
            "        # Apply prediction filter to both dataset and predictions",
            "        if prediction_mask is not None and prediction_mask.any():",
            "            working_dataset = working_dataset[prediction_mask]",
            "            predictions = predictions[prediction_mask]",
            "            if probabilities is not None:",
            "                probabilities = probabilities[prediction_mask]",
            "        else:",
            "            return f'No instances found with the specified prediction criteria.'",
            "    ",
            "    result = []",
            "    ",
            "    # Show results based on whether we filtered by predictions",
            "    if prediction_filters:",
            "        # Show only the instances matching prediction criteria",
            "        filter_desc = ', '.join([f\"{f.feature}={f.value}\" for f in prediction_filters])",
            "        result.append(f'Instances where {filter_desc}:')",
            "        result.append(f'Found {len(working_dataset)} matching instances')",
            "        result.append('')",
            "        ",
            "        # Show sample instances",
            "        result.append('Sample instances:')",
            "        display_count = min(10, len(working_dataset))",
            "        sample_data = working_dataset.head(display_count)",
            "        ",
            "        for idx, (instance_id, row) in enumerate(sample_data.iterrows()):",
            "            result.append(f'Instance {instance_id}:')",
            "            feature_parts = []",
            "            for feature, value in row.items():",
            "                if feature not in ['target', 'y', 'label']:",
            "                    feature_parts.append(f'{feature}={value:.1f}')",
            "            result.append('  ' + ' | '.join(feature_parts))",
            "            if idx < len(sample_data) - 1:",
            "                result.append('')",
            "        ",
            "        if len(working_dataset) > display_count:",
            "            result.append('')",
            "            result.append(f'... and {len(working_dataset) - display_count} more instances')",
            "    else:",
            "        # Show prediction summary",
            "        filter_desc = '' if len(feature_filters) == 0 else f' (with filters)'",
            "        result.append(f'Prediction Summary{filter_desc}:')",
            "        result.append(f'Total instances: {len(working_dataset)}')",
            "        result.append('')",
            "        ",
            "        unique_preds, counts = np.unique(predictions, return_counts=True)",
            "        for pred, count in zip(unique_preds, counts):",
            "            percentage = (count / len(predictions)) * 100",
            "            result.append(f'Predicted {pred}: {count} instances ({percentage:.1f}%)')",
            "        ",
            "        result.append('')",
            "        result.append('Sample instances:')",
            "        sample_data = working_dataset.head(5)",
            "        ",
            "        for instance_id, row in sample_data.iterrows():",
            "            result.append(f'Instance {instance_id}:')",
            "            feature_parts = []",
            "            for feature, value in row.items():",
            "                if feature not in ['target', 'y', 'label']:",
            "                    feature_parts.append(f'{feature}={value:.1f}')",
            "            result.append('  ' + ' | '.join(feature_parts))",
            "            result.append('')",
            "    ",
            "    return '\\n'.join(result)"
        ]
    
    def _generate_explanation_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for model explanations."""
        return [
            "    # Generate explanations for filtered data",
            "    if len(dataset) == 0:",
            "        return 'No data points match the specified criteria.'",
            "",
            "    if explainer is None:",
            "        return 'Explainer not available for generating explanations.'",
            "",
            "    # Take first instance if multiple",
            "    instance = dataset.iloc[0]",
            "    X = instance.drop(labels=[col for col in instance.index if col in ['target', 'y', 'label']], errors='ignore')",
            "",
            "    try:",
            "        if hasattr(explainer, 'explain_instance'):",
            "            explanation = explainer.explain_instance(X.values)",
            "            return f'Explanation for instance {instance.name}: {explanation}'",
            "        else:",
            "            return 'Explanation method not supported by current explainer.'",
            "    except Exception as e:",
            "        return f'Error generating explanation: {str(e)}'"
        ]
    
    def _generate_importance_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for feature importance/relevance analysis - maximum generalizability."""
        return [
            "    # Feature importance and relevance analysis",
            "    feature_names = dataset.drop(columns=[col for col in dataset.columns if col in ['target', 'y', 'label']], errors='ignore').columns",
            "    result = []",
            "    result.append('Feature Relevance for Model Predictions:')",
            "    result.append('')",
            "    ",
            "    # Method 1: Model-based feature importance (if available)",
            "    if hasattr(model, 'feature_importances_'):",
            "        importances = model.feature_importances_",
            "        sorted_idx = np.argsort(importances)[::-1]",
            "        ",
            "        result.append('📊 Model Feature Importance Rankings:')",
            "        for i, idx in enumerate(sorted_idx[:8]):  # Top 8",
            "            importance_pct = importances[idx] * 100",
            "            result.append(f'  {i+1}. {feature_names[idx]}: {importance_pct:.1f}% importance')",
            "        result.append('')",
            "    ",
            "    # Method 2: Correlation with predictions (universal approach)",
            "    try:",
            "        X = dataset.drop(columns=[col for col in dataset.columns if col in ['target', 'y', 'label']], errors='ignore')",
            "        predictions = model.predict(X)",
            "        ",
            "        # Calculate correlation between each feature and predictions",
            "        feature_correlations = []",
            "        for feature in feature_names:",
            "            correlation = np.corrcoef(dataset[feature], predictions)[0, 1]",
            "            if not np.isnan(correlation):",
            "                feature_correlations.append((feature, abs(correlation)))",
            "        ",
            "        # Sort by absolute correlation",
            "        feature_correlations.sort(key=lambda x: x[1], reverse=True)",
            "        ",
            "        result.append('🔗 Feature-Prediction Correlations:')",
            "        for feature, correlation in feature_correlations[:8]:  # Top 8",
            "            correlation_pct = correlation * 100",
            "            result.append(f'  • {feature}: {correlation_pct:.1f}% correlation with predictions')",
            "        result.append('')",
            "    except Exception as e:",
            "        result.append('Could not compute feature-prediction correlations.')",
            "        result.append('')",
            "    ",
            "    # Method 3: Statistical relevance analysis",
            "    result.append('📈 Feature Value Ranges & Patterns:')",
            "    for feature in feature_names[:6]:  # Top 6 features",
            "        feature_std = dataset[feature].std()",
            "        feature_range = dataset[feature].max() - dataset[feature].min()",
            "        result.append(f'  • {feature}: range={feature_range:.1f}, variability={feature_std:.1f}')",
            "    ",
            "    result.append('')",
            "    result.append('💡 Key Insights:')",
            "    result.append('  • Higher importance/correlation = more relevant for predictions')",
            "    result.append('  • Features with high variability often have more predictive power')",
            "    result.append('  • Model considers feature combinations, not just individual features')",
            "    ",
            "    return '\\n'.join(result)"
        ]
    
    def _generate_count_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for counting instances."""
        return [
            "    # Count instances",
            "    filter_desc = '' if len(query.filters) == 0 else f' matching {_describe_filters(query.filters)}'",
            "    return f'Found {len(dataset)} instances{filter_desc} out of {original_size} total.'"
        ]

    def _generate_new_instance_prediction_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for predicting on new instances with specified feature values."""
        return [
            "    # Create new instance prediction",
            "    # Extract feature values from query context",
            "    context = query.context or {}",
            "    original_query = context.get('original_query', '').lower()",
            "    ",
            "    # Parse feature values from AutoGen filters (preferred) or query text",
            "    import re",
            "    feature_values = {}",
            "    feature_names = list(dataset.columns)",
            "    ",
            "    # First, try to extract from AutoGen filters (more reliable)",
            "    if hasattr(query, 'filters') and query.filters:",
            "        for filter_op in query.filters:",
            "            if filter_op.operator == 'eq':  # Only equality filters for new prediction",
            "                actual_feature = find_column_name(filter_op.feature, feature_names)",
            "                if actual_feature in feature_names:",
            "                    feature_values[actual_feature] = float(filter_op.value)",
            "    ",
            "    # Fallback: Look for patterns like 'age = 50', 'bmi = 25', etc. in query text",
            "    if not feature_values:",
            "        for feature in feature_names:",
            "            # Try case-insensitive pattern matching",
            "            pattern = rf'{feature.lower()}\\s*=\\s*([\\d\\.]+)'",
            "            match = re.search(pattern, original_query)",
            "            if match:",
            "                actual_feature = find_column_name(feature, feature_names)",
            "                feature_values[actual_feature] = float(match.group(1))",
            "    ",
            "    if not feature_values:",
            "        return 'No feature values specified for new instance prediction.'",
            "    ",
            "    # Create new instance with specified values (use robust defaults for missing features)",
            "    new_instance = {}",
            "    for feature in feature_names:",
            "        if feature in feature_values:",
            "            new_instance[feature] = feature_values[feature]",
            "        else:",
            "            # Use robust default: mean if available, otherwise median, otherwise 0",
            "            feature_mean = dataset[feature].mean()",
            "            if pandas.isna(feature_mean):",
            "                feature_median = dataset[feature].median()",
            "                if pandas.isna(feature_median):",
            "                    new_instance[feature] = 0.0  # Final fallback",
            "                else:",
            "                    new_instance[feature] = feature_median",
            "            else:",
            "                new_instance[feature] = feature_mean",
            "    ",
            "    # Convert to DataFrame for prediction",
            "    import pandas as pd",
            "    new_df = pd.DataFrame([new_instance])",
            "    ",
            "    # Make prediction",
            "    prediction = model.predict(new_df)[0]",
            "    probabilities = model.predict_proba(new_df)[0] if hasattr(model, 'predict_proba') else None",
            "    ",
            "    # Format result",
            "    result = []",
            "    result.append('New Instance Prediction:')",
            "    ",
            "    # Show specified features",
            "    specified_features = [f'{k}={v}' for k, v in feature_values.items()]",
            "    features_str = ', '.join(specified_features)",
            "    result.append(f'• Specified features: {features_str}')",
            "    result.append(f'• Predicted class: {prediction}')",
            "    ",
            "    if probabilities is not None:",
            "        result.append('• Prediction probabilities:')",
            "        for class_idx, prob in enumerate(probabilities):",
            "            result.append(f'  - Class {class_idx}: {prob:.3f} ({prob*100:.1f}%)')",
            "    ",
            "    return '\\n'.join(result)"
        ]

    def _generate_show_data_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for displaying data points."""
        return [
            "    # Show data points",
            "    if len(dataset) == 0:",
            "        return 'No data points match the specified criteria.'",
            "    ",
            "    result = []",
            "    filter_desc = '' if len(query.filters) == 0 else f' (filtered by {_describe_filters(query.filters)})'",
            "    result.append(f'Data Points{filter_desc}:')",
            "    result.append('')",
            "    ",
            "    # Show up to 10 instances to avoid overwhelming output",
            "    display_data = dataset.head(10)",
            "    ",
            "    for idx, (instance_id, row) in enumerate(display_data.iterrows()):",
            "        result.append(f'Instance {instance_id}:')",
            "        for feature, value in row.items():",
            "            result.append(f'  • {feature}: {value}')",
            "        result.append('')",
            "    ",
            "    if len(dataset) > 10:",
            "        result.append(f'... and {len(dataset) - 10} more instances')",
            "    ",
            "    return '\\n'.join(result)"
        ]

    def _generate_counterfactual_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for counterfactual analysis."""
        return [
            "    # Counterfactual analysis",
            "    if len(dataset) == 0:",
            "        return 'No data points match the specified criteria.'",
            "    ",
            "    # Parse specific changes from query context",
            "    context = query.context or {}",
            "    original_query = context.get('original_query', '').lower()",
            "    ",
            "    # Look for specific change patterns like 'increased bmi by 10'",
            "    import re",
            "    specific_changes = {}",
            "    change_patterns = [",
            "        (r'increas(?:ed?|ing)\\s+(\\w+)\\s+by\\s+([\\d\\.]+)', '+'),",
            "        (r'decreas(?:ed?|ing)\\s+(\\w+)\\s+by\\s+([\\d\\.]+)', '-'),",
            "        (r'rais(?:ed?|ing)\\s+(\\w+)\\s+by\\s+([\\d\\.]+)', '+'),",
            "        (r'lower(?:ed?|ing)\\s+(\\w+)\\s+by\\s+([\\d\\.]+)', '-')",
            "    ]",
            "    ",
            "    for pattern, operation in change_patterns:",
            "        matches = re.findall(pattern, original_query)",
            "        for feature, amount in matches:",
            "            actual_feature = find_column_name(feature, list(dataset.columns))",
            "            if actual_feature in dataset.columns:",
            "                specific_changes[actual_feature] = (operation, float(amount))",
            "    ",
            "    result = []",
            "    filter_desc = '' if len(query.filters) == 0 else f' (filtered by {_describe_filters(query.filters)})'",
            "    result.append(f'Counterfactual Analysis{filter_desc}:')",
            "    result.append(f'Analyzing {len(dataset)} instances')",
            "    result.append('')",
            "    ",
            "    if specific_changes:",
            "        # Apply specific changes to all instances",
            "        result.append('Testing specific changes requested:')",
            "        ",
            "        for feature, (operation, amount) in specific_changes.items():",
            "            # Get original predictions for all instances",
            "            X_original = dataset.drop(columns=[col for col in dataset.columns if col in ['target', 'y', 'label']], errors='ignore')",
            "            original_preds = model.predict(X_original)",
            "            ",
            "            # Apply modification to all instances",
            "            X_modified = X_original.copy()",
            "            if operation == '+':",
            "                X_modified[feature] = X_modified[feature] + amount",
            "                change_desc = f'+{amount}'",
            "            else:",
            "                X_modified[feature] = X_modified[feature] - amount",
            "                change_desc = f'-{amount}'",
            "            ",
            "            # Get new predictions",
            "            new_preds = model.predict(X_modified)",
            "            ",
            "            # Count prediction changes",
            "            changed_instances = sum(original_preds != new_preds)",
            "            total_instances = len(dataset)",
            "            ",
            "            result.append(f'• {feature} {change_desc}:')",
            "            result.append(f'  - {changed_instances}/{total_instances} instances changed predictions')",
            "            ",
            "            if changed_instances > 0:",
            "                # Show some examples of changes",
            "                changed_indices = []",
            "                for i, (orig, new) in enumerate(zip(original_preds, new_preds)):",
            "                    if orig != new and len(changed_indices) < 3:  # Show first 3 examples",
            "                        instance_id = dataset.index[i]",
            "                        old_val = X_original.iloc[i][feature]",
            "                        new_val = X_modified.iloc[i][feature]",
            "                        result.append(f'    Instance {instance_id}: {feature}={old_val:.1f}→{new_val:.1f}, prediction={orig}→{new}')",
            "                        changed_indices.append(i)",
            "                ",
            "                if hasattr(model, 'predict_proba'):",
            "                    # Show probability changes",
            "                    orig_probas = model.predict_proba(X_original)",
            "                    new_probas = model.predict_proba(X_modified)",
            "                    if orig_probas.shape[1] > 1:  # Binary classification",
            "                        avg_prob_change = (new_probas[:, 1] - orig_probas[:, 1]).mean()",
            "                        result.append(f'  - Average probability change: {avg_prob_change:+.3f}')",
            "            else:",
            "                result.append(f'  - No prediction changes observed')",
            "    else:",
            "        # Default: test general modifications on first instance",
            "        if len(dataset) > 0:",
            "            result.append('Testing general feature modifications (first instance):')",
            "            instance = dataset.iloc[0]",
            "            instance_id = dataset.index[0]",
            "            X_original = instance.drop(labels=[col for col in instance.index if col in ['target', 'y', 'label']], errors='ignore')",
            "            ",
            "            original_pred = model.predict(X_original.values.reshape(1, -1))[0]",
            "            result.append(f'• Instance {instance_id} original prediction: {original_pred}')",
            "            ",
            "            counterfactuals = []",
            "            for feature in X_original.index:",
            "                for modifier in [0.8, 1.2]:  # -20% and +20%",
            "                    modified_instance = X_original.copy()",
            "                    modified_instance[feature] *= modifier",
            "                    ",
            "                    new_pred = model.predict(modified_instance.values.reshape(1, -1))[0]",
            "                    if new_pred != original_pred:",
            "                        change_pct = (modifier - 1) * 100",
            "                        counterfactuals.append((",
            "                            feature, change_pct, modified_instance[feature], new_pred",
            "                        ))",
            "            ",
            "            if counterfactuals:",
            "                result.append('Found counterfactual changes:')",
            "                for feature, change_pct, new_value, new_pred in counterfactuals[:5]:",
            "                    result.append(f'• {feature}: {change_pct:+.1f}% → value={new_value:.2f} → prediction={new_pred}')",
            "            else:",
            "                result.append('No simple counterfactuals found with ±20% feature changes.')",
            "    ",
            "    return '\\n'.join(result)"
        ]

    def _generate_similarity_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for similarity analysis between instances."""
        return [
            "    # Similarity analysis",
            "    if len(dataset) < 2:",
            "        return 'Need at least 2 instances for similarity analysis.'",
            "    ",
            "    # Take first two instances",
            "    instance_a = dataset.iloc[0]",
            "    instance_b = dataset.iloc[1]",
            "    id_a, id_b = dataset.index[0], dataset.index[1]",
            "    ",
            "    # Get predictions",
            "    X_a = instance_a.drop(labels=[col for col in instance_a.index if col in ['target', 'y', 'label']], errors='ignore')",
            "    X_b = instance_b.drop(labels=[col for col in instance_b.index if col in ['target', 'y', 'label']], errors='ignore')",
            "    ",
            "    pred_a = model.predict(X_a.values.reshape(1, -1))[0]",
            "    pred_b = model.predict(X_b.values.reshape(1, -1))[0]",
            "    ",
            "    result = []",
            "    result.append(f'Similarity Analysis: Instance {id_a} vs Instance {id_b}')",
            "    result.append(f'• Instance {id_a} prediction: {pred_a}')",
            "    result.append(f'• Instance {id_b} prediction: {pred_b}')",
            "    result.append('')",
            "    ",
            "    # Compare feature values",
            "    result.append('Feature Comparison:')",
            "    for feature in X_a.index:",
            "        val_a, val_b = X_a[feature], X_b[feature]",
            "        diff = abs(val_a - val_b)",
            "        result.append(f'• {feature}: {val_a:.2f} vs {val_b:.2f} (diff: {diff:.2f})')",
            "    ",
            "    # Calculate overall similarity (Euclidean distance)",
            "    import numpy as np",
            "    distance = np.sqrt(sum((X_a - X_b) ** 2))",
            "    similarity = 1 / (1 + distance)  # Convert distance to similarity score",
            "    ",
            "    result.append('')",
            "    result.append(f'Overall similarity score: {similarity:.3f}')",
            "    ",
            "    if pred_a == pred_b:",
            "        result.append(f'Both instances have the same prediction ({pred_a}), likely due to similar feature patterns.')",
            "    else:",
            "        result.append(f'Different predictions ({pred_a} vs {pred_b}) despite similarity score of {similarity:.3f}.')",
            "    ",
            "    return '\\n'.join(result)"
        ]

    def _generate_model_info_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for model information and transparency."""
        return [
            "    # Model information and transparency",
            "    result = []",
            "    result.append('Model Information:')",
            "    ",
            "    # Model type",
            "    model_type = type(model).__name__",
            "    result.append(f'• Model type: {model_type}')",
            "    ",
            "    # Model parameters (if available)",
            "    if hasattr(model, 'get_params'):",
            "        params = model.get_params()",
            "        result.append('• Key parameters:')",
            "        for param, value in list(params.items())[:5]:  # Show first 5 params",
            "            result.append(f'  - {param}: {value}')",
            "    ",
            "    # Feature count",
            "    n_features = len(dataset.columns)",
            "    result.append(f'• Number of features: {n_features}')",
            "    ",
            "    # Feature names",
            "    feature_list = ', '.join(dataset.columns)",
            "    result.append(f'• Features: {feature_list}')",
            "    ",
            "    # Output classes",
            "    if hasattr(model, 'classes_'):",
            "        classes = model.classes_",
            "        result.append(f'• Output classes: {list(classes)}')",
            "    ",
            "    # Training info (if available)",
            "    if hasattr(model, 'n_estimators'):",
            "        result.append(f'• Number of estimators: {model.n_estimators}')",
            "    ",
            "    if hasattr(model, 'max_depth'):",
            "        result.append(f'• Max depth: {model.max_depth}')",
            "    ",
            "    result.append('')",
            "    result.append('Model Logic: This model learns patterns from training data to predict outcomes based on input features.')",
            "    ",
            "    return '\\n'.join(result)"
        ]

    def _generate_interaction_code(self, query: GeneratedQuery) -> List[str]:
        """Generate code for feature interaction analysis."""
        return [
            "    # Feature interaction analysis",
            "    if len(dataset) == 0:",
            "        return 'No data points match the specified criteria.'",
            "    ",
            "    result = []",
            "    result.append('Feature Interaction Analysis:')",
            "    result.append('')",
            "    ",
            "    # Calculate correlation matrix",
            "    correlation_matrix = dataset.corr()",
            "    ",
            "    # Find strongest correlations",
            "    strong_correlations = []",
            "    features = list(dataset.columns)",
            "    ",
            "    for i, feat1 in enumerate(features):",
            "        for j, feat2 in enumerate(features):",
            "            if i < j:  # Avoid duplicates",
            "                corr_value = correlation_matrix.loc[feat1, feat2]",
            "                if abs(corr_value) > 0.3:  # Threshold for 'strong' correlation",
            "                    strong_correlations.append((feat1, feat2, corr_value))",
            "    ",
            "    # Sort by absolute correlation strength",
            "    strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)",
            "    ",
            "    if strong_correlations:",
            "        result.append('Strong feature correlations found:')",
            "        for feat1, feat2, corr in strong_correlations[:5]:  # Top 5",
            "            corr_desc = 'positive' if corr > 0 else 'negative'",
            "            result.append(f'• {feat1} ↔ {feat2}: {corr:.3f} ({corr_desc})')",
            "    else:",
            "        result.append('No strong feature correlations found (threshold: 0.3).')",
            "    ",
            "    result.append('')",
            "    result.append('Note: High correlations indicate features that tend to change together,')",
            "    result.append('which may suggest redundancy or meaningful relationships in the data.')",
            "    ",
            "    return '\\n'.join(result)"
        ]


class AutoGenQueryExtractor:
    """Uses AutoGen to extract structured query components from natural language."""
    
    def __init__(self, autogen_decoder):
        self.decoder = autogen_decoder
    
    def extract_query_components(self, natural_language: str, conversation) -> Optional[GeneratedQuery]:
        """Extract structured query components using AutoGen agents."""
        try:
            # Use AutoGen to get structured extraction
            autogen_response = self.decoder.complete_sync(natural_language, conversation)
            
            # Parse AutoGen response into our structured format
            if autogen_response.get("intent_response"):
                intent_data = autogen_response["intent_response"]
                entities = intent_data.get("entities", {})
                intent = intent_data.get("intent")
                
                # Build filters from entities
                filters = []
                
                # Handle special case for patient_id (instance-specific queries)
                if entities.get("patient_id"):
                    filters.append(FilterOperation("instance", "eq", entities["patient_id"]))
                
                # Handle general feature-based filters
                if entities.get("features") and entities.get("operators") and entities.get("values"):
                    features = entities["features"]
                    operators = entities["operators"] 
                    values = entities["values"]
                    
                    # Convert to our operator format
                    operator_map = {
                        "greater": "gt",
                        "less": "lt", 
                        "greaterequal": "gte",
                        "lessequal": "lte",
                        "equal": "eq",
                        "equals": "eq",
                        "=": "eq",
                        ">": "gt",
                        "<": "lt"
                    }
                    
                    # Skip arithmetic operators when intent suggests counterfactual analysis
                    arithmetic_operators = ["+", "-", "add", "subtract", "increase", "decrease", "increased", "decreased"]
                    
                    for feature, op, value in zip(features, operators, values):
                        # Skip arithmetic operations - these are handled in counterfactual analysis
                        if op.lower() in arithmetic_operators:
                            continue
                        
                        # Normalize prediction-related feature names
                        feature_lower = feature.lower()
                        if any(pred_term in feature_lower for pred_term in ["predict", "classified", "class"]):
                            # Convert any prediction-related term to standard "prediction"
                            normalized_feature = "prediction"
                        else:
                            normalized_feature = feature
                            
                        normalized_op = operator_map.get(op, op)
                        # Only create filter if it's a valid comparison operator and not duplicate
                        if normalized_op in ["gt", "lt", "gte", "lte", "eq", "ne"]:
                            # Check for duplicates based on feature name and value
                            is_duplicate = any(
                                f.feature.lower() == normalized_feature.lower() and f.value == value
                                for f in filters
                            )
                            if not is_duplicate:
                                filters.append(FilterOperation(normalized_feature, normalized_op, value))
                
                # Also check if query mentions specific instance numbers in the original text
                instance_match = re.search(r'instance\s+(\d+)', natural_language.lower())
                if instance_match and not any(f.feature.lower() in ['index', 'instance', 'row', 'id'] for f in filters):
                    instance_num = int(instance_match.group(1))
                    filters.append(FilterOperation("instance", "eq", instance_num))
                
                # Check for prediction filtering patterns FIRST (highest priority)
                prediction_patterns = [
                    (r'(?:model\s+)?predicted?\s+(\d+)', 'prediction', 'eq'),
                    (r'(?:model\s+)?predicts?\s+(\d+)', 'prediction', 'eq'),
                    (r'prediction\s*=\s*(\d+)', 'prediction', 'eq'),
                    (r'prediction\s+(?:is|equals?)\s+(\d+)', 'prediction', 'eq'),
                    (r'classified?\s+as\s+(\d+)', 'prediction', 'eq'),
                    (r'where.*predicted?\s+(\d+)', 'prediction', 'eq'),
                    (r'instances.*predicted?\s+(\d+)', 'prediction', 'eq')
                ]
                
                # Only add prediction filter if none exists yet with same value
                for pattern, feature, operator in prediction_patterns:
                    pred_match = re.search(pattern, natural_language.lower())
                    if pred_match:
                        pred_value = int(pred_match.group(1))
                        # Check for duplicates
                        is_duplicate = any(
                            f.feature.lower() in ['prediction', 'predicted', 'pred'] and f.value == pred_value
                            for f in filters
                        )
                        if not is_duplicate:
                            filters.append(FilterOperation(feature, operator, pred_value))
                        break  # Only process the first matching pattern
                
                # Check for age-related filtering patterns  
                age_patterns = [
                    (r'(?:patients?|people|instances?)\s+older\s+than\s+(\d+)', 'age', 'gt'),
                    (r'(?:patients?|people|instances?)\s+younger\s+than\s+(\d+)', 'age', 'lt'),
                    (r'age\s*>\s*(\d+)', 'age', 'gt'),
                    (r'age\s*<\s*(\d+)', 'age', 'lt'),
                    (r'age\s*>=\s*(\d+)', 'age', 'gte'),
                    (r'age\s*<=\s*(\d+)', 'age', 'lte')
                ]
                
                for pattern, feature, operator in age_patterns:
                    age_match = re.search(pattern, natural_language.lower())
                    if age_match and not any(f.feature == 'age' for f in filters):
                        age_value = int(age_match.group(1))
                        filters.append(FilterOperation(feature, operator, age_value))
                
                # Check for other common filtering patterns
                other_patterns = [
                    (r'bmi\s*>\s*([\\d\\.]+)', 'bmi', 'gt'),
                    (r'bmi\s*<\s*([\\d\\.]+)', 'bmi', 'lt'),
                    (r'glucose\s*>\s*([\\d\\.]+)', 'glucose', 'gt'),
                    (r'glucose\s*<\s*([\\d\\.]+)', 'glucose', 'lt')
                ]
                
                # Only add these patterns if this is NOT a counterfactual query
                if not any(phrase in natural_language.lower() for phrase in ["increased by", "decreased by", "what if", "happen if"]):
                    for pattern, feature, operator in other_patterns:
                        match = re.search(pattern, natural_language.lower())
                        if match and not any(f.feature == feature for f in filters):
                            value = float(match.group(1))
                            filters.append(FilterOperation(feature, operator, value))
                
                # Map intent to operation
                operation_map = {
                    "performance": "accuracy",
                    "data": "statistics", 
                    "predict": "predict",
                    "explain": "explain",
                    "importance": "importance",
                    "show": "show_data",
                    "counterfactual": "counterfactual",
                    "similarity": "similarity", 
                    "model": "model_info",
                    "interaction": "interaction",
                    "new_predict": "new_predict"
                }
                
                # Enhanced intent detection from query text
                query_lower = natural_language.lower()
                
                # Check for counterfactual patterns first (highest priority)
                counterfactual_patterns = [
                    "increased by", "decreased by", "what if", "happen if", "change",
                    "what would happen", "if we increased", "if we decreased",
                    "what could", "minimum change", "counterfactual"
                ]
                
                is_counterfactual = any(pattern in query_lower for pattern in counterfactual_patterns)
                
                # Check for counting queries
                count_patterns = [
                    "how many", "count", "number of", "how many instances", 
                    "how many patients", "how many records", "count instances"
                ]
                
                is_count_query = any(pattern in query_lower for pattern in count_patterns)
                
                # Override intent based on specific patterns (ABSOLUTE PRIORITY)
                # Check for prediction-related queries FIRST (highest priority)
                prediction_keywords = ["predicted", "predicts", "prediction", "classify", "classified"]
                has_prediction_keyword = any(keyword in query_lower for keyword in prediction_keywords)
                
                if is_counterfactual:
                    operation_type = "counterfactual"
                elif is_count_query:
                    operation_type = "count"
                elif has_prediction_keyword and ("instance" in query_lower or "where" in query_lower or "show" in query_lower):
                    operation_type = "predict"  # Prediction filtering takes priority
                elif any(phrase in query_lower for phrase in ["show me", "display", "show data"]) and "instance" in query_lower and not any(word in query_lower for word in ["important", "influential"]):
                    operation_type = "show_data"
                elif any(phrase in query_lower for phrase in ["influential", "important"]) and "instance" in query_lower:
                    operation_type = "explain"  # Show explanations for important instances
                elif any(phrase in query_lower for phrase in ["compare", "similarity", "why same", "why different"]):
                    operation_type = "similarity"
                elif any(phrase in query_lower for phrase in ["system logic", "algorithm", "model type", "what kind"]):
                    operation_type = "model_info"
                elif any(phrase in query_lower for phrase in ["interaction", "correlat", "how features"]):
                    operation_type = "interaction" 
                elif any(phrase in query_lower for phrase in ["relevant", "important", "importance", "influential", "feature relevance", "feature importance", "features are relevant"]):
                    operation_type = "importance"
                elif any(phrase in query_lower for phrase in ["new instance", "predict for", "given age"]) and "=" in query_lower:
                    operation_type = "new_predict"
                else:
                    # Only use AutoGen intent if no pattern override detected
                operation_type = operation_map.get(intent, "statistics")
                
                operation = QueryOperation(operation_type)
                
                # Pass original query in context for feature detection
                context = {
                    "original_query": natural_language,
                    "intent": intent,
                    "entities": entities
                }
                
                return GeneratedQuery(filters=filters, operation=operation, context=context)
                
        except Exception as e:
            logger.error(f"Error extracting query components: {e}")
            
        return None


def _describe_filters(filters: List[FilterOperation]) -> str:
    """Create human-readable description of filters."""
    descriptions = []
    for f in filters:
        op_desc = {
            'gt': '>', 'lt': '<', 'gte': '≥', 'lte': '≤', 'eq': '=', 'ne': '≠'
        }.get(f.operator, f.operator)
        descriptions.append(f"{f.feature} {op_desc} {f.value}")
    return ", ".join(descriptions)


class DynamicQueryExecutor:
    """Executes dynamically generated query code."""
    
    def __init__(self):
        self.code_generator = CodeGenerator()
    
    def execute(self, query: GeneratedQuery, dataset: pd.DataFrame, model, explainer=None, conversation=None) -> str:
        """Execute a generated query safely with maximum generalizability."""
        try:
            # Generate the code
            code = self.code_generator.generate(query)
            logger.info(f"🔧 Generated code for {query.operation.operation_type} operation with {len(query.filters)} filters")
            
            # Create comprehensive execution environment for maximum generalizability
            import numpy as np
            import pandas as pd
            import re
            
            safe_globals = {
                # Core modules
                'pd': pd,
                'np': np,
                're': re,
                'numpy': np,
                'pandas': pd,
                
                # Query context
                '_describe_filters': _describe_filters,
                'find_column_name': find_column_name,
                'query': query,
                
                # Enhanced builtins for comprehensive data analysis
                '__builtins__': {
                    # Basic types and operations
                    'len': len, 'str': str, 'int': int, 'float': float, 'round': round,
                    'min': min, 'max': max, 'sum': sum, 'abs': abs, 'bool': bool,
                    'type': type, 'hasattr': hasattr, 'getattr': getattr, 'isinstance': isinstance,
                    
                    # Collections and iteration
                    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                    'zip': zip, 'enumerate': enumerate, 'range': range,
                    'sorted': sorted, 'reversed': reversed,
                    
                    # Logic and control
                    'any': any, 'all': all, 'filter': filter, 'map': map,
                    
                    # Math operations
                    'pow': pow, 'divmod': divmod,
                    
                    # Import capability for advanced operations
                    '__import__': __import__,
                    
                    # Exception handling
                    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
                }
            }
            
            # Log the generated code for debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Generated code:\n{code}")
            
            # Execute the generated code
            exec(code, safe_globals)
            
            # Call the generated function
            if 'execute_query' not in safe_globals:
                raise ValueError("Generated code did not create execute_query function")
                
            result = safe_globals['execute_query'](dataset, model, explainer, conversation)
            
            logger.info(f"✅ Successfully executed dynamic query: {query.operation.operation_type}")
            return result
            
        except Exception as e:
            logger.error(f"💥 Error executing dynamic query: {e}", exc_info=True)
            
            # Provide detailed error information for debugging
            error_details = [
                f"Query operation: {query.operation.operation_type}",
                f"Number of filters: {len(query.filters)}",
                f"Error type: {type(e).__name__}",
                f"Error message: {str(e)}"
            ]
            
            if query.filters:
                filter_details = [f"{f.feature} {f.operator} {f.value}" for f in query.filters]
                error_details.append(f"Filters: {'; '.join(filter_details)}")
            
            detailed_error = "\n".join(error_details)
            logger.error(f"Detailed error information:\n{detailed_error}")
            
            return f"Dynamic code execution failed:\n{detailed_error}\n\nThis helps identify the issue for maximum generalizability."