"""Lightweight explainer that doesn't require torch.

This provides a minimal explainer interface that works with just LIME and SHAP,
avoiding the heavy torch dependency.
"""
import pandas as pd
import numpy as np
from typing import Callable, Union, Optional, List
import logging

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class LightweightExplainer:
    """A lightweight explainer that uses only LIME and SHAP, no torch required."""
    
    def __init__(self, 
                 prediction_fn: Callable[[np.ndarray], np.ndarray],
                 data: pd.DataFrame,
                 cat_features: Union[List[int], List[str]],
                 class_names: Optional[List[str]] = None):
        """Initialize the lightweight explainer.
        
        Args:
            prediction_fn: Function that takes features and returns predictions
            data: Training data as pandas DataFrame
            cat_features: List of categorical feature indices or names
            class_names: List of class names
        """
        self.prediction_fn = prediction_fn
        self.data = data
        self.class_names = class_names or ['Class 0', 'Class 1']
        
        # Convert categorical features to indices if needed
        if cat_features and isinstance(cat_features[0], str):
            self.cat_features = [data.columns.get_loc(col) for col in cat_features]
        else:
            self.cat_features = cat_features or []
        
        # Initialize explainers
        self._lime_explainer = None
        self._shap_explainer = None
        
        logger.info("Initialized lightweight explainer (LIME + SHAP, no torch)")
    
    def _get_lime_explainer(self):
        """Lazy initialization of LIME explainer."""
        if self._lime_explainer is None and LIME_AVAILABLE:
            self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.data.values,
                feature_names=self.data.columns,
                class_names=self.class_names,
                categorical_features=self.cat_features,
                mode='classification'
            )
        return self._lime_explainer
    
    def _get_shap_explainer(self):
        """Lazy initialization of SHAP explainer."""
        if self._shap_explainer is None and SHAP_AVAILABLE:
            # Use a sample of the data for SHAP background
            background_sample = self.data.sample(min(100, len(self.data)))
            self._shap_explainer = shap.Explainer(
                self.prediction_fn, 
                background_sample.values
            )
        return self._shap_explainer
    
    def explain_instance(self, instance: np.ndarray, method: str = 'lime') -> dict:
        """Explain a single instance.
        
        Args:
            instance: Instance to explain
            method: Explanation method ('lime' or 'shap')
            
        Returns:
            Dictionary with explanation results
        """
        if method.lower() == 'lime':
            return self._explain_with_lime(instance)
        elif method.lower() == 'shap':
            return self._explain_with_shap(instance)
        else:
            # Default to LIME if available, otherwise SHAP
            if LIME_AVAILABLE:
                return self._explain_with_lime(instance)
            elif SHAP_AVAILABLE:
                return self._explain_with_shap(instance)
            else:
                return self._fallback_explanation(instance)
    
    def _explain_with_lime(self, instance: np.ndarray) -> dict:
        """Explain instance with LIME."""
        if not LIME_AVAILABLE:
            logger.warning("LIME not available, using fallback")
            return self._fallback_explanation(instance)
        
        lime_explainer = self._get_lime_explainer()
        if lime_explainer:
            explanation = lime_explainer.explain_instance(
                instance, 
                self.prediction_fn,
                num_features=len(self.data.columns)
            )
            
            # Convert to consistent format
            return {
                'method': 'lime',
                'explanation': explanation.as_list(),
                'score': explanation.score,
                'local_explanation': explanation.local_exp[1] if hasattr(explanation, 'local_exp') else []
            }
        else:
            return self._fallback_explanation(instance)
    
    def _explain_with_shap(self, instance: np.ndarray) -> dict:
        """Explain instance with SHAP."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using fallback")
            return self._fallback_explanation(instance)
        
        shap_explainer = self._get_shap_explainer()
        if shap_explainer:
            shap_values = shap_explainer(instance.reshape(1, -1))
            
            # Convert to consistent format
            feature_importance = list(zip(self.data.columns, shap_values.values[0]))
            
            return {
                'method': 'shap',
                'explanation': feature_importance,
                'score': float(np.sum(np.abs(shap_values.values[0]))),
                'shap_values': shap_values.values[0]
            }
        else:
            return self._fallback_explanation(instance)
    
    def _fallback_explanation(self, instance: np.ndarray) -> dict:
        """Provide a basic fallback explanation when no explainer is available."""
        logger.warning("Using basic fallback explanation (no LIME or SHAP available)")
        
        # Simple feature importance based on prediction sensitivity
        feature_importance = []
        base_prediction = self.prediction_fn(instance.reshape(1, -1))[0]
        
        for i, feature_name in enumerate(self.data.columns):
            # Perturb feature slightly and measure prediction change
            perturbed_instance = instance.copy()
            perturbed_instance[i] *= 1.1  # 10% increase
            
            perturbed_prediction = self.prediction_fn(perturbed_instance.reshape(1, -1))[0]
            sensitivity = abs(perturbed_prediction - base_prediction)
            
            feature_importance.append((feature_name, float(sensitivity)))
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'method': 'fallback',
            'explanation': feature_importance,
            'score': sum(x[1] for x in feature_importance),
            'note': 'Basic sensitivity-based explanation (install lime/shap for better results)'
        }
    
    def get_available_methods(self) -> List[str]:
        """Get list of available explanation methods."""
        methods = []
        if LIME_AVAILABLE:
            methods.append('lime')
        if SHAP_AVAILABLE:
            methods.append('shap')
        if not methods:
            methods.append('fallback')
        return methods

    def summarize_explanations(self, data, filtering_text: str = None, ids_to_regenerate: list = None):
        """Provide explanation summaries compatible with MegaExplainer interface.
        
        This method makes LightweightExplainer compatible with the explain_operation
        function that expects a summarize_explanations method.
        
        Args:
            data: pandas DataFrame containing instances to explain
            filtering_text: Description of any filtering applied
            ids_to_regenerate: List of IDs to regenerate (ignored for lightweight explainer)
            
        Returns:
            Tuple of (full_summary, short_summary) strings
        """
        try:
            if len(data) == 0:
                return ("No instances to explain.", "No instances to explain.")
            
            # Handle single vs multiple instances
            if len(data) == 1:
                # Single instance explanation
                instance = data.iloc[0].values
                explanation = self.explain_instance(instance)
                
                # Format single instance explanation
                method = explanation.get('method', 'unknown')
                feature_importance = explanation.get('explanation', [])
                
                # Create human-readable explanation
                filter_desc = f" for instance where <b>{filtering_text}</b>" if filtering_text else ""
                
                short_summary = f"<b>Model Explanation{filter_desc}:</b><br>"
                short_summary += f"<em>Method: {method.upper()}</em><br><br>"
                
                if feature_importance:
                    short_summary += "Most important features for this prediction:<br>"
                    short_summary += "<ul>"
                    
                    # Show top 5 features
                    top_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:5]
                    for feature, importance in top_features:
                        direction = "increases" if importance > 0 else "decreases"
                        short_summary += f"<li><b>{feature}</b> ({direction} prediction confidence)</li>"
                    
                    short_summary += "</ul>"
                else:
                    short_summary += "No feature importance information available."
                
                if 'note' in explanation:
                    short_summary += f"<br><em>{explanation['note']}</em>"
                
                # Full summary is the same as short for single instances
                full_summary = short_summary
                
            else:
                # Multiple instances - provide aggregate explanation
                filter_desc = f" for instances where <b>{filtering_text}</b>" if filtering_text else ""
                
                short_summary = f"<b>Model Explanations{filter_desc}:</b><br>"
                short_summary += f"<em>Analyzed {len(data)} instances</em><br><br>"
                
                # Aggregate feature importance across instances
                feature_importance_sum = {}
                
                for idx, (_, instance) in enumerate(data.iterrows()):
                    if idx >= 10:  # Limit to first 10 instances for performance
                        break
                        
                    explanation = self.explain_instance(instance.values)
                    feature_importance = explanation.get('explanation', [])
                    
                    for feature, importance in feature_importance:
                        if feature not in feature_importance_sum:
                            feature_importance_sum[feature] = []
                        feature_importance_sum[feature].append(abs(importance))
                
                # Calculate average importance
                avg_importance = []
                for feature, importance_list in feature_importance_sum.items():
                    avg_imp = sum(importance_list) / len(importance_list)
                    avg_importance.append((feature, avg_imp))
                
                if avg_importance:
                    avg_importance.sort(key=lambda x: x[1], reverse=True)
                    
                    short_summary += "Most consistently important features across instances:<br>"
                    short_summary += "<ul>"
                    
                    # Show top 5 features
                    for feature, importance in avg_importance[:5]:
                        short_summary += f"<li><b>{feature}</b> (avg importance: {importance:.3f})</li>"
                    
                    short_summary += "</ul>"
                else:
                    short_summary += "No consistent feature importance patterns found."
                
                short_summary += "<br><em>Note: Analysis limited to first 10 instances for performance.</em>"
                
                # Full summary includes more details
                full_summary = short_summary + "<br><br>"
                full_summary += "For more detailed explanations, try filtering to a single instance using 'filter id X explain'."
            
            return (full_summary, short_summary)
            
        except Exception as e:
            logger.error(f"Error in lightweight explainer summarize_explanations: {e}")
            error_msg = f"Unable to generate explanations: {str(e)}"
            return (error_msg, error_msg)