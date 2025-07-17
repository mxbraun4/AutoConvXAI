"""SHAP (SHapley Additive exPlanations) wrapper for model explanations.

This module provides a standardized interface to SHAP's kernel explainer,
which computes Shapley values from cooperative game theory to fairly
distribute the prediction among features.
"""
import numpy as np
import torch
import shap


class SHAPExplainer(torch.nn.Module):
    """SHAP explainer wrapper using kernel-based approximation.
    
    SHAP provides model-agnostic explanations by computing Shapley values,
    which represent each feature's contribution to moving the prediction
    from the expected value to the actual prediction.
    
    This implementation uses KernelSHAP, which works with any model type
    but is computationally intensive for large datasets.
    """

    def __init__(self,
                 model,
                 data: torch.FloatTensor,
                 link: str = 'identity'):
        """Initialize SHAP explainer with model and background data.

        Args:
            model: The ML model to explain (must be callable)
            data: Background dataset for computing expected values (numpy array or tensor).
                  Gets clustered to 25 samples for efficiency.
            link: Output space link function - 'identity' for probability outputs,
                  'logit' for log-odds outputs
        """
        super().__init__()
        self.model = model

        # Cluster background data to 25 representative samples for efficiency
        # KernelSHAP is O(2^n) so using full dataset would be too slow
        self.data = shap.kmeans(data, 25)

        # Initialize KernelSHAP - a model-agnostic but computationally intensive method
        # Future enhancement: Could use TreeSHAP for tree models, DeepSHAP for neural nets
        self.explainer = shap.KernelExplainer(self.model, self.data, link=link)

    def get_explanation(self, data_x: np.ndarray, label) -> tuple[torch.FloatTensor, float]:
        """Generate SHAP explanation for a single instance.
        
        Computes Shapley values that show how each feature contributes to moving
        the prediction from the expected value (average prediction on background data)
        to the actual prediction for this instance.

        Args:
            data_x: Single instance to explain, shape (1, n_features)
            label: The class label to explain (for multi-class models)

        Returns:
            tuple: (shap_values, score) where:
                - shap_values: torch.FloatTensor of shape (n_features,) with Shapley values
                  (positive = increases prediction, negative = decreases prediction)
                - score: Always 0 (included for interface compatibility with LIME)
        """
        # Compute Shapley values using Monte Carlo approximation
        # nsamples=1000 balances accuracy vs performance (was 10,000)
        shap_vals = self.explainer.shap_values(data_x[0], nsamples=1_000, silent=True)

        # Handle multi-class models - extract values for specific label
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            # Multi-class: shap_vals is list of arrays, one per class
            final_shap_values = torch.FloatTensor(shap_vals[label])
        else:
            # Binary classification: shap_vals is 2D array [n_features, n_classes]
            shap_tensor = torch.FloatTensor(shap_vals)
            
            # Check if we have a 2D tensor [features, classes] - extract specific class
            if len(shap_tensor.shape) == 2 and shap_tensor.shape[1] > 1:
                final_shap_values = shap_tensor[:, label]
            else:
                final_shap_values = shap_tensor.flatten()
            
        # Return with dummy score (0) for interface compatibility
        return final_shap_values, 0
