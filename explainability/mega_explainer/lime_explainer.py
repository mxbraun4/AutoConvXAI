"""LIME explanations wrapper for tabular data.

This module provides a clean interface to the LIME library for generating
local explanations of model predictions on tabular data.
"""
from lime import lime_tabular
import numpy as np
import torch


class Lime(torch.nn.Module):
    """LIME (Local Interpretable Model-agnostic Explanations) wrapper for tabular data.
    
    This class provides a standardized interface to LIME's tabular explainer,
    generating local explanations by learning an interpretable linear model
    around individual predictions.
    
    LIME works by:
    1. Generating perturbed samples around the instance to explain
    2. Getting model predictions on these perturbed samples
    3. Fitting a linear model to approximate the black-box model locally
    4. Using linear model coefficients as feature importance scores
    """

    def __init__(self,
                 model,
                 data: np.ndarray,
                 discrete_features: list,
                 mode: str = "tabular",
                 sample_around_instance: bool = False,
                 kernel_width: float = 0.75,
                 n_samples: int = 10_000,
                 discretize_continuous: bool = False):
        """Initialize LIME explainer with model and dataset configuration.

        Args:
            model: The ML model to explain (must be callable with data arrays)
            data: Background dataset used for generating perturbations (np.ndarray)
            discrete_features: List of indices indicating which features are categorical/discrete
            mode: Explanation mode - currently only "tabular" is supported
            sample_around_instance: If True, sample around the specific instance; 
                                  if False, sample from training data distribution
            kernel_width: Controls locality of explanations - smaller values = more local explanations.
                         Gets scaled by sqrt(n_features) automatically
            n_samples: Number of perturbed samples to generate for fitting linear model
            discretize_continuous: Whether to discretize continuous features into bins
        """
        # Store configuration parameters
        self.data = data
        self.mode = mode
        self.model = model
        self.n_samples = n_samples
        self.discretize_continuous = discretize_continuous
        self.sample_around_instance = sample_around_instance
        self.discrete_features = discrete_features

        # Initialize the underlying LIME explainer
        if self.mode == "tabular":
            # Create LIME tabular explainer with scaled kernel width
            # Kernel width is scaled by sqrt(n_features) to maintain consistent locality
            # across datasets with different numbers of features
            self.explainer = lime_tabular.LimeTabularExplainer(
                self.data,
                mode="classification",  # Assuming classification tasks
                categorical_features=self.discrete_features,
                sample_around_instance=self.sample_around_instance,
                discretize_continuous=self.discretize_continuous,
                kernel_width=kernel_width * np.sqrt(self.data.shape[1]),  # Auto-scale kernel width
            )
        else:
            message = "Currently, only lime tabular explainer is implemented"
            raise NotImplementedError(message)

        super().__init__()

    def get_explanation(self, data_x: np.ndarray, label=None) -> tuple[torch.FloatTensor, float]:
        """Generate LIME explanation for a single instance.
        
        This method uses LIME to explain why the model made a specific prediction
        by fitting a local linear model around the instance and extracting
        feature importance coefficients.

        Args:
            data_x: Single data instance to explain, shape (1, num_features)
            label: The specific class label to explain (if None, explains predicted class)

        Returns:
            tuple: (feature_importances, fidelity_score) where:
                - feature_importances: torch.FloatTensor of shape (num_features,) with 
                  importance scores for each feature (positive = increases prediction, 
                  negative = decreases prediction)
                - fidelity_score: R² score indicating how well the linear model fits 
                  the black-box model locally (higher = more reliable explanation)
                  
        Raises:
            NameError: If the requested label is not found in LIME's explanation
            NotImplementedError: If mode is not "tabular"
        """
        if self.mode == "tabular":
            # Generate LIME explanation for the single instance
            # LIME creates perturbed samples around data_x[0] and fits a linear model
            output = self.explainer.explain_instance(
                data_x[0],  # Single instance (1D array)
                self.model,  # Black-box model to explain
                num_samples=self.n_samples,  # Number of perturbed samples for linear model
                num_features=data_x.shape[1],  # Include all features in explanation
                labels=(label,),  # Specific label to explain
                top_labels=None  # Don't limit to top labels
            )
            
            # Validate that LIME generated explanation for requested label
            if label not in output.local_exp:
                message = (f"label {label} not in local_explanation! "
                           f"Only labels are {output.local_exp.keys()}")
                raise NameError(message)
            
            # Extract feature importance scores for the requested label
            local_explanation = output.local_exp[label]

            # Convert LIME's sparse representation to dense array format
            # LIME returns list of (feature_index, importance_score) tuples
            # We need dense array where position i = importance of feature i
            att_arr = [0.0] * data_x.shape[1]  # Initialize with zeros
            for feat_idx, importance in local_explanation:
                att_arr[feat_idx] = importance
                
            # Return as torch tensor with fidelity score
            # Fidelity score (R²) indicates how well linear model approximates black-box model
            return torch.FloatTensor(att_arr), output.score
        else:
            raise NotImplementedError("Only tabular mode is currently supported")
