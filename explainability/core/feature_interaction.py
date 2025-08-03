"""Feature interaction analysis using partial dependence methods.

Measures how features work together by analyzing effect variations
when features are held at different values.
"""
import copy
from typing import Any

import numpy as np
import pandas as pd


class FeatureInteraction:
    """Measures feature interactions using partial dependence analysis.
    
    Analyzes how feature effects change when other features vary.
    Higher interaction scores indicate synergistic feature relationships.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 prediction_fn: Any,
                 cat_features: list[str],
                 class_ind: int = None,
                 verbose: bool = False):
        """Initialize feature interaction analyzer.

        Args:
            data: Training data for analysis
            prediction_fn: Model prediction function
            cat_features: Categorical feature names
            class_ind: Target class index (None = max across classes)
            verbose: Enable logging
        """
        self.data = data
        self.class_ind = class_ind
        self.prediction_fn = prediction_fn
        self.cat_features = cat_features
        self.verbose = verbose

    def feature_interaction(self,
                            i: str,
                            j: str,
                            sub_sample_pct: float = None,
                            number_sub_samples: int = None):
        """Compute symmetric interaction strength between two features.

        Args:
            i: First feature name
            j: Second feature name  
            sub_sample_pct: Percentage of values to sample (0-100)
            number_sub_samples: Exact sample count (overrides percentage)
            
        Returns:
            float: Interaction strength (higher = stronger interaction)
        """
        # Determine sample size for efficiency
        if number_sub_samples is not None:
            num_sub_samples = number_sub_samples
        else:
            if sub_sample_pct is None:
                num_sub_samples = int(len(self.data) * 0.10)  # Default 10%
            elif sub_sample_pct == 'full':
                num_sub_samples = len(self.data)
            else:
                sub_sample_pct /= 100
                num_sub_samples = int(len(self.data) * sub_sample_pct)

        # Compute bidirectional interaction and average for symmetry
        i_given_j = self.conditional_interaction(i, j, self.data, num_sub_samples)
        j_given_i = self.conditional_interaction(j, i, self.data, num_sub_samples)
        mean_interaction = np.mean([i_given_j, j_given_i])
        return mean_interaction

    def choose_values_to_sample(self, i: str, data: pd.DataFrame, num_sub_samples: int):
        """Select representative feature values for efficient computation.
        
        Args:
            i: Feature name
            data: Dataset
            num_sub_samples: Target sample count
            
        Returns:
            np.array: Representative feature values
        """
        unique_values = np.sort(data[i].unique())

        if len(unique_values) < num_sub_samples:
            return unique_values

        if i in self.cat_features:
            # Random sampling for categorical features
            indices = np.random.choice(len(unique_values), size=num_sub_samples)
            samples = unique_values[indices]
        else:
            # Even spacing for numerical features
            indices = list(range(0, len(unique_values), len(unique_values) // num_sub_samples))
            samples = unique_values[indices]

        return samples

    def conditional_interaction(self, i: str, j: str, data: pd.DataFrame, num_sub_samples: int):
        """Measure how feature i's effect varies when j is held constant.
        
        Args:
            i: Feature whose effect is measured
            j: Feature held constant
            data: Dataset
            num_sub_samples: Number of j values to test
            
        Returns:
            float: Standard deviation of i's effect (interaction strength)
        """
        # Sample values of conditioning feature j
        unique_values_of_j = self.choose_values_to_sample(j, data, num_sub_samples)

        results = []
        for unique_val in unique_values_of_j:
            # Fix j at this value across all data points
            fixed_j_dataset = copy.deepcopy(data)
            fixed_j_dataset[j] = unique_val
            flatness_at_j = self.partial_dependence_flatness(i, fixed_j_dataset, num_sub_samples)
            results.append(flatness_at_j)
        return np.std(results)

    def partial_dependence_flatness(self, i: str, data: pd.DataFrame, num_sub_samples: int) -> float:
        """Measure feature effect variability across its value range.
        
        Args:
            i: Feature name
            data: Dataset
            num_sub_samples: Sample count
            
        Returns:
            float: Effect variability score
        """
        if i in self.cat_features:
            # Range-based measure for categorical features
            _, dependence = self.partial_dependence(i, data, num_sub_samples)
            max_dep, min_dep = np.max(dependence, axis=0), np.min(dependence, axis=0)
            flatness = (max_dep - min_dep) / 4
        else:
            # Variance-based measure for numerical features
            _, dependence = self.partial_dependence(i, data, num_sub_samples)
            mean_dependence = np.mean(dependence, axis=0)
            flatness = np.sum((dependence - mean_dependence) ** 2) / (len(dependence) - 1)

        # Return max across classes if no specific class requested
        if self.class_ind is None:
            return np.max(flatness)

        return flatness[self.class_ind] if hasattr(flatness, '__getitem__') else flatness

    def partial_dependence(self, i: str, data: pd.DataFrame, num_sub_samples: int) -> Any:
        """Compute partial dependence for a feature.
        
        Shows feature's isolated effect by fixing it at different values
        and averaging predictions across all data points.
        
        Args:
            i: Feature name
            data: Dataset
            num_sub_samples: Sample count
            
        Returns:
            tuple: (feature_values, dependence_values)
        """
        unique_column_values = self.choose_values_to_sample(i, data, num_sub_samples)

        pdp = {}
        for unique_value in unique_column_values:
            # Fix feature at this value across all data points
            updated_dataset = copy.deepcopy(data)
            updated_dataset[i] = unique_value

            # Get predictions and compute average
            predictions = self.prediction_fn(updated_dataset)
            average_prediction = np.mean(predictions, axis=0)
            pdp[unique_value] = average_prediction

        feature_vals = np.array(list(pdp.keys()))
        dependence = np.array([pdp[val] for val in feature_vals])
        
        # Sort results by feature value if possible
        try:
            sorted_vals = np.argsort(feature_vals)
            feature_vals = feature_vals[sorted_vals]
            dependence = dependence[sorted_vals]
        except TypeError:
            # Keep original order for mixed data types
            pass

        return feature_vals, dependence