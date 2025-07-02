"""Feature interaction analysis for understanding how features work together.

This module measures feature interactions by analyzing how the effect of one feature
changes when another feature is held at different values. Uses partial dependence
and conditional interaction analysis.
"""
import copy
from typing import Any

import numpy as np
import pandas as pd


class FeatureInteraction:
    """Measures feature interactions using partial dependence analysis.
    
    Computes interaction strength by examining how the effect of feature i changes
    when feature j is held at different values. Higher interaction means the features
    work together synergistically to influence predictions.
    
    Method:
    1. For each value of feature j, fix j at that value across all data
    2. Compute partial dependence of feature i on the modified data
    3. Measure variance in partial dependence across different j values
    4. Higher variance = stronger interaction between features i and j
    """

    def __init__(self,
                 data: pd.DataFrame,
                 prediction_fn: Any,
                 cat_features: list[str],
                 class_ind: int = None,
                 verbose: bool = False):
        """Initialize the feature interaction analyzer.

        Args:
            data: Training data for computing interactions
            prediction_fn: Model prediction function that returns probabilities
            cat_features: List of categorical feature names
            class_ind: Specific class index to analyze (if None, uses max across classes)
            verbose: Enable detailed logging output
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
        """Compute bidirectional interaction strength between two features.

        Measures how much feature i's effect changes when feature j varies, and vice versa.
        Returns the average of both directions to get symmetric interaction strength.

        Args:
            i: Name of first feature to analyze
            j: Name of second feature to analyze  
            sub_sample_pct: Percentage (0-100) of unique values to sample for efficiency
            number_sub_samples: Exact number of values to sample (overrides percentage)
            
        Returns:
            float: Symmetric interaction strength (higher = more interaction)
        """
        # If number sub_samples is set, use this value
        if number_sub_samples is not None:
            num_sub_samples = number_sub_samples
        else:
            # Otherwise, see if percentage is provided
            if sub_sample_pct is None:
                num_sub_samples = int(len(self.data) * 0.10)
            elif sub_sample_pct == 'full':
                num_sub_samples = len(self.data)
            else:
                sub_sample_pct /= 100
                num_sub_samples = int(len(self.data) * sub_sample_pct)

        i_given_j = self.conditional_interaction(i, j, self.data, num_sub_samples)
        j_given_i = self.conditional_interaction(j, i, self.data, num_sub_samples)
        mean_interaction = np.mean([i_given_j, j_given_i])
        return mean_interaction

    def choose_values_to_sample(self, i: str, data: pd.DataFrame, num_sub_samples: int):
        """Select representative values from a feature for efficient computation.
        
        For categorical features: randomly samples values
        For numerical features: evenly spaces values across the range
        
        Args:
            i: Feature name to sample from
            data: Dataset containing the feature
            num_sub_samples: Target number of values to sample
            
        Returns:
            np.array: Selected representative values for the feature
        """
        unique_values = np.sort(data[i].unique())

        if len(unique_values) < num_sub_samples:
            return unique_values

        if i in self.cat_features:
            # Randomly subsample categorical features
            indices = np.random.choice(len(unique_values), size=num_sub_samples)
            samples = unique_values[indices]
        else:
            # For numeric features, space out indices evenly
            indices = list(range(0, len(unique_values), len(unique_values) // num_sub_samples))
            samples = unique_values[indices]

        return samples

    def conditional_interaction(self, i: str, j: str, data: pd.DataFrame, num_sub_samples: int):
        """Measure how feature i's effect varies when feature j is held constant.
        
        For each value of feature j:
        1. Fix j at that value across all data points
        2. Compute partial dependence of feature i 
        3. Measure the 'flatness' (variance) of i's effect
        
        Higher standard deviation = stronger interaction between i and j.
        
        Args:
            i: Feature whose effect we're measuring
            j: Feature we're conditioning on (holding constant)
            data: Dataset to analyze
            num_sub_samples: Number of j values to test
            
        Returns:
            float: Standard deviation of i's effect across different j values
        """
        # Choose sub sample of feature
        unique_values_of_j = self.choose_values_to_sample(j, data, num_sub_samples)

        results = []
        for unique_val in unique_values_of_j:
            fixed_j_dataset = copy.deepcopy(data)
            fixed_j_dataset[j] = unique_val
            flatness_at_j = self.partial_dependence_flatness(i, fixed_j_dataset, num_sub_samples)
            results.append(flatness_at_j)
        return np.std(results)

    def partial_dependence_flatness(self, i: str, data: pd.DataFrame, num_sub_samples: int) -> float:
        """Measure how much a feature's effect varies across its value range.
        
        For categorical features: Uses (max - min) / 4 as flatness measure
        For numerical features: Uses sample variance as flatness measure
        
        Higher flatness = feature has more variable/non-linear effects
        Lower flatness = feature has consistent/linear effects
        
        Args:
            i: Feature name to analyze
            data: Dataset to compute partial dependence on
            num_sub_samples: Number of feature values to sample
            
        Returns:
            float: Flatness score for the specified class (or max across classes)
        """
        if i in self.cat_features:
            _, dependence = self.partial_dependence(i, data, num_sub_samples)
            max_dep, min_dep = np.max(dependence, axis=0), np.min(dependence, axis=0)
            flatness = (max_dep - min_dep) / 4
        else:
            _, dependence = self.partial_dependence(i, data, num_sub_samples)
            mean_dependence = np.mean(dependence, axis=0)

            # The sample std
            flatness = np.sum((dependence - mean_dependence) ** 2) / (len(dependence) - 1)

        # If there are many classes and no label, return the max
        if self.class_ind is None:
            return np.max(flatness)

        return flatness[self.class_ind] if hasattr(flatness, '__getitem__') else flatness

    def partial_dependence(self, i: str, data: pd.DataFrame, num_sub_samples: int) -> Any:
        """Compute partial dependence plot data for a single feature.
        
        Partial dependence shows the marginal effect of a feature on predictions:
        1. For each unique value of feature i
        2. Set all data points to have that value for feature i
        3. Compute average prediction across all modified data points
        4. This reveals i's isolated effect on the model
        
        Args:
            i: Feature name to analyze
            data: Dataset to compute partial dependence on  
            num_sub_samples: Number of feature values to sample
            
        Returns:
            tuple: (feature_values, dependence_values) both sorted by feature value
        """
        unique_column_values = self.choose_values_to_sample(i, data, num_sub_samples)

        pdp = {}
        for unique_value in unique_column_values:
            # substitute unique value into the data frame
            updated_dataset = copy.deepcopy(data)
            updated_dataset[i] = unique_value

            # compute predictions on updated data
            predictions = self.prediction_fn(updated_dataset.to_numpy())
            # compute the average prediction
            average_prediction = np.mean(predictions, axis=0)
            pdp[unique_value] = average_prediction

        feature_vals = np.array(list(pdp.keys()))
        dependence = np.array([pdp[val] for val in feature_vals])
        sorted_vals = np.argsort(feature_vals)

        feature_vals = feature_vals[sorted_vals]
        dependence = dependence[sorted_vals]

        return feature_vals, dependence