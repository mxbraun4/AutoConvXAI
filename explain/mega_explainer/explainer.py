"""Compares many explanations to determine the best one."""
import copy
from dataclasses import dataclass
from functools import partial
from typing import Union, Any

import heapq
import numpy as np
import pandas as pd
import torch

from explain.mega_explainer.lime_explainer import Lime
from explain.mega_explainer.perturbation_methods import NormalPerturbation
from explain.mega_explainer.shap_explainer import SHAPExplainer


@dataclass
class MegaExplanation:
    """The return format for the mega explanation!"""
    list_exp: list  # List of (feature_name, importance_score) tuples, sorted by importance
    score: float  # Confidence/fidelity score of the selected explanation method
    label: int  # The class label being explained (predicted class)
    best_explanation_type: str  # Which method was selected (e.g., "lime_0.75", "shap")
    agree: bool  # Whether multiple explanation methods agreed on feature importance rankings


def conv_disc_inds_to_char_enc(discrete_feature_indices: list[int], n_features: int):
    """Converts an array of discrete feature indices to a char encoding.

    Here, the ith value in the returned array is 'c' or 'd' for whether the feature is
    continuous or discrete respectively.

    Args:
        discrete_feature_indices: An array like [0, 1, 2] where the ith value corresponds to
                                  whether the arr[i] column in the data is discrete.
        n_features: The number of features in the data.
    Returns:
        char_encoding: An encoding like ['c', 'd', 'c'] where the ith value indicates whether
                       that respective column in the data is continuous ('c') or discrete ('d')
    """
    # Check to make sure (1) feature indices are integers and (2) they are unique
    error_message = "Features all must be type int but are not"
    assert all(isinstance(f, int) for f in discrete_feature_indices), error_message
    error_message = "Features indices must be unique but there are repetitions"
    assert len(set(discrete_feature_indices)) == len(discrete_feature_indices), error_message
    # Perform conversion - initialize with placeholder 'e' to catch errors
    char_encoding = ['e'] * n_features
    for i in range(len(char_encoding)):
        if i in discrete_feature_indices:
            char_encoding[i] = 'd'  # 'd' for discrete/categorical features
        else:
            char_encoding[i] = 'c'  # 'c' for continuous/numerical features
    # Safety check - no 'e' should remain if conversion worked correctly
    assert 'e' not in char_encoding, 'Error in char encoding processing!'
    return char_encoding


class Explainer:
    """
    Explainer is the orchestrator class that drives the logic for selecting
    the best possible explanation from the set of explanation methods.
    """

    def __init__(self,
                 explanation_dataset: np.ndarray,
                 explanation_model: Any,
                 feature_names: list[str],
                 discrete_features: list[int],
                 use_selection: bool = True):
        """
        Init.

        Args:
            explanation_dataset: background data, given as numpy array
            explanation_model: the callable black box model. the model should be callable via
                               explanation_model(data) to generate prediction probabilities
            feature_names: the feature names
            discrete_features: The indices of the discrete features in the dataset. Note, in the
                               rest of the repo, we adopt the terminology 'categorical features'.
                               However, in this mega_explainer sub folder, we adopt the term
                               `discrete features` to describe these features.
            use_selection: Whether to use the explanation selection. If false, uses lime.
        """
        # Convert input data to numpy array format for consistency
        if isinstance(explanation_dataset, pd.DataFrame):
            # Deep copy prevents modifying original dataset when converting to numpy
            # This avoids confusing side effects but uses more memory for large datasets
            explanation_dataset = copy.deepcopy(explanation_dataset)
            explanation_dataset = explanation_dataset.to_numpy()
        else:
            arr_type = type(explanation_dataset)
            message = f"Data must be pd.DataFrame or np.ndarray, not {arr_type}"
            assert isinstance(explanation_dataset, np.ndarray), message

        self.data = explanation_dataset
        self.model = explanation_model
        self.feature_names = feature_names

        # Initialize multiple explanation methods to compare and select the best one
        # Strategy: Run several variants of LIME with different kernel widths plus SHAP,
        # then automatically select the most faithful explanation for each instance
        
        # Create LIME template with common parameters
        lime_template = partial(Lime,
                                model=self.model,
                                data=self.data,
                                discrete_features=discrete_features)

        # Multiple LIME variants with different kernel widths for robustness
        # Wider kernels = more global explanations, narrower = more local
        if use_selection:
            kernel_widths = [0.25, 0.50, 0.75, 1.0]  # Range from local to global explanations
        else:
            kernel_widths = [0.75]  # Standard LIME kernel width

        # Build explanation method registry
        available_explanations = {}
        for width in kernel_widths:
            name = f"lime_{round(width, 3)}"
            available_explanations[name] = lime_template(kernel_width=width)

        # Add SHAP as alternative explanation method for comparison
        if use_selection:
            shap_explainer = SHAPExplainer(self.model, self.data)
            available_explanations["shap"] = shap_explainer

        self.explanation_methods = available_explanations

        # TODO(satya): change this to be inputs to __init__
        # Perturbation parameters for testing explanation faithfulness
        # These control how we modify input features to test if explanations are accurate
        self.perturbation_mean = 0.0        # Center perturbations around original values
        self.perturbation_std = 0.05        # Small standard deviation for gentle perturbations
        self.perturbation_flip_percentage = 0.03  # Flip 3% of categorical features randomly
        self.perturbation_max_distance = 0.4     # Maximum distance from original in feature space

        # Convert discrete feature indices to character encoding for perturbation methods
        # This creates ['c', 'd', 'c', ...] array indicating continuous vs discrete features
        self.feature_types = conv_disc_inds_to_char_enc(discrete_feature_indices=discrete_features,
                                                        n_features=self.data.shape[1])

        # Initialize perturbation method for faithfulness testing
        # This will generate slightly modified versions of input data to test explanation quality
        self.perturbation_method = NormalPerturbation("tabular",
                                                      mean=self.perturbation_mean,
                                                      std=self.perturbation_std,
                                                      flip_percentage=self.perturbation_flip_percentage)

    @staticmethod
    def _arr(x) -> np.ndarray:
        """Converts x to a numpy array, handling both torch tensors and other types.
        
        This utility function ensures consistent numpy array format for computations,
        properly detaching torch tensors from the computation graph.
        
        Args:
            x: Input data (torch.Tensor, list, numpy array, etc.)
            
        Returns:
            np.ndarray: The input converted to numpy array format
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    def _compute_faithfulness_auc(self, data, explanation, c_label, k, metric="topk"):
        """Computes AUC for faithfulness scores, perturbing top k (where k is an array).
        
        Faithfulness measures how well the explanation predicts model behavior:
        - High faithfulness = when important features are changed, model predictions change a lot
        - Low faithfulness = changing "important" features doesn't affect model much
        
        Args:
            data: The input instance being explained
            explanation: Feature importance scores from explanation method
            c_label: The class label being explained
            k: Array of top-k values to test (e.g., [2, 3, 4] tests top-2, top-3, top-4 features)
            metric: "topk" to test most important features, other values test least important
        Returns:
            faithfulness: Aggregated faithfulness score across all k values
        """
        faithfulness = 0
        for k_i in k:
            # Start with mask that protects all features from perturbation (all True)
            n_feats = len(explanation)
            top_k_map = torch.ones(n_feats, dtype=torch.bool)

            # Ensure k_i is within valid range
            k_val = max(1, min(k_i, n_feats))

            # Get absolute importance scores to rank features
            exp_abs = torch.abs(explanation) if isinstance(explanation, torch.Tensor) else torch.abs(torch.tensor(explanation))

            # Validate explanation tensor before torch.topk
            if k_val > len(exp_abs):
                raise ValueError(f"Cannot select top-{k_val} features from explanation with only {len(exp_abs)} features")

            # Find the top-k most important features and allow them to be perturbed
            top_indices = torch.topk(exp_abs, k=k_val).indices
            top_k_map[top_indices] = False  # False = allow perturbation

            # Test faithfulness: if explanation is good, perturbing important features
            # should cause large changes in model predictions
            if metric == "topk":
                faithfulness += self._compute_faithfulness_topk(data, c_label, top_k_map)
            else:
                # Test bottom-k features instead (should cause smaller prediction changes)
                faithfulness += self._compute_faithfulness_topk(data, c_label, ~top_k_map)
        return faithfulness

    def _compute_faithfulness_topk(self, x, label, top_k_mask, num_samples: int = 10_000):
        """Approximates the expected local faithfulness of the explanation in a neighborhood.
        
        This is the core faithfulness test: generate many perturbed versions of the input
        where certain features are modified, then measure how much the model's predictions change.
        Good explanations should identify features that, when changed, cause large prediction changes.

        Args:
            x: The original sample to test
            label: The class label being explained
            top_k_mask: Boolean mask where True = keep feature unchanged, False = allow perturbation
            num_samples: Number of perturbed samples to generate for Monte Carlo estimation
        """
        perturb_args = {
            "original_sample": x[0],
            "feature_mask": top_k_mask,  # Controls which features can be perturbed
            "num_samples": num_samples,
            "max_distance": self.perturbation_max_distance,
            "feature_metadata": self.feature_types  # Tells perturbation method feature types
        }
        # Generate many slightly modified versions of the original input
        x_perturbed = self.perturbation_method.get_perturbed_inputs(**perturb_args)

        # Get model predictions for original and perturbed inputs
        # Extract probability for the specific class we're explaining
        y_original = self._arr([i[label] for i in self._arr(self.model(x.reshape(1, -1)))])
        y_perturbed = self._arr([i[label] for i in self._arr(self.model(x_perturbed.float()))])

        # Faithfulness = average absolute change in prediction probability
        # Higher values mean the explanation correctly identified important features
        return np.mean(np.abs(y_original - y_perturbed), axis=0)

    @staticmethod
    def check_exp_data_shape(data_x: np.ndarray) -> np.ndarray:
        """Validates and reshapes data for single-instance explanation.
        
        Ensures the input data represents exactly one instance and reshapes it
        to the expected 2D format (1, n_features) for explanation methods.
        
        Args:
            data_x: Input data to be explained
            
        Returns:
            np.ndarray: Properly shaped data as (1, n_features)
            
        Raises:
            AssertionError: If data contains multiple instances
        """
        # Validate that we have a single instance for explanation
        data_x_shape = data_x.shape
        if len(data_x_shape) > 1:
            n_samples = data_x_shape[0]
            if n_samples > 1:
                message = f"Data must be individual sample, but has shape {data_x_shape}"
                assert len(data_x_shape) == 1, message
        elif len(data_x_shape) == 1:
            # Reshape 1D array to 2D (1, n_features) format expected by explanation methods
            data_x = data_x.reshape(1, -1)
        return data_x

    def explain_instance(self,
                         data: Union[np.ndarray, pd.DataFrame],
                         top_k_starting_pct: float = 0.2,
                         top_k_ending_pct: float = 0.5,
                         epsilon: float = 1e-4,
                         return_fidelities: bool = False) -> MegaExplanation:
        """Computes the best explanation for a single instance using automatic method selection.

        This is the main entry point for generating explanations. It runs multiple explanation
        methods (LIME variants + SHAP), tests their faithfulness and stability, then automatically
        selects and returns the most reliable explanation.

        Process:
        1. Run all available explanation methods on the input
        2. Test faithfulness (how well explanations predict model behavior when features change)
        3. Select best method based on faithfulness scores
        4. Use stability as tiebreaker if faithfulness scores are close
        5. Return the selected explanation with metadata

        Args:
            data: The single instance to explain (will be converted to numpy if DataFrame)
            top_k_starting_pct: Start of range for testing feature importance (default: 20% of features)
            top_k_ending_pct: End of range for testing feature importance (default: 50% of features)
            epsilon: Threshold for determining if faithfulness scores are "close enough" for tiebreaking
            return_fidelities: Whether to return raw fidelity scores along with explanation
            
        Returns:
            MegaExplanation: The best explanation containing feature importances, confidence scores,
                           method used, and whether methods agreed on feature rankings
        """
        if not isinstance(data, np.ndarray):
            try:
                data = data.to_numpy()
            except Exception as exp:
                message = f"Data not type np.ndarray, failed to convert with error {exp}"
                raise NameError(message)

        explanations, scores = {}, {}
        fidelity_scores_topk = {}

        # Makes sure data is formatted correctly
        formatted_data = self.check_exp_data_shape(data)

        # Gets indices of 20-50% of data
        lower_index = int(formatted_data.shape[1]*top_k_starting_pct)
        upper_index = int(formatted_data.shape[1]*top_k_ending_pct)
        k = list(range(lower_index, upper_index))

        # Explain the most likely class
        label = np.argmax(self.model(formatted_data)[0])

        # Iterate over each explanation method and compute fidelity scores of topk
        # and non-topk features per the method
        for method in self.explanation_methods.keys():
            cur_explainer = self.explanation_methods[method]
            cur_expl, score = cur_explainer.get_explanation(formatted_data,
                                                            label=label)

            explanations[method] = cur_expl.squeeze(0)
            scores[method] = score
            # Compute the fidelity auc of the top-k features
            fidelity_scores_topk[method] = self._compute_faithfulness_auc(formatted_data,
                                                                          explanations[method],
                                                                          label,
                                                                          k,
                                                                          metric="topk")

        # Explanation selection logic: choose the most reliable explanation method
        if len(fidelity_scores_topk) >= 2:
            # Get the two explanation methods with highest faithfulness scores
            top2 = heapq.nlargest(2, fidelity_scores_topk, key=fidelity_scores_topk.get)

            # Compare faithfulness scores of top 2 methods
            diff = abs(fidelity_scores_topk[top2[0]] - fidelity_scores_topk[top2[1]])
            
            # If there's a clear winner in faithfulness, use it
            if diff > epsilon:
                best_method = top2[0]
                best_exp = explanations[best_method]
                best_method_score = scores[best_method]
                agree = True  # Methods agreed on which features are important
            else:
                # Faithfulness scores are too close to call - use stability as tiebreaker
                # Stability measures how consistent explanations are across small input changes
                highest_fidelity = self.compute_stability(formatted_data,
                                                          explanations[top2[0]],
                                                          self.explanation_methods[top2[0]],
                                                          label,
                                                          k)

                second_highest_fidelity = self.compute_stability(formatted_data,
                                                                 explanations[top2[1]],
                                                                 self.explanation_methods[top2[1]],
                                                                 label,
                                                                 k)

                agree = False  # Methods disagreed, had to use tiebreaker
                # Choose method with better (lower) stability score
                # Note: lower stability score = more stable explanations
                if highest_fidelity < second_highest_fidelity:
                    best_method = top2[0]
                    best_exp = explanations[best_method]
                    best_method_score = scores[best_method]
                else:
                    best_method = top2[1]
                    best_exp = explanations[best_method]
                    best_method_score = scores[best_method]
        else:
            # Fallback: if only one explanation method available, use standard LIME
            best_method = "lime_0.75"
            best_exp = explanations[best_method]
            best_method_score = scores[best_method]
            agree = True

        # Format return
        # TODO(satya,dylan): figure out a way to get a score metric using fidelity
        final_explanation = self._format_explanation(best_exp.numpy(),
                                                     label,
                                                     best_method_score,
                                                     best_method,
                                                     agree)
        if return_fidelities:
            return final_explanation, fidelity_scores_topk
        else:
            return final_explanation

    def compute_stability(self, data, baseline_explanation, explainer, label, top_k_inds):
        """Computes the AUC stability scores.
        
        Stability measures how consistent an explanation method is when the input is
        slightly perturbed. Stable methods give similar feature rankings even when
        the input data is modified slightly. This is used as a tiebreaker when
        multiple explanation methods have similar faithfulness scores.

        Arguments:
            data: The *single* data point to compute stability for.
            baseline_explanation: The baseline explanation for data.
            explainer: The explanation class to test for stability
            label: The label to explain
            top_k_inds: The indices of the top k features to use for the perturbation process.
        Returns:
            stability: The AUC stability for the top k indices (lower = more stable).
        """
        stability = 0
        for k_i in top_k_inds:
            stability += self.compute_stability_topk(data,
                                                     baseline_explanation,
                                                     explainer,
                                                     label,
                                                     k_i)
        return stability

    def compute_stability_topk(self, data, baseline_explanation, explainer, label, top_k, num_perturbations=100):
        """Computes the stability score for top-k features.
        
        This tests how consistent an explanation method is by:
        1. Getting the top-k most important features from the baseline explanation
        2. Slightly perturbing the input data many times
        3. Re-running the explanation method on each perturbed input
        4. Measuring how much the top-k feature rankings change (using Jaccard similarity)
        
        High stability = top features stay the same even with input perturbations
        Low stability = top features change frequently with small input changes

        Arguments:
            data: Original input data
            baseline_explanation: Feature importance scores from original explanation
            explainer: The explanation method to test for stability
            label: Class label being explained
            top_k: Number of top features to compare for stability
            num_perturbations: Number of perturbed samples to test with
        Returns:
            stability_top_k: Mean Jaccard similarity (higher = more stable, but we return 1-similarity so lower is better)
        """
        # Generate perturbed versions of the input (all features can be changed)
        perturb_args = {
            "original_sample": data[0],
            "feature_mask": torch.tensor([False] * len(baseline_explanation), dtype=torch.bool),  # Allow all features to be perturbed
            "num_samples": num_perturbations,
            "max_distance": self.perturbation_max_distance,
            "feature_metadata": self.feature_types
        }

        # Get many slightly modified versions of the original input
        x_perturbed = self.perturbation_method.get_perturbed_inputs(**perturb_args)

        # Get the top-k most important features from the baseline explanation
        topk_base = torch.argsort(torch.abs(baseline_explanation), descending=True)[:top_k]
        np_topk_base = topk_base.numpy()
        
        stability_value = 0
        # Test explanation consistency across all perturbed inputs
        for perturbed_sample in x_perturbed:
            # Generate explanation for the perturbed input
            explanation_perturbed_input, _ = explainer.get_explanation(perturbed_sample[None, :].numpy(),
                                                                       label=label)

            # Find top-k features in the perturbed explanation
            abs_expl = torch.abs(explanation_perturbed_input)
            topk_perturbed = torch.argsort(abs_expl, descending=True)[:top_k]
            np_topk_perturbed = topk_perturbed.numpy()

            # Compute Jaccard similarity: |intersection| / |union|
            # Higher similarity = more stable (same features ranked highly)
            jaccard_similarity = len(np.intersect1d(np_topk_base, np_topk_perturbed)) / len(
                np.union1d(np_topk_base, np_topk_perturbed))
            stability_value += jaccard_similarity

        # Return average stability across all perturbations
        # Note: we could return 1 - mean_stability to make "lower = more stable"
        # but current code expects higher stability values to be better for tiebreaking
        mean_stability = stability_value / num_perturbations
        return mean_stability

    def _format_explanation(self, explanation: list, label: int, score: float, best_method: str, agree: bool):
        """Formats the explanation in LIME format to be returned.
        
        Converts raw feature importance scores into a structured explanation object
        with feature names paired with their importance scores, sorted by importance.
        """
        list_exp = []

        # Combine feature names with their importance scores into tuples
        # Format: [(feature_name, importance_score), ...]
        for feature_name, feature_imp in zip(self.feature_names, explanation):
            list_exp.append((feature_name, feature_imp))

        # Sort by absolute importance (most important features first)
        # This ensures the explanation shows the most influential features at the top
        list_exp.sort(key=lambda x: abs(x[1]), reverse=True)

        # Package everything into the standard MegaExplanation format
        return_exp = MegaExplanation(list_exp=list_exp,
                                     label=label,
                                     score=score,
                                     best_explanation_type=best_method,
                                     agree=agree)

        return return_exp
