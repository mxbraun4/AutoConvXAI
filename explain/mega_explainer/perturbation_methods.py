"""Data perturbation methods for testing explanation faithfulness and stability.

This module provides methods to generate slightly modified versions of input data
for testing how robust and faithful explanations are.
"""
import torch


class BasePerturbation:
    """Base class for data perturbation methods used in explanation testing."""

    def __init__(self, data_format):
        """Initialize perturbation method.
        
        Args:
            data_format: Type of data to perturb ("tabular" only currently supported)
        """
        assert data_format == "tabular", "Currently, only tabular data is supported!"
        self.data_format = data_format

    def get_perturbed_inputs(self,
                             original_sample: torch.FloatTensor,
                             feature_mask: torch.BoolTensor,
                             num_samples: int,
                             feature_metadata: list,
                             max_distance: int = None) -> torch.tensor:
        """Generate perturbed versions of input sample for explanation testing.

        This method should be overridden by subclasses to implement specific
        perturbation strategies (Gaussian noise, random flips, etc.).
        
        Args:
            original_sample: The input instance to perturb
            feature_mask: Boolean mask indicating which features to keep unchanged
            num_samples: Number of perturbed samples to generate
            feature_metadata: List of 'c'/'d' indicating continuous/discrete features
            max_distance: Maximum allowed distance from original sample
            
        Returns:
            torch.tensor: Generated perturbed samples for testing
        """
        raise NotImplementedError("Subclasses must implement get_perturbed_inputs")


class NormalPerturbation(BasePerturbation):
    """Gaussian perturbation method for continuous features with random flips for discrete features.
    
    This method perturbs data by:
    - Adding Gaussian noise to continuous features 
    - Randomly flipping discrete/categorical features based on flip percentage
    - Respecting feature masks to keep certain features unchanged
    
    Note: Uses same std deviation across all features - may need feature-specific scaling
    for datasets where features have very different scales.
    """

    def __init__(self,
                 data_format,
                 mean: float = 0.0,
                 std: float = 0.05,
                 flip_percentage: float = 0.3):
        """Initialize Gaussian perturbation method.

        Args:
            data_format: Type of data ("tabular" only currently supported)
            mean: Mean of Gaussian noise added to continuous features (usually 0.0)
            std: Standard deviation of Gaussian noise (controls perturbation strength)
            flip_percentage: Probability of flipping each discrete feature (0.0 to 1.0)
        """
        self.mean = mean
        self.std_dev = std
        self.flip_percentage = flip_percentage
        super(NormalPerturbation, self).__init__(data_format)

    def get_perturbed_inputs(self,
                             original_sample: torch.FloatTensor,
                             feature_mask: torch.BoolTensor,
                             num_samples: int,
                             feature_metadata: list,
                             max_distance: int = None) -> torch.tensor:
        """Given a sample and mask, compute perturbations.

        Args:
            original_sample: The original instance
            feature_mask: the indices of the indices to mask where True corresponds to an index
                          that is to be masked. E.g., [False, True, False] means that index 1 will
                          not be perturbed while 0 and 2 _will_ be perturbed.
            num_samples: number of perturbed samples.
            feature_metadata: the list of 'c' or 'd' for whether the feature is categorical or
                              discrete.
            max_distance: the maximum distance between original sample and perturbed samples.
        Returns:
            perturbed_samples: The original_original sample perturbed with Gaussian perturbations
                               num_samples times.
        """
        # Input validation and format conversion
        if not isinstance(original_sample, torch.Tensor):
            original_sample = torch.tensor(original_sample, dtype=torch.float32)
        original_sample = original_sample.flatten()  # Ensure 1D

        if not isinstance(feature_mask, torch.Tensor):
            feature_mask = torch.tensor(feature_mask, dtype=torch.bool)

        # Ensure all arrays have matching lengths
        if len(feature_mask) < len(original_sample):
            # Pad mask with False (allow perturbation) for missing dimensions
            pad_size = len(original_sample) - len(feature_mask)
            pad = torch.zeros(pad_size, dtype=torch.bool)
            feature_mask = torch.cat([feature_mask, pad])
        elif len(feature_mask) > len(original_sample):
            feature_mask = feature_mask[: len(original_sample)]

        # Align feature metadata length with sample
        feature_type = feature_metadata[: len(original_sample)]

        # Create boolean masks for feature types
        continuous_features = torch.tensor([i == 'c' for i in feature_type])
        discrete_features = torch.tensor([i == 'd' for i in feature_type])

        # Generate perturbations based on feature types
        # Step 1: Add Gaussian noise to continuous features
        noise = torch.normal(self.mean, self.std_dev, [num_samples, len(feature_type)])
        perturbations = noise * continuous_features + original_sample

        # Step 2: Handle discrete features with random flips
        # Generate random flip decisions for each discrete feature
        flip_probs = torch.empty(num_samples, len(feature_type)).fill_(self.flip_percentage)
        random_flips = torch.bernoulli(flip_probs)
        
        # Apply flips only to discrete features: flip by subtracting from absolute value
        perturbations = (perturbations * (~discrete_features) +  # Keep continuous perturbations
                        torch.abs((perturbations * discrete_features) -  # For discrete features
                                 (random_flips * discrete_features)))     # Apply random flips

        # Step 3: Respect feature mask - keep masked features unchanged
        # feature_mask: True = keep original, False = allow perturbation
        perturbed_samples = (original_sample * feature_mask +        # Keep masked features
                           perturbations * (~feature_mask))         # Perturb unmasked features

        return perturbed_samples
