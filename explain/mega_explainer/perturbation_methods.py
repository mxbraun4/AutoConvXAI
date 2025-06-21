"""TODO(satya): docstring."""
import torch


class BasePerturbation:
    """Base Class for perturbation methods."""

    def __init__(self, data_format):
        """Initialize generic parameters for the perturbation method."""
        assert data_format == "tabular", "Currently, only tabular data is supported!"
        self.data_format = data_format

    def get_perturbed_inputs(self,
                             original_sample: torch.FloatTensor,
                             feature_mask: torch.BoolTensor,
                             num_samples: int,
                             feature_metadata: list,
                             max_distance: int = None) -> torch.tensor:
        """Logic of the perturbation methods which will return perturbed samples.

        This method should be overwritten.
        """


class NormalPerturbation(BasePerturbation):
    """TODO(satya): docstring.

    TODO(satya): Should we scale the std. based on the size of the feature? This could lead to
    some odd results if the features aren't scaled the same and we apply the same std noise
    across all the features.
    """

    def __init__(self,
                 data_format,
                 mean: float = 0.0,
                 std: float = 0.05,
                 flip_percentage: float = 0.3):
        """Init.

        Args:
            data_format: A string describing the format of the data, i.e., "tabular" for tabular
                         data.
            mean: the mean of the gaussian perturbations
            std: the standard deviation of the gaussian perturbations
            flip_percentage: The percent of features to flip while perturbing
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
        # Ensure tensor dtype/shape consistency -------------------------------------------
        # Convert the incoming numpy array (or list) to a 1-D torch tensor
        if not isinstance(original_sample, torch.Tensor):
            original_sample = torch.tensor(original_sample, dtype=torch.float32)

        # Flatten in case the sample is 2-D (e.g., shape (1, n_features))
        original_sample = original_sample.flatten()

        # Make sure the boolean mask is a torch tensor as well
        if not isinstance(feature_mask, torch.Tensor):
            feature_mask = torch.tensor(feature_mask, dtype=torch.bool)

        # Align lengths (defensive programming – avoid truncating original sample)
        if len(feature_mask) < len(original_sample):
            # Pad feature_mask with False (i.e., allow perturbation) for missing dims
            pad_size = len(original_sample) - len(feature_mask)
            pad = torch.zeros(pad_size, dtype=torch.bool)
            feature_mask = torch.cat([feature_mask, pad])
        elif len(feature_mask) > len(original_sample):
            # Trim mask if somehow longer
            feature_mask = feature_mask[: len(original_sample)]

        # Ensure feature_type length matches
        feature_type = feature_metadata[: len(original_sample)]

        continuous_features = torch.tensor([i == 'c' for i in feature_type])
        discrete_features = torch.tensor([i == 'd' for i in feature_type])

        # -------------------------------------------------------------------------------
        # Generate perturbations --------------------------------------------------------
        mean = self.mean
        std_dev = self.std_dev

        # Continuous columns – add Gaussian noise
        perturbations = (torch.normal(mean, std_dev, [num_samples, len(feature_type)])
                         * continuous_features + original_sample)

        # Discrete columns – randomly flip a percentage of values
        flip_percentage = self.flip_percentage
        p = torch.empty(num_samples, len(feature_type)).fill_(flip_percentage)
        perturbations = (perturbations * (~discrete_features)
                         + torch.abs((perturbations * discrete_features)
                                     - (torch.bernoulli(p) * discrete_features)))

        # Keep the top-K (masked == True) features static
        perturbed_samples = original_sample * feature_mask + perturbations * (~feature_mask)

        return perturbed_samples
