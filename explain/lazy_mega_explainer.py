"""Lazy loading wrapper for MegaExplainer to avoid slow startup times."""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LazyMegaExplainer:
    """Wrapper for MegaExplainer that loads explanations on-demand.
    
    This avoids computing all explanations at startup, which can be very slow.
    Explanations are computed and cached only when actually requested.
    """
    
    def __init__(self, prediction_fn, data, cat_features, class_names):
        """Initialize the lazy explainer.
        
        Args:
            prediction_fn: Model prediction function
            data: Background dataset for explanations
            cat_features: List of categorical feature names
            class_names: List of class names
        """
        # Store initialization parameters
        self.prediction_fn = prediction_fn
        self.background_data = data
        self.cat_features = cat_features
        self.class_names = class_names
        
        # Cache for computed explanations
        self._explanation_cache = {}
        
        # The actual MegaExplainer (created on first use)
        self._mega_explainer = None
        
        logger.info("LazyMegaExplainer initialized (no explanations computed yet)")
    
    def _ensure_explainer(self):
        """Create the actual MegaExplainer if not already created."""
        if self._mega_explainer is None:
            logger.info("Creating MegaExplainer instance...")
            from explain.explanation import MegaExplainer
            
            self._mega_explainer = MegaExplainer(
                prediction_fn=self.prediction_fn,
                data=self.background_data,
                cat_features=self.cat_features,
                class_names=self.class_names
            )
            logger.info("MegaExplainer instance created")
    
    def get_explanations(self, ids: List[int], data: pd.DataFrame, 
                        ids_to_regenerate: Optional[List[int]] = None) -> Dict[int, Any]:
        """Get explanations for the specified IDs.
        
        This method computes explanations on-demand and caches them.
        
        Args:
            ids: List of instance IDs to explain
            data: DataFrame with the instances to explain
            ids_to_regenerate: Optional list of IDs to force regeneration
            
        Returns:
            Dictionary mapping ID to explanation
        """
        self._ensure_explainer()
        
        # Determine which IDs need computation
        ids_to_compute = []
        cached_explanations = {}
        
        for id_val in ids:
            # Check if we need to regenerate
            if ids_to_regenerate and id_val in ids_to_regenerate:
                ids_to_compute.append(id_val)
                # Remove from cache if it exists
                if id_val in self._explanation_cache:
                    del self._explanation_cache[id_val]
            # Check if already cached
            elif id_val in self._explanation_cache:
                cached_explanations[id_val] = self._explanation_cache[id_val]
            else:
                ids_to_compute.append(id_val)
        
        # Compute explanations for IDs that aren't cached
        if ids_to_compute:
            logger.info(f"Computing explanations for {len(ids_to_compute)} instances...")
            
            # Get only the data for IDs that need computation
            data_to_explain = data.loc[data.index.isin(ids_to_compute)]
            
            # Compute explanations
            new_explanations = self._mega_explainer.get_explanations(
                ids=ids_to_compute, 
                data=data_to_explain
            )
            
            # Cache the new explanations
            self._explanation_cache.update(new_explanations)
            
            # Merge with cached explanations
            cached_explanations.update(new_explanations)
            
            logger.info(f"Computed and cached {len(new_explanations)} explanations")
        else:
            logger.info(f"All {len(ids)} explanations retrieved from cache")
        
        return cached_explanations
    
    def summarize_explanations(self, data: pd.DataFrame, instance_ids: List[int],
                              input_string: Optional[str] = None) -> tuple:
        """Summarize explanations for a set of instances.
        
        Args:
            data: DataFrame with the instances
            instance_ids: List of instance IDs to summarize
            input_string: Optional input string for context
            
        Returns:
            Tuple of (full_summary, short_summary)
        """
        self._ensure_explainer()
        
        # Ensure we have explanations for all requested IDs
        explanations = self.get_explanations(instance_ids, data)
        
        # Use the underlying explainer's summarization
        return self._mega_explainer.summarize_explanations(
            data, instance_ids, input_string
        )
    
    def clear_cache(self):
        """Clear the explanation cache."""
        self._explanation_cache.clear()
        logger.info("Explanation cache cleared")
    
    def get_cache_size(self) -> int:
        """Get the number of cached explanations."""
        return len(self._explanation_cache)
    
    @property
    def contents(self):
        """For backward compatibility - return self."""
        return self 