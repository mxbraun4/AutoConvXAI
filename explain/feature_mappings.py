"""Feature name mappings for generalizable natural language processing.

This module provides mappings between natural language terms and actual feature names
in the dataset, making the system more flexible and generalizable.
"""

class FeatureMapper:
    """Maps natural language terms to dataset feature names."""
    
    def __init__(self, dataset_features: list = None):
        """Initialize with optional dataset features for validation."""
        self.dataset_features = dataset_features or []
        
        # Default mappings for common terms
        self.feature_mappings = {
            # Pregnancy-related
            'pregnant': 'pregnancies',
            'pregnancy': 'pregnancies',
            'pregnancies': 'pregnancies',
            
            # Age-related
            'age': 'age',
            'years': 'age',
            'years old': 'age',
            
            # BMI-related
            'bmi': 'bmi',
            'body mass index': 'bmi',
            'weight': 'bmi',
            
            # Glucose-related
            'glucose': 'glucose',
            'sugar': 'glucose',
            'blood sugar': 'glucose',
            
            # Blood pressure
            'bp': 'bloodpressure',
            'blood pressure': 'bloodpressure',
            'pressure': 'bloodpressure',
            
            # Diabetes-related
            'diabetes': 'outcome',
            'diabetic': 'outcome',
            'diagnosis': 'outcome',
        }
        
        # Value mappings for boolean/categorical values
        self.value_mappings = {
            # Boolean mappings
            'yes': '1',
            'no': '0',
            'true': '1',
            'false': '0',
            
            # Pregnancy specific
            'pregnant': 'greater 0',
            'not pregnant': 'equal 0',
            'never pregnant': 'equal 0',
            
            # Diabetes specific
            'diabetic': '1',
            'not diabetic': '0',
            'healthy': '0',
        }
        
    def map_feature_name(self, term: str) -> str:
        """Map a natural language term to a feature name."""
        term_lower = term.lower().strip()
        
        # Direct mapping
        if term_lower in self.feature_mappings:
            mapped = self.feature_mappings[term_lower]
            # Validate if we have dataset features
            if self.dataset_features and mapped not in self.dataset_features:
                # Try to find a close match
                for feature in self.dataset_features:
                    if feature.lower() == mapped.lower():
                        return feature
            return mapped
        
        # Check if it's already a valid feature
        if self.dataset_features:
            # Case-insensitive match
            for feature in self.dataset_features:
                if feature.lower() == term_lower:
                    return feature
        
        # Return as-is if no mapping found
        return term
    
    def map_value(self, value: str, context: str = None) -> str:
        """Map a natural language value to a dataset value."""
        value_lower = value.lower().strip()
        
        # Check context-specific mappings
        if context and context.lower() in ['pregnancies', 'pregnant']:
            if value_lower in ['no', 'false', 'none', '0']:
                return '0'
            elif value_lower in ['yes', 'true', 'some']:
                return '1'  # Or could be 'greater 0' depending on use case
        
        # General mappings
        if value_lower in self.value_mappings:
            return self.value_mappings[value_lower]
        
        # Return as-is if it's already a number
        try:
            float(value)
            return value
        except ValueError:
            return value
    
    def process_filter_command(self, command: str) -> str:
        """Process a filter command to map features and values."""
        parts = command.split()
        
        if len(parts) < 3:
            return command
        
        # Expected format: "filter {feature} {operator} {value}"
        if parts[0] == 'filter' and len(parts) >= 4:
            feature = self.map_feature_name(parts[1])
            operator = parts[2]
            value = ' '.join(parts[3:])  # Handle multi-word values
            
            # Map the value based on context
            mapped_value = self.map_value(value, context=feature)
            
            return f"filter {feature} {operator} {mapped_value}"
        
        return command
    
    def add_mapping(self, natural_term: str, feature_name: str):
        """Add a custom mapping."""
        self.feature_mappings[natural_term.lower()] = feature_name
    
    def add_value_mapping(self, natural_value: str, dataset_value: str):
        """Add a custom value mapping."""
        self.value_mappings[natural_value.lower()] = dataset_value


# Global instance
_feature_mapper = None

def get_feature_mapper(dataset_features: list = None) -> FeatureMapper:
    """Get or create the feature mapper instance."""
    global _feature_mapper
    if _feature_mapper is None or dataset_features:
        _feature_mapper = FeatureMapper(dataset_features)
    return _feature_mapper 