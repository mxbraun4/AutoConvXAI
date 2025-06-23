"""
Generalized Configuration System for TalkToModel

This module provides a flexible configuration system that maintains generalizability
while allowing domain-specific customizations.
"""

from typing import Dict, List, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class GeneralizableDomainConfig:
    """Configuration class that adapts to any domain while maintaining generalizability."""
    
    def __init__(self, domain_name: str = "general"):
        self.domain_name = domain_name
        self.feature_synonyms = {}
        self.class_names = {}
        self.custom_patterns = {}
        self.entity_hints = {}
        
    def add_feature_synonyms(self, synonyms: Dict[str, List[str]]):
        """Add domain-specific feature synonyms.
        
        Example:
            config.add_feature_synonyms({
                'age': ['years old', 'older', 'younger'],
                'income': ['salary', 'earnings', 'wage'],
                'education': ['schooling', 'degree', 'qualification']
            })
        """
        self.feature_synonyms.update(synonyms)
        logger.info(f"Added synonyms for {len(synonyms)} features in {self.domain_name} domain")
    
    def add_class_names(self, class_mapping: Dict[int, str]):
        """Add human-readable class names.
        
        Example:
            config.add_class_names({
                0: "low risk",
                1: "high risk"
            })
        """
        self.class_names.update(class_mapping)
        logger.info(f"Added class names for {len(class_mapping)} classes in {self.domain_name} domain")
    
    def add_entity_hints(self, hints: Dict[str, Any]):
        """Add domain-specific entity recognition hints.
        
        Example:
            config.add_entity_hints({
                'id_synonyms': ['customer id', 'account number', 'case id'],
                'numeric_features': ['amount', 'score', 'rating'],
                'categorical_features': ['status', 'type', 'category']
            })
        """
        self.entity_hints.update(hints)
        logger.info(f"Added entity hints for {self.domain_name} domain")
    
    def add_custom_patterns(self, patterns: Dict[str, str]):
        """Add custom regex patterns for domain-specific terminology.
        
        Example:
            config.add_custom_patterns({
                'risk_indicators': r'(?:risk|danger|hazard|threat)',
                'financial_terms': r'(?:profit|loss|revenue|cost)'
            })
        """
        self.custom_patterns.update(patterns)
        logger.info(f"Added {len(patterns)} custom patterns for {self.domain_name} domain")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            'domain_name': self.domain_name,
            'feature_synonyms': self.feature_synonyms,
            'class_names': self.class_names,
            'custom_patterns': self.custom_patterns,
            'entity_hints': self.entity_hints
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved {self.domain_name} domain config to {filepath}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'GeneralizableDomainConfig':
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            config = cls(data.get('domain_name', 'general'))
            config.feature_synonyms = data.get('feature_synonyms', {})
            config.class_names = data.get('class_names', {})
            config.custom_patterns = data.get('custom_patterns', {})
            config.entity_hints = data.get('entity_hints', {})
            
            logger.info(f"Loaded {config.domain_name} domain config from {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {filepath}: {e}")
            return cls()  # Return default config


# Pre-built domain configurations for common use cases
def create_medical_config() -> GeneralizableDomainConfig:
    """Create configuration optimized for medical/healthcare domains."""
    config = GeneralizableDomainConfig("medical")
    
    config.add_feature_synonyms({
        'age': ['years old', 'older', 'younger', 'elderly'],
        'glucose': ['blood sugar', 'sugar', 'glucose level'],
        'blood_pressure': ['bp', 'pressure', 'hypertension'],
        'bmi': ['body mass index', 'weight', 'obesity'],
        'pregnancies': ['pregnant', 'gravid', 'parity'],
        'insulin': ['insulin level', 'insulin resistance'],
        'diagnosis': ['condition', 'disease', 'illness']
    })
    
    config.add_class_names({
        0: "healthy",
        1: "at risk"
    })
    
    config.add_entity_hints({
        'id_synonyms': ['patient id', 'case number', 'medical record'],
        'measurement_units': ['mg/dl', 'mmhg', 'kg/m2', 'years'],
        'risk_terms': ['risk', 'likelihood', 'probability', 'chance']
    })
    
    return config


def create_financial_config() -> GeneralizableDomainConfig:
    """Create configuration optimized for financial domains."""
    config = GeneralizableDomainConfig("financial")
    
    config.add_feature_synonyms({
        'income': ['salary', 'earnings', 'wage', 'revenue'],
        'credit_score': ['credit rating', 'score', 'creditworthiness'],
        'debt': ['liability', 'obligation', 'loan'],
        'assets': ['wealth', 'holdings', 'property'],
        'age': ['years old', 'older', 'younger'],
        'employment': ['job', 'work', 'occupation']
    })
    
    config.add_class_names({
        0: "low risk",
        1: "high risk"
    })
    
    config.add_entity_hints({
        'id_synonyms': ['customer id', 'account number', 'client id'],
        'currency_terms': ['dollar', 'usd', 'amount', 'value'],
        'risk_terms': ['default', 'risk', 'probability', 'likelihood']
    })
    
    return config


def create_general_config() -> GeneralizableDomainConfig:
    """Create a general-purpose configuration that works across domains."""
    config = GeneralizableDomainConfig("general")
    
    # Universal synonyms that work across domains
    config.add_feature_synonyms({
        'id': ['identifier', 'number', 'index'],
        'age': ['years old', 'older', 'younger'],
        'score': ['rating', 'value', 'level'],
        'status': ['state', 'condition', 'category'],
        'type': ['kind', 'category', 'class']
    })
    
    config.add_class_names({
        0: "negative",
        1: "positive"
    })
    
    config.add_entity_hints({
        'id_synonyms': ['id', 'identifier', 'number', 'index'],
        'comparison_terms': ['versus', 'vs', 'compared to', 'against'],
        'importance_terms': ['important', 'significant', 'influential', 'key']
    })
    
    return config


# Utility function to apply configuration to dispatcher
def apply_config_to_dispatcher(dispatcher, config: GeneralizableDomainConfig):
    """Apply domain configuration to smart action dispatcher."""
    try:
        # Add feature synonyms and class names
        dispatcher.add_domain_knowledge(
            feature_synonyms=config.feature_synonyms,
            class_names=config.class_names
        )
        
        # Apply custom patterns if entity extractor supports it
        if hasattr(dispatcher.entity_extractor, 'add_custom_patterns'):
            dispatcher.entity_extractor.add_custom_patterns(config.custom_patterns)
        
        # Apply entity hints
        if hasattr(dispatcher.entity_extractor, 'add_entity_hints'):
            dispatcher.entity_extractor.add_entity_hints(config.entity_hints)
        
        logger.info(f"Applied {config.domain_name} domain configuration to dispatcher")
        return True
        
    except Exception as e:
        logger.error(f"Error applying configuration: {e}")
        return False


# Example usage for different domains
DOMAIN_CONFIGS = {
    'medical': create_medical_config,
    'financial': create_financial_config,
    'general': create_general_config
}

def get_domain_config(domain_name: str) -> GeneralizableDomainConfig:
    """Get pre-built configuration for a specific domain."""
    if domain_name in DOMAIN_CONFIGS:
        return DOMAIN_CONFIGS[domain_name]()
    else:
        logger.warning(f"Unknown domain '{domain_name}', using general config")
        return create_general_config() 