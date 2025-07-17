"""Function to define feature meanings."""

def define_operation(conversation, parse_text, i, **kwargs):
    """Generates text to define feature.
    
    Handles both old parse_text format and new AutoGen entities format.
    """
    # Try to get feature from AutoGen entities first
    ent_features = kwargs.get('features', []) if kwargs else []
    
    if ent_features and len(ent_features) > 0:
        # Use AutoGen entities (modern approach)
        feature_name = ent_features[0]
    elif i+1 < len(parse_text):
        # Fallback to old parse_text approach
        feature_name = parse_text[i+1]
    else:
        return {
            'type': 'definition_error',
            'message': 'No feature name specified for definition. Try asking "What does BMI mean?" or "Define glucose levels".',
            'available_features': list(conversation.feature_definitions.keys())
        }, 0
    
    # Handle case variations
    feature_name_original = feature_name
    feature_name = feature_name.lower()
    
    # Handle common synonyms and aliases
    synonyms = {
        'target variable': 'y',
        'target': 'y',
        'outcome': 'y',
        'label': 'y',
        'class': 'y',
        'prediction target': 'y',
        'dependent variable': 'y',
        'response variable': 'y'
    }
    
    # Check synonyms first
    if feature_name in synonyms:
        matching_feature = synonyms[feature_name]
    else:
        # Find matching feature (case-insensitive)
        matching_feature = None
        for feat_name in conversation.feature_definitions.keys():
            if feat_name.lower() == feature_name:
                matching_feature = feat_name
                break
        
        if matching_feature is None:
            # Try partial matching
            for feat_name in conversation.feature_definitions.keys():
                if feature_name in feat_name.lower() or feat_name.lower() in feature_name:
                    matching_feature = feat_name
                    break
    
    if matching_feature is None:
        return {
            'type': 'definition_not_found',
            'message': f'Definition for feature "{feature_name_original}" not found.',
            'available_features': list(conversation.feature_definitions.keys()),
            'suggestion': 'Try asking about: ' + ', '.join(list(conversation.feature_definitions.keys())[:3])
        }, 0
    
    feature_definition = conversation.get_feature_definition(matching_feature)
    
    return {
        'type': 'feature_definition',
        'feature_name': matching_feature,
        'definition': feature_definition,
        'medical_context': 'diabetes risk assessment'
    }, 1
