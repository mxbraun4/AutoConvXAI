"""Data summary operation."""


# Note, these are hardcode for compas!
def data_operation(conversation, parse_text, i, **kwargs):
    """Data summary operation - now returns structured data."""
    # Get dataset size information - handle both filtered and unfiltered cases
    if hasattr(conversation, 'temp_dataset') and conversation.temp_dataset:
        df = conversation.temp_dataset.contents['X']
        dataset_size = len(df)
        total_size = len(conversation.get_var('dataset').contents['X'])
        filtered = True
    else:
        # No filtering applied - use full dataset
        df = conversation.get_var('dataset').contents['X']
        dataset_size = len(df)
        total_size = dataset_size
        filtered = False
    
    # Check if user is asking for specific feature statistics or class counts
    query_text = " ".join(parse_text).lower()
    
    # Base result structure
    result = {
        'type': 'data_summary',
        'dataset_size': dataset_size,
        'total_size': total_size,
        'filtered': filtered,
        'features': list(df.columns),
        'description': conversation.describe.get_dataset_description() or "diabetes prediction based on patient health metrics",
        'available_ids': list(df.index)
    }
    
    # ENHANCED: Check for class counting requests (e.g., "how many diabetes", "count instances")
    if any(word in query_text for word in ['how many', 'count', 'instances']):
        # Check if user is asking about specific class labels
        y_data = conversation.get_var('dataset').contents.get('y')
        if y_data is not None:
            # Get class names for better readability
            class_names = getattr(conversation, 'class_names', {})
            
            # Use the appropriate y data based on filtering
            if hasattr(conversation, 'temp_dataset') and conversation.temp_dataset:
                target_data = conversation.temp_dataset.contents.get('y', y_data)
            else:
                target_data = y_data
            
            # Count each class
            value_counts = target_data.value_counts()
            
            result['request_type'] = 'class_distribution'
            result['class_distribution'] = []
            
            for class_val, count in value_counts.items():
                class_name = class_names.get(class_val, str(class_val)) if class_names else str(class_val)
                percentage = round((count / len(target_data)) * 100, conversation.rounding_precision)
                result['class_distribution'].append({
                    'class_value': class_val,
                    'class_name': class_name,
                    'count': count,
                    'percentage': percentage
                })
            
            return result, 1
    
    # Check if a specific feature is mentioned in parse_text 
    specific_feature = None
    if len(parse_text) > 1:
        potential_feature = parse_text[1].lower()
        if potential_feature in [col.lower() for col in df.columns]:
            specific_feature = next(col for col in df.columns if col.lower() == potential_feature)
    
    # Handle specific feature requests (either via keywords or direct mention)
    if specific_feature or any(word in query_text for word in ['average', 'mean', 'statistics']):
        if specific_feature:
            target_feature = specific_feature
        else:
            # Look for feature names in the query text
            target_feature = None
            for feature in df.columns:
                if feature.lower() in query_text:
                    target_feature = feature
                    break
        
        if target_feature:
            result['request_type'] = 'feature_statistics'
            result['target_feature'] = target_feature
            result['feature_stats'] = {
                'mean': round(df[target_feature].mean(), conversation.rounding_precision),
                'std': round(df[target_feature].std(), conversation.rounding_precision),
                'min': round(df[target_feature].min(), conversation.rounding_precision),
                'max': round(df[target_feature].max(), conversation.rounding_precision)
            }
            return result, 1
    
    # If statistics requested but no specific feature found, show all
    if any(word in query_text for word in ['statistics', 'stats', 'summary']):
        result['request_type'] = 'all_statistics'
        result['all_feature_stats'] = {}
        
        for feature in df.columns:
            result['all_feature_stats'][feature] = {
                'mean': round(df[feature].mean(), conversation.rounding_precision),
                'std': round(df[feature].std(), conversation.rounding_precision),
                'min': round(df[feature].min(), conversation.rounding_precision),
                'max': round(df[feature].max(), conversation.rounding_precision)
            }
        
        return result, 1
    
    # Default dataset overview
    result['request_type'] = 'overview'
    
    # Add model performance if available
    try:
        model = conversation.get_var('model').contents
        performance = conversation.describe.get_eval_performance(model, conversation.default_metric)
        if performance:
            result['model_performance'] = performance
    except:
        pass
    
    # Add detailed statistics for followup
    detailed_stats = {}
    for f in df.columns:
        detailed_stats[f] = {
            'mean': round(df[f].mean(), conversation.rounding_precision),
            'std': round(df[f].std(), conversation.rounding_precision),
            'min': round(df[f].min(), conversation.rounding_precision),
            'max': round(df[f].max(), conversation.rounding_precision)
        }
    result['detailed_stats_available'] = detailed_stats
    
    # Store followup description for legacy compatibility
    conversation.store_followup_desc("Detailed statistics available upon request")

    return result, 1
