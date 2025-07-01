"""Function to show data instances.

For single instances, this function prints out the feature values. For many instances,
it returns the mean.
"""
from explain.utils import gen_parse_op_text


def show_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows an instance."""
    data = conversation.temp_dataset.contents['X']

    parse_op = gen_parse_op_text(conversation)
    if len(parse_op) > 0:
        intro_text = f"For the data with <b>{parse_op}</b>,"
    else:
        intro_text = "For all the instances in the data,"
    
    if len(data) == 0:
        return {'type': 'error', 'message': 'There are no instances in the data that meet this description.'}, 0
    
    if len(data) == 1:
        # Single instance - show all feature values
        features = {}
        truncated_features = {}
        
        for i, feature_name in enumerate(data.columns):
            feature_value = data[feature_name].values[0]
            if i < n_features_to_show:
                features[feature_name] = feature_value
            else:
                truncated_features[feature_name] = feature_value
        
        # Store truncated info for follow-up
        if truncated_features:
            truncated_text = "\n".join([f"{name}: {val}" for name, val in truncated_features.items()])
            conversation.store_followup_desc(truncated_text)
        
        result = {
            'type': 'single_instance',
            'instance_id': data.index[0],
            'features': features,
            'has_more_features': len(truncated_features) > 0,
            'total_features': len(data.columns),
            'shown_features': len(features),
            'filter_applied': parse_op
        }
        return result, 1
    else:
        # Multiple instances - show IDs
        instance_ids = list(data.index)
        
        result = {
            'type': 'multiple_instances',
            'instance_ids': instance_ids,
            'total_count': len(data),
            'filter_applied': parse_op,
            'needs_selection': True
        }
        return result, 1
