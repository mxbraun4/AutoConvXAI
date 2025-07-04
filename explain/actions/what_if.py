"""The what if operation.

This operation updates the data according to some what commands.
"""


def convert_categorical_bools(data):
    """Convert categorical boolean strings to numeric values."""
    if data == 'true':
        return 1
    elif data == 'false':
        return 0
    else:
        return data


def is_numeric(feature_name, temp_dataset):
    return feature_name in temp_dataset['numeric']


def is_categorical(feature_name, temp_dataset):
    return feature_name in temp_dataset['cat']


def get_numeric_updates(parse_text, i):
    """Gets the numeric update information."""
    if i+2 >= len(parse_text):
        raise ValueError(f"Missing update operation in what-if query")
    if i+3 >= len(parse_text):
        raise ValueError(f"Missing update value in what-if query")
    
    update_term = parse_text[i+2]
    try:
        update_value = float(parse_text[i+3])
    except ValueError:
        raise ValueError(f"Invalid numeric value '{parse_text[i+3]}' in what-if query")
    
    return update_term, update_value


def update_numeric_feature(temp_data, feature_name, update_term, update_value):
    """Performs the numerical update."""
    new_dataset = temp_data["X"]

    if update_term == "increase":
        new_dataset[feature_name] += update_value
        parse_op = f"{feature_name} is increased by {str(update_value)}"
    elif update_term == "decrease":
        new_dataset[feature_name] -= update_value
        parse_op = f"{feature_name} is decreased by {str(update_value)}"
    elif update_term == "set":
        new_dataset[feature_name] = update_value
        parse_op = f"{feature_name} is set to {str(update_value)}"
    else:
        raise NameError(f"Unknown update operation {update_term}")

    return new_dataset, parse_op


def what_if_operation(conversation, parse_text, i, **kwargs):
    """The what if operation - now uses entity-based approach."""

    # Extract entities from kwargs (passed from AutoGen)
    features = kwargs.get('features', [])
    operators = kwargs.get('operators', [])
    values = kwargs.get('values', [])
    
    if not features:
        return {'type': 'error', 'message': 'No feature specified for what-if analysis!'}, 0
    
    if not values:
        return {'type': 'error', 'message': 'No value specified for what-if analysis!'}, 0
    
    # Get the feature name (handle case insensitive matching)
    feature_name = features[0]
    temp_dataset = conversation.temp_dataset.contents
    
    # Find actual feature name (case insensitive)
    actual_feature = None
    for col in temp_dataset['X'].columns:
        if col.lower() == feature_name.lower():
            actual_feature = col
            break
    
    if actual_feature is None:
        return {'type': 'error', 'message': f'Unknown feature: {feature_name}'}, 0
    
    feature_name = actual_feature
    update_value = values[0]
    
    # Determine the operation type from operators or infer from context
    if operators and operators[0] == '+':
        update_term = "increase"
    elif operators and operators[0] == '-':
        update_term = "decrease"
    else:
        # Default to "set" if no specific operator
        update_term = "set"

    # Apply the what-if change
    if is_numeric(feature_name, temp_dataset):
        try:
            temp_dataset['X'], parse_op = update_numeric_feature(temp_dataset,
                                                                 feature_name,
                                                                 update_term,
                                                                 update_value)
        except Exception as e:
            return {'type': 'error', 'message': f'Error updating {feature_name}: {str(e)}'}, 0
            
    elif is_categorical(feature_name, temp_dataset):
        categorical_val = convert_categorical_bools(update_value)
        temp_dataset['X'][feature_name] = categorical_val
        parse_op = f"{feature_name} is set to {str(categorical_val)}"
        
    elif feature_name == "id":
        return {'type': 'error', 'message': 'What-if updates have no effect on IDs!'}, 0
    else:
        return {'type': 'error', 'message': f'Cannot modify feature {feature_name}'}, 0

    # Get original predictions BEFORE the change for comparison
    model = conversation.get_var('model').contents
    original_data = conversation.temp_dataset.contents['X'].copy()
    
    # Make original predictions (before change)
    try:
        from main import _safe_model_predict
        original_predictions = _safe_model_predict(model, original_data)
        
        # Get prediction probabilities for confidence
        try:
            original_probabilities = model.predict_proba(original_data.values)
        except:
            original_probabilities = None
            
    except Exception as e:
        original_predictions = None
        original_probabilities = None

    # Apply the change to get new predictions
    changed_data = temp_dataset['X']
    
    try:
        new_predictions = _safe_model_predict(model, changed_data)
        
        # Get new prediction probabilities
        try:
            new_probabilities = model.predict_proba(changed_data.values)
        except:
            new_probabilities = None
            
    except Exception as e:
        new_predictions = None
        new_probabilities = None

    # Track regeneration and parse operations
    processed_ids = list(conversation.temp_dataset.contents['X'].index)
    conversation.temp_dataset.contents['ids_to_regenerate'].extend(processed_ids)

    conversation.add_interpretable_parse_op("and")
    conversation.add_interpretable_parse_op(parse_op)

    # Build comparison result
    result = {
        'type': 'what_if_comparison',
        'feature_name': feature_name,
        'operation': update_term,
        'value': update_value,
        'description': parse_op,
        'instances_affected': len(processed_ids)
    }
    
    # Add prediction comparison if we have both before and after
    if original_predictions is not None and new_predictions is not None:
        if len(original_predictions) == 1 and len(new_predictions) == 1:
            # Single instance comparison
            original_class = conversation.class_names.get(original_predictions[0], str(original_predictions[0]))
            new_class = conversation.class_names.get(new_predictions[0], str(new_predictions[0]))
            
            # Calculate confidence scores
            original_confidence = None
            new_confidence = None
            if original_probabilities is not None:
                original_confidence = round(original_probabilities[0][original_predictions[0]] * 100, 2)
            if new_probabilities is not None:
                new_confidence = round(new_probabilities[0][new_predictions[0]] * 100, 2)
                
            result.update({
                'original_prediction': int(original_predictions[0]),
                'original_prediction_class': original_class,
                'original_confidence': original_confidence,
                'new_prediction': int(new_predictions[0]),
                'new_prediction_class': new_class,
                'new_confidence': new_confidence,
                'prediction_changed': bool(original_predictions[0] != new_predictions[0])
            })
        else:
            # Multiple instances - show distribution changes
            import numpy as np
            unique_original = np.unique(original_predictions, return_counts=True)
            unique_new = np.unique(new_predictions, return_counts=True)
            
            original_dist = {}
            for pred, count in zip(unique_original[0], unique_original[1]):
                class_name = conversation.class_names.get(pred, str(pred))
                original_dist[class_name] = round(count / len(original_predictions) * 100, 2)
                
            new_dist = {}
            for pred, count in zip(unique_new[0], unique_new[1]):
                class_name = conversation.class_names.get(pred, str(pred))
                new_dist[class_name] = round(count / len(new_predictions) * 100, 2)
            
            result.update({
                'original_distribution': original_dist,
                'new_distribution': new_dist,
                'total_instances': len(original_predictions)
            })

    return result, 1
