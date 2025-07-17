"""The filtering action.

This action filters some data according to different filtering criteria, e.g., less than or equal
to, greater than, etc. It modifies the temporary dataset in the conversation object, updating that dataset to yield the
correct filtering based on the parse.
"""
import numpy as np
import pandas as pd


def filter_dataset(dataset, bools):
    """Selects x and y of dataset by booleans."""
    # Create a copy to avoid modifying the original dataset
    filtered_dataset = {
        'X': dataset['X'][bools].copy(),
        'y': dataset['y'][bools].copy() if dataset['y'] is not None else None,
        'full_data': dataset['full_data'][bools].copy() if 'full_data' in dataset else None,
        'cat': dataset.get('cat', []).copy(),
        'numeric': dataset.get('numeric', []).copy(), 
        'ids_to_regenerate': dataset.get('ids_to_regenerate', []).copy()
    }
    return filtered_dataset


def format_parse_string(feature_name, feature_value, operation):
    """Formats a string that describes the filtering parse."""
    return f"{feature_name} {operation} {str(feature_value)}"



def prediction_filter(temp_dataset, conversation, feature_name):
    """filters based on the model's prediction"""
    model = conversation.get_var('model').contents
    x_values = temp_dataset['X']

    # compute model predictions
    predictions = model.predict(x_values)

    # feature name is given as string from grammar, bind predictions to str
    # to get correct equivalence
    str_predictions = np.array([str(p) for p in predictions])
    bools = str_predictions == feature_name

    updated_dset = filter_dataset(temp_dataset, bools)
    class_text = conversation.get_class_name_from_label(int(feature_name))
    interpretable_parse_text = f"the model predicts {class_text}"

    return updated_dset, interpretable_parse_text


def label_filter(temp_dataset, conversation, feature_name):
    """Filters based on the labels in the data"""""
    y_values = temp_dataset['y']
    str_y_values = np.array([str(y) for y in y_values])
    bools = feature_name == str_y_values

    updated_dset = filter_dataset(temp_dataset, bools)
    class_text = conversation.get_class_name_from_label(int(feature_name))
    interpretable_parse_text = f"the ground truth label is {class_text}"

    return updated_dset, interpretable_parse_text


def filter_operation(conversation, parse_text, i, is_or=False, **kwargs):
    """The filtering operation.

    This function performs filtering on a data set using ONLY AutoGen-parsed entities.
    It updates the temp_dataset attribute in the conversation object.
    
    NO FALLBACKS - AutoGen must provide structured entities for all filtering operations.

    Arguments:
        is_or: Whether this is an OR operation
        conversation: The conversation object
        parse_text: Legacy parameter (ignored in clean architecture)
        i: Legacy parameter (ignored in clean architecture)
        **kwargs: Must contain AutoGen entities (features, operators, values)
    """
    if is_or:
        # construct a new temp data set to or with
        temp_dataset = conversation.build_temp_dataset(save=False).contents
    else:
        temp_dataset = conversation.temp_dataset.contents

    # Extract AutoGen entities - these are REQUIRED in clean architecture
    ent_features = kwargs.get('features', []) if kwargs else []
    ent_ops = kwargs.get('operators', []) if kwargs else []
    ent_vals = kwargs.get('values', []) if kwargs else []
    ent_prediction_vals = kwargs.get('prediction_values', []) if kwargs else []
    ent_label_vals = kwargs.get('label_values', []) if kwargs else []
    
    # Handle prediction-based filtering (e.g., "show instances where model predicted 1")
    if kwargs.get('filter_type') == 'prediction' and ent_prediction_vals:
        # This is explicit prediction-based filtering using prediction_values
        prediction_value = ent_prediction_vals[0]
        updated_dset, interp_parse_text = prediction_filter(temp_dataset, conversation, str(prediction_value))
        
    # Fallback: If no filter_type but pattern suggests prediction filtering
    elif not ent_features and ent_vals and len(ent_vals) == 1 and not kwargs.get('filter_type'):
        # Pattern suggests prediction filtering when no features specified
        prediction_value = ent_vals[0]
        updated_dset, interp_parse_text = prediction_filter(temp_dataset, conversation, str(prediction_value))
        
    # Handle feature-based filtering using AutoGen entities
    elif ent_features and ent_ops and ent_vals:
        # Handle multiple conditions (e.g., BMI > 30 AND age > 50)
        if len(ent_features) != len(ent_ops) or len(ent_features) != len(ent_vals):
            raise ValueError(f"Mismatched filter conditions: {len(ent_features)} features, {len(ent_ops)} operators, {len(ent_vals)} values")
        
        # Special case: Single ID filtering
        if len(ent_features) == 1 and ent_features[0].lower() == 'id':
            updated_dset, interp_parse_text = _handle_id_filtering(temp_dataset, conversation, ent_vals[0])
        else:
            # Apply multiple feature filters with AND logic
            combined_bools = None
            interp_parts = []
            
            for feature_name, operation, feature_value in zip(ent_features, ent_ops, ent_vals):
                # Regular feature filtering - handle case-insensitive matching
                actual_feature_name = None
                for col in temp_dataset['X'].columns:
                    if col.lower() == feature_name.lower():
                        actual_feature_name = col
                        break
                
                if actual_feature_name is None:
                    raise ValueError(f"Unknown feature name: {feature_name}. Available features: {list(temp_dataset['X'].columns)}")
                
                feature_name = actual_feature_name  # use the actual column name
                    
                # Apply the filtering based on operator
                if operation == '>' or operation == 'greater':
                    condition_bools = temp_dataset['X'][feature_name] > feature_value
                elif operation == '<' or operation == 'less':
                    condition_bools = temp_dataset['X'][feature_name] < feature_value
                elif operation == '=' or operation == '==' or operation == 'equal':
                    condition_bools = temp_dataset['X'][feature_name] == feature_value
                elif operation == '>=' or operation == 'greater_equal':
                    condition_bools = temp_dataset['X'][feature_name] >= feature_value
                elif operation == '<=' or operation == 'less_equal':
                    condition_bools = temp_dataset['X'][feature_name] <= feature_value
                elif operation == '!=' or operation == 'not_equal':
                    condition_bools = temp_dataset['X'][feature_name] != feature_value
                else:
                    raise ValueError(f"Unknown operator: {operation}")
                
                # Combine conditions with AND logic
                if combined_bools is None:
                    combined_bools = condition_bools
                else:
                    combined_bools = combined_bools & condition_bools
                
                # Build interpretable text
                interp_parts.append(format_parse_string(feature_name, feature_value, operation))
            
            # Apply the combined filter
            updated_dset = filter_dataset(temp_dataset, combined_bools)
            interp_parse_text = " and ".join(interp_parts)
    
    # Handle label-based filtering (ground truth filtering)
    elif kwargs.get('filter_type') == 'label' and ent_label_vals:
        label_value = ent_label_vals[0]
        updated_dset, interp_parse_text = label_filter(temp_dataset, conversation, str(label_value))
    
    # Handle ID-based filtering directly
    elif kwargs.get('filter_type') == 'id' and kwargs.get('patient_id') is not None:
        patient_id = kwargs.get('patient_id')
        updated_dset, interp_parse_text = _handle_id_filtering(temp_dataset, conversation, patient_id)
    
    # Handle patient_id filtering when filter_type isn't explicitly set
    elif kwargs.get('patient_id') is not None:
        patient_id = kwargs.get('patient_id')
        updated_dset, interp_parse_text = _handle_id_filtering(temp_dataset, conversation, patient_id)
    
    else:
        # NO FALLBACK - AutoGen must provide proper entities
        raise ValueError(
            "Clean Architecture Violation: AutoGen must provide structured entities for filtering. "
            f"Received: features={ent_features}, operators={ent_ops}, values={ent_vals}, "
            f"prediction_values={ent_prediction_vals}, label_values={ent_label_vals}, "
            f"filter_type={kwargs.get('filter_type')}, patient_id={kwargs.get('patient_id')}, kwargs={kwargs}. "
            "The AutoGen decoder needs to be improved to handle this query type."
        )

    if is_or:
        current_dataset = conversation.temp_dataset.contents
        updated_dset['X'] = pd.concat([updated_dset['X'], current_dataset['X']]).drop_duplicates()
        updated_dset['y'] = pd.concat([updated_dset['y'], current_dataset['y']]).drop_duplicates()
        conversation.add_interpretable_parse_op("or")
    else:
        conversation.add_interpretable_parse_op("and")

    conversation.add_interpretable_parse_op(interp_parse_text)
    conversation.temp_dataset.contents = updated_dset

    # NEW: Mark that this query applied a filter for clearer response communication
    resulting_size = len(updated_dset['X'])
    conversation.mark_query_filter_applied(interp_parse_text, resulting_size)

    # Return meaningful results instead of empty string
    num_instances = len(updated_dset['X'])
    if num_instances == 0:
        return f"No instances found where {interp_parse_text}.", 1
    else:
        # Show brief summary of filtered instances
        if num_instances <= 10:
            # Show IDs of instances for small results
            instance_ids = list(updated_dset['X'].index)
            result_text = f"Found {num_instances} instances where {interp_parse_text}: {instance_ids}"
        else:
            # Show count and sample for large results
            sample_ids = list(updated_dset['X'].index[:10])
            result_text = f"Found {num_instances} instances where {interp_parse_text}. Sample IDs include: {sample_ids}..."
        
        return result_text, 1


def _handle_id_filtering(temp_dataset, conversation, feature_value):
    """Handle ID-based filtering cleanly.
    
    When filtering by ID, we reset to the full dataset first since the user
    is asking about a specific instance regardless of current filters.
    """
    try:
        id_value = int(feature_value)
    except ValueError:
        raise ValueError(f"ID must be a number, got: {feature_value}")
    
    # Use full dataset for ID filtering, not current filtered dataset
    # This ensures we can find any valid instance ID
    full_dataset = conversation.get_var('dataset').contents
    updated_dset = {
        'X': full_dataset['X'].copy(),
        'y': full_dataset['y'].copy() if full_dataset['y'] is not None else None,
        'full_data': full_dataset['full_data'].copy() if 'full_data' in full_dataset else None,
        'cat': full_dataset.get('cat', []).copy(),
        'numeric': full_dataset.get('numeric', []).copy(),
        'ids_to_regenerate': full_dataset.get('ids_to_regenerate', []).copy()
    }
    
    # If id never appears in index, set the data to empty
    if id_value not in list(updated_dset['X'].index):
        updated_dset['X'] = updated_dset['X'].iloc[0:0]  # Empty dataframe with same structure
        updated_dset['y'] = updated_dset['y'].iloc[0:0]  # Empty series with same structure
        
        # Store helpful error information from full dataset
        available_ids = list(full_dataset['X'].index)[:10]
        total_ids = len(list(full_dataset['X'].index))
        
        error_msg = f"ID {id_value} not found. "
        if total_ids > 0:
            if total_ids <= 10:
                error_msg += f"Available IDs: {available_ids}"
            else:
                error_msg += f"Available IDs include: {available_ids}... (showing first 10 of {total_ids})"
        else:
            error_msg += "No instances available."
        
        conversation.last_filter_error = error_msg
    else:
        updated_dset['X'] = updated_dset['X'].loc[[id_value]]
        updated_dset['y'] = updated_dset['y'].loc[[id_value]]
    
    interp_parse_text = f"id equal to {id_value}"
    return updated_dset, interp_parse_text
