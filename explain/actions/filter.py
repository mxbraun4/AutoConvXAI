"""The filtering action.

This action filters some data according to different filtering criteria, e.g., less than or equal
to, greater than, etc. It modifies the temporary dataset in the conversation object, updating that dataset to yield the
correct filtering based on the parse.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


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


def numerical_filter(parse_text, temp_dataset, i, feature_name):
    """Performs numerical filtering.

    All this routine does (though it looks a bit clunky) is look at
    the parse_text and decide which filtering operation to do (e.g.,
    greater than, equal to, etc.) and then performs the operation.
    """
    # Greater than or equal to
    if parse_text[i+2] == 'greater' and i+3 < len(parse_text) and parse_text[i+3] == 'equal':
        print(parse_text)
        if i+5 >= len(parse_text):
            raise ValueError(f"Missing value for filter operation at position {i+5}")
        feature_value = float(parse_text[i+5])
        bools = temp_dataset['X'][feature_name] >= feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "greater than or equal to")
    # Greater than
    elif parse_text[i+2] == 'greater':
        if i+3 >= len(parse_text):
            raise ValueError(f"Missing value for filter operation at position {i+3}")
        # For "filter age greater 50", the value is at i+3
        value_index = i+3
        
        try:
            feature_value = float(parse_text[value_index])
        except ValueError:
            raise ValueError(f"Expected numeric value after 'greater', got '{parse_text[value_index]}'")
            
        bools = temp_dataset['X'][feature_name] > feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "greater than")
    # Less than or equal to
    elif parse_text[i+2] == 'less' and i+3 < len(parse_text) and parse_text[i+3] == 'equal':
        if i+5 >= len(parse_text):
            raise ValueError(f"Missing value for filter operation at position {i+5}")
        feature_value = float(parse_text[i+5])
        bools = temp_dataset['X'][feature_name] <= feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "less than or equal to")
    # Less than
    elif parse_text[i+2] == 'less':
        if i+3 >= len(parse_text):
            raise ValueError(f"Missing value for filter operation at position {i+3}")
        value_index = i+3
        try:
            feature_value = float(parse_text[value_index])
        except ValueError:
            raise ValueError(f"Expected numeric value after 'less', got '{parse_text[value_index]}'")
            
        bools = temp_dataset['X'][feature_name] < feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "less than")
    # Equal to
    elif parse_text[i+2] == 'equal':
        if i+3 >= len(parse_text):
            raise ValueError(f"Missing value for filter operation at position {i+3}")
        value_index = i+3
        try:
            feature_value = float(parse_text[value_index])
        except ValueError:
            raise ValueError(f"Expected numeric value after 'equal', got '{parse_text[value_index]}'")
            
        bools = temp_dataset['X'][feature_name] == feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "equal to")
    # Not equal to
    elif parse_text[i+2] == 'not':
        if i+4 >= len(parse_text):
            raise ValueError(f"Missing value for filter operation at position {i+4}")
        value_index = i+4
        try:
            feature_value = float(parse_text[value_index])
        except ValueError:
            raise ValueError(f"Expected numeric value after 'not equal', got '{parse_text[value_index]}'")
            
        bools = temp_dataset['X'][feature_name] != feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "not equal to")
    else:
        raise NameError(f"Uh oh, looks like something is wrong with {parse_text}")
    return updated_dset, interpretable_parse_text


def categorical_filter(parse_text, temp_dataset, conversation, i, feature_name):
    """Perform categorical filtering of a data set."""
    feature_value = parse_text[i+2]

    interpretable_parse_text = f"{feature_name} equal to {str(feature_value)}"

    if feature_name == "incorrect":
        # In the case the user asks for the incorrect predictions
        data = temp_dataset['X']
        y_values = temp_dataset['y']

        # compute model predictions
        model = conversation.get_var('model').contents
        y_pred = model.predict(data)

        # set bools to when the predictions are not the same
        # this with parse out to when incorrect is true, filter by
        # predictions != ground truth
        if feature_value == "true":
            bools = y_values != y_pred
        else:
            bools = y_values == y_pred
    else:
        if is_numeric_dtype(temp_dataset['X'][feature_name]):
            if feature_value == 'true':
                feature_value = 1
            elif feature_value == 'false':
                feature_value = 0
            else:
                feature_value = float(feature_value)
        bools = temp_dataset['X'][feature_name] == feature_value
    updated_dset = filter_dataset(temp_dataset, bools)
    return updated_dset, interpretable_parse_text


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
        # Use AutoGen entities for clean filtering
        feature_name = ent_features[0]  # Take first feature
        operation = ent_ops[0]  # Take first operator  
        feature_value = ent_vals[0]  # Take first value
        
        # Special case: ID filtering
        if feature_name.lower() == 'id':
            updated_dset, interp_parse_text = _handle_id_filtering(temp_dataset, conversation, feature_value)
        else:
            # Regular feature filtering
            if feature_name not in temp_dataset['X'].columns:
                raise ValueError(f"Unknown feature name: {feature_name}. Available features: {list(temp_dataset['X'].columns)}")
                
            # Apply the filtering based on operator
            if operation == '>' or operation == 'greater':
                bools = temp_dataset['X'][feature_name] > feature_value
            elif operation == '<' or operation == 'less':
                bools = temp_dataset['X'][feature_name] < feature_value
            elif operation == '=' or operation == '==' or operation == 'equal':
                bools = temp_dataset['X'][feature_name] == feature_value
            elif operation == '>=' or operation == 'greater_equal':
                bools = temp_dataset['X'][feature_name] >= feature_value
            elif operation == '<=' or operation == 'less_equal':
                bools = temp_dataset['X'][feature_name] <= feature_value
            elif operation == '!=' or operation == 'not_equal':
                bools = temp_dataset['X'][feature_name] != feature_value
            else:
                raise ValueError(f"Unknown operator: {operation}")
                
            updated_dset = filter_dataset(temp_dataset, bools)
            interp_parse_text = format_parse_string(feature_name, feature_value, operation)
    
    # Handle label-based filtering (ground truth filtering)
    elif kwargs.get('filter_type') == 'label' and ent_label_vals:
        label_value = ent_label_vals[0]
        updated_dset, interp_parse_text = label_filter(temp_dataset, conversation, str(label_value))
    
    else:
        # NO FALLBACK - AutoGen must provide proper entities
        raise ValueError(
            "Clean Architecture Violation: AutoGen must provide structured entities for filtering. "
            f"Received: features={ent_features}, operators={ent_ops}, values={ent_vals}, "
            f"prediction_values={ent_prediction_vals}, label_values={ent_label_vals}, "
            f"filter_type={kwargs.get('filter_type')}, kwargs={kwargs}. "
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
            sample_ids = list(updated_dset['X'].index[:5])
            result_text = f"Found {num_instances} instances where {interp_parse_text}. Sample IDs: {sample_ids}..."
        
        return result_text, 1


def _handle_id_filtering(temp_dataset, conversation, feature_value):
    """Handle ID-based filtering cleanly."""
    try:
        id_value = int(feature_value)
    except ValueError:
        raise ValueError(f"ID must be a number, got: {feature_value}")
    
    updated_dset = temp_dataset.copy()
    
    # If id never appears in index, set the data to empty
    if id_value not in list(updated_dset['X'].index):
        updated_dset['X'] = updated_dset['X'].iloc[0:0]  # Empty dataframe with same structure
        updated_dset['y'] = updated_dset['y'].iloc[0:0]  # Empty series with same structure
        
        # Store helpful error information
        available_ids = list(temp_dataset['X'].index)[:10]
        total_ids = len(list(temp_dataset['X'].index))
        
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
