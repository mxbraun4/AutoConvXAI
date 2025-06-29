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
    """The what if operation."""

    # The temporary dataset to approximate
    temp_dataset = conversation.temp_dataset.contents

    # The feature name to adjust
    if i+1 >= len(parse_text):
        return "No feature specified for what-if analysis!", 0
    
    feature_name = parse_text[i+1]

    # Numerical feature case. Also putting id in here because the operations
    # are the same
    if is_numeric(feature_name, temp_dataset):
        update_term, update_value = get_numeric_updates(parse_text, i)
        temp_dataset['X'], parse_op = update_numeric_feature(temp_dataset,
                                                             feature_name,
                                                             update_term,
                                                             update_value)
    elif is_categorical(feature_name, temp_dataset):
        # handles conversion between true/false and 1/0 for categorical features
        if i+2 >= len(parse_text):
            return f"No value specified for categorical feature {feature_name}!", 0
        
        categorical_val = convert_categorical_bools(parse_text[i+2])
        temp_dataset['X'][feature_name] = categorical_val
        parse_op = f"{feature_name} is set to {str(categorical_val)}"
    elif feature_name == "id":
        # Setting what if updates on ids to no effect. I don't think there's any
        # reason to support this.
        return "What if updates have no effect on id's!", 0
    else:
        raise NameError(f"Parsed unknown feature name {feature_name}")

    processed_ids = list(conversation.temp_dataset.contents['X'].index)
    conversation.temp_dataset.contents['ids_to_regenerate'].extend(processed_ids)

    conversation.add_interpretable_parse_op("and")
    conversation.add_interpretable_parse_op(parse_op)

    return '', 1
