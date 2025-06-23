"""Function to define feature meanings."""

def define_operation(conversation, parse_text, i, **kwargs):
    """Generates text to define feature."""
    if i+1 >= len(parse_text):
        return "No feature name specified for definition.", 0
    
    feature_name = parse_text[i+1]
    feature_definition = conversation.get_feature_definition(feature_name)
    if feature_definition is None:
        return f"Definition for feature name <b>{feature_name}</b> is not specified.", 1
    return_string = f"The feature named <b>{feature_name}</b> is defined as: "
    return_string += "<em>" + feature_definition + "</em>."
    return return_string, 1
