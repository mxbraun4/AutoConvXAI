"""Data summary operation."""


# Note, these are hardcode for compas!
def data_operation(conversation, parse_text, i, **kwargs):
    """Data summary operation."""
    # Get dataset size information
    df = conversation.temp_dataset.contents['X']
    dataset_size = len(df)
    total_size = len(conversation.get_var('dataset').contents['X'])
    
    # Check if user is asking for specific feature statistics
    query_text = " ".join(parse_text).lower()
    
    # Handle specific feature queries like "average age", "mean glucose", etc.
    if any(word in query_text for word in ['average', 'mean']) and 'age' in query_text:
        if 'age' in df.columns:
            avg_age = round(df['age'].mean(), conversation.rounding_precision)
            std_age = round(df['age'].std(), conversation.rounding_precision)
            min_age = round(df['age'].min(), conversation.rounding_precision)
            max_age = round(df['age'].max(), conversation.rounding_precision)
            
            size_desc = f"{dataset_size} patients" if dataset_size == total_size else f"{dataset_size} out of {total_size} patients"
            
            text = f"For the {size_desc} in the dataset:<br><br>"
            text += f"<b>Age Statistics:</b><br>"
            text += f"• Average age: <b>{avg_age} years</b><br>"
            text += f"• Standard deviation: {std_age} years<br>"
            text += f"• Age range: {min_age} to {max_age} years<br><br>"
            
            return text, 1
        else:
            return "Age information is not available in this dataset.<br><br>", 0
    
    # Handle other specific feature queries
    for feature in df.columns:
        if feature.lower() in query_text and any(word in query_text for word in ['average', 'mean']):
            avg_val = round(df[feature].mean(), conversation.rounding_precision)
            std_val = round(df[feature].std(), conversation.rounding_precision)
            min_val = round(df[feature].min(), conversation.rounding_precision)
            max_val = round(df[feature].max(), conversation.rounding_precision)
            
            size_desc = f"{dataset_size} patients" if dataset_size == total_size else f"{dataset_size} out of {total_size} patients"
            
            text = f"For the {size_desc} in the dataset:<br><br>"
            text += f"<b>{feature.title()} Statistics:</b><br>"
            text += f"• Average: <b>{avg_val}</b><br>"
            text += f"• Standard deviation: {std_val}<br>"
            text += f"• Range: {min_val} to {max_val}<br><br>"
            
            return text, 1
    
    # Default dataset description
    description = conversation.describe.get_dataset_description()
    if not description or description == "":
        description = "diabetes prediction based on patient health metrics"
    
    # Create size description
    if dataset_size == total_size:
        size_text = f"<b>{dataset_size} patient records</b>"
    else:
        size_text = f"<b>{dataset_size} out of {total_size} patient records</b> (filtered dataset)"
    
    text = f"The dataset contains {size_text} with information related to <b>{description}</b>.<br><br>"

    # List out the feature names
    f_names = list(df.columns)
    f_string = "<ul>"
    for fn in f_names:
        f_string += f"<li>{fn}</li>"
    f_string += "</ul>"
    text += f"The exact feature names in the data are listed as follows:{f_string}<br><br>"

    # Summarize performance
    model = conversation.get_var('model').contents
    score = conversation.describe.get_eval_performance(model, conversation.default_metric)

    # Note, if no eval data is specified this will return an empty string and nothing will happen.
    if score != "":
        text += score
        text += "<br><br>"

    # Create more in depth description of the data, summarizing a few statistics
    rest_of_text = ""
    rest_of_text += "Here's a more in depth summary of the data.<br><br>"

    for i, f in enumerate(f_names):
        mean = round(df[f].mean(), conversation.rounding_precision)
        std = round(df[f].std(), conversation.rounding_precision)
        min_v = round(df[f].min(), conversation.rounding_precision)
        max_v = round(df[f].max(), conversation.rounding_precision)
        new_feature = (f"{f}: The mean is {mean}, one standard deviation is {std},"
                       f" the minimum value is {min_v}, and the maximum value is {max_v}")
        new_feature += "<br><br>"

        rest_of_text += new_feature

    text += "Let me know if you want to see an in depth description of the dataset statistics.<br><br>"
    conversation.store_followup_desc(rest_of_text)

    return text, 1
