"""Data summary operation."""


# Note, these are hardcode for compas!
def data_operation(conversation, parse_text, i, **kwargs):
    """Data summary operation."""
    # Get dataset size information - handle both filtered and unfiltered cases
    if hasattr(conversation, 'temp_dataset') and conversation.temp_dataset:
        df = conversation.temp_dataset.contents['X']
        dataset_size = len(df)
        total_size = len(conversation.get_var('dataset').contents['X'])
    else:
        # No filtering applied - use full dataset
        df = conversation.get_var('dataset').contents['X']
        dataset_size = len(df)
        total_size = dataset_size
    
    # Check if user is asking for specific feature statistics or class counts
    query_text = " ".join(parse_text).lower()
    
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
            
            size_desc = f"{dataset_size} patients" if dataset_size == total_size else f"{dataset_size} out of {total_size} patients"
            text = f"<b>Class Distribution</b> for {size_desc}:<br><br>"
            
            for class_val, count in value_counts.items():
                class_name = class_names.get(class_val, str(class_val)) if class_names else str(class_val)
                percentage = round((count / len(target_data)) * 100, conversation.rounding_precision)
                text += f"• <b>{class_name}</b>: {count} instances ({percentage}%)<br>"
            
            text += "<br>"
            return text, 1
    
    # ENHANCED: Check if a specific feature is mentioned in parse_text 
    # This handles cases like "data age" where AutoGen correctly identifies the feature
    specific_feature = None
    if len(parse_text) > 1:
        # Check if the second element is a feature name
        potential_feature = parse_text[1].lower()
        if potential_feature in [col.lower() for col in df.columns]:
            specific_feature = next(col for col in df.columns if col.lower() == potential_feature)
    
    # Handle specific feature requests (either via keywords or direct mention)
    if specific_feature or any(word in query_text for word in ['average', 'mean', 'statistics']):
        # If we have a specific feature from parse_text, use that
        if specific_feature:
            target_feature = specific_feature
        else:
            # Otherwise, look for feature names in the query text
            target_feature = None
            for feature in df.columns:
                if feature.lower() in query_text:
                    target_feature = feature
                    break
        
        if target_feature:
            avg_val = round(df[target_feature].mean(), conversation.rounding_precision)
            std_val = round(df[target_feature].std(), conversation.rounding_precision)
            min_val = round(df[target_feature].min(), conversation.rounding_precision)
            max_val = round(df[target_feature].max(), conversation.rounding_precision)
            
            size_desc = f"{dataset_size} patients" if dataset_size == total_size else f"{dataset_size} out of {total_size} patients"
            
            text = f"For the {size_desc} in the dataset:<br><br>"
            text += f"<b>{target_feature.title()} Statistics:</b><br>"
            text += f"• Average: <b>{avg_val}</b><br>"
            text += f"• Standard deviation: {std_val}<br>"
            text += f"• Range: {min_val} to {max_val}<br><br>"
            
            return text, 1
    
    # If statistics requested but no specific feature found, show all
    if any(word in query_text for word in ['statistics', 'stats', 'summary']):
        size_desc = f"{dataset_size} patients" if dataset_size == total_size else f"{dataset_size} out of {total_size} patients"
        
        text = f"<b>Dataset Statistics</b> for {size_desc}:<br><br>"
        
        for feature in df.columns:
            avg_val = round(df[feature].mean(), conversation.rounding_precision)
            std_val = round(df[feature].std(), conversation.rounding_precision)
            min_val = round(df[feature].min(), conversation.rounding_precision)
            max_val = round(df[feature].max(), conversation.rounding_precision)
            
            text += f"<b>{feature.title()}:</b> avg={avg_val}, std={std_val}, range={min_val}-{max_val}<br>"
        
        text += "<br>"
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

    # Show available IDs for filtering (helpful for users)
    available_ids = list(df.index)
    if available_ids:
        if len(available_ids) <= 10:
            id_sample = available_ids
            id_text = f"Available instance IDs: {id_sample}"
        else:
            id_sample = available_ids[:10]
            id_text = f"Sample instance IDs: {id_sample}... (total: {len(available_ids)} instances)"
        
        text += f"<em>{id_text}</em><br><br>"

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
