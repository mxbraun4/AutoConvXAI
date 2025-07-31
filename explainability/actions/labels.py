"""Show data labels"""




def show_labels_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows labels."""
    y_values = conversation.temp_dataset.contents['y']
    
    # Apply target_values scoping if specified
    target_vals = kwargs.get('target_values', []) if kwargs else []
    if target_vals:
        # Scope labels to specific diabetes status
        target_value = target_vals[0]  # Use first target value
        
        # Filter labels to only include patients with specified diabetes status
        y_values = y_values[y_values == target_value]
        
        if len(y_values) == 0:
            target_name = conversation.get_class_name_from_label(target_value)
            return f"There are no {target_name} patients that meet this description.", 0
    
    intro_text = "For the data,"

    if len(y_values) == 0:
        return "There are no instances in the data that meet this description.", 0
    if len(y_values) == 1:
        label = y_values.item()
        label_text = conversation.get_class_name_from_label(label)
        return_string = f"{intro_text} the label is {label_text}."
    else:
        return_string = f"{intro_text} the labels are:\n"
        for index, label in zip(list(y_values.index), y_values):
            label_text = conversation.get_class_name_from_label(label)
            return_string += f"id {index} is labeled {label_text}\n"

    return return_string, 1
