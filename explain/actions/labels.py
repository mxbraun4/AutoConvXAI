"""Show data labels"""




def show_labels_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows labels."""
    y_values = conversation.temp_dataset.contents['y']
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
