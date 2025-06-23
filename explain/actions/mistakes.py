"""Show model mistakes"""
from copy import deepcopy


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from explain.actions.utils import get_parse_filter_text, get_rules


def one_mistake(y_true, y_pred, conversation, intro_text):
    """One mistake text"""
    label = y_true[0]
    prediction = y_pred[0]

    label_text = conversation.get_class_name_from_label(label)
    predict_text = conversation.get_class_name_from_label(prediction)

    if label == prediction:
        correct_text = "correct"
    else:
        correct_text = "incorrect"

    return_string = (f"{intro_text} the model predicts <em>{predict_text}</em> and the ground"
                     f" label is <em>{label_text}</em>, so the model is <b>{correct_text}</b>!")
    return return_string


def sample_mistakes(y_true, y_pred, conversation, intro_text, ids):
    """Sample mistakes sub-operation"""
    if len(y_true) == 1:
        return_string = one_mistake(y_true, y_pred, conversation, intro_text)
    else:
        incorrect_num = np.sum(y_true != y_pred)
        total_num = len(y_true)
        incorrect_data = ids[y_true != y_pred]

        error_rate = round(incorrect_num / total_num, conversation.rounding_precision)
        return_string = (f"{intro_text} the model is incorrect {incorrect_num} out of {total_num} "
                         f"times (error rate {error_rate}). Here are the ids of instances the model"
                         f" predicts incorrectly:<br><br>{incorrect_data}")

    return return_string


def train_tree(data, target, depth: int = 1):
    """Trains a decision tree"""
    dt_string = []
    tries = 0
    while len(dt_string) < 3 and tries < 10:
        tries += 1
        dt = DecisionTreeClassifier(max_depth=depth).fit(data, target)
        dt_string = get_rules(dt,
                              feature_names=list(data.columns),
                              class_names=["correct", "incorrect"])
        depth += 1

    return dt_string


def typical_mistakes(data, y_true, y_pred, conversation, intro_text, ids):
    """Typical mistakes sub-operation"""
    if len(y_true) == 1:
        return_string = one_mistake(y_true, y_pred, conversation, intro_text)
    else:
        incorrect_vals = y_true != y_pred
        return_options = train_tree(data, incorrect_vals)

        if len(return_options) == 0:
            return "I couldn't find any patterns for mistakes the model typically makes."

        return_string = f"{intro_text} the model typically predicts incorrect:<br><br>"
        for rule in return_options:
            return_string += rule + "<br><br>"

    return return_string


def show_mistakes_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows the model mistakes."""
    data = conversation.temp_dataset.contents['X']
    y_true_pd = deepcopy(conversation.temp_dataset.contents['y'])

    if isinstance(y_true_pd, pd.Series):
        y_true = y_true_pd.to_numpy()
    elif isinstance(y_true_pd, list):
        y_true = np.array(y_true_pd)

    # Get ids
    ids = np.array(list(data.index))

    model = conversation.get_var('model').contents

    # The filtering text
    intro_text = get_parse_filter_text(conversation)

    if len(y_true) == 0:
        return "There are no instances in the data that meet this description.<br><br>", 0

    y_pred = model.predict(data)
    if np.sum(y_true == y_pred) == len(y_true):
        if len(y_true) == 1:
            return f"{intro_text} the model predicts correctly!<br><br>", 1
        else:
            return f"{intro_text} the model predicts correctly on all the instances in the data!<br><br>", 1

    # Check if mistake type is specified and within bounds
    mistake_type = None
    if i+1 < len(parse_text):
        mistake_type = parse_text[i+1]
    
    # If no specific type specified, try to infer from the original query
    if mistake_type is None or mistake_type not in ["sample", "typical"]:
        # Check if "typical" is mentioned anywhere in the parse_text
        full_text = " ".join(parse_text).lower()
        if "typical" in full_text:
            mistake_type = "typical"
        elif "sample" in full_text:
            mistake_type = "sample"
        else:
            # Default to typical mistakes as it's more informative
            mistake_type = "typical"
    
    if mistake_type == "sample":
        return_string = sample_mistakes(y_true,
                                        y_pred,
                                        conversation,
                                        intro_text,
                                        ids)
    elif mistake_type == "typical":
        return_string = typical_mistakes(data,
                                         y_true,
                                         y_pred,
                                         conversation,
                                         intro_text,
                                         ids)
    else:
        # This shouldn't happen with the logic above, but safe fallback
        return_string = typical_mistakes(data,
                                         y_true,
                                         y_pred,
                                         conversation,
                                         intro_text,
                                         ids)

    return_string += "<br><br>"
    return return_string, 1
