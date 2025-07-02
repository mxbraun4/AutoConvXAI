import numpy as np
from explain.core.utils import gen_parse_op_text


def predict_likelihood(conversation, parse_text, i, **kwargs):
    """The prediction likelihood operation."""
    predict_proba = conversation.get_var('model_prob_predict').contents
    model = conversation.get_var('model').contents
    data = conversation.temp_dataset.contents['X'].values

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    model_prediction_probabilities = predict_proba(data)
    model_predictions = model.predict(data)
    num_classes = model_prediction_probabilities.shape[1]

    # Format return string
    return_s = ""

    filter_string = gen_parse_op_text(conversation)

    if model_prediction_probabilities.shape[0] == 1:
        return_s += f"The model predicts the instance with {filter_string} as:\n"
        for c in range(num_classes):
            proba = round(model_prediction_probabilities[0, c]*100, conversation.rounding_precision)
            if conversation.class_names is None:
                return_s += f"- class {str(c)}"
            else:
                class_text = conversation.class_names[c]
                return_s += f"- {class_text}"
            return_s += f" with {str(proba)}% probability\n"
    else:
        if len(filter_string) > 0:
            filtering_text = f" where <b>{filter_string}</b>"
        else:
            filtering_text = ""
        return_s += f"Over {data.shape[0]} cases{filtering_text} in the data, the model predicts:\n"
        unique_preds = np.unique(model_predictions)
        for j, uniq_p in enumerate(unique_preds):
            freq = np.sum(uniq_p == model_predictions) / len(model_predictions)
            round_freq = str(round(freq*100, conversation.rounding_precision))

            if conversation.class_names is None:
                return_s += f"- class {uniq_p}, {round_freq}%"
            else:
                class_text = conversation.class_names[uniq_p]
                return_s += f"- {class_text}, {round_freq}%"
            return_s += " of the time\n"
    return return_s, 1
