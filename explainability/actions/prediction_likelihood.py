import numpy as np
from explainability.core.utils import gen_parse_op_text


def predict_likelihood(conversation, parse_text, i, **kwargs):
    """The prediction likelihood operation."""
    predict_proba = conversation.get_var('model_prob_predict').contents
    model = conversation.get_var('model').contents
    
    # Check if we should use stored prediction context (follow-up to a specific prediction)
    if hasattr(conversation, 'last_prediction_instance') and conversation.last_prediction_instance:
        # Use the stored prediction context for follow-up queries
        prediction_context = conversation.last_prediction_instance
        
        # Use the stored data and confidence information
        data = prediction_context['data']
        confidence = prediction_context.get('confidence')
        prediction = prediction_context.get('prediction')
        
        # Get all probabilities for the instance
        try:
            model_prediction_probabilities = predict_proba(data.values)
        except:
            # Fallback for compatibility
            model_prediction_probabilities = model.predict_proba(data.values)
        
        # Create structured response similar to predict.py
        result = {
            'type': 'prediction_likelihood',
            'prediction': int(prediction),
            'prediction_class': conversation.class_names.get(prediction, str(prediction)),
            'confidence': round(confidence * 100, conversation.rounding_precision) if confidence is not None else None,
            'context': 'follow_up_to_prediction',
            'input_features': prediction_context.get('features', {}),
            'all_probabilities': {}
        }
        
        # Add probability breakdown for all classes
        for c in range(model_prediction_probabilities.shape[1]):
            proba = round(model_prediction_probabilities[0, c] * 100, conversation.rounding_precision)
            class_name = conversation.class_names.get(c, f"class {c}")
            result['all_probabilities'][class_name] = proba
        
        return result, 1
    
    # Original behavior for general dataset queries
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
