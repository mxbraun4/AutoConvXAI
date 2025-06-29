"""Prediction operation."""
import numpy as np
from explain.utils import gen_parse_op_text



def predict_operation(conversation, parse_text, i, max_num_preds_to_print=1, **kwargs):
    """The prediction operation."""
    model = conversation.get_var('model').contents
    
    # Check if we have specific feature values to predict on (new instance prediction)
    ent_features = kwargs.get('features', []) if kwargs else []
    ent_ops = kwargs.get('operators', []) if kwargs else []
    ent_vals = kwargs.get('values', []) if kwargs else []
    
    # If we have specific values with '=' operators, create a new instance to predict
    if ent_features and ent_vals and all(op == '=' for op in ent_ops):
        try:
            # Get dataset structure to create a properly formatted instance
            dataset_X = conversation.get_var('dataset').contents['X']
            
            # Create a new instance with mean values as defaults
            new_instance = dataset_X.mean().copy()
            
            # Set the specified feature values
            for feat, val in zip(ent_features, ent_vals):
                if feat in new_instance.index:
                    new_instance[feat] = val
            
            # Convert to array and reshape for single prediction
            data = new_instance.values.reshape(1, -1)
            
            # Make prediction and get probabilities
            model_predictions = model.predict(data)
            try:
                model_probabilities = model.predict_proba(data)
                confidence = model_probabilities[0][model_predictions[0]]
            except:
                confidence = None
            
            # Format response for single instance prediction
            return_s = f"For a new instance with "
            feature_desc = ", ".join([f"{feat}={val}" for feat, val in zip(ent_features, ent_vals)])
            return_s += f"<b>{feature_desc}</b>, "
            
            if conversation.class_names is None:
                prediction_class = str(model_predictions[0])
                return_s += f"the model predicts <b>{prediction_class}</b>"
            else:
                class_text = conversation.class_names[model_predictions[0]]
                return_s += f"the model predicts <b>{class_text}</b>"
            
            # Add confidence if available
            if confidence is not None:
                confidence_pct = round(confidence * 100, conversation.rounding_precision)
                return_s += f" with <b>{confidence_pct}% confidence</b>"
            
            return_s += ".<br>"
            return return_s, 1
            
        except Exception as e:
            # Fall back to original behavior if there's any issue
            pass
    
    # Original behavior for filtering/general predictions
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        # Check if there was a recent filter error for better error messages
        if hasattr(conversation, 'last_filter_error') and conversation.last_filter_error:
            error_msg = conversation.last_filter_error
            # Clear the error after use
            conversation.last_filter_error = None
            return f'No instances found: {error_msg}', 0
        else:
            return 'There are no instances that meet this description!', 0

    model_predictions = model.predict(data)
    
    # Get confidence scores if possible
    try:
        # Convert to numpy array to remove feature names for compatibility
        data_array = data.values if hasattr(data, 'values') else data
        model_probabilities = model.predict_proba(data_array)
    except:
        model_probabilities = None

    # Format return string
    return_s = ""

    filter_string = gen_parse_op_text(conversation)

    if len(model_predictions) == 1:
        return_s += f"The instance with <b>{filter_string}</b> is predicted "
        if conversation.class_names is None:
            prediction_class = str(model_predictions[0])
            return_s += f"<b>{prediction_class}</b>"
        else:
            class_text = conversation.class_names[model_predictions[0]]
            return_s += f"<b>{class_text}</b>"
        
        # Add confidence for single prediction
        if model_probabilities is not None:
            confidence = model_probabilities[0][model_predictions[0]]
            confidence_pct = round(confidence * 100, conversation.rounding_precision)
            return_s += f" with <b>{confidence_pct}% confidence</b>"
        
        return_s += "."
    else:
        intro_text = "For the data,"
        return_s += f"{intro_text} the model predicts:"
        unique_preds = np.unique(model_predictions)
        return_s += "<ul>"
        for j, uniq_p in enumerate(unique_preds):
            return_s += "<li>"
            freq = np.sum(uniq_p == model_predictions) / len(model_predictions)
            round_freq = str(round(freq * 100, conversation.rounding_precision))

            if conversation.class_names is None:
                return_s += f"<b>class {uniq_p}</b>, {round_freq}%"
            else:
                class_text = conversation.class_names[uniq_p]
                return_s += f"<b>{class_text}</b>, {round_freq}%"
            return_s += "</li>"
        return_s += "</ul>"
    return_s += "<br>"
    return return_s, 1
