"""Prediction operation."""
import numpy as np
from explain.core.utils import gen_parse_op_text



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
            
            # Create a new instance with mean values as defaults for unspecified features
            new_instance = dataset_X.mean().copy()
            
            # Set the specified feature values
            for feat, val in zip(ent_features, ent_vals):
                if feat in new_instance.index:
                    new_instance[feat] = val
            
            # Convert to DataFrame to maintain feature names for sklearn compatibility
            import pandas as pd
            data_df = pd.DataFrame([new_instance], columns=dataset_X.columns)
            
            # Make prediction and get probabilities
            from main import _safe_model_predict
            model_predictions = _safe_model_predict(model, data_df)
            try:
                model_probabilities = model.predict_proba(data_df)
                confidence = model_probabilities[0][model_predictions[0]]
            except:
                confidence = None
            
            # Return structured data for single instance prediction
            result = {
                'type': 'single_prediction',
                'input_features': dict(zip(ent_features, ent_vals)),
                'all_features': new_instance.to_dict(),  # Show all feature values used
                'prediction': int(model_predictions[0]),
                'prediction_class': conversation.class_names[model_predictions[0]] if conversation.class_names else str(model_predictions[0]),
                'confidence': round(confidence * 100, conversation.rounding_precision) if confidence is not None else None,
                'instance_type': 'new_instance'
            }
            return result, 1
            
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
            return {'type': 'error', 'message': f'No instances found: {error_msg}'}, 0
        else:
            return {'type': 'error', 'message': 'There are no instances that meet this description!'}, 0

    from main import _safe_model_predict
    model_predictions = _safe_model_predict(model, data)
    
    # Get confidence scores if possible
    try:
        # Convert to numpy array to remove feature names for compatibility
        data_array = data.values if hasattr(data, 'values') else data
        model_probabilities = model.predict_proba(data_array)
    except:
        model_probabilities = None

    # Create structured response
    filter_string = gen_parse_op_text(conversation)

    if len(model_predictions) == 1:
        # Single instance prediction
        confidence = None
        if model_probabilities is not None:
            confidence = round(model_probabilities[0][model_predictions[0]] * 100, conversation.rounding_precision)
        
        result = {
            'type': 'single_prediction',
            'prediction': int(model_predictions[0]),
            'prediction_class': conversation.class_names[model_predictions[0]] if conversation.class_names else str(model_predictions[0]),
            'confidence': confidence,
            'filter_applied': filter_string,
            'instance_type': 'filtered_data'
        }
        return result, 1
    else:
        # Multiple predictions - distribution
        unique_preds = np.unique(model_predictions)
        prediction_dist = {}
        
        for uniq_p in unique_preds:
            freq = np.sum(uniq_p == model_predictions) / len(model_predictions)
            round_freq = round(freq * 100, conversation.rounding_precision)
            class_name = conversation.class_names[uniq_p] if conversation.class_names else f"class {uniq_p}"
            prediction_dist[class_name] = round_freq
        
        result = {
            'type': 'prediction_distribution',
            'distribution': prediction_dist,
            'total_instances': len(model_predictions),
            'filter_applied': filter_string
        }
        return result, 1
