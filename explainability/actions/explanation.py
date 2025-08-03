"""Explanation action for model predictions.

Generates LIME-based explanations for individual instances or groups,
with optimizations for single-instance queries.
"""


def _is_followup_question(parse_text, conversation):
    """Check if query is a follow-up question rather than explanation request.
    
    Args:
        parse_text: User's query text
        conversation: Conversation context
        
    Returns:
        bool: True if this should be handled as follow-up, not new explanation
    """
    
    if not isinstance(parse_text, str):
        return False
    
    text_lower = parse_text.lower()
    
    # Patterns indicating analytical follow-up rather than explanation request
    followup_patterns = [
        # Prediction bias questions
        "underpredicts", "overpredicts", "underestimates", "overestimates",
        "model predicts less", "model predicts more", "conservative", "aggressive",
        
        # Analytical conclusions
        "this means the model", "so the model", "the model seems",
        "does this mean", "therefore the model", "so it appears",
        
        # Performance questions that can be answered from context
        "how accurate", "is this good", "is this bad", "what does this mean",
        
        # Direct follow-ups
        "tell me more", "explain that better", "what about that"
    ]
    
    # Check if any followup pattern matches
    for pattern in followup_patterns:
        if pattern in text_lower:
            return True
    
    return False


def explain_operation(conversation, parse_text, i, **kwargs):
    """Generate LIME explanations for model predictions.
    
    Args:
        conversation: Conversation context with dataset and model
        parse_text: User's query text
        i: Instance index (unused)
        **kwargs: Optional target_values for filtering
        
    Returns:
        tuple: (explanation_dict, success_flag)
    """
    
    # Early exit for follow-up questions to avoid expensive LIME computations
    if parse_text and _is_followup_question(parse_text, conversation):
        from explainability.actions.followup import followup_operation
        return followup_operation(conversation, parse_text, i, **kwargs)
    
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return {'type': 'error', 'message': 'There are no instances that meet this description!'}, 0

    # Filter by target class if requested (e.g., "explain diabetes cases")
    target_vals = kwargs.get('target_values', []) if kwargs else []
    if target_vals:
        y_data = conversation.temp_dataset.contents['y']
        target_value = target_vals[0]
        
        # Filter to only patients with specified outcome
        matching_indices = y_data[y_data == target_value].index
        data = data.loc[matching_indices]
        
        if len(data) == 0:
            target_name = conversation.get_class_name_from_label(target_value)
            return {'type': 'error', 'message': f'There are no {target_name} patients that meet this description!'}, 0
        
        # Update dataset for downstream operations
        conversation.temp_dataset.contents['X'] = data
        conversation.temp_dataset.contents['y'] = y_data.loc[matching_indices]

    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    model = conversation.get_var('model').contents
    
    # Generate predictions and confidence scores
    from app.core import _safe_model_predict
    predictions = _safe_model_predict(model, data)
    try:
        probabilities = model.predict_proba(data)
    except:
        probabilities = None

    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    
    # Use optimized path for single instances, full pipeline for multiple
    if len(data) == 1:
        structured_explanation = _extract_structured_explanation_fast(data, predictions, probabilities, 
                                                                     mega_explainer_exp, conversation)
    else:
        # Generate comprehensive summary for multiple instances
        full_summary, short_summary = mega_explainer_exp.summarize_explanations(data,
                                                                                filtering_text="",
                                                                                ids_to_regenerate=regen)
        conversation.store_followup_desc(full_summary)
        
        structured_explanation = _extract_structured_explanation(data, predictions, probabilities, 
                                                                mega_explainer_exp, conversation)
    
    return structured_explanation, 1


def _extract_structured_explanation_fast(data, predictions, probabilities, mega_explainer, conversation):
    """Fast LIME explanation for single instance.
    
    Uses optimized fast_explain_instance method for better performance.
    
    Returns:
        dict: Structured explanation with prediction and feature importance
    """
    patient_data = data.iloc[0]
    patient_features = dict(patient_data)
    
    # Extract prediction details
    prediction = int(predictions[0])
    prediction_class = conversation.class_names.get(prediction, str(prediction))
    
    # Calculate confidence scores
    confidence = None
    if probabilities is not None and len(probabilities) > 0:
        confidence = round(max(probabilities[0]) * 100, 2)
        prob_scores = {
            'No Diabetes': round(probabilities[0][0] * 100, 1),
            'Diabetes': round(probabilities[0][1] * 100, 1)
        }
    else:
        prob_scores = None
    
    # Generate LIME feature importance using fast method
    feature_importance = []
    try:
        explanation = mega_explainer.fast_explain_instance(data.iloc[0:1])
        
        # Sort by absolute influence strength
        feature_scores = explanation.list_exp
        feature_scores_sorted = sorted(feature_scores, key=lambda x: abs(x[1]), reverse=True)
        
        for feature_name, influence_score in feature_scores_sorted:
            feature_importance.append({
                'feature': feature_name,
                'influence': round(influence_score, 3),
                'direction': 'positive' if influence_score >= 0 else 'negative'
            })
    except Exception as e:
        feature_importance = [{'error': f'Could not retrieve LIME scores: {str(e)}'}]
    
    return {
        'type': 'single_explanation',
        'instance_id': data.index[0],
        'prediction': prediction,
        'prediction_class': prediction_class,
        'confidence': confidence,
        'probability_scores': prob_scores,
        'patient_features': patient_features,
        'feature_importance': feature_importance,
        'explanation_method': 'LIME (Fast)'
    }


def _extract_structured_explanation(data, predictions, probabilities, mega_explainer, conversation):
    """Extract structured explanations for single or multiple instances.
    
    Returns:
        dict: Structured explanation data for formatting
    """
    
    if len(data) == 1:
        # Single instance - standard LIME explanation
        patient_data = data.iloc[0]
        patient_features = dict(patient_data)
        
        prediction = int(predictions[0])
        prediction_class = conversation.class_names.get(prediction, str(prediction))
        
        # Calculate confidence scores
        confidence = None
        if probabilities is not None and len(probabilities) > 0:
            confidence = round(max(probabilities[0]) * 100, 2)
            prob_scores = {
                'No Diabetes': round(probabilities[0][0] * 100, 1),
                'Diabetes': round(probabilities[0][1] * 100, 1)
            }
        else:
            prob_scores = None
        
        # Generate LIME feature importance
        feature_importance = []
        try:
            ids = list(data.index)
            explanations = mega_explainer.get_explanations(ids, data, ids_to_regenerate=[])
            
            if explanations and len(explanations) > 0:
                first_id = ids[0]
                explanation = explanations[first_id]
                
                # Sort by absolute influence
                feature_scores = explanation.list_exp
                feature_scores_sorted = sorted(feature_scores, key=lambda x: abs(x[1]), reverse=True)
                
                for feature_name, influence_score in feature_scores_sorted:
                    feature_importance.append({
                        'feature': feature_name,
                        'influence': round(influence_score, 3),
                        'direction': 'positive' if influence_score >= 0 else 'negative'
                    })
        except Exception as e:
            feature_importance = [{'error': f'Could not retrieve LIME scores: {str(e)}'}]
        
        return {
            'type': 'single_explanation',
            'instance_id': data.index[0],
            'prediction': prediction,
            'prediction_class': prediction_class,
            'confidence': confidence,
            'probability_scores': prob_scores,
            'patient_features': patient_features,
            'feature_importance': feature_importance,
            'explanation_method': 'LIME'
        }
    else:
        # Multiple instances - summary analysis
        prediction_summary = {}
        for pred_class in [0, 1]:
            class_name = conversation.class_names.get(pred_class, str(pred_class))
            count = sum(predictions == pred_class)
            percentage = round(count / len(predictions) * 100, 1)
            prediction_summary[class_name] = {'count': count, 'percentage': percentage}
        
        # Extract patterns across instances
        patterns = _analyze_patterns_structured(data, predictions)
        
        return {
            'type': 'multiple_explanations',
            'total_instances': len(data),
            'prediction_summary': prediction_summary,
            'patterns': patterns,
            'explanation_method': 'LIME'
        }


def _analyze_patterns_structured(data, predictions):
    """Analyze feature patterns across multiple instances.
    
    Args:
        data: DataFrame with patient features
        predictions: Array of model predictions
        
    Returns:
        list: Pattern analysis for diabetic vs non-diabetic groups
    """
    
    patterns = []
    
    # Compare age patterns between predicted groups
    if 'Age' in data.columns:
        diabetic_ages = data[predictions == 1]['Age'] if any(predictions == 1) else []
        non_diabetic_ages = data[predictions == 0]['Age'] if any(predictions == 0) else []
        
        if len(diabetic_ages) > 0 and len(non_diabetic_ages) > 0:
            avg_diabetic_age = round(diabetic_ages.mean(), 1)
            avg_non_diabetic_age = round(non_diabetic_ages.mean(), 1)
            
            patterns.append({
                'feature': 'Age',
                'pattern': f'Average age for diabetes predictions: {avg_diabetic_age} vs non-diabetes: {avg_non_diabetic_age}',
                'trend': 'higher_age_diabetes_risk' if avg_diabetic_age > avg_non_diabetic_age + 2 else 'no_clear_trend'
            })
    
    # Compare glucose patterns between predicted groups
    if 'Glucose' in data.columns:
        diabetic_glucose = data[predictions == 1]['Glucose'] if any(predictions == 1) else []
        non_diabetic_glucose = data[predictions == 0]['Glucose'] if any(predictions == 0) else []
        
        if len(diabetic_glucose) > 0 and len(non_diabetic_glucose) > 0:
            avg_diabetic_glucose = round(diabetic_glucose.mean(), 1)
            avg_non_diabetic_glucose = round(non_diabetic_glucose.mean(), 1)
            
            patterns.append({
                'feature': 'Glucose',
                'pattern': f'Average glucose for diabetes predictions: {avg_diabetic_glucose} vs non-diabetes: {avg_non_diabetic_glucose}',
                'trend': 'higher_glucose_diabetes_risk' if avg_diabetic_glucose > avg_non_diabetic_glucose + 10 else 'no_clear_trend'
            })
    
    return patterns

