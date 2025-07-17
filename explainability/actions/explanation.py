"""Explanation action.

This action controls the explanation generation operations.
"""


def _is_followup_question(parse_text, conversation):
    """Check if the parse_text is actually a follow-up question that should be handled differently."""
    
    if not isinstance(parse_text, str):
        return False
    
    text_lower = parse_text.lower()
    
    # Patterns that indicate this is a follow-up analytical question, not a request for new explanations
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
    """The explanation operation."""
    
    # EARLY EXIT: Check if this is a follow-up question that doesn't need new explanations
    if parse_text and _is_followup_question(parse_text, conversation):
        # Delegate to followup action instead of running expensive explanations
        from explainability.actions.followup import followup_operation
        return followup_operation(conversation, parse_text, i, **kwargs)
    
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return {'type': 'error', 'message': 'There are no instances that meet this description!'}, 0

    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    model = conversation.get_var('model').contents
    
    # Get predictions and probabilities
    from app.core import _safe_model_predict
    predictions = _safe_model_predict(model, data)
    try:
        probabilities = model.predict_proba(data)
    except:
        probabilities = None

    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    
    # PERFORMANCE OPTIMIZATION: Use fast path for single instances
    if len(data) == 1:
        # Skip expensive summarize_explanations for single instances
        # Use fast explanation method instead
        structured_explanation = _extract_structured_explanation_fast(data, predictions, probabilities, 
                                                                     mega_explainer_exp, conversation)
    else:
        # Use full explanation pipeline for multiple instances
        full_summary, short_summary = mega_explainer_exp.summarize_explanations(data,
                                                                                filtering_text="",
                                                                                ids_to_regenerate=regen)
        # Store full summary for follow-up
        conversation.store_followup_desc(full_summary)
        
        structured_explanation = _extract_structured_explanation(data, predictions, probabilities, 
                                                                mega_explainer_exp, conversation)
    
    return structured_explanation, 1


def _extract_structured_explanation_fast(data, predictions, probabilities, mega_explainer, conversation):
    """Fast extraction for single instance using optimized explanation method."""
    # Single instance explanation using fast path
    patient_data = data.iloc[0]
    patient_features = dict(patient_data)
    
    # Get prediction info
    prediction = int(predictions[0])
    prediction_class = conversation.class_names.get(prediction, str(prediction))
    
    # Get confidence if available
    confidence = None
    if probabilities is not None and len(probabilities) > 0:
        confidence = round(max(probabilities[0]) * 100, 2)
        prob_scores = {
            'No Diabetes': round(probabilities[0][0] * 100, 1),
            'Diabetes': round(probabilities[0][1] * 100, 1)
        }
    else:
        prob_scores = None
    
    # Get LIME feature importance using FAST method
    feature_importance = []
    try:
        # Use fast_explain_instance for single instances - much faster!
        explanation = mega_explainer.fast_explain_instance(data.iloc[0:1])
        
        # Sort features by absolute importance
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
    """Extract structured data from LIME explanations."""
    
    if len(data) == 1:
        # Single instance explanation
        patient_data = data.iloc[0]
        patient_features = dict(patient_data)
        
        # Get prediction info
        prediction = int(predictions[0])
        prediction_class = conversation.class_names.get(prediction, str(prediction))
        
        # Get confidence if available
        confidence = None
        if probabilities is not None and len(probabilities) > 0:
            confidence = round(max(probabilities[0]) * 100, 2)
            prob_scores = {
                'No Diabetes': round(probabilities[0][0] * 100, 1),
                'Diabetes': round(probabilities[0][1] * 100, 1)
            }
        else:
            prob_scores = None
        
        # Get LIME feature importance
        feature_importance = []
        try:
            ids = list(data.index)
            explanations = mega_explainer.get_explanations(ids, data, ids_to_regenerate=[])
            
            if explanations and len(explanations) > 0:
                first_id = ids[0]
                explanation = explanations[first_id]
                
                # Sort features by absolute importance
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
        # Multiple instances explanation
        prediction_summary = {}
        for pred_class in [0, 1]:
            class_name = conversation.class_names.get(pred_class, str(pred_class))
            count = sum(predictions == pred_class)
            percentage = round(count / len(predictions) * 100, 1)
            prediction_summary[class_name] = {'count': count, 'percentage': percentage}
        
        # Analyze patterns
        patterns = _analyze_patterns_structured(data, predictions)
        
        return {
            'type': 'multiple_explanations',
            'total_instances': len(data),
            'prediction_summary': prediction_summary,
            'patterns': patterns,
            'explanation_method': 'LIME'
        }


def _analyze_patterns_structured(data, predictions):
    """Analyze patterns across multiple instances and return structured data."""
    
    patterns = []
    
    # Analyze age patterns if available
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
    
    # Analyze glucose patterns if available
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

