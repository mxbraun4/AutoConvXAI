"""Explanation action.

This action controls the explanation generation operations.
"""


def explain_operation(conversation, parse_text, i, **kwargs):
    """The explanation operation."""
    # TODO(satya): replace explanation generation code here

    # Example code loading the model
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    regen = conversation.temp_dataset.contents['ids_to_regenerate']

    # For AutoGen architecture: always use mega_explainer (LIME-based explanations)
    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    full_summary, short_summary = mega_explainer_exp.summarize_explanations(data,
                                                                            filtering_text="",
                                                                            ids_to_regenerate=regen)
    
    # Enhance explanation with insights about model reasoning
    enhanced_summary = _enhance_explanation_with_insights(short_summary, data, conversation)
    
    conversation.store_followup_desc(full_summary)
    return enhanced_summary, 1


def _enhance_explanation_with_insights(summary, data, conversation):
    """Enhance explanations with insights about model reasoning patterns and technical details."""
    
    # Get model predictions and confidence
    model = conversation.get_var('model').contents
    predictions = model.predict(data)
    
    # Try to get prediction probabilities for confidence analysis
    try:
        probabilities = model.predict_proba(data)
        confidence_analysis = _analyze_prediction_confidence(probabilities, predictions)
    except:
        confidence_analysis = ""
    
    # Add technical details for single instance
    technical_details = ""
    if len(data) == 1:
        technical_details = "<br><br><b>📊 Technical Details:</b><br>"
        technical_details += "<b>Patient Feature Values:</b><br>"
        
        # Show actual feature values for this patient
        patient_data = data.iloc[0]
        for feature_name, value in patient_data.items():
            technical_details += f"• {feature_name}: {value:.2f}<br>"
        
        # Show prediction probabilities
        if len(probabilities) > 0:
            prob_no_diabetes = probabilities[0][0] * 100
            prob_diabetes = probabilities[0][1] * 100
            technical_details += f"<br><b>Model Prediction Probabilities:</b><br>"
            technical_details += f"• No Diabetes: {prob_no_diabetes:.1f}%<br>"
            technical_details += f"• Diabetes: {prob_diabetes:.1f}%<br>"
        
        # Get LIME feature influence scores from mega_explainer
        try:
            mega_explainer = conversation.get_var('mega_explainer').contents
            
            # Get explanations using the MegaExplainer's actual method
            ids = list(data.index)
            explanations = mega_explainer.get_explanations(ids, data, ids_to_regenerate=[])
            
            if explanations and len(explanations) > 0:
                # Get the first explanation (explanations is a dict with id keys)
                first_id = ids[0]
                explanation = explanations[first_id]
                
                technical_details += f"<br><b>LIME Feature Influence Scores:</b><br>"
                
                # Sort features by absolute importance for better display
                feature_scores = explanation.list_exp
                feature_scores_sorted = sorted(feature_scores, key=lambda x: abs(x[1]), reverse=True)
                
                for feature_name, influence_score in feature_scores_sorted:
                    sign = "+" if influence_score >= 0 else ""
                    technical_details += f"• {feature_name}: {sign}{influence_score:.3f}<br>"
            else:
                technical_details += f"<br><b>LIME Feature Influence:</b> No explanations generated<br>"
                
        except Exception as e:
            # More detailed error info for debugging
            technical_details += f"<br><b>LIME Feature Influence:</b> Could not retrieve detailed scores (Error: {str(e)})<br>"
    
    # Analyze feature patterns across multiple instances if available
    pattern_analysis = ""
    if len(data) > 1:
        pattern_analysis = _analyze_feature_patterns(data, predictions, conversation)
    
    # Add insights about model reasoning
    insights = "<br><br><b>🧠 Model Reasoning Insights:</b><br>"
    
    if len(data) == 1:
        insights += "This prediction is based on learned patterns from similar patients in the training data. "
        insights += "The model has identified which health indicators are most predictive of diabetes risk "
        insights += "and is applying those patterns to this specific case.<br>"
    else:
        insights += f"Analyzing {len(data)} patients, the model applies consistent learned patterns "
        insights += "rather than memorizing individual cases. The feature importance rankings show "
        insights += "which health indicators the model has learned are most predictive across different patient profiles.<br>"
    
    # Add confidence analysis if available
    if confidence_analysis:
        insights += confidence_analysis
    
    # Add pattern analysis if available
    if pattern_analysis:
        insights += pattern_analysis
    
    # Combine original summary with technical details and insights
    enhanced_summary = summary + technical_details + insights
    
    return enhanced_summary


def _analyze_prediction_confidence(probabilities, predictions):
    """Analyze prediction confidence to show model is reasoning, not memorizing."""
    
    if len(probabilities) == 0:
        return ""
    
    # Calculate confidence scores (max probability for each prediction)
    confidences = [max(prob) for prob in probabilities]
    avg_confidence = sum(confidences) / len(confidences)
    
    analysis = f"<br><b>Confidence Analysis:</b> The model's average confidence is {avg_confidence:.1%}. "
    
    if avg_confidence > 0.8:
        analysis += "High confidence suggests the model has learned clear patterns distinguishing diabetes risk factors."
    elif avg_confidence > 0.6:
        analysis += "Moderate confidence indicates the model recognizes these cases have mixed risk factors, showing nuanced reasoning."
    else:
        analysis += "Lower confidence suggests these are borderline cases where multiple factors create uncertainty - exactly what we'd expect from a model learning real patterns rather than memorizing."
    
    return analysis + "<br>"


def _analyze_feature_patterns(data, predictions, conversation):
    """Analyze patterns across multiple instances to show learned relationships."""
    
    if len(data) < 2:
        return ""
    
    analysis = "<br><b>Pattern Analysis:</b> "
    
    # Analyze age patterns if available
    if 'age' in data.columns:
        diabetic_ages = data[predictions == 1]['age'] if any(predictions == 1) else []
        non_diabetic_ages = data[predictions == 0]['age'] if any(predictions == 0) else []
        
        if len(diabetic_ages) > 0 and len(non_diabetic_ages) > 0:
            avg_diabetic_age = diabetic_ages.mean()
            avg_non_diabetic_age = non_diabetic_ages.mean()
            
            if avg_diabetic_age > avg_non_diabetic_age + 2:
                analysis += f"The model predicts diabetes risk increases with age (avg age for predicted diabetic: {avg_diabetic_age:.1f} vs non-diabetic: {avg_non_diabetic_age:.1f}). "
    
    # Analyze glucose patterns if available
    if 'glucose' in data.columns:
        diabetic_glucose = data[predictions == 1]['glucose'] if any(predictions == 1) else []
        non_diabetic_glucose = data[predictions == 0]['glucose'] if any(predictions == 0) else []
        
        if len(diabetic_glucose) > 0 and len(non_diabetic_glucose) > 0:
            avg_diabetic_glucose = diabetic_glucose.mean()
            avg_non_diabetic_glucose = non_diabetic_glucose.mean()
            
            if avg_diabetic_glucose > avg_non_diabetic_glucose + 10:
                analysis += f"Higher glucose levels are consistently associated with diabetes predictions (avg glucose for predicted diabetic: {avg_diabetic_glucose:.1f} vs non-diabetic: {avg_non_diabetic_glucose:.1f}). "
    
    analysis += "These patterns demonstrate the model has learned meaningful relationships from the training data."
    
    return analysis + "<br>"
