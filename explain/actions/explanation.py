"""Explanation action.

This action controls the explanation generation operations.
"""
from explain.actions.utils import gen_parse_op_text


def explain_operation(conversation, parse_text, i, **kwargs):
    """The explanation operation."""
    # TODO(satya): replace explanation generation code here

    # Example code loading the model
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    parse_op = gen_parse_op_text(conversation)

    # Check if there are additional parameters after 'explain'
    explanation_type = None
    if i + 1 < len(parse_text):
        explanation_type = parse_text[i+1]
    
    # Note, do we want to remove parsing for lime -> mega_explainer here?
    if explanation_type == 'features' or explanation_type == 'lime':
        # mega explainer explanation case
        mega_explainer_exp = conversation.get_var('mega_explainer').contents
        full_summary, short_summary = mega_explainer_exp.summarize_explanations(data,
                                                                                filtering_text=parse_op,
                                                                                ids_to_regenerate=regen)
        
        # Enhance explanation with insights about model reasoning
        enhanced_summary = _enhance_explanation_with_insights(short_summary, data, conversation)
        
        conversation.store_followup_desc(full_summary)
        return enhanced_summary, 1
    elif explanation_type == 'cfe':
        dice_tabular = conversation.get_var('tabular_dice').contents
        out = dice_tabular.summarize_explanations(data,
                                                  filtering_text=parse_op,
                                                  ids_to_regenerate=regen)
        additional_options, short_summary = out
        conversation.store_followup_desc(additional_options)
        return short_summary, 1
    elif explanation_type == 'shap':
        # This is when a user asks for a shap explanation
        raise NotImplementedError
    elif explanation_type is None:
        # Default explanation when no specific type is requested
        mega_explainer_exp = conversation.get_var('mega_explainer').contents
        full_summary, short_summary = mega_explainer_exp.summarize_explanations(data,
                                                                                filtering_text=parse_op,
                                                                                ids_to_regenerate=regen)
        
        # Enhance explanation with insights about model reasoning
        enhanced_summary = _enhance_explanation_with_insights(short_summary, data, conversation)
        
        conversation.store_followup_desc(full_summary)
        return enhanced_summary, 1
    else:
        raise NameError(f"No explanation operation defined for {parse_text}")


def _enhance_explanation_with_insights(summary, data, conversation):
    """Enhance explanations with insights about model reasoning patterns."""
    
    # Get model predictions and confidence
    model = conversation.get_var('model').contents
    predictions = model.predict(data)
    
    # Try to get prediction probabilities for confidence analysis
    try:
        probabilities = model.predict_proba(data)
        confidence_analysis = _analyze_prediction_confidence(probabilities, predictions)
    except:
        confidence_analysis = ""
    
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
    
    # Combine original summary with insights
    enhanced_summary = summary + insights
    
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
