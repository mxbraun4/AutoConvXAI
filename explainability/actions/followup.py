"""Followup operation."""
import re


def followup_operation(conversation, parse_text, i, **kwargs):
    """Follow up operation.

    If there's an explicit option to followup, this command deals with it.
    Enhanced to handle analytical follow-ups using conversation context.
    """
    # Check for analytical follow-up patterns first
    if parse_text and isinstance(parse_text, str):
        analytical_response = _handle_analytical_followup(conversation, parse_text)
        if analytical_response:
            return analytical_response, 1
    
    # Fall back to traditional followup handling
    follow_up_text = conversation.get_followup_desc()
    if follow_up_text == "":
        return "Sorry, I'm a bit unsure what you mean... try again?", 0
    else:
        return follow_up_text, 1


def _handle_analytical_followup(conversation, parse_text):
    """Handle analytical follow-up questions using conversation context."""
    
    # Convert to lowercase for pattern matching
    text_lower = parse_text.lower()
    
    # Pattern: Model underprediction/overprediction analysis
    if any(pattern in text_lower for pattern in [
        "underpredicts", "overpredicts", "underestimates", "overestimates",
        "model predicts less", "model predicts more", "conservative", "aggressive"
    ]):
        return _analyze_prediction_bias(conversation, text_lower)
    
    # Pattern: General model performance conclusions
    if any(pattern in text_lower for pattern in [
        "this means the model", "so the model", "the model seems",
        "does this mean", "therefore the model", "so it appears"
    ]):
        return _analyze_model_conclusions(conversation, text_lower)
    
    # Pattern: Accuracy or performance questions
    if any(pattern in text_lower for pattern in [
        "how accurate", "is this good", "is this bad", "what does this mean"
    ]):
        return _analyze_performance_context(conversation, text_lower)
    
    return None


def _analyze_prediction_bias(conversation, text_lower):
    """Analyze prediction bias from conversation context."""
    
    # Try to get ground truth and prediction counts from context
    try:
        # Check if we have recent results about ground truth vs predictions
        dataset = conversation.get_var('dataset')
        if dataset and hasattr(dataset, 'contents'):
            # Get ground truth labels
            y_true = dataset.contents.get('y', [])
            if y_true:
                ground_truth_positive = sum(y_true)
                total_instances = len(y_true)
                
                # Get model predictions if available
                model = conversation.get_var('model')
                if model and hasattr(dataset, 'contents'):
                    X = dataset.contents.get('X', [])
                    if X:
                        predictions = model.predict(X)
                        predicted_positive = sum(predictions)
                        
                        # Compare ground truth vs predictions
                        gt_percentage = (ground_truth_positive / total_instances) * 100
                        pred_percentage = (predicted_positive / total_instances) * 100
                        
                        if "underpredict" in text_lower or "underestimate" in text_lower:
                            if predicted_positive < ground_truth_positive:
                                return (f"Yes, you're correct. The model underpredicts diabetes cases. "
                                       f"Ground truth shows {ground_truth_positive} cases ({gt_percentage:.1f}%) "
                                       f"but the model only predicts {predicted_positive} cases ({pred_percentage:.1f}%). "
                                       f"This suggests the model is conservative and may miss some positive cases.")
                            else:
                                return (f"Actually, the model doesn't underpredict. "
                                       f"Ground truth: {ground_truth_positive} cases ({gt_percentage:.1f}%), "
                                       f"Predictions: {predicted_positive} cases ({pred_percentage:.1f}%).")
                        
                        elif "overpredict" in text_lower or "overestimate" in text_lower:
                            if predicted_positive > ground_truth_positive:
                                return (f"Yes, the model overpredicts diabetes cases. "
                                       f"Ground truth shows {ground_truth_positive} cases ({gt_percentage:.1f}%) "
                                       f"but the model predicts {predicted_positive} cases ({pred_percentage:.1f}%). "
                                       f"This suggests the model is aggressive and may have false positives.")
                            else:
                                return (f"Actually, the model doesn't overpredict. "
                                       f"Ground truth: {ground_truth_positive} cases ({gt_percentage:.1f}%), "
                                       f"Predictions: {predicted_positive} cases ({pred_percentage:.1f}%).")
                        
    except Exception as e:
        # If we can't get the data, provide a generic response
        pass
    
    # Fallback for underprediction/overprediction questions
    if "underpredict" in text_lower:
        return ("Based on the previous results, if the model predicts fewer positive cases than the ground truth, "
                "then yes, it would be underpredicting. This could mean the model is conservative and might miss some cases.")
    elif "overpredict" in text_lower:
        return ("Based on the previous results, if the model predicts more positive cases than the ground truth, "
                "then yes, it would be overpredicting. This could mean the model is aggressive and might have false positives.")
    
    return None


def _analyze_model_conclusions(conversation, text_lower):
    """Analyze general model conclusions from context."""
    
    # Generic responses for model analysis conclusions
    if "conservative" in text_lower:
        return ("A conservative model tends to predict fewer positive cases to avoid false positives. "
                "This can be good for minimizing false alarms but might miss some true cases.")
    elif "aggressive" in text_lower:
        return ("An aggressive model tends to predict more positive cases to avoid missing true cases. "
                "This can be good for catching all cases but might have more false positives.")
    elif "biased" in text_lower:
        return ("Model bias can occur when the training data or model architecture systematically "
                "favors certain outcomes. This should be evaluated by comparing predictions to ground truth.")
    
    return ("I can help analyze model behavior based on the data we've seen. "
            "What specific aspect of the model's performance would you like to understand?")


def _analyze_performance_context(conversation, text_lower):
    """Analyze performance-related questions using context."""
    
    # Generic performance analysis
    if "accurate" in text_lower:
        return ("Model accuracy depends on the specific use case and domain. "
                "For medical diagnosis, we typically want high sensitivity (catching all cases) "
                "and high specificity (avoiding false positives). The balance depends on the clinical context.")
    elif "good" in text_lower or "bad" in text_lower:
        return ("Model performance should be evaluated in context. Consider factors like: "
                "1) The cost of false positives vs false negatives, "
                "2) The base rate of the condition, "
                "3) How the model will be used in practice.")
    
    return ("I can help interpret model performance metrics. "
            "What specific aspect would you like me to explain?")
