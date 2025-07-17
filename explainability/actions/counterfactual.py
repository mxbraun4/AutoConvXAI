"""Counterfactual action.

This action controls the counterfactual explanation generation operations using DICE-ML.
"""


def counterfactual_operation(conversation, parse_text, i, **kwargs):
    """Generate counterfactual explanations using TabularDice."""
    data = conversation.temp_dataset.contents['X']

    if len(data) == 0:
        return {'type': 'error', 'message': 'There are no instances that meet this description!'}, 0
    
    if len(data) > 1:
        return {'type': 'error', 'message': 'Counterfactual explanations can only be generated for single instances. Please narrow down your selection to a single instance.'}, 0

    # Get the TabularDice explainer
    dice_explainer = conversation.get_var('tabular_dice').contents
    model = conversation.get_var('model').contents
    
    # Get predictions for the current instance
    from app.core import _safe_model_predict
    predictions = _safe_model_predict(model, data)
    original_prediction = predictions[0]
    original_class = conversation.class_names.get(original_prediction, str(original_prediction))
    
    try:
        # Generate counterfactuals
        cfes = dice_explainer.run_explanation(data)
        instance_id = list(data.index)[0]
        cfe = cfes[instance_id]
        
        # Get the counterfactual examples
        final_cfes = cfe.cf_examples_list[0].final_cfs_df
        
        if dice_explainer.temp_outcome_name in final_cfes.columns:
            final_cfes.pop(dice_explainer.temp_outcome_name)
        
        # Get predictions for counterfactuals
        new_predictions = dice_explainer.model.predict(final_cfes)
        
        # Build structured counterfactual data
        counterfactuals = []
        original_instance = data.iloc[0]
        
        for i, cf_id in enumerate(final_cfes.index):
            cf_instance = final_cfes.loc[cf_id]
            cf_prediction = new_predictions[i]
            cf_class = conversation.class_names.get(cf_prediction, str(cf_prediction))
            
            # Get the changes needed
            changes = []
            for feature in cf_instance.index:
                orig_val = original_instance[feature]
                cf_val = cf_instance[feature]
                
                if isinstance(cf_val, str):
                    cf_val = float(cf_val)
                
                if orig_val != cf_val:
                    change_direction = "increase" if cf_val > orig_val else "decrease"
                    changes.append({
                        'feature': feature,
                        'original_value': round(orig_val, 2),
                        'counterfactual_value': round(cf_val, 2),
                        'change_direction': change_direction,
                        'change_amount': round(abs(cf_val - orig_val), 2)
                    })
            
            counterfactuals.append({
                'counterfactual_id': i + 1,
                'new_prediction': cf_prediction,
                'new_prediction_class': cf_class,
                'changes_required': changes,
                'counterfactual_features': dict(cf_instance)
            })
        
        # Generate summary using TabularDice's built-in method
        full_summary, short_summary = dice_explainer.summarize_explanations(data)
        conversation.store_followup_desc(full_summary)
        
        return {
            'type': 'counterfactual_explanation',
            'instance_id': instance_id,
            'original_prediction': original_prediction,
            'original_prediction_class': original_class,
            'original_features': dict(original_instance),
            'counterfactuals': counterfactuals,
            'total_counterfactuals': len(counterfactuals),
            'explanation_method': 'DICE-ML',
            'summary': short_summary
        }, 1
        
    except Exception as e:
        return {'type': 'error', 'message': f'Error generating counterfactuals: {str(e)}'}, 0


def alternatives_operation(conversation, parse_text, i, **kwargs):
    """Alias for counterfactual_operation to handle 'alternatives' queries."""
    return counterfactual_operation(conversation, parse_text, i, **kwargs)


def scenarios_operation(conversation, parse_text, i, **kwargs):
    """Alias for counterfactual_operation to handle 'scenarios' queries."""
    return counterfactual_operation(conversation, parse_text, i, **kwargs)