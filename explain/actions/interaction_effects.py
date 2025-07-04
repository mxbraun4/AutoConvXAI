"""Measure interaction effects between features."""
import numpy as np

from explain.core.feature_interaction import FeatureInteraction
from explain.core.utils import gen_parse_op_text

NUM_FEATURES_TO_COMPUTE_INTERACTIONS = 4
NUM_TO_SAMPLE = 40


def measure_interaction_effects(conversation, parse_text, i, **kwargs):
    """Analyze feature interaction effects.

    This function identifies which features work together and how they influence
    model predictions synergistically. Returns structured data about feature interactions.
    
    Can analyze specific feature pairs (from AutoGen entities) or discover top interactions.

    Arguments:
        conversation: The conversation object
        parse_text: The parse text for the question
        i: Index in the parse
        **kwargs: additional kwargs including AutoGen entities (features list)
    """
    
    # Filtering text
    parse_op = gen_parse_op_text(conversation)

    # The filtered dataset
    data = conversation.temp_dataset.contents['X']

    if len(data) == 0:
        return {
            'type': 'interaction_effects',
            'error': 'No instances meet this description',
            'filter_applied': parse_op
        }, 0

    # Probability predicting function
    predict_proba = conversation.get_var('model_prob_predict').contents

    # Categorical features
    cat_features = conversation.get_var('dataset').contents['cat']
    cat_feature_names = [data.columns[i] for i in cat_features]

    interaction_explainer = FeatureInteraction(data=data,
                                               prediction_fn=predict_proba,
                                               cat_features=cat_feature_names)

    # PERFORMANCE OPTIMIZATION: Use sampling for feature selection
    ids = list(data.index)
    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    
    # Sample data for feature importance estimation (much faster)
    sample_size = min(5, len(data))  # Reduced to only 5 instances for speed
    sample_indices = np.random.choice(len(data), size=sample_size, replace=False)
    sample_data = data.iloc[sample_indices]
    sample_ids = [ids[i] for i in sample_indices]
    
    # Use fast explanations for much better performance
    explanations = {}
    for sample_id in sample_ids:
        single_instance = sample_data.loc[[sample_id]]
        try:
            # Use fast_explain_instance for much faster processing
            fast_explanation = mega_explainer_exp.fast_explain_instance(single_instance)
            explanations[sample_id] = fast_explanation
        except Exception as e:
            # Fallback if needed
            single_explanations = mega_explainer_exp.get_explanations([sample_id], single_instance, ids_to_regenerate=[], save_to_cache=False)
            explanations[sample_id] = single_explanations[sample_id]

    # Check if specific features are requested via AutoGen entities
    requested_features = kwargs.get('features', [])
    
    if requested_features and len(requested_features) >= 2:
        # User requested specific features - analyze those
        # Validate that the requested features exist in the dataset
        available_features = list(data.columns)
        valid_features = []
        
        for feature in requested_features:
            # Case-insensitive matching
            matching_feature = None
            for col in available_features:
                if col.lower() == feature.lower():
                    matching_feature = col
                    break
            
            if matching_feature:
                valid_features.append(matching_feature)
            else:
                return {
                    'type': 'interaction_effects',
                    'error': f'Feature "{feature}" not found in dataset. Available features: {available_features}',
                    'filter_applied': parse_op
                }, 0
        
        if len(valid_features) < 2:
            return {
                'type': 'interaction_effects', 
                'error': f'Need at least 2 valid features for interaction analysis. Found: {valid_features}',
                'filter_applied': parse_op
            }, 0
        
        # Analyze the specific feature pair(s)
        interactions = []
        feature_pairs = []
        
        # If exactly 2 features, analyze their interaction
        if len(valid_features) == 2:
            f1, f2 = valid_features[0], valid_features[1]
            try:
                i_score = interaction_explainer.feature_interaction(f1, f2, number_sub_samples=NUM_TO_SAMPLE)
                interactions.append(i_score)
                feature_pairs.append((f1, f2))
            except Exception as e:
                return {
                    'type': 'interaction_effects',
                    'error': f'Failed to analyze interaction between {f1} and {f2}: {str(e)}',
                    'filter_applied': parse_op
                }, 0
        else:
            # If more than 2 features, analyze all pairs
            for idx1, f1 in enumerate(valid_features):
                for idx2 in range(idx1 + 1, len(valid_features)):
                    f2 = valid_features[idx2]
                    try:
                        i_score = interaction_explainer.feature_interaction(f1, f2, number_sub_samples=NUM_TO_SAMPLE)
                        interactions.append(i_score)
                        feature_pairs.append((f1, f2))
                    except Exception as e:
                        # Continue with other pairs if one fails
                        continue
        
    else:
        # No specific features requested - use automatic top feature discovery
        # Store the feature importance arrays from sampled explanations
        feature_importances = []
        for current_id in sample_ids:
            list_exp = explanations[current_id].list_exp
            list_imps = [coef[1] for coef in list_exp]
            feature_importances.append(list_imps)
        
        feature_importances = np.array(feature_importances)
        mean_feature_importances = np.mean(np.abs(feature_importances), axis=0)

        # Get the names of the top features
        topk_features = np.argsort(mean_feature_importances)[-NUM_FEATURES_TO_COMPUTE_INTERACTIONS:]
        topk_names = [data.columns[j] for j in topk_features]

        interactions = []
        feature_pairs = []
        
        for idx1, f1 in enumerate(topk_names):
            for idx2 in range(idx1 + 1, len(topk_names)):
                f2 = topk_names[idx2]
                try:
                    i_score = interaction_explainer.feature_interaction(f1, f2, number_sub_samples=NUM_TO_SAMPLE)
                    interactions.append(i_score)
                    feature_pairs.append((f1, f2))
                except Exception as e:
                    # Continue with other pairs if one fails
                    continue

    # Sort interactions by strength
    sorted_indices = np.argsort(interactions)[::-1]  # Descending order
    
    # Create structured response
    result = {
        'type': 'interaction_effects',
        'filter_applied': len(parse_op) > 0,
        'filter_description': parse_op,
        'total_instances': len(ids),
        'sampled_instances': sample_size,
        'sampling_used': sample_size < len(ids),
        'analyzed_features': topk_names,
        'interactions': []
    }
    
    # Add top interactions
    for idx in sorted_indices:
        f1, f2 = feature_pairs[idx]
        strength = interactions[idx]
        
        result['interactions'].append({
            'feature1': f1,
            'feature2': f2,
            'interaction_strength': round(strength, conversation.rounding_precision),
            'description': f"{f1} and {f2} interaction"
        })
    
    # Add summary statistics
    if interactions:
        result['summary'] = {
            'strongest_interaction': round(max(interactions), conversation.rounding_precision),
            'average_interaction': round(np.mean(interactions), conversation.rounding_precision),
            'total_pairs_analyzed': len(interactions)
        }
    
    return result, 1