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

    Arguments:
        conversation: The conversation object
        parse_text: The parse text for the question
        i: Index in the parse
        **kwargs: additional kwargs
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

    # Figure out which features to use by taking top k features
    ids = list(data.index)
    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    explanations = mega_explainer_exp.get_explanations(ids,
                                                       data,
                                                       ids_to_regenerate=regen,
                                                       save_to_cache=False)

    # Store the feature importance arrays
    feature_importances = []
    for current_id in ids:
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
            i_score = interaction_explainer.feature_interaction(f1,
                                                                f2,
                                                                number_sub_samples=NUM_TO_SAMPLE)
            interactions.append(i_score)
            feature_pairs.append((f1, f2))

    # Sort interactions by strength
    sorted_indices = np.argsort(interactions)[::-1]  # Descending order
    
    # Create structured response
    result = {
        'type': 'interaction_effects',
        'filter_applied': len(parse_op) > 0,
        'filter_description': parse_op,
        'total_instances': len(ids),
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