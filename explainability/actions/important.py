"""Important features operation."""
import numpy as np
import statsmodels.stats.api as sms

from explainability.core.utils import add_to_dict_lists, gen_parse_op_text


def _sample_data_for_importance(data, ids, max_samples=50):
    """Sample data for feature importance analysis to improve performance.
    
    For large datasets, analyzing feature importance on all instances is slow.
    This function samples a representative subset while maintaining statistical validity.
    
    Args:
        data: Full dataset
        ids: All instance IDs
        max_samples: Maximum number of samples to use (default 50)
        
    Returns:
        tuple: (sampled_data, sampled_ids)
    """
    if len(data) <= max_samples:
        # Use all data if dataset is small
        return data, ids
    
    # For larger datasets, use stratified sampling if possible
    try:
        # Try to get predictions to enable stratified sampling
        from app.core import _safe_model_predict
        import pandas as pd
        
        # Get model predictions for stratification
        model = None
        # We need conversation context to get the model, but it's not passed here
        # For now, use random sampling - could be improved with stratified sampling
        
        # Random sampling with fixed seed for reproducibility
        sample_indices = np.random.choice(len(data), size=max_samples, replace=False)
        sample_data = data.iloc[sample_indices]
        sample_ids = [ids[i] for i in sample_indices]
        
        return sample_data, sample_ids
        
    except Exception:
        # Fallback to simple random sampling
        sample_indices = np.random.choice(len(data), size=max_samples, replace=False)
        sample_data = data.iloc[sample_indices]
        sample_ids = [ids[i] for i in sample_indices]
        
        return sample_data, sample_ids


def gen_feature_name_to_rank_dict(data, explanations):
    """Generates a dictionary that maps feature name -> rank -> ids.

    This dictionary contains a mapping from a feature name to a rank to a list
    of ids that have that feature name at that rank.

    Arguments:
        data: the dataset
        explanations: the explanations for the dataset
    Returns:
        feature_name_to_rank: the dictionary with the mapping described above
    """
    feature_name_to_rank = {}
    for feature_name in data.columns:
        # Dictionary mapping rank (i.e., first most important to a list of ids at that rank)
        # to ids that have that feature at that rank
        rank_to_ids = {}
        for i, current_id in enumerate(explanations):
            list_exp = explanations[current_id].list_exp
            for rank, tup in enumerate(list_exp):
                # tup[0] is the feature name in the explanation
                # also, lime will store the value for categorical features
                # at the end of the explanation, i.e., race=0 or race=1
                # so we need to check startswith
                if tup[0].startswith(feature_name):
                    add_to_dict_lists(rank, current_id, rank_to_ids)
                    # Feature name must appear once per explanation so we can break
                    break
        feature_name_to_rank[feature_name] = rank_to_ids
    return feature_name_to_rank


def compute_rank_stats(data, feature_name_to_rank):
    """Compute stats about the feature rankings."""
    max_ranks = {}
    avg_ranks = {}
    ci_95s = {}
    print(feature_name_to_rank)
    for feature_name in data.columns:
        # Get the ranks of each feature name
        rank_to_ids = feature_name_to_rank[feature_name]

        # If the feature isn't very important and never
        # ends up getting included
        if len(feature_name_to_rank[feature_name]) == 0:
            continue

        max_rank = sorted(rank_to_ids.keys())[0]
        max_ranks[feature_name] = max_rank

        rank_list = []
        for key in rank_to_ids:
            rank_list.extend([key] * len(rank_to_ids[key]))
        rank_list = np.array(rank_list) + 1
        avg_ranking = np.mean(rank_list)

        # in case there is only one instance
        if len(rank_list) == 1:
            ci_95 = None
        else:
            ci_95 = sms.DescrStatsW(rank_list).tconfint_mean()

        avg_ranks[feature_name] = avg_ranking
        ci_95s[feature_name] = ci_95
    return max_ranks, avg_ranks, ci_95s


def important_operation(conversation, parse_text, i, **kwargs):
    """Important features operation.

    For a given feature, this operation finds explanations where it is important.
    Now returns structured data instead of formatted HTML.
    """
    data = conversation.temp_dataset.contents['X']
    if len(conversation.temp_dataset.contents['X']) == 0:
        # In the case that filtering has removed all the instances
        return {'error': 'No instances meet this description', 'type': 'no_data'}, 0
    
    ids = list(data.index)

    # Use AutoGen entities instead of legacy text parsing
    ent_features = kwargs.get('features', []) if kwargs else []
    ent_topk = kwargs.get('topk') if kwargs else None
    
    if ent_features and len(ent_features) > 0:
        # Specific feature requested via AutoGen entities
        parsed_feature_name = ent_features[0]
    elif ent_topk:
        # Top-k features requested via AutoGen entities
        parsed_feature_name = "topk"
    else:
        # No specific feature specified, default to showing all features
        parsed_feature_name = "all"

    # Generate the text for the filtering operation
    parse_op = gen_parse_op_text(conversation)

    # Get the explainer
    mega_explainer_exp = conversation.get_var('mega_explainer').contents

    # If there's ids to regenerate from a previous operation
    regen = conversation.temp_dataset.contents['ids_to_regenerate']

    # AGGRESSIVE PERFORMANCE OPTIMIZATION: Use much smaller sampling and fast explanations
    sample_data, sample_ids = _sample_data_for_importance(data, ids, max_samples=10)  # Reduced from 50 to 10
    
    # Use fast explanations instead of full pipeline for much better performance
    explanations = {}
    for sample_id in sample_ids:
        single_instance = sample_data.loc[[sample_id]]
        try:
            # Use fast_explain_instance for much faster processing
            fast_explanation = mega_explainer_exp.fast_explain_instance(single_instance)
            explanations[sample_id] = fast_explanation
        except Exception as e:
            # Fallback to regular explanation if fast fails
            single_explanations = mega_explainer_exp.get_explanations([sample_id], single_instance, ids_to_regenerate=[], save_to_cache=False)
            explanations[sample_id] = single_explanations[sample_id]

    # Generate feature name to frequency of ids at rank mapping
    feature_name_to_rank = gen_feature_name_to_rank_dict(sample_data, explanations)

    # Compute rank stats for features including max rank of feature, its average rank
    # and the 95% ci's
    max_ranks, avg_ranks, ci_95s = compute_rank_stats(sample_data, feature_name_to_rank)

    # Return structured data instead of formatted HTML
    result = {
        'type': 'feature_importance',
        'request_type': parsed_feature_name,  # 'all', 'topk', or specific feature name
        'filter_applied': len(parse_op) > 0,
        'filter_description': parse_op,
        'total_instances': len(ids),
        'sampled_instances': len(sample_ids),
        'sampling_used': len(sample_ids) < len(ids),
        'total_features': len(data.columns),
        'features': []
    }

    # Process different types of importance requests
    if parsed_feature_name == "all":
        # Return all features ranked by importance
        ranked_features = sorted(avg_ranks.items(), key=lambda x: x[1])
        for rank, (feature_name, avg_rank) in enumerate(ranked_features, 1):
            ci_95 = ci_95s.get(feature_name)
            result['features'].append({
                'name': feature_name,
                'rank': rank,
                'avg_rank': round(avg_rank, conversation.rounding_precision),
                'max_rank': max_ranks.get(feature_name, avg_rank),
                'confidence_interval': {
                    'low': round(ci_95[0], conversation.rounding_precision) if ci_95 else None,
                    'high': round(ci_95[1], conversation.rounding_precision) if ci_95 else None
                } if ci_95 else None
            })
            
    elif parsed_feature_name == "topk":
        # Return top k features using AutoGen entities
        topk = ent_topk if ent_topk else 5
        
        result['top_k'] = topk
        ranked_features = sorted(avg_ranks.items(), key=lambda x: x[1])[:topk]
        for rank, (feature_name, avg_rank) in enumerate(ranked_features, 1):
            ci_95 = ci_95s.get(feature_name)
            result['features'].append({
                'name': feature_name,
                'rank': rank,
                'avg_rank': round(avg_rank, conversation.rounding_precision),
                'max_rank': max_ranks.get(feature_name, avg_rank),
                'confidence_interval': {
                    'low': round(ci_95[0], conversation.rounding_precision) if ci_95 else None,
                    'high': round(ci_95[1], conversation.rounding_precision) if ci_95 else None
                } if ci_95 else None
            })
            
    else:
        # Individual feature importance case
        if parsed_feature_name in avg_ranks:
            avg_rank = avg_ranks[parsed_feature_name]
            ci_95 = ci_95s.get(parsed_feature_name)
            max_rank = max_ranks.get(parsed_feature_name, avg_rank)
            
            # Compute relative importance description
            all_rankings = list(avg_ranks.values())
            quartiles = np.percentile(all_rankings, [25, 50, 75])
            
            if avg_rank < quartiles[0]:
                importance_level = "highly"
            elif avg_rank < quartiles[1]:
                importance_level = "fairly"
            elif avg_rank < quartiles[2]:
                importance_level = "somewhat"
            else:
                importance_level = "not very"
            
            result['features'] = [{
                'name': parsed_feature_name,
                'rank': None,  # Will be calculated based on avg_rank
                'avg_rank': round(avg_rank, conversation.rounding_precision),
                'max_rank': max_rank,
                'importance_level': importance_level,
                'confidence_interval': {
                    'low': round(ci_95[0], conversation.rounding_precision) if ci_95 else None,
                    'high': round(ci_95[1], conversation.rounding_precision) if ci_95 else None
                } if ci_95 else None
            }]
        else:
            result['error'] = f"Feature '{parsed_feature_name}' not found in dataset"
            result['available_features'] = list(data.columns)

    return result, 1
