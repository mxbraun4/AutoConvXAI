"""Shows the feature statistics"""
from copy import deepcopy


import numpy as np


def compute_stats(df, labels, f, conversation):
    """Computes the feature stats"""
    if f == "target":
        labels = deepcopy(labels).to_numpy()
        label_stats = {}
        for label in conversation.class_names:
            freq = np.count_nonzero(label == labels) / len(labels)
            r_freq = round(freq*100, conversation.rounding_precision)
            name = conversation.get_class_name_from_label(label)
            label_stats[name] = r_freq
        return {'type': 'categorical', 'distribution': label_stats}
    else:
        feature = df[f]
        return {
            'type': 'numerical',
            'mean': round(feature.mean(), conversation.rounding_precision),
            'std': round(feature.std(), conversation.rounding_precision),
            'min': round(feature.min(), conversation.rounding_precision),
            'max': round(feature.max(), conversation.rounding_precision)
        }



def feature_stats(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows the feature stats."""
    data = conversation.temp_dataset.contents['X']
    label = conversation.temp_dataset.contents['y']
    
    if i+1 >= len(parse_text):
        return {'type': 'error', 'message': 'No feature name specified for statistics.'}, 0
    
    feature_name = parse_text[i+1]

    if len(data) == 1:
        value = data[feature_name].item()
        return {
            'type': 'single_value',
            'feature_name': feature_name,
            'value': value,
            'data_size': 1
        }, 1

    # Compute feature statistics
    stats = compute_stats(data, label, feature_name, conversation)
    
    return {
        'type': 'feature_statistics',
        'feature_name': feature_name if feature_name != 'target' else 'labels',
        'statistics': stats,
        'data_size': len(data)
    }, 1
