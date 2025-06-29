"""Score operation.

This operation computes a score metric on the data or the eval data.
"""
from explain.actions.utils import gen_parse_op_text


def score_operation(conversation, parse_text, i, **kwargs):
    """
    Score operation for model performance evaluation.
    
    This function determines whether to compute performance on:
    1. Filtered data (default - respects user's filtering intent)
    2. Full dataset (only when user explicitly asks for "overall" or "global" performance)
    
    Design Philosophy:
        Simplified logic that respects user intent. Only use full dataset when 
        explicitly requested with keywords like "overall", "global", etc.
        Otherwise, always use the filtered dataset to respect user's filtering intent.
    """

    # Get the name of the metric
    metric = parse_text[i+1] if len(parse_text) > i+1 else "default"

    if metric == "default":
        metric = conversation.default_metric

    model = conversation.get_var('model').contents

    # Only use full dataset if user explicitly asks for overall/global performance
    # Keywords that explicitly indicate global performance requests
    global_keywords = ['overall', 'total', 'global', 'all', 'entire', 'whole', 'complete']
    
    # Check if any global keywords appear in the parse text
    full_parse_string = ' '.join(parse_text).lower()
    use_full_dataset = any(keyword in full_parse_string for keyword in global_keywords)
    
    if use_full_dataset:
        # User explicitly asked for overall/global performance
        full_data = conversation.get_var('dataset').contents['X']
        y_true = conversation.get_var('dataset').contents['y']
        filter_string = ""  # No filtering applied
        is_global_query = True
        data_name = "the <b>entire dataset</b>"
    else:
        # Default: Use filtered dataset (respects user's filtering intent)
        full_data = conversation.temp_dataset.contents['X']
        y_true = conversation.temp_dataset.contents['y']
        filter_string = gen_parse_op_text(conversation)
        is_global_query = False
        
        # Format data description for output
        if len(filter_string) <= 0:
            data_name = "the <b>entire dataset</b>"
        else:
            data_name = f"instances where <b>{filter_string}</b>"

    # -----------------------------------------------
    # NEW: Apply on-the-fly filtering when the query
    # directly specifies feature/operator/value but
    # no prior filter action has been called (e.g.,
    # "accuracy where age > 50"). We inspect kwargs
    # passed through action_args.
    # -----------------------------------------------
    ent_features = kwargs.get('features', []) if kwargs else []
    ent_ops = kwargs.get('operators', []) if kwargs else []
    ent_vals = kwargs.get('values', []) if kwargs else []

    if ent_features and ent_ops and ent_vals:
        try:
            for feat, op, val in zip(ent_features, ent_ops, ent_vals):
                # Handle case-insensitive feature matching
                actual_feat = None
                for col in full_data.columns:
                    if col.lower() == feat.lower():
                        actual_feat = col
                        break
                
                if actual_feat is None:
                    continue  # skip unknown feature
                
                feat = actual_feat  # use the actual column name
                
                # Apply the filter
                if op == '>':
                    mask = full_data[feat] > val
                elif op == '<':
                    mask = full_data[feat] < val
                elif op == '=' or op == '==':
                    mask = full_data[feat] == val
                else:
                    continue
                
                full_data = full_data[mask]
                y_true = y_true[mask]
                
            # Update data description
            filter_parts = [f"{f} {o} {v}" for f, o, v in zip(ent_features, ent_ops, ent_vals)]
            filter_string = " and ".join(filter_parts)
            data_name = f"instances where <b>{filter_string}</b>" if filter_string else data_name
        except Exception:
            pass  # fall back silently if any issue

    # Remove target column if present (models expect only features, not target)
    if 'y' in full_data.columns:
        data = full_data.drop(columns=['y'])
    else:
        data = full_data

    # Convert to numpy array to remove feature names (sklearn 1.0.2 compatibility)
    data_array = data.values

    # Compute predictions on the selected dataset
    y_pred = model.predict(data_array)
    
    # Generate performance description
    text = conversation.describe.get_score_text(y_true,
                                                y_pred,
                                                metric,
                                                conversation.rounding_precision,
                                                data_name)

    # Add debugging information when useful
    if is_global_query:
        text += f"<br><em>Note: Overall model performance computed on {len(data)} total instances.</em>"
    elif len(filter_string) > 0:
        text += f"<br><em>Note: Performance computed on {len(data)} filtered instances.</em>"

    text += "<br><br>"
    return text, 1
