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
        data = conversation.get_var('dataset').contents['X']
        y_true = conversation.get_var('dataset').contents['y']
        filter_string = ""  # No filtering applied
        is_global_query = True
        data_name = "the <b>entire dataset</b>"
    else:
        # Default: Use filtered dataset (respects user's filtering intent)
        data = conversation.temp_dataset.contents['X']
        y_true = conversation.temp_dataset.contents['y']
        filter_string = gen_parse_op_text(conversation)
        is_global_query = False
        
        # Format data description for output
        if len(filter_string) <= 0:
            data_name = "the <b>entire dataset</b>"
        else:
            data_name = f"instances where <b>{filter_string}</b>"

    # Compute predictions on the selected dataset
    y_pred = model.predict(data)
    
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
