"""Utils"""


def add_to_dict_lists(key, value, dictionary):
    """Stores values in list corresponding to key in place."""
    if key not in dictionary:
        dictionary[key] = [value]
    else:
        dictionary[key].append(value)


def gen_parse_op_text(conversation):
    """Generate text describing current filtering operations.
    
    This function creates a human-readable description of the current
    filtering operations applied to the dataset. It combines the
    parsing operations stored in the conversation object.
    
    Args:
        conversation: The conversation object containing parse_operation list
        
    Returns:
        str: Formatted text describing the filtering operations
    """
    if not hasattr(conversation, 'parse_operation'):
        return ""
    
    parse_ops = conversation.parse_operation
    if not parse_ops:
        return ""
    
    # Filter out logical operators ('and', 'or') and combine remaining operations
    meaningful_ops = [op for op in parse_ops if op not in ['and', 'or']]
    
    if not meaningful_ops:
        return ""
    
    # Join operations with appropriate conjunctions
    if len(meaningful_ops) == 1:
        return meaningful_ops[0]
    elif len(meaningful_ops) == 2:
        return f"{meaningful_ops[0]} and {meaningful_ops[1]}"
    else:
        # For multiple operations, join all but last with commas, last with 'and'
        return ", ".join(meaningful_ops[:-1]) + f", and {meaningful_ops[-1]}"
