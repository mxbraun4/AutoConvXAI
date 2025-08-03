"""Utility functions for data manipulation and text generation."""


def add_to_dict_lists(key, value, dictionary):
    """Add a value to a list stored in a dictionary, creating the list if needed.
    
    Args:
        key: Dictionary key to store the value under
        value: Value to append to the list at the given key
        dictionary: Dictionary to modify (updated in place)
    """
    if key not in dictionary:
        dictionary[key] = [value]
    else:
        dictionary[key].append(value)


def gen_parse_op_text(conversation):
    """Generate human-readable text describing current filtering operations.
    
    Args:
        conversation: Conversation object containing parse_operation list
        
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
