"""Utility functions for data manipulation and text generation.

This module provides helper functions used across the explanation system:
- Dictionary manipulation for storing lists of values
- Text generation for filtering operation descriptions
"""


def add_to_dict_lists(key, value, dictionary):
    """Add a value to a list stored in a dictionary, creating the list if needed.
    
    This function treats dictionary values as lists and appends new values.
    If the key doesn't exist, it creates a new list with the value.
    Modifies the dictionary in place.
    
    Args:
        key: Dictionary key to store the value under
        value: Value to append to the list at the given key
        dictionary: Dictionary to modify (updated in place)
        
    Example:
        data = {}
        add_to_dict_lists('scores', 0.85, data)  # data = {'scores': [0.85]}
        add_to_dict_lists('scores', 0.92, data)  # data = {'scores': [0.85, 0.92]}
    """
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
