"""Describe self operation.

This operation gives some description of the system.
"""


def self_operation(conversation, parse_text, i, **kwargs):
    """Self description."""
    
    # Handle casual responses if provided
    if 'casual_response' in kwargs:
        return kwargs['casual_response'], 1

    objective = conversation.describe.get_dataset_objective()
    text = f"I'm a machine learning model trained to {objective}."

    dataset = conversation.describe.get_dataset_description()
    text += f" I was trained on a {dataset} dataset."

    return text, 1
