"""Describes the model."""


def model_operation(conversation, parse_text, i, **kwargs):
    """Model description."""

    objective = conversation.describe.get_dataset_objective()
    model = conversation.describe.get_model_description()
    
    return {
        'type': 'model_info',
        'model_type': model,
        'objective': objective,
        'task': 'classification'
    }, 1
