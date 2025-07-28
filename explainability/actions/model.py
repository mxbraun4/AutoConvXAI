"""Describes the model and system capabilities."""


def model_operation(conversation, parse_text, i, **kwargs):
    """Model and system description - handles both technical specs and conversational capabilities."""

    # Handle casual responses if provided
    if 'casual_response' in kwargs:
        return kwargs['casual_response'], 1

    # Get model and dataset information
    objective = conversation.describe.get_dataset_objective()
    model_type = conversation.describe.get_model_description()
    dataset_desc = conversation.describe.get_dataset_description()
    
    # Create comprehensive response covering both technical and conversational aspects
    response = {
        'type': 'model_and_system_info',
        'model_technical_specs': {
            'model_type': model_type,
            'objective': objective,
            'task': 'classification',
            'dataset': dataset_desc
        },
        'system_description': f"I'm a machine learning model trained to {objective}. I was trained on a {dataset_desc} dataset.",
        'capabilities': [
            "Generate predictions for new cases",
            "Explain reasoning behind predictions", 
            "Show feature importance and interactions",
            "Perform what-if analysis and counterfactuals",
            "Display model performance metrics",
            "Filter and analyze specific patient data",
            "Provide data statistics and summaries",
            "Define medical and technical terms"
        ],
        'available_features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'target_classes': ['No Diabetes', 'Diabetes']
    }
    
    return response, 1
