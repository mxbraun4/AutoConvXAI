"""Describes the AI assistant capabilities and system information."""


def self_operation(conversation, parse_text, i, **kwargs):
    """AI assistant self-description - handles capabilities, limitations, and system information."""

    # Handle casual responses if provided
    if 'casual_response' in kwargs:
        return kwargs['casual_response'], 1

    # Get model and dataset information
    objective = conversation.describe.get_dataset_objective()
    model_type = conversation.describe.get_model_description()
    dataset_desc = conversation.describe.get_dataset_description()
    
    # Create response focused on AI assistant capabilities and role
    response = {
        'type': 'assistant_capabilities',
        'system_description': f"I'm an AI assistant that helps you understand and analyze a machine learning model trained to {objective}. I work with a {dataset_desc} dataset.",
        'capabilities': [
            "Generate predictions for new cases",
            "Explain reasoning behind predictions", 
            "Show feature importance and interactions",
            "Perform what-if analysis and counterfactuals",
            "Display model performance metrics",
            "Filter and analyze specific patient data",
            "Provide data statistics and summaries",
            "Define medical and technical terms",
            "Answer questions about diabetes prediction",
            "Help interpret model results"
        ],
        'limitations': [
            "Cannot provide medical advice or diagnosis",
            "Results are based on training data patterns",
            "Cannot access external medical databases",
            "Predictions should be validated by medical professionals"
        ],
        'role': f"I serve as an interface to help you understand how the diabetes prediction model works and interpret its results.",
        'available_features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'target_classes': ['No Diabetes', 'Diabetes']
    }
    
    return response, 1
