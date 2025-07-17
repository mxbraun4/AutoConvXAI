# TalkToModel Architecture

## Overview
TalkToModel follows a clean 3-component architecture:

```
Natural Language → AutoGen → Actions → LLM Formatter
```

## Components

### 1. AutoGen Decoder (`nlu/autogen_decoder.py`)
- Parses natural language queries into structured intents
- Uses multi-agent collaboration for intent understanding
- Extracts entities and action parameters

### 2. Action Dispatcher (`app/core.py`)
- Executes explainability actions based on parsed intents
- Manages conversation context and filtering
- Handles model predictions and explanations

### 3. LLM Formatter (`formatter/llm_formatter.py`)
- Formats raw action results into natural language
- Tailors responses based on detected intent
- Maintains conversational context

## Directory Structure

```
/
├── app/                    # Flask application
│   ├── __init__.py        # App factory
│   ├── routes.py          # Route definitions
│   ├── core.py            # Business logic
│   └── conversation.py    # Conversation management
├── ui/                    # User Interface Layer
│   ├── static/           # Web assets
│   │   ├── css/          # Stylesheets
│   │   └── js/           # JavaScript files
│   └── templates/        # HTML templates
├── nlu/                   # Natural Language Understanding
│   └── autogen_decoder.py # Intent parser
├── explainability/        # Explainability Module
│   ├── actions/          # Action implementations
│   ├── core/             # Core explainer classes
│   └── mega_explainer/   # LIME/SHAP explainers
├── formatter/             # Response Formatting
│   └── llm_formatter.py  # LLM-based formatter
├── data/                  # Dataset and models
├── evaluation/            # Testing and evaluation
├── models/                # Trained models
├── config/                # Configuration
└── docs/                  # Documentation
```

## Key Classes

### Conversation (`app/conversation.py`)
- Manages conversational state and memory
- Handles dataset filtering and context
- Maintains feature definitions and metadata

### SimpleActionDispatcher (`app/core.py`)
- Routes actions to appropriate handlers
- Manages action execution and error handling
- Provides unified interface to explainability actions

### LLMFormatter (`app/core.py`)
- Transforms structured results into natural language
- Uses OpenAI API for intelligent formatting
- Applies intent-specific formatting rules

## Data Flow

1. **User Input**: Natural language query received
2. **Intent Parsing**: AutoGen decoder analyzes query
3. **Action Execution**: Dispatcher calls appropriate action
4. **Result Formatting**: LLM formatter creates response
5. **Context Update**: Conversation state updated
6. **Response**: Natural language response returned

## Configuration

Configuration is managed through:
- Environment variables (API keys, models)
- `config/settings.py` for application settings
- Feature definitions in conversation class

## Testing

- Unit tests in `tests/` directory
- Evaluation scripts in `evaluation/` directory
- Test configuration in `config/settings.py`