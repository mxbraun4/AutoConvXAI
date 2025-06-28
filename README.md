# AutoGen TalkToModel

A clean, autogen-only implementation of TalkToModel for natural language understanding using multi-agent systems.

## Features

- **Multi-Agent Architecture**: Uses AutoGen framework with specialized agents for intent extraction, action planning, and validation
- **Natural Language Understanding**: Converts conversational queries into structured actions
- **Clean Codebase**: Focused only on autogen functionality with minimal dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TalkToModel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GPT_MODEL="gpt-4o"  # optional, defaults to gpt-4o
```

## Usage

### Simple Flask App

Run the simple autogen-only Flask application:

```bash
python3 simple_autogen_app.py
```

Then visit http://localhost:5000 or use the API:

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the most important feature?"}'
```

### AutoGen Example

Run the standalone autogen example:

```bash
python3 tests/autogen_example.py
```

### Programmatic Usage

```python
from explain.autogen_decoder import AutoGenDecoder

# Initialize decoder
decoder = AutoGenDecoder(
    api_key="your-openai-api-key",
    model="gpt-4o",
    max_rounds=3
)

# Process a query
result = decoder.complete_sync("What are the most important features?", conversation=None)
print(result)
```

## Architecture

The system uses a three-agent architecture:

1. **Intent Extraction Agent**: Analyzes user queries to extract intentions and entities
2. **Action Planning Agent**: Translates intents into executable action syntax
3. **Validation Agent**: Validates and corrects generated actions

## Files Structure

- `explain/autogen_decoder.py` - Main AutoGen decoder implementation
- `simple_autogen_app.py` - Simple Flask app for testing
- `tests/autogen_example.py` - Example usage
- `requirements.txt` - Python dependencies
- `Dockerfile*` - Container configurations

## Docker

Build and run with Docker:

```bash
docker build -t autogen-talktomodel .
docker run -e OPENAI_API_KEY="your-key" -p 5000:5000 autogen-talktomodel
```

## Contributing

This is a cleaned-up version focused only on AutoGen functionality. The codebase has been simplified to remove unnecessary dependencies and focus on the core multi-agent natural language understanding capabilities.