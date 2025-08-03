# AutoConvXAI: Interactive Explanations in AI

**A Multi-Agent Conversational XAI System**

*Bachelor's Thesis by Maximilian Braun*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![AutoGen](https://img.shields.io/badge/AutoGen-0.6.4-purple.svg)](https://github.com/microsoft/autogen)

## Overview

This project presents a novel approach to intent parsing in Conversational XAI and bridges the gap between complex AI models and human understanding by providing an intuitive, interactive dialogue interface that democratizes access to machine learning explanations.


## Architecture

The system implements a clean **3-component pipeline**:

```
Natural Language Query → AutoGen Multi-Agent System → Action Execution → LLM Formatter → Natural Response
```

### Core Components

1. **AutoGen Multi-Agent Decoder** (`nlu/autogen_decoder.py`)
   - **Extraction Agent**: Analyzes user queries for intentions and entities
   - **Validation Agent**: Ensures action correctness and handles edge cases
   - Collaborative decision-making with configurable conversation rounds

2. **Action Dispatcher** (`app/core.py`)
   - Routes parsed intents to appropriate explainability functions
   - Manages conversational context and data filtering
   - Handles 20+ specialized explanation actions

3. **LLM Response Formatter** (`formatter/llm_formatter.py`)
   - Transforms technical results into natural language
   - Maintains conversational flow and context
   - Adapts tone and detail level based on user intent

### Explainability Actions

| Category | Actions | Description |
|----------|---------|-------------|
| **Data Exploration** | `data_summary`, `show_data`, `feature_stats` | Dataset understanding and statistics |
| **Model Analysis** | `important`, `score`, `mistakes`, `interaction_effects` | Model performance and behavior |
| **Predictions** | `predict`, `prediction_likelihood`, `what_if` | Individual and scenario-based predictions |
| **Counterfactuals** | `counterfactual` | Alternative scenarios and decision boundaries |
| **Context Management** | `filter`, `define`, `labels` | Data filtering and feature definitions |

## Key Features

- **Natural Language Interface**: Ask questions like "What are the most important features?" or "What if this patient had lower BMI?"
- **Multi-Agent Intelligence**: Two specialized agents collaborate for robust query understanding
- **Comprehensive Explanations**: LIME, SHAP, counterfactuals, feature importance, and statistical analysis
- **Context Awareness**: Maintains conversation history and applies filters across interactions
- **Real-time Processing**: Fast response times with efficient caching
- **Web Interface**: Clean, accessible chat interface with sample questions
- **Extensible Design**: Easy to add new explanation methods and actions

## Quick Start

**Docker recommended** - see [Docker Usage](#docker-usage) section for easiest setup.

### Prerequisites

- Python 3.8+
- OpenAI API key
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mxbraun4/AutoConvXAI.git
   cd AutoConvXAI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export GPT_MODEL="gpt-4o"  # Optional, defaults to gpt-4o-mini
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

5. **Access the interface**:
   - Web UI: http://localhost:5000
   - API: POST to `/query` with JSON `{"query": "your question"}`

## Research Evaluation

The system underwent comprehensive evaluation comparing **multi-agent** vs **single-agent** approaches:

### Evaluation Metrics
- **Parsing Accuracy**: Intent extraction and action parameter accuracy
- **Response Quality**: Naturalness and informativeness of explanations
- **Conversation Flow**: Context maintenance and coherence
- **Robustness**: Handling of edge cases and ambiguous queries

### Results Summary
- **Multi-Agent System**: 85.47% parsing accuracy
- **Single-Agent Baseline**: 84.46% parsing accuracy

*Detailed results available in `evaluation/results/`*

## Development

### Project Structure
```
├── app/                    # Flask application core
│   ├── conversation.py     # Context and state management
│   ├── core.py            # Action dispatcher and business logic
│   └── routes.py          # API endpoints
├── nlu/                   # Natural Language Understanding
│   └── autogen_decoder.py # Multi-agent intent parser
├── explainability/        # ML Explanation Engine
│   ├── actions/          # 20+ explanation functions
│   ├── core/             # Base explainer classes
│   └── mega_explainer/   # LIME/SHAP implementations
├── formatter/             # Response Generation
│   └── llm_formatter.py  # Natural language formatter
├── ui/                    # Web Interface
│   ├── static/           # CSS, JavaScript
│   └── templates/        # HTML templates
├── data/                  # Datasets and models
├── evaluation/            # Research evaluation scripts
└── docs/                  # Documentation
```

### Running Tests

```bash
python evaluation/run_full_evaluation.py
python evaluation/parsing_accuracy/autogen_evaluator.py
```

## Academic Context

This work contributes to the growing field of **Explainable AI (XAI)** and **Human-AI Interaction**:

### Research Contributions
1. **Novel Multi-Agent Architecture** for natural language explanations
2. **Comprehensive Evaluation Framework** for conversational explainability systems
3. **Practical Implementation** demonstrating real-world applicability

### Related Work
- **TalkToModel** (Slack et al., 2022): Original concept foundation
- **AutoGen** (Microsoft, 2023): Multi-agent conversation framework

*Based on TalkToModel framework by Slack et al. (2022)*

## Docker Usage

### Prerequisites
- Docker installed ([Docker Desktop](https://www.docker.com/products/docker-desktop/) recommended)
- OpenAI API key
- 4GB+ RAM, 2GB free disk space

### Quick Start
```bash
# Build images
docker build -t ttm-gpt4 .
docker build -t ttm-gpt4-test .

# Run web application
docker run -p 5000:5000 -e OPENAI_API_KEY="your-key" ttm-gpt4
# Access at http://localhost:5000

# Run evaluations
docker run -e OPENAI_API_KEY="your-key" -v $(pwd):/app -w /app ttm-gpt4-test python evaluation/run_full_evaluation.py
```

### Docker Images
- `ttm-gpt4`: Main web application
- `ttm-gpt4-test`: Evaluation and testing suite

## Contributing

This is a research project developed for academic purposes. For questions about the implementation or research methodology:

1. **Issues**: Use GitHub issues for bugs or questions
2. **Extensions**: Fork the repository for your own research

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Original TalkToModel** framework by Slack, Krishna, Lakkaraju, and Singh
- **Microsoft AutoGen** for multi-agent conversation capabilities  

## Contact

**Maximilian Braun**  
Bachelor's Thesis Project  
*Interactive Explanations in AI*

maximilian3.braun@stud.uni-regensburg.de

---

*This README provides comprehensive documentation for AutoConvXAI, developed as part of a bachelor's thesis exploring conversational XAI for machine learning explainability.*