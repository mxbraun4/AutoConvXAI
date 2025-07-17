# TalkToModel API Documentation

## Overview
TalkToModel provides a conversational interface for machine learning model explanations using natural language queries.

## Endpoints

### GET /
**Description**: Render the main chat interface
**Response**: HTML page with chat interface

### POST /query
**Description**: Process queries via JSON API
**Request Body**:
```json
{
  "query": "What are the most important features?"
}
```
**Response**:
```json
{
  "response": "The most important features are...",
  "action_used": "important",
  "raw_result": "Raw action result"
}
```

### POST /get_bot_response
**Description**: Main chat endpoint (used by web interface)
**Request Body**:
```json
{
  "userInput": "Tell me about the dataset"
}
```
**Response**: Plain text in format "response<>log_info"

### POST /sample_prompt
**Description**: Generate sample prompts for suggestion buttons
**Request Body**:
```json
{
  "action": "important"
}
```
**Response**: Plain text sample prompt

### POST /log_feedback
**Description**: Log user feedback
**Request Body**: Form data with `feedback` field
**Response**:
```json
{
  "status": "logged"
}
```

## Query Types

### Data Exploration
- `"Tell me about this dataset"`
- `"How many patients are in the dataset?"`
- `"What are the feature statistics?"`

### Model Analysis
- `"What are the most important features?"`
- `"How well does the model perform?"`
- `"Show me the model's mistakes"`

### Predictions
- `"Predict for age=50, BMI=30"`
- `"What if BMI was 25 instead?"`
- `"Show counterfactuals for patient 5"`

### Filtering
- `"Show patients with age > 50"`
- `"Filter to high BMI patients"`

## Error Handling
All endpoints return appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (missing parameters)
- 500: Internal Server Error