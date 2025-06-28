# AutoGen's Role in the Generalized Framework

## Complete Role Breakdown

AutoGen handles **4 critical stages** of natural language understanding, while the code generator handles execution. Here's the exact division of responsibilities:

## AutoGen's Responsibilities

### 1. **Intent Classification** 
**Agent:** Intent Extraction Agent  
**Input:** Raw natural language  
**Output:** Structured intent classification

```python
# Input: "how accurate is the model on instances with age > 50"
# AutoGen Intent Agent Output:
{
  "intent": "performance",           # Classifies as performance query
  "confidence": 0.95,               # Confidence in classification
  "reasoning": "User seeks model evaluation metrics"
}
```

**What AutoGen Does:**
- Analyzes semantic meaning of the query
- Classifies into intent categories (performance, data, predict, explain, importance)
- Handles complex conversational patterns
- Determines context switching vs continuation

### 2. **Entity Extraction & Parsing**
**Agent:** Intent Extraction Agent  
**Input:** Natural language + context  
**Output:** Structured entities

```python
# AutoGen Entity Extraction Output:
{
  "entities": {
    "features": ["age"],              # Extracts feature names
    "operators": ["greater"],         # Parses comparison operators  
    "values": [50],                  # Extracts numeric values
    "patient_id": null,              # Identifies specific IDs if mentioned
    "explanation_type": null,        # Determines explanation method
    "context_reset": true            # Decides if filters should reset
  }
}
```

**What AutoGen Does:**
- **Feature Recognition:** Maps "age", "BMI", "glucose" to dataset columns
- **Operator Parsing:** Converts ">", "greater than", "over" to "greater"
- **Value Extraction:** Pulls out numeric thresholds (50, 30, 140, etc.)
- **Context Analysis:** Determines if query continues previous context or starts fresh
- **Ambiguity Resolution:** Handles "older patients" → "age > X"

### 3. **Action Planning & Translation**
**Agent:** Action Planning Agent  
**Input:** Structured intent + entities  
**Output:** Planned action sequence

```python
# AutoGen Action Planning Output:
{
  "action": "filter age greater 50 score accuracy",
  "reasoning": "User wants performance metrics for age-filtered subset",
  "confidence": 0.88
}
```

**What AutoGen Does:**
- **Sequence Planning:** Determines that filters come before operations
- **Syntax Generation:** Creates action syntax from structured components
- **Logic Ordering:** Ensures operations happen in correct order
- **Complex Combinations:** Handles multiple filters + operations

### 4. **Validation & Error Correction**
**Agent:** Validation Agent  
**Input:** Planned action  
**Output:** Validated/corrected action

```python
# AutoGen Validation Output:
{
  "valid": true,
  "corrected_action": "filter age greater 50 score accuracy",
  "issues": [],                     # Any problems found
  "confidence": 0.92
}
```

**What AutoGen Does:**
- **Syntax Validation:** Ensures action follows correct grammar
- **Semantic Validation:** Checks if features exist in dataset
- **Value Range Checking:** Validates numeric ranges are reasonable
- **Error Correction:** Fixes common mistakes automatically

## Code Generator's Responsibilities

### 5. **Structure Conversion**
**Component:** AutoGenQueryExtractor  
**Input:** AutoGen's structured output  
**Output:** GeneratedQuery object

```python
# Converts AutoGen output to internal structure:
GeneratedQuery(
    filters=[FilterOperation("age", "gt", 50)],
    operation=QueryOperation("accuracy")
)
```

### 6. **Code Generation**
**Component:** CodeGenerator  
**Input:** GeneratedQuery  
**Output:** Executable Python code

```python
# Generated executable code:
def execute_query(dataset, model, explainer=None, conversation=None):
    original_dataset = dataset.copy()
    original_size = len(original_dataset)
    
    # Apply filters
    filtered_dataset = dataset[dataset['age'] > 50]
    dataset = filtered_dataset
    
    # Calculate accuracy
    X = dataset.drop(columns=['y'], errors='ignore')
    y_true = dataset.get('y')
    y_pred = model.predict(X)
    accuracy = (y_pred == y_true).mean() * 100
    
    return f'Accuracy (age > 50): {accuracy:.2f}% ({len(dataset)}/{original_size} samples)'
```

### 7. **Safe Execution**
**Component:** DynamicQueryExecutor  
**Input:** Generated code + data  
**Output:** Execution results

## Complete Pipeline Example

```
User Input: "how accurate is the model on instances with age > 50"

┌─────────────────┐
│ AutoGen Stage 1 │ Intent Classification
│ Intent Agent    │ → "performance" intent
└─────────────────┘

┌─────────────────┐
│ AutoGen Stage 2 │ Entity Extraction  
│ Intent Agent    │ → features: ["age"], operators: ["greater"], values: [50]
└─────────────────┘

┌─────────────────┐
│ AutoGen Stage 3 │ Action Planning
│ Action Agent    │ → "filter age greater 50 score accuracy"
└─────────────────┘

┌─────────────────┐
│ AutoGen Stage 4 │ Validation
│ Validation Agent│ → Validates and confirms action
└─────────────────┘

┌─────────────────┐
│ Code Gen Stage 1│ Structure Conversion
│ Query Extractor │ → GeneratedQuery(filters=[...], operation=...)
└─────────────────┘

┌─────────────────┐
│ Code Gen Stage 2│ Code Generation  
│ Code Generator  │ → Python function with pandas operations
└─────────────────┘

┌─────────────────┐
│ Code Gen Stage 3│ Safe Execution
│ Query Executor  │ → "Accuracy (age > 50): 77.3% (234/768 samples)"
└─────────────────┘
```

## Why This Division Works

### AutoGen's Strengths:
- **Language Understanding:** Sophisticated NLU with conversational context
- **Ambiguity Resolution:** Handles unclear or informal language
- **Context Management:** Tracks conversation state and context switches
- **Multi-Agent Collaboration:** Specialized agents for different aspects
- **Error Recovery:** Built-in validation and correction

### Code Generator's Strengths:
- **Execution Optimization:** Pandas-optimized code generation
- **Type Safety:** Ensures correct data types and operations
- **Security:** Safe execution environment
- **Extensibility:** Easy to add new operations and operators
- **Performance:** Compiled pandas operations vs interpreted strings

## Benefits of This Architecture

1. **Best of Both Worlds:** AutoGen's sophisticated NLU + optimized code execution
2. **Clear Separation:** Each component does what it's best at
3. **Maintainability:** Changes to NLU don't affect execution and vice versa
4. **Extensibility:** Easy to extend either the understanding or execution parts
5. **Debugging:** Clear pipeline for tracking where issues occur
6. **Performance:** AutoGen for understanding, compiled code for execution

## Summary

**AutoGen is responsible for the entire natural language understanding pipeline:**
- 🧠 Understanding what the user wants (intent)
- 🔍 Extracting the specific details (entities)
- 📋 Planning how to accomplish it (action)
- ✅ Validating the plan is correct (validation)

**Code Generator is responsible for execution optimization:**
- 🔧 Converting to internal structures
- 💻 Generating optimized Python code  
- 🚀 Safely executing the operations

This creates a truly generalizable system where AutoGen handles the complexity of language understanding while the code generator ensures optimal execution.