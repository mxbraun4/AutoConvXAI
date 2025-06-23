# 🤖 Generalized Tool-Augmented Agent System

A more flexible, extensible approach to data exploration that reduces hardcoded rules and automatically adapts to any dataset structure.

## 🚀 What's New?

Instead of hardcoded action mappings like `"filter age greater 50"`, the generalized system uses:

- **🔍 Dynamic Schema Discovery**: Automatically discovers dataset structure and capabilities
- **🛠️ Tool-Augmented Agents**: Agents have access to pandas, sklearn, and visualization tools
- **📝 Code Generation**: Generates actual Python code for data operations
- **🎯 Self-Describing Operations**: Discovers available operations dynamically

## 📊 Traditional vs Generalized Comparison

| Feature | Traditional | Generalized |
|---------|-------------|-------------|
| **Adaptability** | ❌ Hardcoded for diabetes data | ✅ Adapts to any dataset |
| **Extensibility** | ❌ Requires code changes | ✅ Self-extending |
| **Maintainability** | ❌ Brittle special rules | ✅ Clean, modular design |
| **Error Handling** | ❌ Cryptic failures | ✅ Automatic fallbacks |
| **Performance** | ✅ Fast for known patterns | ⚠️ Slightly slower (code generation) |
| **Testing** | ✅ Well-tested | ⚠️ Experimental |

## 🛠️ How to Use

### 1. **Enable Generalized Mode**

```python
# In your bot configuration
bot = EnhancedExplainBot(
    use_generalized_actions=True,  # Enable generalized mode
    # ... other config ...
)
```

Or use the gin config:
```bash
python flask_app_gpt4.py --gin_file=configs/generalized-mode-config.gin
```

### 2. **Test Both Systems**

```bash
# Compare traditional vs generalized
python test_generalized_vs_traditional.py

# Quick test of generalized only
python test_generalized_vs_traditional.py quick
```

### 3. **Switch Back to Traditional**

```python
# Disable generalized mode
bot = EnhancedExplainBot(
    use_generalized_actions=False,  # Use traditional hardcoded actions
    # ... other config ...
)
```

## 🧪 Example Queries

The generalized system handles these naturally:

```python
queries = [
    "How large is the dataset?",
    "What does the model score on instances with age > 50?",
    "What are the most important features for prediction?", 
    "Predict for patient with id 2",
    "Show me patients with pregnancies > 2",
    "What's the accuracy on non-pregnant women?",
    "Explain why patient 5 got this prediction"
]
```

## 🔧 Technical Architecture

### Core Components

1. **`DatasetSchema`**: Dynamic schema discovery
   ```python
   schema = DatasetSchema.discover(df, target_col)
   # Automatically discovers: features, types, ranges, categories
   ```

2. **`DataOperationTool`**: Base class for tools
   - `FilteringTool`: Data filtering operations
   - `ModelAnalysisTool`: Model evaluation and prediction
   - `FeatureAnalysisTool`: Feature importance analysis
   - `DataSummaryTool`: Dataset statistics

3. **`GeneralizedActionPlanner`**: Orchestrates tool selection
   ```python
   planner = GeneralizedActionPlanner()
   operations = planner.plan_operations(intent, entities)
   results = planner.execute_plan(operations, context)
   ```

### Code Generation Example

For "What's the model accuracy on age > 50?":

**Traditional**: Hardcoded `filter age greater 50 score accuracy`

**Generalized**: Generates pandas code:
```python
# Filter dataset based on conditions
filtered_df = df[(df['age'] > 50)]
print(f"Applied filter: (df['age'] > 50)")

# Evaluate model performance  
X = filtered_df[feature_columns]
y_true = filtered_df[target_column]
y_pred = model.predict(X)
accuracy = accuracy_score(y_true, y_pred)
result = f"Model accuracy on {len(X)} instances: {accuracy:.1%}"
```

## 🔀 Migration Strategy

### Phase 1: **Parallel Testing** (Current)
- Both systems available
- Switch with `use_generalized_actions` flag
- Compare results side-by-side

### Phase 2: **Gradual Adoption**
- Generalized as default for new datasets
- Traditional as fallback for edge cases
- Extensive testing and refinement

### Phase 3: **Full Migration** (Future)
- Generalized system becomes primary
- Traditional system deprecated
- Legacy support for critical workflows

## 🐛 Debugging

### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Generated Code
```python
# In generalized mode, generated code is logged:
# DEBUG - Generated code:
# # Filter dataset based on conditions
# filtered_df = df[(df['age'] > 50)]
# ...
```

### Fallback Behavior
The generalized system automatically falls back to traditional mode if:
- Code generation fails
- Tool execution errors
- No suitable tools found

## 🚧 Limitations & Future Work

### Current Limitations
- ⚠️ Experimental (needs more testing)
- ⚠️ Slightly slower than hardcoded actions
- ⚠️ Limited visualization tools

### Planned Improvements
- 📊 Advanced visualization tools (matplotlib, plotly)
- 🔍 Statistical analysis tools (scipy, statsmodels)
- 🤖 Custom tool creation interface
- 📈 Performance optimizations
- 🧠 Memory of previous successful operations

## 💡 Why This Matters

Your frustration with "low level and not generalized" approaches is exactly what this solves:

❌ **Before**: `if parts[0] == "important" and len(parts) == 1: return "important all"`

✅ **After**: Dynamic tool selection based on intent and dataset schema

❌ **Before**: Hardcoded feature mappings and domain rules

✅ **After**: Self-discovering schema and adaptive operations

❌ **Before**: Brittle parsing with lots of special cases

✅ **After**: Natural language → executable code generation

## 🎯 Next Steps

1. **Try it out**: Run the test script to see the difference
2. **Enable for your use case**: Update your config with `use_generalized_actions=True`
3. **Provide feedback**: Test with your queries and report issues
4. **Extend**: Add custom tools for your specific domain needs

The AutoGen multi-agent foundation you already have is perfect for this generalized approach. This just makes it much more flexible and maintainable! 🚀 