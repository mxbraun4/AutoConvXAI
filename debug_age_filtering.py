#!/usr/bin/env python3
"""
Debug script to analyze age filtering issue.
This script traces through the entire pipeline to understand why age filtering isn't working.
"""

def simulate_autogen_response():
    """Simulate what AutoGen should return for the query 'how accurate is the model on instances with age > 50'"""
    return {
        'intent': 'performance',
        'entities': {
            'features': ['age'],
            'operators': ['>'],
            'values': [50],
            'filter_type': 'feature'
        }
    }

def test_score_action_filtering():
    """Test how the score action processes the filtering entities"""
    print("=== TESTING SCORE ACTION FILTERING ===")
    
    # Simulate the action arguments that would be passed to score_operation
    action_args = {
        'features': ['age'],
        'operators': ['>'], 
        'values': [50],
        'parse_text': ['how', 'accurate', 'is', 'the', 'model', 'on', 'instances', 'with', 'age', '>', '50']
    }
    
    print("Action arguments:", action_args)
    
    # Check what the score action looks for
    ent_features = action_args.get('features', [])
    ent_ops = action_args.get('operators', [])
    ent_vals = action_args.get('values', [])
    
    print("Extracted entities:")
    print("  Features:", ent_features)
    print("  Operators:", ent_ops)
    print("  Values:", ent_vals)
    
    # Check if filtering would be applied
    if ent_features and ent_ops and ent_vals:
        print("✅ Score action SHOULD apply filtering")
        for feat, op, val in zip(ent_features, ent_ops, ent_vals):
            print(f"  Filter: {feat} {op} {val}")
    else:
        print("❌ Score action would NOT apply filtering")
        
    return ent_features, ent_ops, ent_vals

def analyze_main_py_flow():
    """Analyze how main.py processes the query"""
    print("\n=== ANALYZING MAIN.PY FLOW ===")
    
    # 1. Intent mapping
    intent = 'performance'
    intent_to_action = {
        'data': 'data',
        'performance': 'score', 
        'predict': 'predict',
        'explain': 'explain',
        'important': 'important',
        'filter': 'filter',
        'casual': 'function'
    }
    
    action_name = intent_to_action[intent]
    print(f"Intent '{intent}' maps to action '{action_name}'")
    
    # 2. Entity extraction
    entities = {
        'features': ['age'],
        'operators': ['>'],
        'values': [50],
        'filter_type': 'feature'
    }
    
    # 3. Action args construction
    user_tokens = ['how', 'accurate', 'is', 'the', 'model', 'on', 'instances', 'with', 'age', '>', '50']
    action_args = {
        'features': entities.get('features', []),
        'operators': entities.get('operators', []),
        'values': entities.get('values', []),
        'patient_id': entities.get('patient_id'),
        'filter_type': entities.get('filter_type'),
        'prediction_values': entities.get('prediction_values', []),
        'label_values': entities.get('label_values', []),
        'parse_text': user_tokens
    }
    
    print("Action args passed to score action:", action_args)
    
    return action_args

if __name__ == "__main__":
    print("Debugging age filtering issue for query: 'how accurate is the model on instances with age > 50'")
    print("="*80)
    
    # Test AutoGen response
    autogen_response = simulate_autogen_response()
    print("Expected AutoGen response:", autogen_response)
    
    # Test main.py flow
    action_args = analyze_main_py_flow()
    
    # Test score action filtering logic
    test_score_action_filtering()
    
    print("\n=== ANALYSIS SUMMARY ===")
    print("1. AutoGen should extract: features=['age'], operators=['>'], values=[50]")
    print("2. main.py should pass these entities to the score action")
    print("3. score.py should detect entities and apply filtering (lines 65-94)")
    print("4. If this isn't working, the issue is likely in:")
    print("   - AutoGen entity extraction (autogen_decoder.py)")
    print("   - Feature name mapping ('age' vs 'Age')")
    print("   - Score action entity processing")