#!/usr/bin/env python3
"""
Test case sensitivity fix for feature filtering
"""
import pandas as pd

def test_case_sensitivity_fix():
    """Test that the case sensitivity fix works"""
    
    # Create mock dataset with PascalCase columns like the real diabetes dataset
    data = {
        'Age': [25, 35, 45, 55, 65],
        'Glucose': [80, 120, 140, 160, 180],
        'BMI': [22.5, 25.0, 27.5, 30.0, 32.5]
    }
    df = pd.DataFrame(data)
    
    print("📊 Test Dataset:")
    print(f"Columns: {list(df.columns)}")
    print(f"Data:\n{df}")
    print()
    
    # Test the case-insensitive column matching logic from our fix
    def find_actual_feature_name(feature_name, columns):
        """Replicate the logic we added to score.py and filter.py"""
        actual_feat = None
        for col in columns:
            if col.lower() == feature_name.lower():
                actual_feat = col
                break
        return actual_feat
    
    # Test cases
    test_cases = [
        ("age", "Age"),      # lowercase -> PascalCase
        ("Age", "Age"),      # exact match
        ("AGE", "Age"),      # uppercase -> PascalCase  
        ("glucose", "Glucose"),
        ("bmi", "BMI"),
        ("nonexistent", None)
    ]
    
    print("🔍 Testing case-insensitive feature matching:")
    for input_feat, expected in test_cases:
        result = find_actual_feature_name(input_feat, df.columns)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_feat}' -> '{result}' (expected: '{expected}')")
    
    print()
    
    # Test filtering logic
    print("🎯 Testing filtering logic:")
    
    # Test Age > 50 filtering (the case that was failing)
    feature_name = "age"  # This is what AutoGen extracts (lowercase)
    actual_feature = find_actual_feature_name(feature_name, df.columns)
    
    if actual_feature:
        mask = df[actual_feature] > 50
        filtered_data = df[mask]
        print(f"✅ Filter 'age > 50' found column '{actual_feature}'")
        print(f"   Original data: {len(df)} rows")
        print(f"   Filtered data: {len(filtered_data)} rows")
        print(f"   Filtered result:\n{filtered_data}")
    else:
        print(f"❌ Filter 'age > 50' failed - could not find column")
    
    print()
    print("🎉 Case sensitivity fix test completed!")
    
    return True

if __name__ == "__main__":
    test_case_sensitivity_fix()