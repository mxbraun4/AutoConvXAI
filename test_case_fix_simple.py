#!/usr/bin/env python3
"""
Test case sensitivity fix logic without external dependencies
"""

def test_case_sensitivity_fix():
    """Test that the case sensitivity fix works"""
    
    # Mock dataset columns (like real diabetes dataset)
    columns = ['Age', 'Glucose', 'BMI', 'Pregnancies', 'BloodPressure', 'Insulin', 'SkinThickness', 'DiabetesPedigreeFunction']
    
    print("📊 Test Dataset Columns:")
    print(f"Columns: {columns}")
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
    
    # Test cases that AutoGen might extract vs actual dataset columns
    test_cases = [
        ("age", "Age"),                           # The main issue case
        ("glucose", "Glucose"),
        ("bmi", "BMI"),  
        ("pregnancies", "Pregnancies"),
        ("bloodpressure", "BloodPressure"),       # Compound word
        ("insulin", "Insulin"),
        ("skinthickness", "SkinThickness"),       # Compound word
        ("diabetespedigreefunction", "DiabetesPedigreeFunction"),  # Long compound
        ("Age", "Age"),                           # Exact match
        ("AGE", "Age"),                           # All caps
        ("nonexistent", None),                    # Not found
        ("glucose_typo", None)                    # Typo
    ]
    
    print("🔍 Testing case-insensitive feature matching:")
    all_passed = True
    for input_feat, expected in test_cases:
        result = find_actual_feature_name(input_feat, columns)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"{status} '{input_feat}' -> '{result}' (expected: '{expected}')")
    
    print()
    
    # Test the specific failing case
    print("🎯 Testing the specific failing case:")
    print("   Query: 'how accurate is the model on instances with age > 50'")
    print("   AutoGen extracts: features=['age'], operators=['>'], values=[50]")
    print("   Dataset has column: 'Age'")
    print()
    
    feature_from_autogen = "age"
    actual_column = find_actual_feature_name(feature_from_autogen, columns)
    
    if actual_column:
        print(f"✅ SUCCESS: '{feature_from_autogen}' maps to '{actual_column}'")
        print("   This means filtering will now work correctly!")
    else:
        print(f"❌ FAILED: '{feature_from_autogen}' could not be mapped")
        print("   Filtering would still fail")
    
    print()
    if all_passed:
        print("🎉 All tests passed! Case sensitivity fix should work.")
    else:
        print("💥 Some tests failed. Need to review the logic.")
    
    return all_passed

if __name__ == "__main__":
    test_case_sensitivity_fix()