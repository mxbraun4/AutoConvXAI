#!/usr/bin/env python3
"""
Test the clean architecture without requiring external dependencies
"""
import sys
import os

def test_imports():
    """Test core imports are working"""
    print("🧪 Testing Clean Architecture...")
    
    try:
        # Test basic Python imports
        import json
        import logging
        print("✅ Basic Python modules working")
        
        # Test if files exist
        required_files = [
            'explain/autogen_decoder.py',
            'explain/dynamic_code_generation.py',
            'main.py',
            'simple_autogen_app.py',
            'data/diabetes.csv'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} exists")
            else:
                print(f"❌ {file_path} missing")
                return False
        
        # Test syntax of key files
        key_files = [
            'explain/autogen_decoder.py',
            'explain/dynamic_code_generation.py',
            'main.py'
        ]
        
        for file_path in key_files:
            try:
                with open(file_path, 'r') as f:
                    compile(f.read(), file_path, 'exec')
                print(f"✅ {file_path} syntax valid")
            except SyntaxError as e:
                print(f"❌ {file_path} syntax error: {e}")
                return False
        
        print("\n🎉 Clean Architecture Foundation Tests PASSED!")
        print("\nArchitecture Summary:")
        print("Natural Language → AutoGen → Dynamic Actions → Response")
        print("\nCore Components:")
        print("- main.py: Clean Flask app with simple UI")
        print("- autogen_decoder.py: Multi-agent natural language understanding")
        print("- dynamic_code_generation.py: Generates executable actions")
        print("- simple_autogen_app.py: Alternative minimal AutoGen-only app")
        print("\nTo run:")
        print("1. Install dependencies: pip install -r requirements-clean.txt")
        print("2. Set OPENAI_API_KEY environment variable")
        print("3. Run: python main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)