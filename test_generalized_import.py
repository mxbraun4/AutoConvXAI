#!/usr/bin/env python3
"""
Test script to verify generalized_agents.py can be imported successfully
"""
import sys
import traceback

def test_import():
    """Test importing the generalized agents module."""
    print("🧪 Testing generalized_agents import...")
    
    try:
        # Test basic import
        print("1. Testing module import...")
        import explain.generalized_agents as ga
        print("   ✅ Module imported successfully")
        
        # Test function import
        print("2. Testing function import...")
        from explain.generalized_agents import generalized_action_dispatcher
        print("   ✅ Function imported successfully")
        
        # Test if permutation_importance flag is set correctly
        print("3. Testing dependency flags...")
        print(f"   PERMUTATION_IMPORTANCE_AVAILABLE: {ga.PERMUTATION_IMPORTANCE_AVAILABLE}")
        print(f"   VISUALIZATION_AVAILABLE: {ga.VISUALIZATION_AVAILABLE}")
        
        # Test creating a basic planner
        print("4. Testing basic functionality...")
        planner = ga.GeneralizedActionPlanner()
        print("   ✅ GeneralizedActionPlanner created successfully")
        
        # Test mock function call
        print("5. Testing mock dispatcher call...")
        # We can't call the real dispatcher without a conversation object,
        # but we can test the function exists and is callable
        assert callable(generalized_action_dispatcher)
        print("   ✅ generalized_action_dispatcher is callable")
        
        print("\n🎉 All tests passed! Generalized agents should work now.")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_import()
    sys.exit(0 if success else 1) 