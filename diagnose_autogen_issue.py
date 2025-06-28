"""Diagnostic script to test AutoGen and OpenAI connectivity."""

import os
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_autogen_imports():
    """Test if AutoGen can be imported properly."""
    print("🔍 Testing AutoGen imports...")
    
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        print("✅ Modern AutoGen imports successful")
        return True
    except ImportError as e:
        print(f"❌ Modern AutoGen import failed: {e}")
        
        try:
            from autogen.agentchat.agents import AssistantAgent
            from autogen.agentchat.teams import RoundRobinGroupChat
            from autogen.models.openai import OpenAIChatCompletionClient
            print("✅ Legacy AutoGen imports successful")
            return True
        except ImportError as e2:
            print(f"❌ Legacy AutoGen import failed: {e2}")
            return False

def test_openai_basic():
    """Test basic OpenAI connectivity."""
    print("\n🔍 Testing basic OpenAI connectivity...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Simple test call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )
        
        print(f"✅ OpenAI test successful: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

def test_autogen_simple():
    """Test simple AutoGen functionality."""
    print("\n🔍 Testing simple AutoGen functionality...")
    
    if not test_autogen_imports():
        return False
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        
        print("Creating model client...")
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            api_key=api_key,
        )
        
        print("Creating agent...")
        agent = AssistantAgent(
            name="TestAgent",
            model_client=model_client,
            system_message="You are a test agent. Respond with exactly: 'Agent test successful'"
        )
        
        print("✅ AutoGen setup successful")
        return True
        
    except Exception as e:
        print(f"❌ AutoGen setup failed: {e}")
        return False

async def test_autogen_collaboration():
    """Test AutoGen agent collaboration."""
    print("\n🔍 Testing AutoGen agent collaboration...")
    
    if not test_autogen_imports():
        return False
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_agentchat.conditions import MaxMessageTermination
        
        print("Creating model client...")
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            api_key=api_key,
        )
        
        print("Creating test agent...")
        agent = AssistantAgent(
            name="TestAgent",
            model_client=model_client,
            system_message="You are a test agent. Respond with exactly: 'Collaboration test successful'"
        )
        
        print("Creating team...")
        team = RoundRobinGroupChat(
            participants=[agent],
            termination_condition=MaxMessageTermination(1)
        )
        
        print("Starting collaboration test...")
        result = await team.run(task="Say hello")
        
        print(f"✅ Collaboration test completed with {len(result.messages)} messages")
        if result.messages:
            print(f"Last message: {result.messages[-1].content}")
        return True
        
    except Exception as e:
        print(f"❌ Collaboration test failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def main():
    """Run all diagnostic tests."""
    print("🩺 AUTOGEN DIAGNOSTIC TESTS")
    print("=" * 50)
    
    results = []
    
    # Test 1: AutoGen imports
    results.append(test_autogen_imports())
    
    # Test 2: OpenAI basic connectivity
    results.append(test_openai_basic())
    
    # Test 3: AutoGen setup
    results.append(test_autogen_simple())
    
    # Test 4: AutoGen collaboration (async)
    try:
        result = asyncio.run(test_autogen_collaboration())
        results.append(result)
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        results.append(False)
    
    print(f"\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed - AutoGen should work correctly")
    elif passed >= 2:
        print("⚠️ Partial functionality - some components working")
    else:
        print("❌ Major issues detected - check installation and API key")
    
    return passed == total

if __name__ == "__main__":
    main() 