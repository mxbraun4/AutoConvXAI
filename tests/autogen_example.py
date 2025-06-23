"""Example usage of AutoGen Multi-Agent Decoder for TalkToModel.

This script demonstrates how to use the multi-agent approach with specialized agents.
"""
import os
import asyncio
from explain.autogen_decoder import AutoGenDecoder
from explain.conversation import Conversation
from explain.enhanced_logic import EnhancedExplainBot


async def test_autogen_decoder():
    """Test the AutoGen decoder with sample queries."""
    
    # Initialize decoder
    decoder = AutoGenDecoder(
        api_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-4o",
        max_rounds=4
    )
    
    # Create a mock conversation object for testing
    # In real usage, this would come from EnhancedExplainBot
    class MockConversation:
        def __init__(self):
            self.class_names = ["0", "1"]
        
        def get_var(self, name):
            class MockVar:
                def __init__(self):
                    self.contents = {
                        'X': range(768),  # Mock dataset size
                        'cat': ['pregnancies'],
                        'numeric': ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'diabetespedigreefunction', 'age']
                    }
            return MockVar()
    
    conversation = MockConversation()
    
    # Test queries
    test_queries = [
        "How accurate is the model?",
        "Give me the prediction for patient 5",
        "Show predictions for patients with age over 50 and BMI greater than 25",
        "Why did patient 10 get diagnosed with diabetes?",
        "What are the most important features?",
        "Show me cases where the model made mistakes",
        "Hello there!"
    ]
    
    print("🤖 Testing AutoGen Multi-Agent Decoder")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n👤 User: {query}")
        try:
            result = await decoder.complete(query, conversation)
            
            if result.get('direct_response'):
                print(f"🤖 Response: {result['direct_response']}")
            else:
                action = result.get('generation', 'No action generated')
                method = result.get('method', 'unknown')
                confidence = result.get('confidence', 0)
                print(f"🎯 Action: {action}")
                print(f"📊 Method: {method} (confidence: {confidence:.2f})")
                
                # Show agent reasoning if available
                if 'agent_reasoning' in result:
                    print(f"🧠 Agent Reasoning: {result['agent_reasoning']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Clean up
    await decoder.close()
    print("\n✅ Testing complete!")


def test_enhanced_bot_with_autogen():
    """Test the full EnhancedExplainBot with AutoGen enabled."""
    
    print("\n🚀 Testing Full Bot with AutoGen")
    print("=" * 60)
    
    # Initialize bot with AutoGen enabled
    bot = EnhancedExplainBot(
        model_file_path="data/diabetes_model_logistic_regression.pkl",
        dataset_file_path="data/diabetes.csv",
        background_dataset_file_path="data/diabetes.csv",
        dataset_index_column=None,
        target_variable_name="y",
        categorical_features=["pregnancies"],
        numerical_features=["glucose", "bloodpressure", "skinthickness", "insulin", "bmi", "diabetespedigreefunction", "age"],
        remove_underscores=True,
        name="diabetes_autogen",
        use_autogen=True  # Enable AutoGen!
    )
    
    # Test a few queries
    test_queries = [
        "How accurate is the model?",
        "Give me the prediction for patient 1"
    ]
    
    for query in test_queries:
        print(f"\n👤 User: {query}")
        try:
            response = bot.update_state(query, bot.conversation)
            print(f"🤖 Bot: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Full bot testing complete!")


if __name__ == "__main__":
    # Test the decoder directly
    asyncio.run(test_autogen_decoder())
    
    # Test the full bot integration
    test_enhanced_bot_with_autogen() 