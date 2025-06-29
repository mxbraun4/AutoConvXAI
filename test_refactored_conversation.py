#!/usr/bin/env python3
"""
Test script to verify the refactored SimpleConversation class works correctly
"""
import pandas as pd
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_refactored_conversation():
    """Test that refactored SimpleConversation preserves functionality"""
    try:
        # Load dataset and model (same as main.py)
        logger.info("Loading dataset and model...")
        dataset = pd.read_csv('data/diabetes.csv')
        with open('data/diabetes_model_logistic_regression.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Import the refactored classes
        from main import SimpleConversation, DatasetManager, VariableStore, ConversationHistory, ExplainerManager, MetadataManager, FilterStateManager
        
        logger.info("Testing individual managers...")
        
        # Test DatasetManager
        dataset_manager = DatasetManager(dataset)
        assert dataset_manager.X_data is not None, "DatasetManager should have X_data"
        assert dataset_manager.y_data is not None, "DatasetManager should have y_data"
        dataset_contents = dataset_manager.get_dataset_contents()
        assert 'X' in dataset_contents, "Dataset contents should have X"
        assert 'y' in dataset_contents, "Dataset contents should have y"
        logger.info("✅ DatasetManager works correctly")
        
        # Test VariableStore
        var_store = VariableStore()
        var_store.add_var('test', 'test_value')
        assert var_store.get_var('test').contents == 'test_value', "VariableStore should store and retrieve variables"
        logger.info("✅ VariableStore works correctly")
        
        # Test ConversationHistory
        history = ConversationHistory()
        history.add_turn("test query", "test response")
        assert len(history.history) == 1, "ConversationHistory should track turns"
        history.store_followup_desc("test followup")
        assert history.get_followup_desc() == "test followup", "ConversationHistory should store followup"
        logger.info("✅ ConversationHistory works correctly")
        
        # Test ExplainerManager
        explainer_manager = ExplainerManager(model, dataset_manager.X_data)
        assert explainer_manager.get_explainer() is not None, "ExplainerManager should create explainer"
        logger.info("✅ ExplainerManager works correctly")
        
        # Test MetadataManager
        metadata = MetadataManager()
        assert metadata.get_class_name_from_label(0) == "No Diabetes", "MetadataManager should map labels to names"
        assert metadata.rounding_precision == 2, "MetadataManager should have default precision"
        logger.info("✅ MetadataManager works correctly")
        
        # Test FilterStateManager
        filter_manager = FilterStateManager(dataset_manager, var_store)
        filter_manager.add_interpretable_parse_op("test operation")
        assert len(filter_manager.parse_operation) == 1, "FilterStateManager should track operations"
        logger.info("✅ FilterStateManager works correctly")
        
        logger.info("Testing refactored SimpleConversation...")
        
        # Test SimpleConversation with composition
        conversation = SimpleConversation(dataset, model)
        
        # Test that all the essential methods still work
        assert conversation.get_var('dataset') is not None, "Should have dataset variable"
        assert conversation.get_var('model') is not None, "Should have model variable"
        assert conversation.get_var('mega_explainer') is not None, "Should have mega_explainer variable"
        
        # Test conversation functionality
        conversation.add_turn("test query", "test response")
        assert len(conversation.history) == 1, "Should track conversation history"
        
        # Test metadata access
        assert conversation.get_class_name_from_label(1) == "Diabetes", "Should map labels correctly"
        assert conversation.rounding_precision == 2, "Should have metadata attributes"
        
        # Test filtering functionality
        conversation.add_interpretable_parse_op("filter operation")
        assert len(conversation.parse_operation) == 1, "Should track parse operations"
        
        # Test dataset reset
        conversation.reset_temp_dataset()
        assert len(conversation.parse_operation) == 0, "Should clear parse operations on reset"
        
        logger.info("✅ SimpleConversation refactored successfully!")
        logger.info("✅ All tests passed! Refactoring preserved functionality.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing refactored SimpleConversation class...")
    success = test_refactored_conversation()
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
        exit(1)