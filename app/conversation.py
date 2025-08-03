"""Conversation management for TalkToModel"""
import os
import logging
import pandas as pd
from explainability.core.explanation import MegaExplainer, TabularDice

logger = logging.getLogger(__name__)

class Conversation:
    """Conversational context with memory for multi-turn interactions"""
    
    def __init__(self, dataset, model):
        """Initialize conversation with dataset and model.
        
        Args:
            dataset: Pandas DataFrame containing the data
            model: Trained machine learning model for predictions
        """
        # Dataset preparation - automatically detect target column
        self.target_col = 'y' if 'y' in dataset.columns else ('Outcome' if 'Outcome' in dataset.columns else None)
        if self.target_col:
            # Split features and target for supervised learning
            self.X_data = dataset.drop(self.target_col, axis=1)
            self.y_data = dataset[self.target_col]
            self.full_data = dataset
        else:
            # Handle datasets without target column
            self.X_data = dataset
            self.y_data = None
            self.full_data = dataset
        
        # Conversational memory system - tracks dialogue context
        self.history = []  # Store query-response pairs
        self.followup = ""  # Store follow-up suggestions
        self.parse_operation = []  # Track parsing operations for transparency
        self.last_parse_string = []  # Store last parsed strings
        
        # Enhanced filter state tracking - maintains context across queries
        self.last_action = None  # Name of the last action performed
        self.last_action_args = None  # Arguments used in last action
        self.last_filter_applied = None  # Details of last filter operation
        self.last_result = None  # Result of last action
        self.conversation_turns = []  # Detailed turn-by-turn conversation log
        
        # Filter state management - tracks data filtering for user clarity
        self.filter_state = {
            'has_inherited_filter': False,  # Was data filtered from previous operations?
            'current_filter_description': '',  # Human-readable description of current filter
            'query_applied_filter': False,  # Did this query apply a new filter?
            'query_filter_description': '',  # Description of filter applied by current query
            'original_size': len(dataset),  # Original dataset size
            'current_size': len(dataset)  # Current filtered dataset size
        }
        
        # Configuration settings - domain-specific metadata
        self.rounding_precision = 2  # Decimal places for numeric outputs
        self.default_metric = "accuracy"  # Default evaluation metric
        self.class_names = {0: "No Diabetes", 1: "Diabetes"}  # Human-readable class labels
        # Feature definitions with medical context for better explanations
        self.feature_definitions = {
            'Pregnancies': 'Number of times pregnant. Higher pregnancy count may increase diabetes risk due to gestational diabetes and hormonal changes.',
            'Glucose': 'Plasma glucose concentration after a 2-hour oral glucose tolerance test (mg/dL). Normal: <140, Prediabetes: 140-199, Diabetes: ≥200.',
            'BloodPressure': 'Diastolic blood pressure measured in mmHg. Normal: <80, Elevated: 80-89, High: ≥90. High blood pressure often accompanies diabetes.',
            'SkinThickness': 'Triceps skinfold thickness measured in millimeters. Used to estimate body fat percentage and insulin resistance.',
            'Insulin': 'Serum insulin level measured 2 hours after glucose load (μU/mL). Normal: 16-166. Higher levels may indicate insulin resistance.',
            'BMI': 'Body Mass Index calculated as weight(kg)/height(m)². Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30. Higher BMI increases diabetes risk.',
            'DiabetesPedigreeFunction': 'Diabetes pedigree function score representing genetic predisposition based on family history. Higher values indicate stronger genetic risk.',
            'Age': 'Age in years. Diabetes risk increases with age, especially after 45. Type 2 diabetes is more common in older adults.',
            'y': 'Target variable indicating diabetes diagnosis. 0 = No Diabetes, 1 = Diabetes. Based on WHO criteria and clinical diagnosis.'
        }
        self.username = "user"  # Default username for conversation context
        
        # Initialize core components
        self.stored_vars = self._setup_variables(model)  # Setup variables for action functions
        self._setup_explainer(model)  # Initialize LIME and SHAP explainers
        self.temp_dataset = self._create_temp_dataset()  # Working copy of dataset for filtering
        
        # Backward compatibility object for legacy describe functionality
        describe_obj = type('Describe', (), {})()
        describe_obj.get_dataset_description = lambda: "diabetes prediction based on patient health metrics"
        describe_obj.get_dataset_objective = lambda: "predict diabetes risk in patients based on health measurements"
        describe_obj.get_model_description = lambda: "Logistic Regression classifier"
        describe_obj.get_eval_performance = lambda model, metric: ""
        describe_obj.get_score_text = lambda y_true, y_pred, metric, precision, data_name: f"Model accuracy: {(y_true == y_pred).mean():.3f} on {data_name}"
        self.describe = describe_obj
    
    def _setup_variables(self, model):
        """Setup variables that action functions expect to find.
        
        Creates variable objects containing dataset, model, and utility functions
        that are used by the explainability action system.
        
        Args:
            model: Trained ML model for predictions
            
        Returns:
            dict: Dictionary of variable objects for action functions
        """
        dataset_contents = {
            'X': self.X_data,
            'y': self.y_data,
            'full_data': self.full_data,
            'cat': [],  # All numeric features for diabetes dataset
            'numeric': list(self.X_data.columns),
            'ids_to_regenerate': []
        }
        
        # Create prediction probability function for feature interaction analysis
        def prediction_probability_function(x, *args, **kwargs):
            """Wrapper for model probability predictions"""
            return model.predict_proba(x)
        
        return {
            'dataset': type('Variable', (), {'contents': dataset_contents})(),
            'model': type('Variable', (), {'contents': model})(),
            'model_prob_predict': type('Variable', (), {'contents': prediction_probability_function})(),
            'mega_explainer': None  # Will be set after explainer setup
        }
    
    def _setup_explainer(self, model):
        """Initialize LIME/SHAP explainer and TabularDice for counterfactuals.
        
        Sets up explainability tools with caching for performance.
        
        Args:
            model: Trained ML model to explain
        """
        # Wrapper function for model predictions in explainer
        def prediction_function(x):
            """Prediction function for explainer tools"""
            return model.predict_proba(x)
        
        # Setup caching directory for explainer performance
        cache_dir = os.path.join(os.getcwd(), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "mega-explainer-tabular.pkl")
        
        # Initialize MegaExplainer with LIME and SHAP capabilities
        mega_explainer = MegaExplainer(
            prediction_fn=prediction_function,
            data=self.X_data,
            cat_features=[],  # All features are numeric for diabetes dataset
            cache_location=cache_path,
            class_names=["No Diabetes", "Diabetes"],
            use_selection=True  # Enable SHAP for better explanations
        )
        
        self.stored_vars['mega_explainer'] = type('Variable', (), {'contents': mega_explainer})()
        
        # Initialize TabularDice for counterfactual explanations ("what if" scenarios)
        dice_cache_path = os.path.join(cache_dir, "dice-tabular.pkl")
        tabular_dice = TabularDice(
            model=model,
            data=self.X_data,
            num_features=list(self.X_data.columns),
            num_cfes_per_instance=10,  # Generate 10 counterfactuals per instance
            num_in_short_summary=3,  # Show top 3 in summary
            desired_class="opposite",  # Find counterfactuals for opposite class
            cache_location=dice_cache_path,
            class_names={0: "No Diabetes", 1: "Diabetes"}
        )
        
        self.stored_vars['tabular_dice'] = type('Variable', (), {'contents': tabular_dice})()
    
    def _create_temp_dataset(self):
        """Create temporary dataset for filtering operations.
        
        Creates a working copy of the dataset that can be filtered
        without affecting the original data.
        
        Returns:
            Variable object containing dataset copy
        """
        original_contents = self.stored_vars['dataset'].contents
        temp_contents = {
            'X': original_contents['X'].copy(),
            'y': original_contents['y'].copy() if original_contents['y'] is not None else None,
            'full_data': original_contents['full_data'].copy(),
            'cat': original_contents['cat'].copy(),
            'numeric': original_contents['numeric'].copy(),
            'ids_to_regenerate': original_contents['ids_to_regenerate'].copy()
        }
        return type('Variable', (), {'contents': temp_contents})()
    
    def reset_temp_dataset(self):
        """Reset temporary dataset to original full dataset.
        
        Clears any applied filters and restores the complete dataset
        for fresh analysis. Also resets filter state tracking.
        """
        import copy
        original_contents = self.stored_vars['dataset'].contents
        
        reset_contents = {
            'X': original_contents['X'].copy(),
            'y': original_contents['y'].copy() if original_contents['y'] is not None else None,
            'full_data': original_contents['full_data'].copy(),
            'cat': original_contents['cat'].copy(),
            'numeric': original_contents['numeric'].copy(),
            'ids_to_regenerate': original_contents['ids_to_regenerate'].copy()
        }
        
        self.temp_dataset = type('Variable', (), {'contents': reset_contents})()
        self.parse_operation = []
        
        # Reset filter state tracking to clean slate
        self.filter_state.update({
            'has_inherited_filter': False,
            'current_filter_description': '',
            'query_applied_filter': False,
            'query_filter_description': '',
            'current_size': self.filter_state['original_size']
        })
        
        logger.info(f"Reset temp_dataset to full dataset: {len(self.temp_dataset.contents['X'])} instances")
    
    def mark_query_filter_applied(self, filter_description, resulting_size):
        """Mark that the current query applied a new filter to the data.
        
        Args:
            filter_description: Human-readable description of the filter applied
            resulting_size: Number of records after filtering
        """
        self.filter_state.update({
            'query_applied_filter': True,
            'query_filter_description': filter_description,
            'current_size': resulting_size
        })
    
    def mark_inherited_filter(self, filter_description, current_size):
        """Mark that data was already filtered from previous operations.
        
        Args:
            filter_description: Description of existing filter state
            current_size: Current number of records in filtered dataset
        """
        self.filter_state.update({
            'has_inherited_filter': True,
            'current_filter_description': filter_description,
            'query_applied_filter': False,
            'query_filter_description': '',
            'current_size': current_size
        })
    
    def get_filter_context_for_response(self):
        """Get current filter context for response formatting.
        
        Provides information about data filtering state to help format
        responses appropriately for user understanding.
        
        Returns:
            dict: Filter context information including type, description, and sizes
        """
        if self.filter_state['query_applied_filter']:
            return {
                'type': 'query_filter',
                'description': self.filter_state['query_filter_description'],
                'original_size': self.filter_state['original_size'],
                'filtered_size': self.filter_state['current_size']
            }
        elif self.filter_state['has_inherited_filter']:
            return {
                'type': 'inherited_filter',
                'description': self.filter_state['current_filter_description'],
                'original_size': self.filter_state['original_size'],
                'filtered_size': self.filter_state['current_size']
            }
        else:
            return {
                'type': 'no_filter',
                'description': '',
                'original_size': self.filter_state['original_size'],
                'filtered_size': self.filter_state['current_size']
            }
    
    # Enhanced interface methods with conversational memory
    def add_turn(self, query, response, action_name=None, action_args=None, action_result=None):
        """Add a conversation turn to memory for context tracking.
        
        Args:
            query: User's query text
            response: System's response text
            action_name: Name of action performed (optional)
            action_args: Arguments used in action (optional)
            action_result: Result of action execution (optional)
        """
        turn_data = {
            'query': query, 
            'response': response,
            'action_name': action_name,
            'action_args': action_args,
            'timestamp': len(self.conversation_turns)
        }
        self.history.append({'query': query, 'response': response})
        self.conversation_turns.append(turn_data)
        
        # Update last operation tracking for context awareness
        if action_name:
            self.last_action = action_name
            self.last_action_args = action_args
            self.last_result = action_result
            
            # Track filter operations specifically for data state management
            if action_name == 'filter':
                self.last_filter_applied = action_args
    
    def get_var(self, name):
        """Retrieve a stored variable by name."""
        return self.stored_vars.get(name)
    
    def add_var(self, name, contents, kind=None):
        """Store a new variable for use by action functions."""
        var_obj = type('Variable', (), {'contents': contents})()
        self.stored_vars[name] = var_obj
    
    def store_followup_desc(self, desc):
        """Store follow-up suggestion for next user interaction."""
        self.followup = desc
    
    def get_followup_desc(self):
        """Retrieve stored follow-up suggestion."""
        return self.followup
    
    def add_interpretable_parse_op(self, text):
        """Add human-readable parsing operation for transparency."""
        self.parse_operation.append(text)
    
    def get_class_name_from_label(self, label):
        """Convert numeric class label to human-readable name."""
        return self.class_names.get(label, str(label))
    
    def get_feature_definition(self, feature_name):
        """Get medical definition and context for a feature."""
        return self.feature_definitions.get(feature_name, "")
    
    def build_temp_dataset(self, save=True):
        """Build temporary dataset - returns dataset variable."""
        return self.get_var('dataset')