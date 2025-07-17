"""Conversation management for TalkToModel"""
import os
import logging
import pandas as pd
from explainability.core.explanation import MegaExplainer, TabularDice

logger = logging.getLogger(__name__)

class Conversation:
    """Conversational context with memory for multi-turn interactions"""
    
    def __init__(self, dataset, model):
        # Dataset preparation (consolidated from DatasetManager)
        self.target_col = 'y' if 'y' in dataset.columns else ('Outcome' if 'Outcome' in dataset.columns else None)
        if self.target_col:
            self.X_data = dataset.drop(self.target_col, axis=1)
            self.y_data = dataset[self.target_col]
            self.full_data = dataset
        else:
            self.X_data = dataset
            self.y_data = None
            self.full_data = dataset
        
        # Conversational memory system
        self.history = []
        self.followup = ""
        self.parse_operation = []
        self.last_parse_string = []
        
        # NEW: Enhanced filter state tracking
        self.last_action = None
        self.last_action_args = None
        self.last_filter_applied = None
        self.last_result = None
        self.conversation_turns = []
        
        # NEW: Filter state management for clearer communication
        self.filter_state = {
            'has_inherited_filter': False,  # Was data filtered from previous operations?
            'current_filter_description': '',  # Human-readable description of current filter
            'query_applied_filter': False,  # Did this query apply a new filter?
            'query_filter_description': '',  # Description of filter applied by current query
            'original_size': len(dataset),
            'current_size': len(dataset)
        }
        
        # Configuration (consolidated from MetadataManager)
        self.rounding_precision = 2
        self.default_metric = "accuracy" 
        self.class_names = {0: "No Diabetes", 1: "Diabetes"}
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
        self.username = "user"
        
        # Variables (consolidated from VariableStore)
        self.stored_vars = self._setup_variables(model)
        
        # Setup explainer (consolidated from ExplainerManager)
        self._setup_explainer(model)
        
        # Initialize temp dataset
        self.temp_dataset = self._create_temp_dataset()
        
        # Create describe object for backward compatibility
        describe_obj = type('Describe', (), {})()
        describe_obj.get_dataset_description = lambda: "diabetes prediction based on patient health metrics"
        describe_obj.get_dataset_objective = lambda: "predict diabetes risk in patients based on health measurements"
        describe_obj.get_model_description = lambda: "Logistic Regression classifier"
        describe_obj.get_eval_performance = lambda model, metric: ""
        describe_obj.get_score_text = lambda y_true, y_pred, metric, precision, data_name: f"Model accuracy: {(y_true == y_pred).mean():.3f} on {data_name}"
        self.describe = describe_obj
    
    def _setup_variables(self, model):
        """Setup variables that actions expect"""
        dataset_contents = {
            'X': self.X_data,
            'y': self.y_data,
            'full_data': self.full_data,
            'cat': [],  # All numeric features for diabetes dataset
            'numeric': list(self.X_data.columns),
            'ids_to_regenerate': []
        }
        
        # Create prediction probability function for interaction analysis
        def prediction_probability_function(x, *args, **kwargs):
            return model.predict_proba(x)
        
        return {
            'dataset': type('Variable', (), {'contents': dataset_contents})(),
            'model': type('Variable', (), {'contents': model})(),
            'model_prob_predict': type('Variable', (), {'contents': prediction_probability_function})(),
            'mega_explainer': None  # Will be set after explainer setup
        }
    
    def _setup_explainer(self, model):
        """Initialize LIME explainer and TabularDice explainer"""
        def prediction_function(x):
            return model.predict_proba(x)
        
        cache_dir = os.path.join(os.getcwd(), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "mega-explainer-tabular.pkl")
        
        mega_explainer = MegaExplainer(
            prediction_fn=prediction_function,
            data=self.X_data,
            cat_features=[],
            cache_location=cache_path,
            class_names=["No Diabetes", "Diabetes"],
            use_selection=True  # Enable SHAP for better explanations
        )
        
        self.stored_vars['mega_explainer'] = type('Variable', (), {'contents': mega_explainer})()
        
        # Initialize TabularDice for counterfactual explanations
        dice_cache_path = os.path.join(cache_dir, "dice-tabular.pkl")
        tabular_dice = TabularDice(
            model=model,
            data=self.X_data,
            num_features=list(self.X_data.columns),
            num_cfes_per_instance=10,
            num_in_short_summary=3,
            desired_class="opposite",
            cache_location=dice_cache_path,
            class_names={0: "No Diabetes", 1: "Diabetes"}
        )
        
        self.stored_vars['tabular_dice'] = type('Variable', (), {'contents': tabular_dice})()
    
    def _create_temp_dataset(self):
        """Create temp dataset object"""
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
        """Reset temp dataset to full dataset"""
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
        
        # NEW: Reset filter state
        self.filter_state.update({
            'has_inherited_filter': False,
            'current_filter_description': '',
            'query_applied_filter': False,
            'query_filter_description': '',
            'current_size': self.filter_state['original_size']
        })
        
        logger.info(f"Reset temp_dataset to full dataset: {len(self.temp_dataset.contents['X'])} instances")
    
    def mark_query_filter_applied(self, filter_description, resulting_size):
        """Mark that this query applied a new filter"""
        self.filter_state.update({
            'query_applied_filter': True,
            'query_filter_description': filter_description,
            'current_size': resulting_size
        })
    
    def mark_inherited_filter(self, filter_description, current_size):
        """Mark that data was already filtered from previous operations"""
        self.filter_state.update({
            'has_inherited_filter': True,
            'current_filter_description': filter_description,
            'query_applied_filter': False,
            'query_filter_description': '',
            'current_size': current_size
        })
    
    def get_filter_context_for_response(self):
        """Get filter context for response formatting"""
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
        turn_data = {
            'query': query, 
            'response': response,
            'action_name': action_name,
            'action_args': action_args,
            'timestamp': len(self.conversation_turns)
        }
        self.history.append({'query': query, 'response': response})
        self.conversation_turns.append(turn_data)
        
        # Update last operation tracking
        if action_name:
            self.last_action = action_name
            self.last_action_args = action_args
            self.last_result = action_result
            
            # Track filter operations specifically
            if action_name == 'filter':
                self.last_filter_applied = action_args
    
    def get_var(self, name):
        return self.stored_vars.get(name)
    
    def add_var(self, name, contents, kind=None):
        var_obj = type('Variable', (), {'contents': contents})()
        self.stored_vars[name] = var_obj
    
    def store_followup_desc(self, desc):
        self.followup = desc
    
    def get_followup_desc(self):
        return self.followup
    
    def add_interpretable_parse_op(self, text):
        self.parse_operation.append(text)
    
    def get_class_name_from_label(self, label):
        return self.class_names.get(label, str(label))
    
    def get_feature_definition(self, feature_name):
        return self.feature_definitions.get(feature_name, "")
    
    def build_temp_dataset(self, save=True):
        return self.get_var('dataset')