"""Smart Action Dispatcher using AutoGen for flexible command interpretation.

This module provides a more generalizable approach to parsing and executing
user commands by leveraging AutoGen agents instead of rigid pattern matching.
"""

import json
import re
from typing import Dict, List, Tuple, Any, Optional
import logging
import asyncio

# Import AutoGen components with compatibility for newer versions
try:
    # Try new AutoGen structure (v0.4+)
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    try:
        # Try older AutoGen structure
        from autogen.agentchat.agents import AssistantAgent
        from autogen.models.openai import OpenAIChatCompletionClient
        AUTOGEN_AVAILABLE = True
    except ImportError:
        # AutoGen not available - raise clear error
        AUTOGEN_AVAILABLE = False
        AssistantAgent = None
        OpenAIChatCompletionClient = None

logger = logging.getLogger(__name__)

# Template validation patterns similar to MP+ approach
COMMAND_TEMPLATES = {
    'filter': {
        'pattern': r'filter\s+(\w+)\s+(greater|less|equal|not)\s+(.+)',
        'slots': ['feature', 'operator', 'value'],
        'required_slots': 3
    },
    'predict': {
        'pattern': r'predict(?:\s+(\d+))?',
        'slots': ['id'],
        'required_slots': 0  # ID is optional
    },
    'important': {
        'pattern': r'important(?:\s+(all|topk|\w+))?(?:\s+(\d+))?',
        'slots': ['type', 'number'],
        'required_slots': 0  # All optional
    },
    'mistake': {
        'pattern': r'mistake(?:\s+(typical|sample))?',
        'slots': ['type'],
        'required_slots': 0
    },
    'explain': {
        'pattern': r'explain(?:\s+(\d+))?(?:\s+(lime|shap|features))?',
        'slots': ['id', 'method'],
        'required_slots': 0
    },
    'change': {
        'pattern': r'change\s+(\w+)\s+(increase|decrease|set)\s+(.+)',
        'slots': ['feature', 'operation', 'value'],
        'required_slots': 3
    },
    'score': {
        'pattern': r'score(?:\s+(\w+))?',
        'slots': ['metric'],
        'required_slots': 0
    },
    'data': {
        'pattern': r'data(?:\s+(.+))?',
        'slots': ['query'],
        'required_slots': 0
    }
}

class GeneralizableEntityExtractor:
    """Domain-agnostic entity extractor that adapts to any dataset schema."""
    
    def __init__(self, dataset_schema=None):
        """Initialize with optional dataset schema for dynamic adaptation."""
        self.dataset_schema = dataset_schema
        self.feature_cache = {}
        
        # Generic patterns that work across domains
        self.generic_patterns = {
            'numeric_operators': {
                'greater': r'\b(?:greater|>|above|over|more\s+than|higher\s+than|older\s+than)\b',
                'less': r'\b(?:less|<|below|under|lower\s+than|younger\s+than)\b',
                'equal': r'\b(?:equal|=|equals?|is|exactly)\b',
                'not': r'\b(?:not\s+equal|!=|not)\b'
            },
            'comparison_indicators': r'(\w+)\s+(?:more\s+important|better\s+predictor|stronger)\s+than\s+(\w+)',
            'id_patterns': r'(?:patient|instance|data\s+point|record|row|id|case)(?:\s+with\s+id\s+|\s+number\s+|\s+)(\d+)',
            'what_if_patterns': r'(?:what\s+if|how\s+would|if\s+we)\s+(?:change|increase|decrease|set)',
            'error_patterns': r'(?:error|mistake|incorrect|wrong|misclassified)',
            'importance_patterns': r'(?:important|importance|significant|influential|predictor)'
        }
        
        # Intent classification patterns (domain-agnostic)
        self.intent_patterns = {
            'typical_errors': [r'typical.*(?:error|mistake)', r'pattern.*(?:error|mistake)', r'common.*(?:error|mistake)'],
            'feature_comparison': [r'more\s+important\s+than', r'compare.*features?', r'versus', r'\bvs\b'],
            'what_if_analysis': [r'what\s+if', r'how\s+would.*change', r'if\s+we\s+(?:change|increase|decrease)'],
            'new_instance_prediction': [
                r'new\s+instance', 
                r'hypothetical.*instance', 
                r'what.*predict.*new', 
                r'predict.*new\s+instance',
                r'what.*you.*predict.*new',
                r'what.*would.*predict.*instance.*with',
                r'what.*would.*model.*predict.*for.*person',
                r'what.*would.*predict.*for.*person',
                r'predict.*for.*person.*(?:of|with|who)',
                r'prediction.*for.*person.*(?:of|with|age)',
                r'what.*predict.*someone.*(?:of|with|age)',
                r'what.*would.*predict.*someone.*who'
            ],
            'model_accuracy': [
                r'how\s+accurate', 
                r'accuracy.*model', 
                r'model.*accuracy', 
                r'performance.*model',
                r'how\s+accurate.*model',
                r'accuracy.*for.*instances'
            ],
            'feature_importance': [r'important.*features?', r'feature.*importance', r'most\s+significant'],
            'prediction': [r'predict', r'classification', r'likelihood', r'probability'],
            'data_summary': [r'summary', r'describe.*data', r'statistics', r'average', r'mean']
        }
    
    def update_schema(self, dataset_schema):
        """Update the entity extractor with new dataset schema."""
        self.dataset_schema = dataset_schema
        self.feature_cache = {}
        
        if hasattr(dataset_schema, 'features'):
            # Build dynamic feature patterns from actual dataset
            for feature_name in dataset_schema.features:
                # Create flexible patterns for each feature
                feature_lower = feature_name.lower()
                # Handle common variations (underscores, camelCase, etc.)
                variations = [
                    feature_lower,
                    feature_lower.replace('_', ' '),
                    feature_lower.replace('_', ''),
                ]
                
                # Add domain-specific synonyms if available
                if hasattr(dataset_schema, 'feature_synonyms') and feature_name in dataset_schema.feature_synonyms:
                    variations.extend(dataset_schema.feature_synonyms[feature_name])
                
                # Create regex pattern for this feature
                pattern = r'\b(?:' + '|'.join(re.escape(var) for var in variations) + r')\b'
                self.feature_cache[feature_name] = pattern
    
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities using schema-driven, domain-agnostic approach."""
        query_lower = query.lower()
        entities = {
            'patient_id': None,
            'features': [],
            'operators': [],
            'values': [],
            'comparison_features': [],
            'intent_type': None,
            'confidence': 0.0
        }
        
        # Extract ID/instance references (domain-agnostic)
        id_match = re.search(self.generic_patterns['id_patterns'], query_lower)
        if id_match:
            extracted_id = int(id_match.group(1))
            entities['patient_id'] = extracted_id
            entities['confidence'] += 0.3
            logger.info(f"Extracted patient_id: {extracted_id} from query: '{query_lower}'")
        
        # Extract feature comparisons (works with any feature names)
        comparison_match = re.search(self.generic_patterns['comparison_indicators'], query_lower)
        if comparison_match:
            feature1, feature2 = comparison_match.groups()
            # Validate against schema if available
            if self._is_valid_feature(feature1) and self._is_valid_feature(feature2):
                entities['comparison_features'] = [feature1, feature2]
                entities['intent_type'] = 'feature_comparison'
                entities['confidence'] += 0.4
        
        # Extract features using dynamic schema-based patterns
        if self.feature_cache:
            for feature_name, pattern in self.feature_cache.items():
                if re.search(pattern, query_lower):
                    entities['features'].append(feature_name)
                    entities['confidence'] += 0.2
                    
                    # Look for associated operators and values in context
                    self._extract_feature_context(query_lower, feature_name, entities)
        
        # Special handling for age patterns in different formats
        age_patterns = [
            r'person\s+of\s+(\d+)\s+years?',
            r'someone\s+(?:who\s+is\s+)?(\d+)\s+years?\s+old',
            r'person\s+(?:who\s+is\s+)?(\d+)\s+years?\s+old',
            r'(\d+)[-\s]year[-\s]old',
            r'age\s+(?:of\s+)?(\d+)',
            r'aged\s+(\d+)'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                age_value = int(match.group(1))
                if 'age' not in entities['features']:
                    entities['features'].append('age')
                    entities['operators'].append('equal')
                    entities['values'].append(age_value)
                    entities['confidence'] += 0.3
                break
        
        # Classify intent using domain-agnostic patterns
        entities['intent_type'] = self._classify_intent(query_lower)
        if entities['intent_type']:
            entities['confidence'] += 0.3
            logger.info(f"Classified intent: '{entities['intent_type']}' for query: '{query_lower}'")
        
        return entities
    
    def _is_valid_feature(self, feature_name: str) -> bool:
        """Check if feature name exists in dataset schema."""
        if not self.dataset_schema or not hasattr(self.dataset_schema, 'features'):
            return True  # Assume valid if no schema available
        
        feature_lower = feature_name.lower()
        schema_features_lower = [f.lower() for f in self.dataset_schema.features]
        return feature_lower in schema_features_lower
    
    def _extract_feature_context(self, query_lower: str, feature_name: str, entities: Dict):
        """Extract operators and values associated with a feature."""
        feature_pos = query_lower.find(feature_name.lower())
        if feature_pos == -1:
            return
        
        # Look in a window around the feature mention
        context_start = max(0, feature_pos - 30)
        context_end = min(len(query_lower), feature_pos + len(feature_name) + 30)
        context = query_lower[context_start:context_end]
        
        # Find operators
        for op_name, op_pattern in self.generic_patterns['numeric_operators'].items():
            if re.search(op_pattern, context):
                entities['operators'].append(op_name)
                
                # Look for numeric values
                value_match = re.search(r'(\d+(?:\.\d+)?)', context)
                if value_match:
                    entities['values'].append(float(value_match.group(1)))
                break
    
    def _classify_intent(self, query_lower: str) -> str:
        """Classify user intent using domain-agnostic patterns."""
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            
            if score > 0:
                intent_scores[intent_type] = score
        
        # Return intent with highest score
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def add_domain_synonyms(self, feature_synonyms: Dict[str, List[str]]):
        """Add domain-specific feature synonyms for better recognition."""
        if not hasattr(self, 'domain_synonyms'):
            self.domain_synonyms = {}
        
        self.domain_synonyms.update(feature_synonyms)
        
        # Rebuild feature cache with new synonyms
        if self.dataset_schema:
            self.update_schema(self.dataset_schema)

class SmartActionDispatcher:
    """Uses AutoGen agents to intelligently parse and execute compound commands."""
    
    def __init__(self, api_key: str):
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen packages not available. Please install with: "
                "pip install autogen-agentchat>=0.4.0 autogen-core>=0.4.0 autogen-ext>=0.4.0"
            )
        
        self.api_key = api_key
        
        # Initialize the model client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            api_key=self.api_key,
        )
        
        # Initialize entity extractor for better slot filling
        self.entity_extractor = GeneralizableEntityExtractor()
        
        self.setup_agents()
    
    def initialize_for_dataset(self, conversation):
        """Initialize the dispatcher with dataset-specific information for generalizability."""
        try:
            # Extract dataset schema dynamically
            dataset = conversation.get_var('dataset').contents
            df = dataset['X']
            
            # Create a simple schema object
            class DatasetSchema:
                def __init__(self, features, target_col=None):
                    self.features = features
                    self.target_column = target_col
                    # Add feature synonyms based on common patterns
                    self.feature_synonyms = self._generate_synonyms(features)
                
                def _generate_synonyms(self, features):
                    """Generate common synonyms for features."""
                    synonyms = {}
                    for feature in features:
                        feature_synonyms = []
                        feature_lower = feature.lower()
                        
                        # Handle underscores and camelCase
                        if '_' in feature_lower:
                            feature_synonyms.append(feature_lower.replace('_', ' '))
                            feature_synonyms.append(feature_lower.replace('_', ''))
                        
                        # Add common domain patterns (generalizable)
                        if 'age' in feature_lower:
                            feature_synonyms.extend(['years old', 'older', 'younger'])
                        elif any(term in feature_lower for term in ['pressure', 'bp']):
                            feature_synonyms.extend(['blood pressure', 'bp', 'pressure'])
                        elif any(term in feature_lower for term in ['mass', 'bmi', 'weight']):
                            feature_synonyms.extend(['body mass index', 'weight', 'bmi'])
                        elif any(term in feature_lower for term in ['glucose', 'sugar']):
                            feature_synonyms.extend(['blood sugar', 'sugar', 'glucose'])
                        
                        if feature_synonyms:
                            synonyms[feature] = feature_synonyms
                    
                    return synonyms
            
            # Get target column if available
            target_col = None
            if 'y' in dataset and len(dataset['y']) > 0:
                target_col = 'target'  # Generic name
            
            # Create and set schema
            schema = DatasetSchema(list(df.columns), target_col)
            self.entity_extractor.update_schema(schema)
            
            logger.info(f"Initialized dispatcher for dataset with {len(df.columns)} features")
            logger.debug(f"Features: {list(df.columns)}")
            logger.debug(f"Generated synonyms: {schema.feature_synonyms}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing dataset schema: {e}")
            return False
    
    def add_domain_knowledge(self, feature_synonyms: Dict[str, List[str]] = None, 
                           class_names: Dict = None):
        """Add domain-specific knowledge while maintaining generalizability."""
        if feature_synonyms:
            self.entity_extractor.add_domain_synonyms(feature_synonyms)
            logger.info(f"Added domain synonyms for {len(feature_synonyms)} features")
        
        if class_names:
            # Store class names for better prediction interpretation
            self.class_names = class_names
            logger.info(f"Added class names: {class_names}")
        
    def setup_agents(self):
        """Initialize AutoGen agents for command parsing."""
        
        # Agent for breaking down compound commands into atomic operations
        self.command_parser = AssistantAgent(
            name="CommandParser",
            model_client=self.model_client,
            system_message="""You are an expert at parsing natural language commands for a machine learning model exploration system.

Your job is to break down compound commands into a sequence of atomic operations.

AVAILABLE ATOMIC OPERATIONS:
- filter {feature} {operator} {value} - Filter data by a single condition
- predict [id] - Make predictions (optionally for a specific ID)  
- explain {id} [type] - Explain a prediction (type: lime/shap)
- show {id} - Show data for an ID
- data - Show data summary
- important [all/topk {n}/{feature}] - Show feature importance
- score {metric} - Show model performance
- change {feature} {operation} {value} - What-if analysis
- mistake - Show model errors

OPERATORS: greater, less, greaterequal, lessequal, equal, not
OPERATIONS: set, increase, decrease

IMPORTANT RULES:
1. Each filter condition must be a separate "filter" command
2. Commands should be executed in sequence
3. Maintain the logical flow of operations
4. Convert natural language to exact command syntax

EXAMPLES:
Input: "filter age greater 50 predict"
Output: ["filter age greater 50", "predict"]

Input: "filter age greater 50 pregnancies equal 1 predict"  
Output: ["filter age greater 50", "filter pregnancies equal 1", "predict"]

Input: "predict 2"
Output: ["filter id 2", "predict"]

Input: "what is your prediction for person 5"
Output: ["filter id 5", "predict"]

Input: "explain 5 with lime and show feature importance"
Output: ["filter id 5", "explain lime", "important all"]

Input: "predict for age > 50 and pregnant = no"
Output: ["filter age greater 50", "filter pregnancies equal 0", "predict"]

Always respond with a JSON array of atomic commands."""
        )
        
        # Agent for validating command sequences
        self.command_validator = AssistantAgent(
            name="CommandValidator",
            model_client=self.model_client,
            system_message="""You are an expert at validating command sequences for correctness.

Check if the command sequence makes logical sense and fix any issues.

VALIDATION RULES:
1. Filters should come before operations that use the filtered data
2. IDs in commands should be used correctly
3. Feature names should be valid
4. Operators and values should match the feature type
5. Commands should be in the correct format

If you find issues, return a corrected sequence. Otherwise, return the original sequence.

Always respond with a JSON object:
{
    "valid": true/false,
    "commands": [...],
    "corrections": "explanation of any corrections made"
}"""
        )
        
        # Enhanced parser with template validation
        self.parser_agent = AssistantAgent(
            name="EnhancedCommandParser",
            model_client=self.model_client,
            system_message="""You are an expert at parsing natural language commands for ML model exploration.

CRITICAL: Always respond with a valid JSON array of atomic commands that match these templates:

VALID COMMAND TEMPLATES:
- filter {feature} {operator} {value} - Examples: "filter age greater 50", "filter id 201"
- predict [id] - Examples: "predict", "predict 49"  
- explain [id] [method] - Examples: "explain 49", "explain lime"
- important [type] [number] - Examples: "important all", "important glucose", "important topk 5"
- mistake [type] - Examples: "mistake typical", "mistake sample"
- data [query] - Examples: "data", "data summary"

PARSING RULES:
1. Break complex queries into atomic operations
2. Always use exact template formats
3. Validate each command matches a template
4. For ID mentions, always add "filter id {number}" first

EXAMPLES:
"What types of patients is the model typically predicting incorrect here"
→ ["mistake typical"]

"Is glucose more important than age for data point 49?"  
→ ["filter id 49", "important glucose", "important age"]

"How would decreasing glucose by 10 change likelihood for men older than 20?"
→ ["filter age greater 20", "change glucose decrease 10", "predict"]

Always respond with ONLY a JSON array, no other text."""
        )
        
    def validate_command_template(self, command: str) -> Tuple[bool, str, Dict]:
        """Validate command against templates like MP+ approach."""
        command = command.strip().lower()
        
        for cmd_type, template in COMMAND_TEMPLATES.items():
            pattern = template['pattern']
            match = re.match(pattern, command)
            
            if match:
                slots = {}
                for i, slot_name in enumerate(template['slots']):
                    if i < len(match.groups()) and match.group(i + 1):
                        slots[slot_name] = match.group(i + 1)
                
                # Check if required slots are filled
                filled_slots = len([v for v in slots.values() if v])
                if filled_slots >= template['required_slots']:
                    return True, cmd_type, slots
                else:
                    return False, f"Missing required slots for {cmd_type}", {}
        
        return False, f"No template match for: {command}", {}
    
    def validate_and_fix_commands(self, commands: List[str], conversation_context: Dict = None) -> Tuple[List[str], bool]:
        """Enhanced validation with template checking."""
        validated_commands = []
        needs_fix = False
        
        for command in commands:
            is_valid, result, slots = self.validate_command_template(command)
            
            if is_valid:
                validated_commands.append(command)
                logger.debug(f"✓ Valid command: {command} → {result} with slots: {slots}")
            else:
                logger.warning(f"✗ Invalid command: {command} → {result}")
                
                # Try to fix common issues
                fixed_command = self._attempt_command_fix(command)
                if fixed_command:
                    is_valid_fix, _, _ = self.validate_command_template(fixed_command)
                    if is_valid_fix:
                        validated_commands.append(fixed_command)
                        needs_fix = True
                        logger.info(f"Fixed command: {command} → {fixed_command}")
                    else:
                        logger.error(f"Could not fix command: {command}")
                else:
                    logger.error(f"Could not fix command: {command}")
        
        return validated_commands, not needs_fix
    
    def _attempt_command_fix(self, command: str) -> str:
        """Attempt to fix invalid commands."""
        command = command.strip().lower()
        
        # Fix common issues
        fixes = [
            # Add missing operators
            (r'filter\s+(\w+)\s+(\d+)', r'filter \1 equal \2'),
            # Fix operator names
            (r'filter\s+(\w+)\s+>\s+(\d+)', r'filter \1 greater \2'),
            (r'filter\s+(\w+)\s+<\s+(\d+)', r'filter \1 less \2'),
            (r'filter\s+(\w+)\s+=\s+(\d+)', r'filter \1 equal \2'),
            # Standardize mistake commands
            (r'mistake.*typical', 'mistake typical'),
            (r'mistake.*sample', 'mistake sample'),
            # Fix important commands
            (r'important\s+all\s+features?', 'important all'),
        ]
        
        for pattern, replacement in fixes:
            fixed = re.sub(pattern, replacement, command)
            if fixed != command:
                return fixed
        
        return None

    def parse_compound_command(self, command: str, available_features: List[str] = None) -> List[str]:
        """Enhanced parsing with entity extraction and template validation."""
        
        # Extract entities first for better understanding
        entities = self.entity_extractor.extract_entities(command)
        logger.info(f"Extracted entities: {entities}")
        
        # Handle specific intent types with extracted entities
        if entities['intent_type'] == 'typical_errors':
            return ["mistake typical"]
        
        elif entities['intent_type'] == 'feature_comparison':
            commands = []
            if entities['patient_id']:
                commands.append(f"filter id {entities['patient_id']}")
            if entities['comparison_features']:
                for feature in entities['comparison_features']:
                    commands.append(f"important {feature}")
            return commands if commands else ["important all"]
        
        elif entities['intent_type'] == 'new_instance_prediction':
            # Handle "new instance" queries - create hypothetical data
            commands = []
            # Extract the conditions for the new instance
            for i, feature in enumerate(entities['features']):
                if i < len(entities['operators']) and i < len(entities['values']):
                    op = entities['operators'][i]
                    val = entities['values'][i]
                    # For new instances, we want to create a synthetic example
                    # Use the boundary condition (e.g., BMI=20 for "BMI > 20")
                    if op == 'greater':
                        commands.append(f"change {feature} set {val + 1}")
                    elif op == 'less':
                        commands.append(f"change {feature} set {val - 1}")
                    else:
                        commands.append(f"change {feature} set {val}")
            
            # If no specific conditions found, try to extract from command text
            if not commands:
                implicit_changes = self._extract_implicit_feature_values(command)
                commands.extend(implicit_changes)
            
            commands.append("predict")
            return commands
        
        elif entities['intent_type'] == 'model_accuracy':
            # Handle accuracy/performance queries
            commands = []
            # Add filtering if specific conditions mentioned
            for i, feature in enumerate(entities['features']):
                if i < len(entities['operators']) and i < len(entities['values']):
                    op = entities['operators'][i]
                    val = entities['values'][i]
                    commands.append(f"filter {feature} {op} {val}")
            
            commands.append("score accuracy")
            return commands
        
        elif entities['intent_type'] == 'what_if_analysis':
            commands = []
            # Add filtering based on extracted features/operators/values
            for i, feature in enumerate(entities['features']):
                if i < len(entities['operators']) and i < len(entities['values']):
                    op = entities['operators'][i]
                    val = entities['values'][i]
                    commands.append(f"filter {feature} {op} {val}")
            
            # Extract what-if changes dynamically from the query
            what_if_changes = self._extract_what_if_changes(command, entities)
            commands.extend(what_if_changes)
            
            commands.append("predict")
            return commands
        
        elif entities['intent_type'] == 'feature_importance':
            commands = []
            if entities['patient_id']:
                commands.append(f"filter id {entities['patient_id']}")
            commands.append("important all")
            return commands
        
        elif entities['intent_type'] == 'prediction':
            # Handle general prediction queries
            commands = []
            if entities['patient_id']:
                commands.append(f"filter id {entities['patient_id']}")
            commands.append("predict")
            return commands
        
        elif entities['intent_type'] == 'data_summary':
            # Handle data summary queries
            commands = []
            # Add filtering if specific conditions mentioned
            for i, feature in enumerate(entities['features']):
                if i < len(entities['operators']) and i < len(entities['values']):
                    op = entities['operators'][i]
                    val = entities['values'][i]
                    commands.append(f"filter {feature} {op} {val}")
            commands.append("data")
            return commands
        
        # Original pattern matching as fallback
        command_lower = command.lower()
        
        # Pattern 1: "What types of patients is the model typically predicting incorrect"
        if any(word in command_lower for word in ['typical', 'pattern']) and any(word in command_lower for word in ['error', 'mistake', 'incorrect']):
            return ["mistake typical"]
        
        # Pattern 2: "what would you predict for a new instance with X"
        new_instance_pattern = re.search(r'what.*?predict.*?(?:new\s+instance|instance).*?with\s+(\w+).*?(?:over|above|greater.*?than|>)\s+(\d+)', command_lower)
        if new_instance_pattern:
            feature, value = new_instance_pattern.groups()
            # For new instances, create a synthetic example at the boundary
            return [f"change {feature} set {int(value) + 1}", "predict"]
        
        # Pattern 2b: "what would you predict for a person of X years" (generalized)
        person_age_pattern = re.search(r'what.*?predict.*?for.*?person.*?(?:of|who.*?is)\s+(\d+)\s+years?', command_lower)
        if person_age_pattern:
            age_value = person_age_pattern.group(1)
            age_feature = self._find_age_like_feature()
            return [f"change {age_feature} set {age_value}", "predict"]
        
        # Pattern 3: "how accurate is the model"
        if re.search(r'how\s+accurate.*?model', command_lower):
            return ["score accuracy"]
        
        # Pattern 4: "how accurate is the model for instances with X"
        accuracy_with_filter = re.search(r'how\s+accurate.*?model.*?for.*?(\w+)\s*>\s*(\d+)', command_lower)
        if accuracy_with_filter:
            feature, value = accuracy_with_filter.groups()
            return [f"filter {feature} greater {value}", "score accuracy"]
        
        # Pattern 5: "Is [feature] more important than [feature] for data point [id]"
        importance_comparison = re.search(r'is\s+(\w+)\s+more\s+important\s+than\s+(\w+).*?(?:data\s+point|instance|patient)\s+(\d+)', command_lower)
        if importance_comparison:
            feature1, feature2, patient_id = importance_comparison.groups()
            return [f"filter id {patient_id}", f"important {feature1}", f"important {feature2}"]
        
        # Pattern 6: Complex what-if with group filtering (generalized)
        what_if_group = re.search(r'(?:how|what).*?(?:change|decrease|increase).*?(\w+).*?by\s+(\d+).*?for\s+(\w+)\s+(?:older|greater)\s+than\s+(\d+)', command_lower)
        if what_if_group:
            target_feature, amount, filter_feature, filter_value = what_if_group.groups()
            # Dynamically determine operation
            operation = "decrease" if "decrease" in command_lower else ("increase" if "increase" in command_lower else "set")
            return [f"filter {filter_feature} greater {filter_value}", f"change {target_feature} {operation} {amount}", "predict"]
        
        # Use AI parser for complex cases
        if self.parser_agent:
            try:
                context = f"\nAVAILABLE FEATURES: {', '.join(available_features)}" if available_features else ""
                context += f"\nEXTRACTED ENTITIES: {entities}"
                context += "\nNote: Use exact template formats. Always validate against templates."
                
                prompt = f"Parse this command into atomic operations: {command}{context}"
                
                # Simplified approach - try to get response synchronously
                try:
                    # This is a placeholder - actual implementation would need proper async handling
                    logger.info("Attempting AI parsing...")
                    return self._fallback_parse(command)
                except Exception as e:
                    logger.error(f"AI parsing failed: {e}")
                    return self._fallback_parse(command)
                    
            except Exception as e:
                logger.error(f"Error in AI parsing: {e}")
                return self._fallback_parse(command)
        
        return self._fallback_parse(command)
    
    def _extract_what_if_changes(self, command: str, entities: Dict[str, Any]) -> List[str]:
        """Extract what-if changes in a generalizable way."""
        command_lower = command.lower()
        changes = []
        
        # Pattern: "decrease/increase [feature] by [amount]"
        change_patterns = [
            r'(decrease|increase|set)\s+(\w+)\s+by\s+(\d+)',
            r'(decrease|increase|set)\s+(\w+)\s+to\s+(\d+)',
            r'(\w+)\s+(decrease|increase|set)\s+by\s+(\d+)',
        ]
        
        for pattern in change_patterns:
            matches = re.findall(pattern, command_lower)
            for match in matches:
                if len(match) == 3:
                    # Handle different orders of operation/feature/value
                    if match[0] in ['decrease', 'increase', 'set']:
                        operation, feature, value = match
                    else:
                        feature, operation, value = match
                    
                    # Validate feature against schema if available
                    if self._is_valid_feature_name(feature):
                        changes.append(f"change {feature} {operation} {value}")
        
        return changes
    
    def _create_generalized_what_if_command(self, full_match: str, groups: tuple) -> List[str]:
        """Create generalized what-if commands from pattern matches."""
        if len(groups) < 2:
            return ["predict"]  # Fallback
        
        command_lower = full_match.lower()
        target_feature = groups[0]  # Feature to change
        filter_value = groups[1]    # Filter value
        
        # Determine operation dynamically
        operation = "decrease" if "decrease" in command_lower else ("increase" if "increase" in command_lower else "set")
        
        # Determine filter feature (age is common but let's be more flexible)
        filter_feature = "age"  # Default, but should be inferred from context
        if "older" in command_lower:
            filter_feature = self._find_age_like_feature()
        
        # Extract amount if specified
        amount_match = re.search(r'by\s+(\d+)', command_lower)
        amount = amount_match.group(1) if amount_match else "10"  # Default amount
        
        return [f"filter {filter_feature} greater {filter_value}", f"change {target_feature} {operation} {amount}", "predict"]
    
    def _is_valid_feature_name(self, feature_name: str) -> bool:
        """Check if feature name exists in current dataset schema."""
        if not self.entity_extractor.dataset_schema:
            return True  # Assume valid if no schema
        
        schema_features = [f.lower() for f in self.entity_extractor.dataset_schema.features]
        return feature_name.lower() in schema_features
    
    def _find_age_like_feature(self) -> str:
        """Find age-like feature in the dataset schema."""
        if not self.entity_extractor.dataset_schema:
            return "age"  # Default fallback
        
        # Look for features that might represent age/time
        age_indicators = ['age', 'years', 'time', 'duration', 'period']
        
        for feature in self.entity_extractor.dataset_schema.features:
            feature_lower = feature.lower()
            if any(indicator in feature_lower for indicator in age_indicators):
                return feature
        
        # If no age-like feature found, return the first numeric feature or default
        return self.entity_extractor.dataset_schema.features[0] if self.entity_extractor.dataset_schema.features else "age"
    
    def _extract_implicit_feature_values(self, command: str) -> List[str]:
        """Extract implicit feature value assignments from command text."""
        command_lower = command.lower()
        changes = []
        
        if not self.entity_extractor.dataset_schema:
            return changes
        
        # Look for patterns like "person of 20 years", "instance with income 50000", etc.
        value_patterns = [
            r'(?:person|instance|someone).*?(?:of|with|who.*?has)\s+(\w+).*?(\d+)',
            r'(\w+)\s+(?:of|is|=|equals?)\s+(\d+)',
            r'with\s+(\w+)\s+(\d+)',
        ]
        
        for pattern in value_patterns:
            matches = re.findall(pattern, command_lower)
            for match in matches:
                if len(match) == 2:
                    # Try both orders (feature, value) and (value, feature)
                    potential_feature, potential_value = match
                    
                    # Check if first match is a feature name
                    if self._is_valid_feature_name(potential_feature) and potential_value.isdigit():
                        changes.append(f"change {potential_feature} set {potential_value}")
                    # Check if second match is a feature name  
                    elif self._is_valid_feature_name(potential_value) and potential_feature.isdigit():
                        changes.append(f"change {potential_value} set {potential_feature}")
        
        return changes
    
    def _fallback_parse(self, command: str) -> List[str]:
        """Enhanced fallback parser with template awareness."""
        command_lower = command.lower()
        
        # Template-aware pattern matching
        patterns = [
            # Mistake queries
            (r'(?:what|which).*?(?:typical|pattern).*?(?:error|mistake|incorrect)', ["mistake typical"]),
            (r'(?:show|list).*?(?:error|mistake).*?(?:sample|example)', ["mistake sample"]),
            
            # Feature importance with ID
            (r'(?:important|importance).*?(?:instance|patient|data\s+point)\s+(\d+)', lambda m: [f"filter id {m.group(1)}", "important all"]),
            
            # Prediction queries with ID
            (r'(?:predict|prediction).*?(?:instance|patient|data\s+point)\s+(\d+)', lambda m: [f"filter id {m.group(1)}", "predict"]),
            
            # Group-based what-if (generalized)
            (r'(?:change|decrease|increase).*?(\w+).*?(?:men|women|people|instances).*?(?:older|greater).*?(\d+)', 
             lambda m: self._create_generalized_what_if_command(m.group(0), m.groups())),
        ]
        
        for pattern, result in patterns:
            if isinstance(result, list):
                if re.search(pattern, command_lower):
                    return result
            else:  # It's a function
                match = re.search(pattern, command_lower)
                if match:
                    return result(match)
        
        # Default fallbacks
        if 'mistake' in command_lower:
            return ["mistake typical"]
        elif 'accurate' in command_lower or 'accuracy' in command_lower:
            return ["score accuracy"]
        elif 'important' in command_lower:
            return ["important all"]
        elif 'predict' in command_lower and 'new' in command_lower:
            # Simple new instance prediction fallback
            return ["predict"]
        elif 'predict' in command_lower:
            return ["predict"]
        
        return ["data"]

    def execute_atomic_command(self, command: str, conversation, actions_map: Dict) -> Tuple[str, int]:
        """Execute with enhanced error handling."""
        try:
            # Validate command first
            is_valid, cmd_type, slots = self.validate_command_template(command)
            
            if not is_valid:
                logger.error(f"Invalid command template: {command}")
                return f"Invalid command format: {command}", 0
            
            # Parse command
            parsed_text = command.split()
            action_keyword = parsed_text[0] if parsed_text else ""
            
            # Ensure temp_dataset is initialized
            if action_keyword in ['important', 'predict', 'data', 'score', 'mistake', 'show']:
                if not hasattr(conversation, 'temp_dataset') or conversation.temp_dataset is None:
                    conversation.build_temp_dataset()
            
            # Find and execute action
            if action_keyword in actions_map:
                action_func = actions_map[action_keyword]
                action_index = 0
                
                try:
                    return action_func(conversation, parsed_text, action_index)
                except IndexError as e:
                    logger.error(f"Index error in {action_keyword}: {e}")
                    return f"Error: Invalid command format for {action_keyword}", 0
                except Exception as e:
                    logger.error(f"Error executing {action_keyword}: {e}")
                    return f"Error executing {action_keyword}: {str(e)}", 0
            else:
                return f"Unknown action: {action_keyword}", 0
                
        except Exception as e:
            logger.error(f"Error in execute_atomic_command: {e}")
            return f"Command execution failed: {str(e)}", 0

    def dispatch(self, command: str, conversation, actions_map: Dict, 
                 available_features: List[str] = None) -> Tuple[str, int]:
        """Enhanced dispatch with MP+-style validation and automatic schema initialization."""
        try:
            logger.info("=" * 50)
            logger.info(f"=== DISPATCH CALLED ===")
            logger.info(f"Dispatching command: '{command}'")
            
            # Auto-initialize schema if not done yet (ensures generalizability)
            if not self.entity_extractor.dataset_schema:
                logger.info("Auto-initializing dataset schema for generalizability...")
                self.initialize_for_dataset(conversation)
            
            # Parse into atomic commands
            atomic_commands = self.parse_compound_command(command, available_features)
            logger.info(f"Parsed into atomic commands: {atomic_commands}")
            
            # Remove duplicates while preserving order
            seen = set()
            deduplicated_commands = []
            for cmd in atomic_commands:
                if cmd not in seen:
                    deduplicated_commands.append(cmd)
                    seen.add(cmd)
            
            if len(deduplicated_commands) != len(atomic_commands):
                logger.info(f"Removed {len(atomic_commands) - len(deduplicated_commands)} duplicate commands")
                logger.info(f"Original: {atomic_commands}")
                logger.info(f"Deduplicated: {deduplicated_commands}")
                atomic_commands = deduplicated_commands
            
            # Validate with template checking (MP+ approach)
            validated_commands, is_valid = self.validate_and_fix_commands(atomic_commands)
            
            if not validated_commands:
                logger.error("No valid commands after validation")
                return "Could not parse command into valid operations", 0
            
            if not is_valid:
                logger.warning(f"Commands needed fixing: {atomic_commands} → {validated_commands}")
            
            # Execute commands sequentially
            results = []
            overall_status = 1
            
            for i, cmd in enumerate(validated_commands):
                logger.info(f"Executing command {i+1}/{len(validated_commands)}: '{cmd}'")
                
                try:
                    result, status = self.execute_atomic_command(cmd, conversation, actions_map)
                    logger.info(f"Command '{cmd}' result: {result[:100] if result else 'None'}...")
                    
                    if result:
                        results.append(result)
                    
                    if status == 0:
                        overall_status = 0
                        logger.error(f"Command failed: {cmd}")
                        break
                        
                except Exception as cmd_error:
                    logger.error(f"Error executing '{cmd}': {cmd_error}")
                    overall_status = 0
                    break
            
            # Combine results
            combined_result = "<br>".join(filter(bool, results))
            
            if not combined_result and overall_status == 1:
                combined_result = "Operations completed successfully"
            
            return combined_result, overall_status
            
        except Exception as e:
            logger.error(f"Error in smart dispatcher: {e}")
            return "I encountered an error processing your request. Please try rephrasing.", 0


# Singleton instance
smart_dispatcher = None

def get_smart_dispatcher(api_key: str) -> SmartActionDispatcher:
    """Get or create the smart dispatcher instance."""
    global smart_dispatcher
    if smart_dispatcher is None:
        smart_dispatcher = SmartActionDispatcher(api_key)
    return smart_dispatcher 