"""Multi-Agent Natural Language Understanding System for Conversational Machine Learning

This module implements a sophisticated multi-agent architecture for parsing and understanding
natural language queries in the context of machine learning model exploration. The system
is based on the AutoGen framework and employs a three-stage processing pipeline that
separates concerns between intent understanding, action planning, and validation.

Theoretical Foundation:
    The architecture follows the principle of computational specialization, where different
    agents are responsible for distinct aspects of the natural language understanding task.
    This approach draws from distributed cognitive systems theory and multi-agent AI research,
    allowing for better error isolation, improved maintainability, and enhanced robustness.

Architecture Overview:
    1. Intent Extraction Agent: Performs semantic analysis to identify user goals and extract
       relevant entities from natural language input
    2. Action Planning Agent: Translates extracted intents and entities into executable
       action syntax using domain-specific grammar rules
    3. Validation Agent: Performs syntactic and semantic validation of generated actions,
       ensuring consistency with the underlying data model and system constraints

Research Contributions:
    - Novel application of multi-agent systems to conversational ML interfaces
    - Robust error handling through redundant validation layers
    - Flexible architecture that can adapt to different domain vocabularies
    - Integration of modern LLMs with structured action systems

Author: [Your Name] - PhD Thesis Research
Institution: [Your Institution]
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple

# Configure logging for research and debugging purposes
logger = logging.getLogger(__name__)

# AutoGen Framework Integration
# The AutoGen library provides the foundational multi-agent infrastructure for our system.
# We implement version-agnostic imports to ensure compatibility across different releases,
# as the API has evolved significantly between versions 0.4 and 0.6+
try:
    # Modern AutoGen architecture (v0.4+)
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
    logger.info("Successfully imported modern AutoGen components (v0.4+)")
except ImportError:
    try:
        # Legacy AutoGen architecture (pre-v0.4)
        from autogen.agentchat.agents import AssistantAgent
        from autogen.agentchat.teams import RoundRobinGroupChat
        from autogen.models.openai import OpenAIChatCompletionClient
        AUTOGEN_AVAILABLE = True
        logger.info("Successfully imported legacy AutoGen components")
    except ImportError:
        # Graceful degradation when AutoGen is not available
        # This allows the system to fail gracefully and provide meaningful error messages
        AUTOGEN_AVAILABLE = False
        AssistantAgent = None
        RoundRobinGroupChat = None
        OpenAIChatCompletionClient = None
        logger.warning("AutoGen framework not available - decoder will be disabled")


class AutoGenDecoder:
    """
    Multi-Agent Natural Language Understanding Decoder
    
    This class implements the core multi-agent architecture for natural language understanding
    in conversational machine learning interfaces. The system employs three specialized agents
    that work collaboratively to transform user queries into executable actions.
    
    Design Philosophy:
        The architecture is based on the principle of separation of concerns, where each agent
        has a well-defined responsibility. This design choice provides several advantages:
        
        1. Maintainability: Each agent can be modified independently without affecting others
        2. Debuggability: Issues can be traced to specific processing stages
        3. Extensibility: New agents can be added to handle additional processing requirements
        4. Robustness: Failures in one agent don't necessarily cascade to others
    
    Research Context:
        This implementation addresses the challenge of bridging the gap between natural language
        queries and structured action execution in machine learning systems. Traditional
        approaches using rule-based parsing or single-model solutions often lack the flexibility
        needed for robust conversational interfaces.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o",
                 max_rounds: int = 3):
        """
        Initialize the Multi-Agent Natural Language Understanding System
        
        The initialization process sets up the foundational infrastructure for multi-agent
        collaboration, including model client configuration and agent instantiation.
        
        Args:
            api_key: OpenAI API key for LLM access. If not provided, attempts to read
                    from OPENAI_API_KEY environment variable
            model: Language model identifier. GPT-4o is chosen for its strong reasoning
                  capabilities and consistent JSON output formatting
            max_rounds: Maximum conversation rounds between agents. This parameter controls
                       the trade-off between processing depth and computational efficiency
        
        Design Rationale:
            - GPT-4o is selected for its superior performance on structured reasoning tasks
            - Round-robin communication ensures all agents contribute to the final decision
            - Environment variable fallback provides flexibility in deployment scenarios
        """
        # Validate AutoGen availability before proceeding
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen framework is required for multi-agent functionality. "
                "Install with: pip install autogen-agentchat>=0.4.0 autogen-core>=0.4.0 autogen-ext>=0.4.0"
            )
        
        # Configure authentication and model parameters
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.max_rounds = max_rounds
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required for LLM-based processing. "
                "Set OPENAI_API_KEY environment variable or provide api_key parameter."
            )
        
        # Initialize the model client for agent communication
        # The OpenAI client provides the underlying LLM capabilities for all agents
        self.model_client = OpenAIChatCompletionClient(
            model=self.model,
            api_key=self.api_key,
        )
        
        # Initialize the specialized agent network
        self._setup_agent_architecture()
        
        logger.info(f"Initialized AutoGenDecoder with model={model}, max_rounds={max_rounds}")
    
    def _validate_and_fix_action_syntax(self, action_syntax: str) -> str:
        """
        Perform Final Action Syntax Validation and Correction
        
        This method implements a final validation layer that addresses edge cases not
        handled by the validation agent. It focuses on critical syntax errors that
        could cause system failures while preserving the flexibility for downstream
        components to handle complex cases.
        
        Design Philosophy:
            The validation approach is conservative - it only fixes clear, unambiguous
            errors while leaving more complex cases to the smart action dispatcher.
            This design prevents over-correction while ensuring system stability.
        
        Args:
            action_syntax: Raw action string from the action planning agent
            
        Returns:
            Validated and potentially corrected action syntax
            
        Research Note:
            This validation layer emerged from empirical analysis of common failure modes
            in the multi-agent system. The most frequent issues involve incomplete action
            specifications and invalid action keywords.
        """
        # Handle null or invalid input gracefully
        if not action_syntax or not isinstance(action_syntax, str):
            logger.warning("Received invalid action syntax input, defaulting to 'explain'")
            return "explain"
        
        # Normalize input for consistent processing
        action_syntax = action_syntax.strip().lower()
        parts = action_syntax.split()
        
        if not parts:
            logger.warning("Empty action syntax after normalization, defaulting to 'explain'")
            return "explain"
        
        # Handle specific known issues based on empirical observations
        
        # Issue: Incorrect ID filtering syntax - "filter id equal 2" should be "filter id 2"
        if (len(parts) >= 4 and parts[0] == "filter" and parts[1] == "id" and parts[2] == "equal"):
            # Fix: Remove the "equal" operator for ID filtering
            corrected_parts = [parts[0], parts[1]] + parts[3:]  # Skip the "equal" part
            corrected_syntax = " ".join(corrected_parts)
            logger.info(f"Correcting ID filter syntax: '{action_syntax}' → '{corrected_syntax}'")
            return corrected_syntax
        
        # Issue: Incomplete 'important' commands are common due to ambiguous user intent
        if parts[0] == "important" and len(parts) == 1:
            logger.info("Correcting incomplete 'important' command to 'important all'")
            return "important all"
        
        # Issue: Invalid action keywords cause downstream failures
        # We maintain a whitelist of valid actions based on the system's capabilities
        valid_actions = [
            "filter", "predict", "explain", "important", "score", 
            "show", "change", "mistake", "data"
        ]
        
        if parts[0] not in valid_actions:
            logger.warning(f"Invalid action keyword '{parts[0]}', defaulting to 'explain'")
            return "explain"  # Safe fallback that always succeeds
        
        # For all other cases, trust the downstream smart dispatcher to handle complexity
        # This design choice prevents over-validation while maintaining system robustness
        return action_syntax

    def _setup_agent_architecture(self):
        """
        Initialize the Specialized Agent Network
        
        This method configures the three core agents that form the natural language
        understanding pipeline. Each agent is designed with specific expertise and
        operating parameters optimized for their particular task.
        
        Agent Architecture Design:
            The three-agent architecture represents a novel application of multi-agent
            systems to conversational AI. The design is inspired by cognitive science
            models of human language processing, where different brain regions specialize
            in specific aspects of language understanding.
        
        Research Innovation:
            Unlike traditional single-model approaches, this architecture allows for:
            - Specialized prompt engineering per processing stage
            - Independent optimization of each processing component
            - Robust error handling through redundant validation
            - Clear separation of concerns for maintainability
        """
        
        # Agent 1: Intent Extraction and Entity Recognition
        # This agent performs the initial semantic analysis of user input
        self.intent_extraction_agent = AssistantAgent(
            name="IntentExtractor",
            model_client=self.model_client,
            system_message=self._create_intent_extraction_prompt()
        )
        
        # Agent 2: Action Planning and Syntax Generation  
        # This agent translates semantic understanding into executable actions
        self.action_planning_agent = AssistantAgent(
            name="ActionPlanner", 
            model_client=self.model_client,
            system_message=self._create_action_planning_prompt()
        )
        
        # Agent 3: Validation and Error Correction
        # This agent ensures output quality and system compatibility
        self.validation_agent = AssistantAgent(
            name="ActionValidator",
            model_client=self.model_client,
            system_message=self._create_validation_prompt()
        )
        
        logger.info("Successfully initialized three-agent architecture")

    def _create_intent_extraction_prompt(self) -> str:
        """
        Generate the Intent Extraction Agent System Prompt
        
        This prompt defines the first stage of our natural language understanding pipeline.
        The intent extraction agent is responsible for semantic analysis and entity recognition,
        transforming unstructured natural language into structured semantic representations.
        
        Prompt Engineering Methodology:
            The prompt is carefully engineered based on:
            1. Empirical analysis of common user query patterns
            2. Error analysis from initial system iterations  
            3. Best practices from NLU research literature
            4. Domain-specific requirements for ML model exploration
        
        Returns:
            Carefully crafted system prompt for intent extraction
        """
        return """You are a specialized Natural Language Understanding agent for machine learning model exploration systems.

RESEARCH CONTEXT:
Your role is the first stage in a multi-agent natural language understanding pipeline. You perform semantic analysis to extract user intentions and identify relevant entities from conversational queries about machine learning models.

CRITICAL CONTEXT AWARENESS:
You must determine if the current query is:
1. CONTINUING from previous context (e.g., asking about the same filtered subset)
2. SWITCHING to a new context (e.g., asking about overall/general performance after a specific instance query)

CONTEXT SWITCHING INDICATORS:
- Keywords like "overall", "global", "total", "entire", "all", "general" often indicate a context switch
- Questions about model-wide metrics after instance-specific queries are context switches  
- New topic introductions are context switches
- References to "the model" without specific qualifiers often mean the entire model

TASK SPECIFICATION:
Analyze user queries and extract:
1. Primary intent (what the user wants to accomplish)
2. Relevant entities (specific parameters, IDs, features mentioned)  
3. Contextual modifiers (comparison operators, explanation types)
4. Context continuation flag (is this continuing from previous filtered context?)

INTENT TAXONOMY (Based on ML Exploration Literature):
- explain: User seeks model explanation or interpretation (keywords: why, how, reason, because, explain)
- predict: User wants prediction results or forecasts (keywords: predict, prediction, forecast, what would)
- filter: User wants to subset or explore data (keywords: filter, show me, for patients who)
- performance: User seeks model evaluation metrics (keywords: accuracy, performance, how good, how well, overall accuracy, total accuracy, global accuracy)
- importance: User wants feature importance analysis (keywords: important, importance, significant, significance, key features, most relevant, relevant features, what features, which features, feature ranking, top features, feature matter)
- whatif: User seeks counterfactual analysis (keywords: what if, change, modify, different)
- mistakes: User wants error analysis (keywords: mistakes, errors, wrong predictions, false)
- data: User seeks data exploration or summary (keywords: data, dataset, summary, overview)
- casual: Conversational interaction without analytical intent (keywords: hi, hello, thanks, how are you)

ENTITY EXTRACTION FRAMEWORK:
Extract structured information including:
- patient_id: Specific case identifiers mentioned by user
- features: Column names or attributes referenced  
- operators: Comparison operators (explicit or implied)
- values: Threshold values or specific parameters
- explanation_type: Requested explanation method (lime, shap, general)
- context_reset: Boolean indicating if filters should be reset (true for context switches)

RESPONSE FORMAT (Strict JSON):
{
  "intent": "primary_intent_classification", 
  "entities": {
    "patient_id": numeric_id_or_null,
    "features": ["extracted", "feature", "names"],
    "operators": ["comparison", "operators"], 
    "values": [numeric_values],
    "explanation_type": "method_type_or_null",
    "context_reset": true_or_false
  },
  "confidence": confidence_score_0_to_1,
  "reasoning": "brief_semantic_analysis_explanation"
}

CRITICAL: Respond only with valid JSON matching this exact format."""

    def _create_action_planning_prompt(self) -> str:
        """
        Generate the Action Planning Agent System Prompt
        
        This prompt defines the second stage of our pipeline, responsible for translating
        semantic representations into executable action syntax. The action planning agent
        must understand both the domain-specific action grammar and the logical sequencing
        of operations.
        
        Grammar Design Philosophy:
            The action grammar is designed to be both expressive and unambiguous. It follows
            principles from formal language theory while remaining intuitive for debugging
            and system maintenance.
        
        Returns:
            Comprehensive system prompt for action planning
        """
        return """You are a specialized Action Planning agent in a multi-agent natural language understanding system.

RESEARCH CONTEXT:  
Your role is the second stage of our NLU pipeline. You receive structured intent and entity information from the Intent Extraction agent and translate it into precise, executable action commands using our domain-specific action grammar.

CRITICAL CONTEXT MANAGEMENT:
When the Intent Extraction agent indicates context_reset: true, you MUST:
1. Start fresh without assuming previous filters apply
2. NOT prepend filter commands from previous context
3. Generate actions that operate on the full dataset

When context_reset: false or not specified:
1. Assume previous filters still apply
2. Generate actions that work with the current filtered context

ACTION GRAMMAR SPECIFICATION:
Our action language follows a formal grammar designed for machine learning model exploration:

Core Actions:
- filter {feature} {operator} {value} : Apply data filtering constraints
- predict [target_id] : Generate model predictions  
- explain {target_id} [method] : Provide model explanations
- important {scope} : Analyze feature importance
- score {metric} : Evaluate model performance
- show {target_id} : Display data instances
- change {feature} {operation} {value} : Perform what-if analysis
- mistake : Analyze prediction errors
- data : Summarize dataset characteristics

Operators: {greater, less, greaterequal, lessequal, equal}
Scopes: {all, topk {n}, {feature_name}}  
Methods: {lime, shap, general}
Operations: {set, increase, decrease}

LOGICAL SEQUENCING RULES:
1. Filtering operations must precede dependent analysis operations
2. Instance-specific operations require prior instance identification
3. Complex conditions decompose into sequential simple operations
4. Context resets do NOT require explicit filter clearing - the system handles this

TRANSLATION EXAMPLES (Intent → Action):
Intent: explain, entities: {patient_id: 5} 
→ "filter id 5 explain"

Intent: predict, entities: {patient_id: 2}
→ "filter id 2 predict"

Intent: predict, entities: {features: ["age"], operators: ["greater"], values: [50]}
→ "filter age greater 50 predict"  

Intent: importance, entities: {number: 3}
→ "important topk 3"

Intent: importance, entities: {} (general importance query)
→ "important all"

Intent: performance, entities: {context_reset: true} (after filtered query)
→ "score accuracy" (no filter prefix, operates on full dataset)

Intent: predict, entities: {features: ["age", "pregnancies"], operators: ["greater", "equal"], values: [50, 0]}
→ "filter age greater 50 filter pregnancies equal 0 predict"

CRITICAL ID FILTERING SYNTAX:
- For patient IDs: "filter id {number}" NOT "filter id equal {number}"
- Examples: "filter id 2", "filter id 100", "filter id 5"
- ID filtering does NOT use the "equal" operator - it's a special case

DOMAIN-SPECIFIC MAPPINGS:
- "pregnant = no" → "pregnancies equal 0"  
- "pregnant = yes" → "pregnancies greater 0"
- Performance queries → "score accuracy" (default metric)
- Overall performance queries → "score overall accuracy" or "score global accuracy"
- Unspecified importance → "important all"

FEATURE IMPORTANCE QUERY PATTERNS:
- "what are the most relevant features" → "important all"
- "what features are most important" → "important all"
- "which features matter most" → "important all"
- "most relevant features" → "important all"
- "key features" → "important all"
- "significant features" → "important all"

RESPONSE FORMAT (Strict JSON):
{
  "action": "generated_action_command",
  "reasoning": "translation_logic_explanation", 
  "confidence": confidence_score_0_to_1
}

CRITICAL: Respond only with valid JSON matching this exact format."""

    def _create_validation_prompt(self) -> str:
        """
        Generate the Validation Agent System Prompt
        
        This prompt defines the final stage of our pipeline, implementing quality assurance
        and error correction. The validation agent serves as a safety layer, ensuring that
        generated actions are syntactically correct and semantically valid within our
        domain constraints.
        
        Validation Strategy:
            The validation approach is based on both syntactic analysis (grammar compliance)
            and semantic validation (domain constraint satisfaction). This dual approach
            ensures both system stability and logical consistency.
        
        Returns:
            Detailed system prompt for action validation
        """
        return """You are a specialized Validation agent in a multi-agent natural language understanding system.

RESEARCH CONTEXT:
Your role is the final quality assurance stage of our NLU pipeline. You receive action commands from the Action Planning agent and perform comprehensive validation to ensure syntactic correctness and semantic consistency with our domain model.

DOMAIN MODEL CONSTRAINTS:
Dataset Schema:
- Features: pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age
- Target Classes: 0 (no diabetes), 1 (diabetes)  
- Valid Patient IDs: Positive integers (typically 1-768)
- Data Types: All features are numeric, requiring appropriate value ranges

VALIDATION CHECKLIST:
1. Syntactic Validation:
   - Action keywords match defined grammar
   - Parameter counts align with action specifications
   - Operators conform to allowed set
   
2. Semantic Validation:  
   - Feature names exist in dataset schema
   - Value ranges are medically plausible
   - Patient IDs are valid integers
   - Logical sequence of operations makes sense

3. Consistency Validation:
   - Compound actions maintain logical coherence
   - Filter operations precede dependent analysis
   - Entity references are properly scoped

COMMON CORRECTION PATTERNS:
- "important" → "important all" (specificity requirement)
- Case normalization: "BMI" → "bmi", "Age" → "age"
- Operator normalization: ">" → "greater", "<" → "less"
- Syntax fixes: Ensure proper spacing between tokens

COMPOUND ACTION HANDLING:
The system accepts compound actions (e.g., "filter age greater 50 predict") as they are processed by downstream smart dispatchers. Focus validation on individual components rather than decomposition.

RESPONSE FORMAT (Strict JSON):
{
  "valid": boolean_validation_result,
  "corrected_action": "corrected_command_if_needed",
  "issues": ["list", "of", "identified", "problems"],
  "confidence": confidence_score_0_to_1
}

VALIDATION PHILOSOPHY:
If the action is valid, return it unchanged. If issues are found, provide the minimally corrected version that addresses all identified problems while preserving user intent.

CRITICAL: Respond only with valid JSON matching this exact format."""

    def _build_contextual_prompt(self, user_query: str, conversation) -> str:
        """
        Construct Contextual Information for Agent Processing
        
        This method builds comprehensive context that helps agents make informed decisions
        about user queries. The context includes dataset characteristics, current system
        state, and relevant metadata that influences processing decisions.
        
        Context Engineering Approach:
            Context construction follows principles from situated cognition theory, where
            understanding is enhanced through environmental awareness. The context provides
            agents with the situational information needed for accurate interpretation.
        
        Args:
            user_query: Raw user input requiring processing
            conversation: Current conversation state and system context
            
        Returns:
            Formatted context string for agent consumption
        """
        context_components = [f"USER QUERY: {user_query}"]
        
        # Extract dataset characteristics for informed processing
        try:
            dataset = conversation.get_var('dataset')
            if dataset:
                # Dataset size information helps with ID validation
                data_size = len(dataset.contents.get('X', []))
                context_components.append(f"DATASET SIZE: {data_size} patient records")
                
                # Feature inventory for entity validation  
                categorical_features = dataset.contents.get('cat', [])
                numerical_features = dataset.contents.get('numeric', [])
                all_features = categorical_features + numerical_features
                context_components.append(f"AVAILABLE FEATURES: {', '.join(all_features)}")
                
                # Target class information for prediction context
                if hasattr(conversation, 'class_names'):
                    context_components.append(f"TARGET CLASSES: {', '.join(map(str, conversation.class_names))}")
                    
        except Exception as e:
            # Graceful degradation when context extraction fails
            context_components.append("DATASET CONTEXT: Context extraction failed - proceeding with limited information")
            logger.warning(f"Context extraction error: {e}")
        
        # ADD PREVIOUS QUERY CONTEXT
        # Track what was filtered in the previous query to help agents understand context switches
        try:
            if hasattr(conversation, 'temp_dataset') and conversation.temp_dataset:
                temp_size = len(conversation.temp_dataset.contents.get('X', []))
                full_size = len(conversation.get_var('dataset').contents.get('X', []))
                
                if temp_size < full_size:
                    # Dataset is filtered - inform agents
                    context_components.append(f"PREVIOUS FILTERING: Dataset is currently filtered to {temp_size} out of {full_size} records")
                    
                    # Add details about the filtering if available
                    if hasattr(conversation, 'parse_operation') and conversation.parse_operation:
                        filter_desc = ' '.join(conversation.parse_operation)
                        context_components.append(f"CURRENT FILTER: {filter_desc}")
                    
                    # Add last parse to understand previous query
                    if hasattr(conversation, 'last_parse_string') and conversation.last_parse_string:
                        last_parse = conversation.last_parse_string[-1] if conversation.last_parse_string else ""
                        if last_parse:
                            context_components.append(f"PREVIOUS QUERY ACTION: {last_parse}")
        except Exception as e:
            logger.warning(f"Could not extract filtering context: {e}")
        
        return "\n".join(context_components)
    
    async def complete(self, user_query: str, conversation, grammar: str = None) -> Dict[str, Any]:
        """
        Execute Multi-Agent Natural Language Understanding Pipeline
        
        This method orchestrates the complete natural language understanding process,
        coordinating the three specialized agents to transform user queries into
        executable actions. The process follows a structured pipeline with robust
        error handling and fallback mechanisms.
        
        Pipeline Architecture:
            1. Context Construction: Build comprehensive situational context
            2. Multi-Agent Collaboration: Execute round-robin agent communication  
            3. Response Integration: Combine agent outputs into final result
            4. Quality Assurance: Apply final validation and error correction
            5. Fallback Processing: Handle edge cases and error conditions
        
        Args:
            user_query: Natural language input from user
            conversation: System context and conversation state  
            grammar: Legacy parameter maintained for API compatibility
            
        Returns:
            Structured response compatible with existing action execution system
            
        Research Note:
            The asynchronous design enables concurrent processing and improves system
            responsiveness, particularly important for interactive ML exploration.
        """
        try:
            # Stage 1: Context Construction and Preparation
            contextual_prompt = self._build_contextual_prompt(user_query, conversation)
            logger.info(f"Processing query: {user_query[:100]}...")
            
            # Stage 2: Multi-Agent Team Configuration
            # The termination condition implementation handles API version differences
            # across AutoGen releases, ensuring broad compatibility
            termination_handler = self._configure_termination_condition()
            
            team_configuration = self._create_agent_team(termination_handler)
            
            # Stage 3: Execute Collaborative Processing Pipeline
            processing_prompt = (
                f"{contextual_prompt}\n\n"
                f"Execute the complete natural language understanding pipeline:\n"
                f"1. Intent Extraction → 2. Action Planning → 3. Validation → 4. Final Output"
            )
            
            collaboration_result = await team_configuration.run(task=processing_prompt)
            
            # Stage 4: Response Processing and Integration
            final_response = self._process_agent_responses(collaboration_result)
            
            if final_response:
                return final_response
                
            # Stage 5: Fallback Processing for Edge Cases
            fallback_response = self._execute_fallback_processing(collaboration_result)
            return fallback_response
            
        except Exception as e:
            # Comprehensive error handling with graceful degradation
            logger.error(f"Multi-agent processing failed: {e}")
            return self._create_error_response(str(e))

    def _configure_termination_condition(self):
        """
        Configure Agent Team Termination Conditions
        
        This method handles the complex task of configuring termination conditions
        across different AutoGen API versions. The AutoGen framework has evolved
        significantly, with API changes affecting termination parameter names.
        
        Compatibility Strategy:
            Use runtime introspection to determine the correct parameter names,
            ensuring compatibility across AutoGen versions while maintaining
            functionality. This approach prevents deployment issues due to API changes.
        
        Returns:
            Configured termination condition object
        """
        try:
            # Import termination classes with version-aware fallbacks
            try:
                from autogen_agentchat.conditions import MaxMessageTermination
            except ModuleNotFoundError:
                from autogen.stop.conditions import MaxMessageTermination
                
            return MaxMessageTermination(self.max_rounds)
            
        except Exception as e:
            logger.warning(f"Termination condition configuration failed: {e}")
            return None

    def _create_agent_team(self, termination_handler):
        """
        Instantiate Multi-Agent Collaboration Team
        
        Creates the collaborative team structure using AutoGen's RoundRobinGroupChat,
        which ensures each agent contributes to the final decision while maintaining
        controlled communication flow.
        
        Args:
            termination_handler: Configured termination condition
            
        Returns:
            Configured agent team ready for collaborative processing
        """
        import inspect
        
        # Use runtime inspection to handle API parameter variations
        team_parameters = {
            "participants": [
                self.intent_extraction_agent,
                self.action_planning_agent, 
                self.validation_agent,
            ]
        }
        
        # Add termination condition with version-appropriate parameter name
        if termination_handler:
            signature = inspect.signature(RoundRobinGroupChat)
            if "termination_condition" in signature.parameters:
                team_parameters["termination_condition"] = termination_handler
            elif "termination" in signature.parameters:
                team_parameters["termination"] = termination_handler
            elif "termination_checker" in signature.parameters:
                team_parameters["termination_checker"] = termination_handler
        
        return RoundRobinGroupChat(**team_parameters)

    def _process_agent_responses(self, collaboration_result) -> Optional[Dict[str, Any]]:
        """
        Process and Integrate Multi-Agent Responses
        
        This method implements sophisticated response processing logic to extract
        and integrate the outputs from all three agents. The processing handles
        various response formats and ensures robust information extraction.
        
        Response Processing Strategy:
            1. Parse structured JSON responses from each agent
            2. Handle partial responses when some agents fail
            3. Implement cascading fallback for missing information
            4. Apply final quality assurance measures
        
        Args:
            collaboration_result: Raw output from agent collaboration
            
        Returns:
            Integrated response or None if processing fails
        """
        # Initialize response containers
        intent_response = None
        action_response = None  
        validation_response = None
        
        logger.info(f"Processing {len(collaboration_result.messages)} agent messages")
        
        # Extract structured responses from agent communications
        for message_index, message in enumerate(collaboration_result.messages):
            extracted_response = self._extract_json_response(message, message_index)
            
            if extracted_response:
                # Classify response by content structure
                response_type = self._classify_response_type(extracted_response)
                
                if response_type == "intent" and not intent_response:
                    intent_response = extracted_response
                    # Handle casual conversation early termination
                    if intent_response.get('intent') == 'casual':
                        return self._create_casual_response()
                        
                elif response_type == "action" and not action_response:
                    action_response = extracted_response
                    # Handle null action cases (conversational queries)
                    if action_response.get('action') is None:
                        return self._create_casual_response()
                        
                elif response_type == "validation" and not validation_response:
                    validation_response = extracted_response
        
        # Integrate responses based on available information
        return self._integrate_agent_outputs(intent_response, action_response, validation_response)

    def _extract_json_response(self, message, message_index: int) -> Optional[Dict]:
        """
        Extract Structured JSON from Agent Message
        
        Implements robust JSON extraction that handles various formatting approaches
        used by different LLM models. The extraction process is fault-tolerant and
        supports multiple JSON embedding patterns.
        
        Args:
            message: Agent message containing potential JSON response
            message_index: Message position for debugging purposes
            
        Returns:
            Extracted JSON dictionary or None if extraction fails
        """
        try:
            if not (hasattr(message, 'content') and message.content and '{' in message.content):
                return None
                
            content = message.content
            
            # Pattern 1: JSON code block extraction
            import re
            json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_block_match:
                try:
                    return json.loads(json_block_match.group(1))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON block parsing failed for message {message_index}: {e}")
            
            # Pattern 2: Direct JSON extraction
            if content.strip().startswith('{') and content.strip().endswith('}'):
                try:
                    return json.loads(content.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"Direct JSON parsing failed for message {message_index}: {e}")
                    
            # Pattern 3: Embedded JSON search
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, content)
            for match in json_matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            logger.warning(f"JSON extraction error for message {message_index}: {e}")
            
        return None

    def _classify_response_type(self, response: Dict) -> str:
        """
        Classify Agent Response by Content Structure
        
        Determines which agent generated a response based on the presence of
        specific keys in the JSON structure. This classification enables
        proper response routing and integration.
        
        Args:
            response: Parsed JSON response from agent
            
        Returns:
            Response type classification string
        """
        if 'intent' in response:
            return "intent"
        elif 'action' in response:
            return "action"
        elif 'valid' in response:
            return "validation"
        else:
            return "unknown"

    def _create_casual_response(self) -> Dict[str, Any]:
        """
        Generate Response for Casual Conversational Queries
        
        Creates appropriate responses for non-analytical user interactions,
        such as greetings or general questions about system capabilities.
        
        Returns:
            Formatted casual conversation response
        """
        return {
            "generation": None,
            "direct_response": (
                "Hello! I'm an AI assistant specialized in machine learning model exploration "
                "for diabetes risk assessment. I can help you analyze patient data, understand "
                "model predictions, explore feature importance, and perform what-if analyses. "
                "What would you like to investigate?"
            ),
            "method": "autogen_conversational",
            "confidence": 0.95
        }

    def _integrate_agent_outputs(self, intent_response, action_response, validation_response) -> Optional[Dict[str, Any]]:
        """
        Integrate Multi-Agent Outputs into Unified Response
        
        This method implements the core integration logic that combines outputs
        from multiple agents into a coherent, actionable response. The integration
        process handles various scenarios including complete responses, partial
        responses, and error conditions.
        
        Integration Strategy:
            1. Complete Integration: All three agents provided valid responses
            2. Partial Integration: Subset of agents provided responses  
            3. Quality Assurance: Apply final validation and correction
            4. Confidence Scoring: Aggregate confidence measures
        
        Args:
            intent_response: Output from intent extraction agent
            action_response: Output from action planning agent
            validation_response: Output from validation agent
            
        Returns:
            Integrated response ready for action execution
        """
        # Scenario 1: Complete agent collaboration (ideal case)
        if intent_response and action_response and validation_response:
            return self._create_complete_response(intent_response, action_response, validation_response)
            
        # Scenario 2: Partial collaboration (fallback case)
        elif intent_response and action_response:
            return self._create_partial_response(intent_response, action_response)
            
        # Scenario 3: Insufficient information for integration
        else:
            logger.warning("Insufficient agent responses for integration")
            return None

    def _create_complete_response(self, intent_response, action_response, validation_response) -> Dict[str, Any]:
        """
        Create Response from Complete Agent Collaboration
        
        Constructs the final response when all three agents have successfully
        contributed to the processing pipeline. This represents the optimal
        processing scenario with full validation and error correction.
        
        Args:
            intent_response: Validated intent extraction results
            action_response: Generated action commands
            validation_response: Validation results and corrections
            
        Returns:
            Complete integrated response with full metadata
        """
        # Extract final action with validation corrections
        final_action = validation_response.get('corrected_action', action_response.get('action', 'explain'))
        
        # Apply additional programmatic validation as safety measure
        original_action = final_action
        final_action = self._validate_and_fix_action_syntax(final_action)
        
        # Log any additional corrections applied
        if original_action != final_action:
            logger.info(f"Applied additional correction: '{original_action}' → '{final_action}'")
        
        # Calculate aggregate confidence score
        confidence_score = min(
            intent_response.get('confidence', 0.8),
            action_response.get('confidence', 0.8),
            validation_response.get('confidence', 0.8)
        )
        
        return {
            "generation": f"parsed: {final_action}[e]",
            "confidence": confidence_score,
            "method": "autogen_complete_pipeline",
            "intent_response": intent_response,  # Include full intent response for context reset detection
            "agent_reasoning": {
                "intent_analysis": intent_response.get('reasoning', ''),
                "action_planning": action_response.get('reasoning', ''),
                "validation_results": (
                    f"Valid: {validation_response.get('valid', False)}, "
                    f"Issues: {validation_response.get('issues', [])}"
                )
            },
            "final_action": final_action,
            "validation_passed": validation_response.get('valid', False),
            "identified_issues": validation_response.get('issues', [])
        }

    def _create_partial_response(self, intent_response, action_response) -> Dict[str, Any]:
        """
        Create Response from Partial Agent Collaboration
        
        Handles scenarios where the validation agent failed to provide output
        but intent extraction and action planning succeeded. This fallback
        maintains system functionality while noting the validation limitation.
        
        Args:
            intent_response: Intent extraction results  
            action_response: Action planning results
            
        Returns:
            Partial response with available information
        """
        final_action = action_response.get('action', 'explain')
        
        # Apply safety validation without validation agent input
        original_action = final_action
        final_action = self._validate_and_fix_action_syntax(final_action)
        
        if original_action != final_action:
            logger.info(f"Applied safety correction: '{original_action}' → '{final_action}'")
        
        # Calculate confidence with penalty for missing validation
        confidence_score = min(
            intent_response.get('confidence', 0.8),
            action_response.get('confidence', 0.8)
        ) * 0.9  # Penalty for missing validation
        
        return {
            "generation": f"parsed: {final_action}[e]",
            "confidence": confidence_score,
            "method": "autogen_partial_pipeline",
            "intent_response": intent_response,  # Include full intent response for context reset detection
            "agent_reasoning": {
                "intent_analysis": intent_response.get('reasoning', ''),
                "action_planning": action_response.get('reasoning', ''),
                "validation_results": "Validation agent response unavailable"
            },
            "final_action": final_action,
            "validation_passed": True,  # Assume valid with programmatic validation
            "identified_issues": []
        }

    def _execute_fallback_processing(self, collaboration_result) -> Dict[str, Any]:
        """
        Execute Emergency Fallback Processing
        
        When structured agent responses are unavailable, this method attempts
        to extract actionable information from raw agent communications using
        pattern matching and heuristic analysis.
        
        Fallback Strategy:
            1. Pattern-based action extraction from message content
            2. Keyword-based intent inference
            3. Safe default action selection
            4. Degraded confidence reporting
        
        Args:
            collaboration_result: Raw collaboration output
            
        Returns:
            Emergency fallback response
        """
        extracted_action = None
        
        # Define action detection patterns based on empirical analysis
        action_keywords = ['filter', 'predict', 'explain', 'important', 'score', 'show', 'change', 'mistake', 'data']
        
        # Search for recognizable action patterns in agent messages
        for message in collaboration_result.messages:
            if extracted_action:
                break
                
            try:
                if hasattr(message, 'content') and message.content:
                    content = message.content
                    
                    # Pattern-based extraction using keyword detection
                    for keyword in action_keywords:
                        if keyword in content.lower():
                            # Extract potential action lines
                            lines = content.split('\n')
                            for line in lines:
                                if keyword in line.lower() and not line.strip().startswith('#'):
                                    # Clean and validate extracted line
                                    action_candidate = line.strip().strip('"').strip("'")
                                    if action_candidate:
                                        extracted_action = action_candidate
                                        logger.info(f"Extracted action via fallback: {extracted_action}")
                                        break
                            if extracted_action:
                                break
                                
            except (AttributeError, TypeError) as e:
                logger.warning(f"Fallback extraction error: {e}")
                continue
        
        # Ultimate safety fallback
        if not extracted_action:
            extracted_action = "explain"  
            logger.warning("Using ultimate safety fallback: 'explain'")
        
        return {
            "generation": f"parsed: {extracted_action}[e]",
            "confidence": 0.60,  # Reduced confidence for fallback processing
            "method": "autogen_emergency_fallback",
            "fallback_extraction": extracted_action,
            "raw_message_count": len(collaboration_result.messages)
        }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create Graceful Error Response
        
        Generates a safe fallback response when the multi-agent system
        encounters unrecoverable errors. The response maintains system
        stability while providing diagnostic information.
        
        Args:
            error_message: Description of the encountered error
            
        Returns:
            Safe error response with diagnostic information
        """
        logger.error(f"Creating error response for: {error_message}")
        
        return {
            "generation": f"parsed: explain[e]",  # Safe default action
            "error": error_message,
            "confidence": 0.1,  # Minimal confidence for error state
            "method": "autogen_error_recovery",
            "timestamp": logger.handlers[0].formatter.formatTime(logger.makeRecord(
                logger.name, logger.ERROR, __file__, 0, error_message, (), None
            )) if logger.handlers else "unknown"
        }
    
    def complete_sync(self, user_query: str, conversation, grammar: str = None) -> Dict[str, Any]:
        """
        Synchronous Wrapper for Asynchronous Multi-Agent Processing
        
        This method provides a synchronous interface to the asynchronous multi-agent
        system, handling the complexities of event loop management in different
        runtime environments (particularly Flask applications).
        
        Concurrency Strategy:
            The implementation uses multiple strategies to handle async/sync integration:
            1. Existing event loop detection and adaptation
            2. Thread-based isolation for running event loops
            3. Timeout management for system responsiveness
            4. Graceful error handling and fallback responses
        
        Args:
            user_query: User's natural language input
            conversation: System context and state
            grammar: Legacy parameter for API compatibility
            
        Returns:
            Processed response from multi-agent system
            
        Technical Note:
            The sync/async integration addresses a common challenge in web applications
            where Flask's synchronous nature conflicts with AutoGen's async requirements.
        """
        try:
            # Strategy 1: Detect existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Flask/web server context - use thread isolation
                return self._execute_in_thread(user_query, conversation, grammar)
            else:
                # CLI or standalone context - use existing loop
                return loop.run_until_complete(self.complete(user_query, conversation, grammar))
                
        except RuntimeError:
            # Strategy 2: No event loop exists - create new one
            return self._execute_with_new_loop(user_query, conversation, grammar)
            
        except Exception as e:
            # Strategy 3: Ultimate fallback for unexpected errors
            logger.error(f"Sync/async integration failed: {e}")
            return self._create_error_response(f"Concurrency management error: {str(e)}")

    def _execute_in_thread(self, user_query: str, conversation, grammar: str = None) -> Dict[str, Any]:
        """Execute processing in isolated thread to avoid event loop conflicts."""
        import concurrent.futures
        import threading
        
        def run_in_isolated_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.complete(user_query, conversation, grammar))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_isolated_thread)
            return future.result(timeout=30)  # 30-second timeout for responsiveness

    def _execute_with_new_loop(self, user_query: str, conversation, grammar: str = None) -> Dict[str, Any]:
        """Execute processing with new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.complete(user_query, conversation, grammar))
        finally:
            loop.close()
    
    async def cleanup(self):
        """
        Clean Up System Resources
        
        Properly releases resources used by the multi-agent system,
        including LLM client connections and agent references.
        """
        try:
            await self.model_client.close()
            logger.info("AutoGen decoder resources cleaned up successfully")
        except Exception as e:
            logger.warning(f"Resource cleanup warning: {e}")


# Factory Functions for System Integration
# These functions provide clean interfaces for integrating the multi-agent system
# with existing codebases while maintaining backward compatibility.

def create_autogen_decoder(**kwargs) -> AutoGenDecoder:
    """
    Factory Function for AutoGen Decoder Instantiation
    
    Provides a clean interface for creating decoder instances with
    appropriate parameter validation and error handling.
    
    Args:
        **kwargs: Configuration parameters for decoder initialization
        
    Returns:
        Configured AutoGenDecoder instance
    """
    return AutoGenDecoder(**kwargs)


def get_autogen_predict_func(api_key: str = None, model: str = "gpt-4o"):
    """
    Generate Prediction Function Compatible with Legacy Interfaces
    
    Creates a prediction function that maintains compatibility with existing
    decoder interfaces while leveraging the new multi-agent architecture.
    This enables gradual migration of existing systems.
    
    Args:
        api_key: OpenAI API key for LLM access
        model: Language model identifier
        
    Returns:
        Compatible prediction function for legacy system integration
        
    Design Note:
        This function bridges the gap between old and new architectures,
        allowing for seamless integration without requiring extensive
        refactoring of existing codebases.
    """
    decoder = AutoGenDecoder(api_key=api_key, model=model)
    
    def prediction_function(prompt: str, grammar: str = None, conversation=None):
        """
        Legacy-Compatible Prediction Function
        
        Extracts user queries from legacy prompt formats and processes
        them through the multi-agent pipeline.
        """
        # Extract user query from legacy prompt format
        if "Query:" in prompt:
            user_query = prompt.split("Query:")[-1].strip()
        else:
            user_query = prompt
            
        return decoder.complete_sync(user_query, conversation, grammar)
    
    return prediction_function 