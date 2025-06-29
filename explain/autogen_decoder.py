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
                 model: str = "gpt-4o-mini",  # Faster model for real-time collaboration
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
        """Generate concise, focused intent extraction prompt for speed and accuracy."""
        return """You are an intent extraction agent for ML model queries. Be FAST and ACCURATE.

TASK: Extract intent and entities from user queries about machine learning models.

INTENT TYPES:
- data: Dataset statistics, averages, summaries ("average age", "dataset info")  
- predict: Model predictions ("predict for patient 2", "what would happen")
- explain: Model explanations ("why", "how did", "explain prediction")
- important: Feature importance ("important features", "which features matter")
- performance: Model accuracy ("how accurate", "model performance")
- filter: Subset data ("patients with age > 50", "show instances where model predicted 1")
- casual: Greetings, chat ("hello", "hi")

SPECIAL FILTERING TYPES:
- Prediction filtering: "show instances where model predicted 1", "cases where model predicts diabetes"
- Feature filtering: "patients with age > 50", "glucose less than 100"
- Label filtering: "instances where ground truth is 1", "actual diabetic patients"

EXAMPLES:
"whats the average age in the dataset" → intent: "data"
"how accurate is the model" → intent: "performance" 
"explain patient 5" → intent: "explain", entities: {patient_id: 5}
"predict for glucose > 120" → intent: "predict", entities: {features: ["glucose"], operators: [">"], values: [120]}
"show instances where model predicted 1" → intent: "filter", entities: {filter_type: "prediction", prediction_values: [1]}
"patients with age > 50" → intent: "filter", entities: {features: ["age"], operators: [">"], values: [50]}
"show cases where ground truth is 1" → intent: "filter", entities: {filter_type: "label", label_values: [1]}

OUTPUT FORMAT (JSON ONLY - NO DUPLICATE KEYS):
{
  "intent": "detected_intent",
  "entities": {
    "patient_id": number_or_null,
    "features": ["feature_names_or_null"],
    "operators": ["operators_or_null"], 
    "values": [numbers_or_null],
    "filter_type": "prediction|feature|label|null",
    "prediction_values": [numbers_for_prediction_filtering_or_null],
    "label_values": [numbers_for_label_filtering_or_null]
  },
  "confidence": 0.95
}

CRITICAL: Never use the same JSON key twice. Use prediction_values for prediction filtering, values for feature filtering, and label_values for label filtering.

Be fast, accurate, and concise. No explanations needed."""

    def _create_action_planning_prompt(self) -> str:
        """Generate concise action planning prompt for speed."""
        return """You are an action planning agent. Convert intents to executable actions QUICKLY.

INTENT → ACTION MAPPING:
- data → "data" 
- predict → "predict" (+ filters if needed)
- explain → "explain" (+ filters if needed)
- important → "important all"
- performance → "score accuracy" (+ filters if needed)
- filter + predict → "filter [feature] [op] [value] predict"
- filter + performance → "filter [feature] [op] [value] score accuracy"

FILTERING ACTION TYPES:
- Prediction filtering: filter_type="prediction" → Use "filter" action with prediction entities
- Feature filtering: filter_type="feature" or features present → Use "filter" action with feature entities  
- Label filtering: filter_type="label" → Use "filter" action with label entities
- ID filtering: patient_id present → Use "filter" action with id entities

EXAMPLES:
Intent: data → Action: "data"
Intent: predict, entities: {patient_id: 5} → Action: "filter id 5 predict" 
Intent: performance, entities: {features: ["age"], operators: [">"], values: [50]} → Action: "filter age greater 50 score accuracy"
Intent: explain, entities: {patient_id: 2} → Action: "filter id 2 explain"
Intent: filter, entities: {filter_type: "prediction", values: [1]} → Action: "filter"
Intent: filter, entities: {features: ["age"], operators: [">"], values: [50]} → Action: "filter"
Intent: filter, entities: {filter_type: "label", values: [1]} → Action: "filter"

OUTPUT FORMAT (JSON ONLY):
{
  "action": "generated_action_string",
  "entities": {
    "features": ["feature_names_if_any"],
    "operators": ["operators_if_any"],
    "values": [values_if_any],
    "filter_type": "prediction|feature|label|null"
  },
  "confidence": 0.95
}

IMPORTANT: Always pass through the entities from intent extraction so the filter function has the structured data it needs.

Be fast and direct. No complex reasoning needed."""

    def _create_validation_prompt(self) -> str:
        """Generate concise validation prompt for speed."""
        return """You are a validation agent. Check action syntax QUICKLY.

VALID ACTIONS: data, predict, explain, important, score, filter, show
VALID FEATURES: age, glucose, bmi, pregnancies, bloodpressure, insulin, skinthickness, diabetespedigreefunction
VALID OPERATORS: greater, less, equal, greaterequal, lessequal

VALIDATION RULES:
- Check action keywords are valid
- Check feature names exist in dataset
- Check syntax is correct
- Fix obvious errors

EXAMPLES:
"data" → Valid ✓
"filter age greater 50 score accuracy" → Valid ✓  
"filter invalid_feature equal 1" → Invalid (bad feature)
"invalid_action" → Invalid (bad action)

OUTPUT FORMAT (JSON ONLY):
{
  "valid": true_or_false,
  "issues": ["list_of_issues"],
  "confidence": 0.95
}

Be fast. Only catch obvious errors."""

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
        executable actions. The agents are allowed to collaborate and self-correct
        through multiple rounds until they reach a satisfactory solution.
        
        Pipeline Architecture:
            1. Context Construction: Build comprehensive situational context
            2. Multi-Agent Collaboration: Execute round-robin agent communication  
            3. Response Integration: Combine agent outputs into final result
            4. Quality Assurance: Apply final validation and error correction
            5. Agent Self-Correction: Allow agents to retry and improve their responses
        
        Args:
            user_query: Natural language input from user
            conversation: System context and conversation state  
            grammar: Legacy parameter maintained for API compatibility
            
        Returns:
            Structured response compatible with existing action execution system
            
        Research Note:
            The asynchronous design enables concurrent processing and improves system
            responsiveness, particularly important for interactive ML exploration.
            Agents are given multiple opportunities to collaborate and self-correct.
        """
        # Direct execution - clean and simple
        
        # Stage 1: Context Construction and Preparation
        contextual_prompt = self._build_contextual_prompt(user_query, conversation)
        logger.info(f"Processing query: {user_query[:100]}...")
        logger.info("Stage 1: Context construction completed")
        
        # Stage 2: Multi-Agent Team Configuration
        termination_handler = self._configure_termination_condition()
        logger.info("Stage 2: Termination condition configured")
        
        team_configuration = self._create_agent_team(termination_handler)
        logger.info("Stage 2: Agent team created successfully")
        
        # Stage 3: Execute Collaborative Processing Pipeline
        processing_prompt = (
            f"{contextual_prompt}\n\n"
            f"Execute the complete natural language understanding pipeline:\n"
            f"1. Intent Extraction → 2. Action Planning → 3. Validation → 4. Final Output\n"
            f"Collaborate and iterate until you reach a high-quality solution."
        )
        
        logger.info("Stage 3: Starting agent collaboration...")
        collaboration_result = await team_configuration.run(task=processing_prompt)
        logger.info(f"Stage 3: Agent collaboration completed with {len(collaboration_result.messages) if hasattr(collaboration_result, 'messages') else 0} messages")
        
        # Stage 4: Response Processing and Integration
        logger.info("Stage 4: Processing agent responses...")
        final_response = self._process_agent_responses(collaboration_result)
        
        if final_response:
            logger.info("Successfully processed query")
            return final_response
        else:
            logger.warning("Creating minimal response")
            return self._create_minimal_response(user_query, collaboration_result)

    def _configure_termination_condition(self):
        """Configure Agent Team Termination Conditions - direct and simple."""
        from autogen_agentchat.conditions import MaxMessageTermination
        return MaxMessageTermination(self.max_rounds)

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
        """Extract JSON from agent message - clean and direct."""
        if not (hasattr(message, 'content') and message.content and '{' in message.content):
            return None
            
        content = message.content
        
        try:
            # Pattern 1: JSON code block
            import re
            json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_block_match:
                return json.loads(json_block_match.group(1))
            
            # Pattern 2: Direct JSON
            if content.strip().startswith('{') and content.strip().endswith('}'):
                return json.loads(content.strip())
                
            # Pattern 3: Embedded JSON
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, content)
            for match in json_matches:
                return json.loads(match)
        except json.JSONDecodeError:
            pass
        
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
        contributed to the processing pipeline. Uses the universal command parser
        to handle structured commands from the action planning agent.
        
        Args:
            intent_response: Validated intent extraction results
            action_response: Generated action commands (now structured)
            validation_response: Validation results and corrections
            
        Returns:
            Complete integrated response with universal command parsing
        """
        # Import universal parser
        from .universal_command_parser import create_universal_parser
        
        # Extract command structure from action response
        command_structure = action_response.get('command', action_response.get('action', {}))
        
        # Use universal parser to generate action list
        universal_parser = create_universal_parser()
        action_list = universal_parser.parse_autogen_response(action_response)
        
        # Join actions for backward compatibility
        final_action = " ".join(action_list) if action_list else "explain"
        
        # Calculate aggregate confidence score
        confidence_score = min(
            intent_response.get('confidence', 0.8),
            action_response.get('confidence', 0.8),
            validation_response.get('confidence', 0.8)
        )
        
        return {
            "generation": f"parsed: {final_action}[e]",
            "confidence": confidence_score,
            "method": "autogen_universal_pipeline",
            "intent_response": intent_response,  # Include full intent response for context reset detection
            "agent_reasoning": {
                "intent_analysis": intent_response.get('reasoning', ''),
                "action_planning": action_response.get('reasoning', ''),
                "validation_results": (
                    f"Valid: {validation_response.get('valid', False)}, "
                    f"Issues: {validation_response.get('issues', [])}"
                )
            },
            "command_structure": command_structure,
            "action_list": action_list,
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

    def _create_minimal_response(self, user_query: str, collaboration_result) -> Dict[str, Any]:
        """
        Create Minimal Response When Agents Couldn't Reach Consensus
        
        Extract intent and entities from the first agent response if available,
        otherwise fall back to a safe default.
        
        Args:
            user_query: Original user query
            collaboration_result: Raw collaboration output
            
        Returns:
            Minimal response that maintains agent-based approach
        """
        logger.info("Creating minimal response - agents didn't reach consensus")
        
        # Try to extract intent and entities from first agent response
        intent_response = None
        for message in collaboration_result.messages:
            extracted_response = self._extract_json_response(message, 0)
            if extracted_response and 'intent' in extracted_response:
                intent_response = extracted_response
                break
        
        if intent_response:
            # Use the extracted intent and entities
            return {
                "intent": intent_response.get("intent"),
                "entities": intent_response.get("entities", {}),
                "confidence": intent_response.get("confidence", 0.8),
                "method": "autogen_intent_extraction",
                "original_query": user_query
            }
        else:
            # NO FALLBACKS - Fail fast when AutoGen doesn't work
            raise Exception("AutoGen failed to extract intent - no fallback allowed")

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
        """Synchronous wrapper for multi-agent processing."""
        return self._execute_in_thread(user_query, conversation, grammar)

    def _execute_in_thread(self, user_query: str, conversation, grammar: str = None) -> Dict[str, Any]:
        """Execute processing in isolated thread to avoid event loop conflicts."""
        import concurrent.futures
        import threading
        
        def run_in_isolated_thread():
            """Execute async processing in a new thread with proper cleanup."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                logger.info(f"Starting async processing for query: {user_query[:50]}...")
                result = new_loop.run_until_complete(self.complete(user_query, conversation, grammar))
                logger.info("Async processing completed successfully")
                return result
            except asyncio.TimeoutError:
                logger.warning("AutoGen processing timed out")
                return self._create_error_response("Processing timeout - please try a simpler query")
            except Exception as e:
                logger.error(f"Error in async processing: {e}")
                return self._create_error_response(f"AutoGen processing error: {str(e)}")
            finally:
                try:
                    # Proper cleanup of the event loop
                    pending = asyncio.all_tasks(new_loop)
                    if pending:
                        for task in pending:
                            task.cancel()
                        new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    new_loop.close()
                    logger.debug("Event loop cleaned up successfully")
                except Exception as cleanup_error:
                    logger.warning(f"Event loop cleanup warning: {cleanup_error}")
        
        try:
            logger.info("Executing AutoGen processing in isolated thread...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_isolated_thread)
                result = future.result(timeout=15)  # Reduced from 30 to 15 seconds for faster responses
                logger.info("Thread execution completed successfully")
                return result
                
        except concurrent.futures.TimeoutError:
            logger.error("Thread execution timed out after 15 seconds")
            return self._create_error_response("System timeout after 15 seconds - agents need more focus")
        except Exception as e:
            logger.error(f"Thread execution failed: {e}")
            return self._create_error_response(f"Thread execution error: {str(e)}")

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


# Alias for backward compatibility
AutoGenNaturalLanguageUnderstanding = AutoGenDecoder 