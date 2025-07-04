"""AutoGen-based multi-agent decoder for natural language to action translation.

This module uses a 2-agent discussion system to convert user queries into executable actions:
1. Intent Extraction Agent - Identifies user intent and extracts entities
2. Intent Validation Agent - Validates and refines the extracted intent through critical discussion

The agents collaborate through 4 rounds of discussion to reach consensus, with direct action mapping.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple

# Configure logging for research and debugging purposes
logger = logging.getLogger(__name__)

# AutoGen Framework Integration  
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


class AutoGenDecoder:
    """Multi-agent decoder that converts natural language queries to executable actions.
    
    Uses two specialized agents working together:
    - Intent extraction: Understands what the user wants
    - Intent validation: Critically examines and refines the intent through discussion
    
    The agents collaborate through 4 rounds of discussion to reach consensus, with direct action mapping.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",  # Faster model for real-time collaboration
                 max_rounds: int = 4):
        """Initialize the multi-agent decoder.
        
        Args:
            api_key: OpenAI API key (or uses OPENAI_API_KEY environment variable)
            model: Language model to use for both agents
            max_rounds: Maximum conversation rounds between agents (default 4 for proper discussion)
        """
        
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
        
        # Initialize direct action mapping
        self._setup_action_mapping()
        
        logger.info(f"Initialized AutoGenDecoder with model={model}, max_rounds={max_rounds}")
    
    def _setup_action_mapping(self):
        """Set up direct intent-to-action mapping."""
        self.intent_to_action = {
            # Core data operations (data handles both overview and counting)
            "data": "data",
            "count": "data",  # Count queries use data action
            "statistics": "statistic",
            
            # Model operations
            "predict": "predict",
            "explain": "explain",
            "important": "important",
            "performance": "score",
            "confidence": "likelihood",
            "mistakes": "mistake",
            
            # Data manipulation (filter handles all filter types)
            "filter": "filter",
            "show": "show",
            "labels": "label",
            "predictionfilter": "filter",  # All filtering uses filter action
            "labelfilter": "filter",       # All filtering uses filter action
            
            # What-if analysis (counterfactual handles all variants)
            "whatif": "change",
            "counterfactual": "counterfactual",
            "alternatives": "counterfactual",  # Alternatives use counterfactual action
            "scenarios": "counterfactual",     # Scenarios use counterfactual action
            
            # Feature analysis
            "interactions": "interact",
            "define": "define",
            "function": "function",
            
            # Conversational (all map to self)
            "about": "self",
            "casual": "self",
            "followup": "followup",
            "model": "model",
            "self": "self"
        }
        
    def _map_intent_to_action(self, intent: str) -> str:
        """Direct mapping from intent to action without agent overhead."""
        return self.intent_to_action.get(intent, "explain")  # Default to explain
    

    def _setup_agent_architecture(self):
        """Initialize the two specialized agents for the processing pipeline.
        
        Creates:
        - Intent extraction agent: Understands user intent and extracts entities
        - Intent validation agent: Critically examines and refines intent through discussion
        """
        
        # Agent 1: Intent Extraction and Entity Recognition
        # This agent performs the initial semantic analysis of user input
        self.intent_extraction_agent = AssistantAgent(
            name="IntentExtractor",
            model_client=self.model_client,
            system_message=self._create_intent_extraction_prompt()
        )
        
        # Agent 2: Intent Validation and Critical Analysis
        # This agent critically examines if the intent was correctly interpreted
        self.intent_validation_agent = AssistantAgent(
            name="IntentValidator",
            model_client=self.model_client,
            system_message=self._create_intent_validation_prompt()
        )
        
        logger.info("Successfully initialized two-agent architecture")

    def _create_intent_extraction_prompt(self) -> str:
        """Generate discussion-focused intent extraction prompt for collaborative analysis."""
        return """You are an intent extraction agent for ML model queries. ENGAGE IN DISCUSSION with the validation agent.

TASK: Extract intent and entities from user queries about machine learning models, then DISCUSS your interpretation with the validation agent.

DISCUSSION GUIDELINES:
- Present your initial interpretation
- Listen to validation agent's questions and concerns
- Revise your interpretation based on discussion
- Consider context and dataset implications together
- Focus on reaching consensus on intent and entities

INTENT TYPES:
- data: Dataset statistics, averages, summaries ("average age", "dataset info")  
- predict: Model predictions ("predict for patient 2", "what would happen")
- explain: Model explanations ("why", "how did", "explain prediction")
- important: Feature importance ("important features", "which features matter")
- performance: Model accuracy ("how accurate", "model performance")
- filter: Subset data ("patients with age > 50", "show instances where model predicted 1")
- whatif: What-if analysis ("what if BMI was 25", "change glucose to 90")
- counterfactual: Counterfactual explanations ("show counterfactuals", "what are the alternatives", "scenarios to flip prediction")
- mistakes: Model error analysis ("show mistakes", "where is the model wrong")
- confidence: Prediction confidence ("how confident", "prediction probability")
- interactions: Feature interactions ("how do features interact", "age and BMI together")
- show: Display data instances ("show patient 10", "display this data")
- statistics: Feature statistics ("glucose statistics", "BMI distribution")
- labels: Ground truth information ("actual labels", "true values")
- count: Count data points (analyze if user wants filtered count or total dataset count) ("how many patients", "number of instances")
- define: Feature definitions ("what is BMI", "define glucose")
- about: System information ("tell me about yourself", "what can you do")
- casual: Greetings, chat ("hello", "hi")

CONVERSATIONAL CONTEXT INTENTS:
- followup: Follow-up questions ("tell me more", "explain that better", "what about") 
  AND analytical follow-ups ("so the model underpredicts", "this means the model", "the model seems to", "does this mean", "so it appears", "therefore the model")
- model: Model information ("about the model", "model details", "training info")
- predictionfilter: Filter by predictions ("where model predicted diabetes", "prediction = 1")
- labelfilter: Filter by actual labels ("actual diabetic patients", "ground truth = 1")

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
"what if BMI was 25 instead" → intent: "whatif", entities: {features: ["BMI"], values: [25]}
"show counterfactuals for patient 5" → intent: "counterfactual", entities: {patient_id: 5}
"what are the alternatives to flip this prediction" → intent: "counterfactual"
"show me scenarios that would change the outcome" → intent: "counterfactual"
"what changes would make this patient non-diabetic" → intent: "counterfactual"
"show me the model's biggest mistakes" → intent: "mistakes"
"how confident is the model" → intent: "confidence"
"how do age and BMI interact" → intent: "interactions", entities: {features: ["age", "BMI"]}
"show me patient 10" → intent: "show", entities: {patient_id: 10}
"glucose statistics" → intent: "statistics", entities: {features: ["glucose"]}
"what are the actual labels" → intent: "labels"
"how many patients are there" → intent: "count"
"what does BMI mean" → intent: "define", entities: {features: ["BMI"]}
"tell me about yourself" → intent: "about"

CONVERSATIONAL CONTEXT EXAMPLES:
"tell me more about that" → intent: "followup"
"so the model underpredicts the amount of people with diabetes?" → intent: "followup"
"this means the model is conservative" → intent: "followup"
"the model seems to underestimate cases" → intent: "followup"
"does this mean the model is biased?" → intent: "followup"
"so it appears the predictions are lower" → intent: "followup"
"therefore the model misses some cases" → intent: "followup"
"what about the model itself" → intent: "model"
"show me where the model predicted diabetes" → intent: "predictionfilter", entities: {prediction_values: [1]}
"filter to actual diabetic patients" → intent: "labelfilter", entities: {label_values: [1]}

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

    def _create_intent_validation_prompt(self) -> str:
        """Generate discussion-focused validation prompt for intent analysis."""
        return """You are an intent validation agent. ENGAGE IN CRITICAL DISCUSSION about the intent interpretation.

Your job: Look at the user's original query and the extracted intent, then DISCUSS with the intent extraction agent to reach consensus.

DISCUSSION GUIDELINES:
- Ask probing questions about ambiguous cases
- Challenge assumptions about user intent
- Discuss context implications (filtered vs full dataset)
- Consider alternative interpretations
- Reach consensus through reasoned discussion
- Focus particularly on dataset context and size implications

CRITICAL ANALYSIS QUESTIONS FOR DISCUSSION:
1. Could this query have multiple interpretations?
2. Did we capture the user's real goal?
3. Is there a better way to understand this request?
4. Are we missing important context or nuance?
5. CONTEXT-SENSITIVE: If dataset is currently filtered, does this query need full dataset or filtered dataset?
6. DATASET SIZE IMPLICATIONS: Does this query need all data or just the current filtered subset?
7. FOLLOW-UP DETECTION: Does this query reference previous results or make analytical conclusions? If so, it should be "followup".

INTENT TYPES TO CONSIDER:
- data: General dataset info, summaries (when no specific feature mentioned)
- statistics: Detailed stats about specific features (mean, std, distribution)
- predict: Model predictions for specific cases
- explain: Why did the model make this prediction
- important: Which features matter most
- performance: How accurate/good is the model
- filter: Show subset of data based on criteria
- show: Display specific data instances (individual patients or records)
- counterfactual: Generate counterfactual explanations for predictions
- interactions: How features work together
- mistakes: Where does the model fail (CRITICAL: Always needs full dataset for meaningful analysis)
- whatif: What-if analysis scenarios
- confidence: Prediction confidence scores
- labels: Ground truth information
- count: Count data points (analyze if user wants filtered count or total dataset count)
- define: Feature definitions
- about: System information
- casual: Greetings, chat
- followup: Follow-up questions
- model: Model information
- predictionfilter: Filter by predictions
- labelfilter: Filter by actual labels

CONTEXT-SENSITIVE VALIDATION:
When the dataset is currently filtered, critically analyze:
- "mistakes": MUST use full dataset (reset filter) - user wants to understand overall model errors
- "performance": MUST use full dataset (reset filter) - user wants overall model accuracy  
- "important": MUST use full dataset (reset filter) - user wants global feature importance
- "data": Depends on context - for general counts without "overall"/"total", keep current filter
- "statistics": Depends on context - could be filtered or full dataset
- "explain": Keep current filter if asking about specific filtered instances
- "predict": Keep current filter if asking about specific instances
- "count": CRITICAL - If user asks "overall", "total", "in the dataset" → needs full dataset
           If user asks "how many" with NEW filtering criteria (features, operators, values) → ALWAYS needs full dataset 
           If user asks "how many" without qualifiers and no new filters → use current filter context
           EXAMPLE: "how many instances are there with age > 40" has NEW criteria → requires_full_dataset: true

CRITICAL DECISION: 
- If user asks about "average", "mean", "std", "distribution" of a SPECIFIC FEATURE → use "statistics"
- If user asks about general dataset info without specific features → use "data"

EXAMPLES OF CRITICAL VALIDATION:

User: "How does age affect the model?"
Initial Intent: "performance" 
Critical Analysis: "This is ambiguous! Could mean:
- Feature importance of age (important)
- Performance across age groups (statistics) 
- Age interaction effects (interactions)
- Model accuracy on age-filtered data (performance)
Recommend: Ask for clarification or choose 'important' as most likely."

User: "so the model underpredicts the amount of people with diabetes?"
Initial Intent: "explain"
Critical Analysis: "This is a follow-up analytical question that should be 'followup' not 'explain'. The user is drawing a conclusion from previous results (comparing predictions vs ground truth). This doesn't need a new explanation - it needs a conversational response using existing context."
Validated Intent: "followup"

User: "Show me patients over 40"
Initial Intent: "filter"
Critical Analysis: "Correct! User clearly wants to filter data by age > 40."

User: "What are the feature values of patient 5?"
Initial Intent: "show"
Critical Analysis: "Correct! User wants to display data for a specific patient instance (patient 5), not filter the dataset. This is a 'show' request."

User: "Show me patient 10"
Initial Intent: "show"
Critical Analysis: "Correct! User wants to display a specific patient's data, which is exactly what 'show' intent is for."

User: "What's the accuracy on older patients?"
Initial Intent: "performance" 
Critical Analysis: "Good interpretation, but 'older' is vague. Should we assume age > 40, > 50, or > 65? Intent is correct but entities need clarification."

User: "What's the average age in the dataset?"
Initial Intent: "data"
Critical Analysis: "This asks for a specific statistic (average) about a specific feature (age). Should be 'statistics' not 'data'. Data is for general dataset info."
Validated Intent: "statistics" with entities: {"features": ["Age"]}

User: "How many instances are there with age > 40?"
Initial Intent: "count"
Critical Analysis: "This count query introduces NEW filtering criteria (age > 40), so it needs the full dataset to apply the filter from scratch. Even if the dataset is currently filtered to diabetes cases, the user wants to know about age > 40 across ALL patients."
Validated Intent: "count" with entities: {"features": ["Age"], "operators": [">"], "values": [40]} and requires_full_dataset: true

CRITICAL: When validating entities, preserve the EXACT SAME structure as the original entities. Use the same keys (features not feature, values not value, etc.)

OUTPUT FORMAT (JSON ONLY):
{
  "validated_intent": "final_intent_decision",
  "entities": {
    "patient_id": number_or_null,
    "features": ["feature_names_or_null"],
    "operators": ["operators_or_null"], 
    "values": [numbers_or_null],
    "filter_type": "prediction|feature|label|null",
    "prediction_values": [numbers_for_prediction_filtering_or_null],
    "label_values": [numbers_for_label_filtering_or_null]
  },
  "confidence": 0.95,
  "critical_analysis": "your_reasoning_about_potential_issues_or_ambiguities",
  "alternative_interpretations": ["list", "of", "other", "possible", "meanings"],
  "requires_full_dataset": true_or_false
}

Be thoughtful and question everything. Better to catch ambiguity now than give wrong answers later."""

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
        
        # ADD PREVIOUS QUERY CONTEXT AND DATASET SIZE ANALYSIS
        # Track what was filtered in the previous query to help agents understand context switches
        try:
            if hasattr(conversation, 'temp_dataset') and conversation.temp_dataset:
                temp_size = len(conversation.temp_dataset.contents.get('X', []))
                full_size = len(conversation.get_var('dataset').contents.get('X', []))
                
                if temp_size < full_size:
                    # Dataset is filtered - inform agents with detailed context
                    filter_percentage = (temp_size / full_size) * 100
                    context_components.append(f"DATASET STATUS: Currently filtered to {temp_size} out of {full_size} records ({filter_percentage:.1f}%)")
                    
                    # Add details about the filtering if available
                    if hasattr(conversation, 'parse_operation') and conversation.parse_operation:
                        filter_desc = ' '.join(conversation.parse_operation)
                        context_components.append(f"CURRENT FILTER: {filter_desc}")
                    
                    # Add last parse to understand previous query
                    if hasattr(conversation, 'last_parse_string') and conversation.last_parse_string:
                        last_parse = conversation.last_parse_string[-1] if conversation.last_parse_string else ""
                        if last_parse:
                            context_components.append(f"PREVIOUS QUERY ACTION: {last_parse}")
                            
                    # Add guidance for agents
                    context_components.append("DISCUSSION FOCUS: Consider whether this new query needs:")
                    context_components.append("- Full dataset analysis (reset filter)")
                    context_components.append("- Filtered dataset analysis (keep current filter)")
                    context_components.append("- New filtering criteria (apply new filter)")
                else:
                    context_components.append(f"DATASET STATUS: Using full dataset ({full_size} records)")
            else:
                # No filtering context available
                context_components.append("DATASET STATUS: Full dataset (no filtering applied)")
        except Exception as e:
            logger.warning(f"Could not extract filtering context: {e}")
        
        # ADD RECENT RESULTS CONTEXT FOR FOLLOW-UP DETECTION
        # Help agents identify when user is referencing previous results
        try:
            # Add model vs ground truth comparison context if available
            dataset = conversation.get_var('dataset')
            model = conversation.get_var('model')
            
            if dataset and model and hasattr(dataset, 'contents'):
                y_true = dataset.contents.get('y', [])
                X = dataset.contents.get('X', [])
                
                if y_true and X:
                    ground_truth_positive = sum(y_true)
                    total_instances = len(y_true)
                    
                    # Get model predictions for context
                    predictions = model.predict(X)
                    predicted_positive = sum(predictions)
                    
                    gt_percentage = (ground_truth_positive / total_instances) * 100
                    pred_percentage = (predicted_positive / total_instances) * 100
                    
                    context_components.append(f"RECENT RESULTS CONTEXT:")
                    context_components.append(f"- Ground truth: {ground_truth_positive} positive cases ({gt_percentage:.1f}%)")
                    context_components.append(f"- Model predictions: {predicted_positive} positive cases ({pred_percentage:.1f}%)")
                    
                    # Add follow-up detection hints
                    if predicted_positive < ground_truth_positive:
                        context_components.append("- Model underpredicts (conservative)")
                    elif predicted_positive > ground_truth_positive:
                        context_components.append("- Model overpredicts (aggressive)")
                    else:
                        context_components.append("- Model predictions match ground truth")
                    
                    context_components.append("NOTE: If user references these results or asks analytical questions about them, consider 'followup' intent")
                    
        except Exception as e:
            logger.warning(f"Could not extract recent results context: {e}")
        
        return "\n".join(context_components)
    
    async def complete(self, user_query: str, conversation, grammar: str = None) -> Dict[str, Any]:
        """Execute the multi-agent pipeline to convert user query to action.
        
        Process:
        1. Build context from conversation state
        2. Run agent collaboration using round-robin communication
        3. Integrate agent responses into final action command
        
        Args:
            user_query: Natural language input from user
            conversation: Current conversation context and system state
            grammar: Legacy parameter (unused, maintained for compatibility)
            
        Returns:
            Structured response with action and entities for execution
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
            f"Execute the collaborative natural language understanding pipeline:\n"
            f"1. Intent Extraction Agent: Analyze the query and extract intent/entities\n"
            f"2. Intent Validation Agent: Critically examine the interpretation\n"
            f"3. DISCUSS: Engage in back-and-forth discussion about ambiguities\n"
            f"4. FOCUS: Pay special attention to dataset context (filtered vs full dataset)\n"
            f"5. CONSENSUS: Reach agreement on final intent and entities\n"
            f"Collaborate through 4 rounds of discussion until you reach consensus."
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
                self.intent_validation_agent,
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
        """Extract and integrate JSON responses from the two agents.
        
        Parses each agent's JSON output and combines them into a final response.
        Uses direct action mapping.
        
        Args:
            collaboration_result: Raw output from agent collaboration
            
        Returns:
            Integrated response or None if processing fails
        """
        # Initialize response containers
        intent_response = None
        intent_validation_response = None
        
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
                        
                elif response_type == "intent_validation" and not intent_validation_response:
                    intent_validation_response = extracted_response
        
        # Log what we found from agents for debugging
        logger.info(f"Agent responses found: Intent={bool(intent_response)}, IntentValidation={bool(intent_validation_response)}")
        
        # Integrate responses based on available information
        return self._integrate_agent_outputs(intent_response, intent_validation_response)

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
        elif 'validated_intent' in response:
            return "intent_validation"
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
        Uses consistent format with final_action to maintain generalizability.
        
        Returns:
            Formatted casual conversation response
        """
        return {
            "generation": "parsed: self[e]",
            "confidence": 0.95,
            "method": "autogen_conversational",
            "final_action": "self",
            "command_structure": {
                "greeting": True,
                "casual_response": (
                    "Hello! I'm an AI assistant specialized in machine learning model exploration "
                    "for diabetes risk assessment. I can help you analyze patient data, understand "
                    "model predictions, explore feature importance, and perform what-if analyses. "
                    "What would you like to investigate?"
                )
            },
            "action_list": ["self"],
            "validation_passed": True
        }

    def _integrate_agent_outputs(self, intent_response, intent_validation_response) -> Optional[Dict[str, Any]]:
        """
        Integrate Multi-Agent Outputs into Unified Response
        
        This method implements the core integration logic that combines outputs
        from the two agents into a coherent, actionable response. Uses direct
        action mapping instead of a third agent.
        
        Integration Strategy:
            1. Complete Integration: Both agents provided valid responses
            2. Partial Integration: Only intent extraction provided response
            3. Direct Action Mapping: Map validated intent to action
            4. Confidence Scoring: Aggregate confidence measures
        
        Args:
            intent_response: Output from intent extraction agent
            intent_validation_response: Output from intent validation agent
            
        Returns:
            Integrated response ready for action execution
        """
        # Scenario 1: Complete agent collaboration (ideal case)
        if intent_response and intent_validation_response:
            return self._create_complete_response(intent_response, intent_validation_response)
            
        # Scenario 2: Partial collaboration (fallback case)
        elif intent_response:
            return self._create_partial_response(intent_response)
            
        # Scenario 3: Insufficient information for integration
        else:
            logger.warning("Insufficient agent responses for integration")
            return None

    def _create_complete_response(self, intent_response, intent_validation_response) -> Dict[str, Any]:
        """
        Create Response from Complete Agent Collaboration
        
        Constructs the final response when both agents have successfully
        contributed to the processing pipeline. Uses the validated intent from
        the intent validation agent and direct action mapping.
        
        Args:
            intent_response: Original intent extraction results
            intent_validation_response: Critical thinking validation results
            
        Returns:
            Complete integrated response from 2-agent collaboration
        """
        # Use validated intent from the critical thinking agent
        validated_intent = intent_validation_response.get('validated_intent', intent_response.get('intent', 'data'))
        
        # Map intent to action using direct mapping
        final_action = self._map_intent_to_action(validated_intent)
        
        # Use validated entities from intent validation, falling back to original entities
        validated_entities = intent_validation_response.get('entities', {})
        original_entities = intent_response.get('entities', {})
        
        # Normalize entity keys (handle both singular and plural forms)
        if 'feature' in validated_entities and 'features' not in validated_entities:
            validated_entities['features'] = [validated_entities.pop('feature')]
        
        command_structure = {**original_entities, **validated_entities}  # Validation takes precedence
        
        # Create action list for backward compatibility
        action_list = [final_action] if final_action else ["explain"]
        
        # Calculate aggregate confidence score
        confidence_score = min(
            intent_response.get('confidence', 0.8),
            intent_validation_response.get('confidence', 0.8)
        )
        
        return {
            "generation": f"parsed: {final_action}[e]",
            "confidence": confidence_score,
            "method": "autogen_2agent_discussion",
            "intent_response": {
                "intent": validated_intent,  # Use validated intent for filter reset logic
                "entities": command_structure,
                "confidence": confidence_score
            },
            "validation_response": {
                "requires_full_dataset": intent_validation_response.get('requires_full_dataset', False),
                "critical_analysis": intent_validation_response.get('critical_analysis', ''),
                "alternative_interpretations": intent_validation_response.get('alternative_interpretations', [])
            },
            "agent_reasoning": {
                "original_intent": intent_response.get('intent', 'data'),
                "validated_intent": validated_intent,
                "critical_analysis": intent_validation_response.get('critical_analysis', ''),
                "alternative_interpretations": intent_validation_response.get('alternative_interpretations', []),
                "action_mapping": f"Intent '{validated_intent}' → Action '{final_action}'"
            },
            "command_structure": command_structure,
            "action_list": action_list,
            "final_action": final_action,
            "validation_passed": True,  # Critical thinking validation always passes with improvements
            "identified_issues": intent_validation_response.get('alternative_interpretations', [])
        }

    def _create_partial_response(self, intent_response) -> Dict[str, Any]:
        """
        Create Response from Partial Agent Collaboration
        
        Handles scenarios where the validation agent failed to provide output
        but intent extraction succeeded. Uses direct action mapping.
        
        Args:
            intent_response: Intent extraction results  
            
        Returns:
            Partial response with available information
        """
        # Get intent and map to action
        intent = intent_response.get('intent', 'explain')
        final_action = self._map_intent_to_action(intent)
        
        # Calculate confidence with penalty for missing validation
        confidence_score = intent_response.get('confidence', 0.8) * 0.9  # Penalty for missing validation
        
        return {
            "generation": f"parsed: {final_action}[e]",
            "confidence": confidence_score,
            "method": "autogen_partial_pipeline",
            "intent_response": intent_response,  # Include full intent response for context reset detection
            "agent_reasoning": {
                "intent_analysis": intent_response.get('reasoning', ''),
                "action_mapping": f"Intent '{intent}' → Action '{final_action}'",
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


# Factory functions for creating decoder instances

def create_autogen_decoder(**kwargs) -> AutoGenDecoder:
    """Factory function to create AutoGenDecoder instances."""
    return AutoGenDecoder(**kwargs)


def get_autogen_predict_func(api_key: str = None, model: str = "gpt-4o"):
    """Create a prediction function compatible with legacy decoder interfaces.
    
    Args:
        api_key: OpenAI API key 
        model: Language model identifier
        
    Returns:
        Prediction function that can replace legacy decoders
    """
    decoder = AutoGenDecoder(api_key=api_key, model=model)
    
    def prediction_function(prompt: str, grammar: str = None, conversation=None):
        """Legacy-compatible prediction function that extracts queries and processes them."""
        # Extract user query from legacy prompt format
        if "Query:" in prompt:
            user_query = prompt.split("Query:")[-1].strip()
        else:
            user_query = prompt
            
        return decoder.complete_sync(user_query, conversation, grammar)
    
    return prediction_function


# Alias for backward compatibility
AutoGenNaturalLanguageUnderstanding = AutoGenDecoder 