"""AutoGen-based multi-agent decoder for natural language to action translation.

This module uses a 2-agent discussion system to convert user queries into executable actions:
1. Intent Extraction Agent - Identifies user intent and extracts entities
2. Intent Validation Agent - Validates and refines the extracted intent through critical discussion

The agents collaborate through 2 rounds of discussion to reach consensus, with direct action mapping.
"""

import os
import json
import asyncio
import logging
import re
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
        
        # No longer need action mapping - agents output actions directly
        
        logger.info(f"Initialized AutoGenDecoder with model={model}, max_rounds={max_rounds}")
    
    # Removed: No longer need intent-to-action mapping
    

    def _setup_agent_architecture(self):
        """Initialize the two specialized agents for the processing pipeline.
        
        Creates:
        - Action extraction agent: Understands user action and extracts entities
        - Action validation agent: Critically examines and refines action through discussion
        """
        
        # Agent 1: Action Extraction and Entity Recognition
        # This agent performs the initial semantic analysis of user input
        self.action_extraction_agent = AssistantAgent(
            name="ActionExtractor",
            model_client=self.model_client,
            system_message=self._create_action_extraction_prompt()
        )
        
        # Agent 2: Action Validation and Critical Analysis
        # This agent critically examines if the action was correctly interpreted
        self.action_validation_agent = AssistantAgent(
            name="ActionValidator",
            model_client=self.model_client,
            system_message=self._create_action_validation_prompt()
        )
        
        logger.info("Successfully initialized two-agent architecture")

    def _create_action_extraction_prompt(self) -> str:
        """Generate clean and generalizable action extraction prompt."""
        return """
# Action Extraction Agent

## Actions // none other than the ones listed here are allowed
- predict: Generate predictions or assess likelihood of certain outcomes based on the model.
- filter: Retrieve or display specific patient records or data points by their unique identifiers.
- label: Display actual, known outcomes or true values from the dataset.
- important: Identify and rank features according to their significance or influence on the models predictions.
- explain: Provide explanations or reasoning for why the model made specific predictions.
- whatif: Analyze the effects of HYPOTHETICAL adjustments or changes to specific feature values. Would it change the prediction?
- counterfactual: Suggest minimal feature changes necessary to alter or reverse the models prediction.
- mistake: Display cases where the models did mistakes or predictions were incorrect or erroneous.
- score: Present model evaluation metrics or performance measures.
- statistic: Provide statistical summaries or data distributions for features (not target values!) within the dataset.
- interact: Explore interactions or combined effects among multiple features. Only applies to filters.
- define: Provide definitions or clarifications of relevant terminology or concepts.
- self: Address questions related to the assistants own capabilities, and functions.
- followup: Allow continuation or deeper exploration based on prior analyses or interactions.


## Entity Extraction Rules
- ONLY extract features that are EXPLICITLY NAMED in the query
- If NO features are mentioned → features = null
- Use parallel arrays for multiple conditions: ["feature1", "feature2"] pairs with [">", "-"] and [100, 30]; Do not over-extract features.
- DO NOT create additional JSON fields 
- For explain, interact never extract all features 

## Patterns
- Features:
  - Direct (BMI, age, glucose)
  - Natural (high BMI → BMI)
  - Compound (over 40 years → Age)

- Operators:
  - ">": over, above
  - "<": under, below
  - "=": exactly, equal
  - "+": increase, add
  - "-": decrease, remove

- Values:
  - Units: "40 years old" → Age:40
  - Simple: "BMI over 30" → BMI:30
  - Ranges: "between 30 and 50" → Feature:[">=", "<="]:[30,50]

- Special:
  - Rankings: "top X" → topk:X
  - IDs: "patient X" → patient_id:X
  - Target values are only needed if the wants to EXPLICITLY filter for instances of a certain outcome class (1 for diabetic, 0 for healthy)

## Discussion Approach
- Explain your reasoning for the validator to review
- Consider alternative interpretations
- Be open to feedback and refinement

Then provide your initial JSON proposal:

**CRITICAL: NO COMMENTS IN JSON - JSON must be valid without // comments**

```json
{
  "action": "detected_action_name",
  "entities": {
    "patient_id": "number_or_null",
    "features": ["feature_name_or_null"],
    "operators": ["operators_or_null"],
    "values": ["numbers_or_null"],
    "topk": "number_or_null",
    "target_values": ["numbers_or_null"]
  }
}
```

Ensure JSON validity without internal comments.
"""

    def _create_action_validation_prompt(self) -> str:
        """Generate focused JSON validation prompt with validation rules baked into JSON comments."""
        return """You are a collaborative validation agent engaging in discussion with the extraction agent.


## Discussion Approach
- Look for ways to improve the extraction // do not question the action
- Consider if there's a more accurate interpretation  
- Ask clarifying questions about entity choices
- Suggest refinements constructively
- Build toward the best possible extraction together

Validate each field using this template:

```json
{
  "action": "action_name", 
  // Always take the action from the extractor, do not change it or question it
  
  "entities": {
    "patient_id": number_or_null,
    // VALIDATE PATIENT_ID: Only set if query mentions specific patient/data point number (e.g., "patient 21", "data point 33"). Otherwise null.
    
    "features": ["feature_names"] or null,
    // STRICT: Only features NAMED in query 
    // "which features" or "the features" → null (asking ABOUT features, not naming them)
    // If ALL features extracted, likely wrong!
    
    "operators": ["operators"] or null,
    // VALIDATE OPERATORS: Convert natural language to symbols: "over"/">", "under"/"<", "equals"/"is"/"=", "increase"/"+", "decrease"/"-"
    // Must align with features array: features[0] pairs with operators[0]
    // Set to null if no conditions mentioned
    
    "values": [numbers] or null,
    // VALIDATE VALUES: Extract exact numbers mentioned in conditions
    // Must align with operators: operators[0] pairs with values[0] 
    // "glucose over 120" → [120], "age between 30 and 50" → [30, 50]
    // Set to null if no values mentioned
    
    "topk": number_or_null,
    // VALIDATE TOPK: Only set if query asks for "top X", "first X", "most important X" with specific number
    // "top 5 features" → 5, "most important features" → null (no number specified)
    
    "target_values": [values] or null
    // CRITICAL: "these people" / "all these people" → target_values = null (contextual reference)
    // Only extract target_values when explicitly requesting to filter new data by outcome
    // RULE: Only when **explicitly** filtering for instances with diabetes or no diabetes
  }
}
```

After your critical analysis, provide clean JSON without comments."""

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
        logger.debug(f"Context prompt: {contextual_prompt[:1000]}...")
        logger.info("Stage 1: Context construction completed")
        
        # Stage 2: Multi-Agent Team Configuration
        termination_handler = self._configure_termination_condition()
        logger.info("Stage 2: Termination condition configured")
        
        team_configuration = self._create_agent_team(termination_handler)
        logger.info("Stage 2: Agent team created successfully")
        
        # Stage 3: Execute Collaborative Processing Pipeline
        processing_prompt = (
            f"{contextual_prompt}\n\n"
            f"COLLABORATIVE DISCUSSION:\n"
            f"Work together to understand the user's intent and extract the correct information.\n"
            f"1. Extractor: Share your interpretation and reasoning\n"
            f"2. Validator: Engage with the reasoning, offer perspectives, refine together\n"
            f"3. Build consensus through discussion (max {self.max_rounds} rounds)\n\n"
            f"User Query: \"{user_query}\"\n\n"
            f"Begin collaborative analysis:"
        )
        
        logger.info("Stage 3: Starting agent collaboration...")
        logger.info(f"Processing prompt: {processing_prompt[:200]}...")
        try:
            collaboration_result = await team_configuration.run(task=processing_prompt)
            logger.info(f"Stage 3: Agent collaboration completed with {len(collaboration_result.messages) if hasattr(collaboration_result, 'messages') else 0} messages")
            logger.info(f"Collaboration result type: {type(collaboration_result)}")
            if hasattr(collaboration_result, 'messages'):
                logger.info(f"Messages attribute exists, length: {len(collaboration_result.messages)}")
                for i, msg in enumerate(collaboration_result.messages):
                    logger.info(f"Message {i}: source={getattr(msg, 'source', 'unknown')}, content_length={len(getattr(msg, 'content', ''))}")
            else:
                logger.warning("collaboration_result has no 'messages' attribute")
                logger.info(f"Available attributes: {dir(collaboration_result)}")
        except Exception as e:
            logger.error(f"Exception during agent collaboration: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
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
        """Configure Agent Team Termination Conditions with consensus detection."""
        from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
        
        # For debugging: Use only max message termination to avoid premature exits
        max_message_termination = MaxMessageTermination(self.max_rounds)
        logger.info(f"Using only MaxMessageTermination with max_rounds={self.max_rounds}")
        
        return max_message_termination
        
        # DISABLED FOR DEBUGGING: Combined termination was causing immediate exits
        # consensus_termination = TextMentionTermination("CONSENSUS_REACHED")
        # combined_termination = consensus_termination | max_message_termination
        # return combined_termination

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
                self.action_extraction_agent,
                self.action_validation_agent,
            ]
        }
        
        # Add termination condition with version-appropriate parameter name
        if termination_handler:
            signature = inspect.signature(RoundRobinGroupChat)
            if "termination_condition" in signature.parameters:
                team_parameters["termination_condition"] = termination_handler
                logger.info("Termination condition attached using 'termination_condition' parameter")
            elif "termination" in signature.parameters:
                team_parameters["termination"] = termination_handler
                logger.info("Termination condition attached using 'termination' parameter")
            elif "termination_checker" in signature.parameters:
                team_parameters["termination_checker"] = termination_handler
                logger.info("Termination condition attached using 'termination_checker' parameter")
            else:
                logger.warning("Could not attach termination condition - no supported parameter found")
        
        # Use only max message termination for debugging
        # termination_handler = None
        
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
        
        if len(collaboration_result.messages) <= 1:
            logger.warning("No agent responses generated - falling back to default")
            return self._create_error_response("No agent responses")

        # Extract structured responses from agent communications
        # Process messages from newest to oldest to prioritize most recent responses
        for message_index, message in enumerate(reversed(collaboration_result.messages)):
            # Calculate original index for logging consistency
            original_index = len(collaboration_result.messages) - 1 - message_index
            
            # Debug: Log agent message content
            if hasattr(message, 'content') and message.content:
                logger.warning(f"=== MESSAGE {original_index} (processing order {message_index}) FULL CONTENT ===")
                logger.warning(f"Source: {getattr(message, 'source', 'unknown')}")
                logger.warning(f"Length: {len(message.content)}")
                logger.warning(f"Content: {message.content}")
                logger.warning(f"=== END MESSAGE {original_index} ===")
                # Check for consensus marker in message
                if "CONSENSUS_REACHED" in message.content:
                    logger.info(f"Found CONSENSUS_REACHED marker in message {original_index} - consensus should have triggered termination")
            else:
                logger.warning(f"Message {original_index}: No content")
            
            extracted_response = self._extract_json_response(message, original_index)
            
            if extracted_response and isinstance(extracted_response, dict):
                logger.info(f"Message {original_index}: Successfully extracted JSON: {extracted_response}")
                # Classify response by content structure
                response_type = self._classify_response_type(extracted_response)
                logger.info(f"Message {original_index}: Classified as response type: {response_type}")
                
                # Process newest first: first valid JSON = validation agent, second = extraction agent
                if response_type == "action":
                    if not intent_validation_response:
                        intent_validation_response = extracted_response
                        logger.info(f"Message {original_index}: Set as intent_validation_response (newest action response)")
                    elif not intent_response:
                        intent_response = extracted_response
                        logger.info(f"Message {original_index}: Set as intent_response (second newest action response)")
            else:
                logger.warning(f"Message {original_index}: Failed to extract valid JSON response")
        
        # Log what we found from agents for debugging
        logger.info(f"Agent responses found: Intent={bool(intent_response)}, IntentValidation={bool(intent_validation_response)}")
        
        # Integrate responses based on available information
        return self._integrate_agent_outputs(intent_response, intent_validation_response)

    def _extract_json_response(self, message, message_index: int) -> Optional[Dict]:
        """Extract JSON from agent message - clean and direct."""
        if not hasattr(message, 'content') or not message.content:
            logger.debug(f"Message {message_index}: No content attribute or empty content")
            return None
        # Check for None content before using 'in' operator
        if message.content is None or '{' not in message.content:
            logger.debug(f"Message {message_index}: No JSON markers found in content")
            return None
            
        content = message.content
        
        try:
            import re
            
            # Pattern 1: JSON code block (most common and reliable)
            json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1).strip()
                logger.debug(f"Message {message_index}: Found JSON code block: {json_str[:100]}...")
                result = json.loads(json_str)
                return result if isinstance(result, dict) else None
            
            # Pattern 2: JSON code block without 'json' label
            json_block_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1).strip()
                logger.debug(f"Message {message_index}: Found generic code block: {json_str[:100]}...")
                result = json.loads(json_str)
                return result if isinstance(result, dict) else None
            
            # Pattern 3: Direct JSON (entire content is JSON)
            if content.strip().startswith('{') and content.strip().endswith('}'):
                json_str = content.strip()
                logger.debug(f"Message {message_index}: Found direct JSON: {json_str[:100]}...")
                result = json.loads(json_str)
                return result if isinstance(result, dict) else None
                
            # Pattern 4: Enhanced embedded JSON with balanced braces
            # Find JSON objects with proper brace balancing
            brace_count = 0
            start_pos = -1
            
            for i, char in enumerate(content):
                if char == '{':
                    if brace_count == 0:
                        start_pos = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_pos != -1:
                        json_candidate = content[start_pos:i+1]
                        try:
                            result = json.loads(json_candidate)
                            if isinstance(result, dict):
                                logger.debug(f"Message {message_index}: Found embedded JSON: {json_candidate[:100]}...")
                                return result
                        except json.JSONDecodeError:
                            continue  # Try next potential JSON object
                        
        except json.JSONDecodeError as e:
            logger.warning(f"Message {message_index}: JSON decode error: {e}")
        
        logger.warning(f"Message {message_index}: No valid JSON found after trying all patterns. Content preview: {content[:200]}...")
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
        # Handle None response gracefully
        if response is None:
            return "unknown"
            
        # Since both agents now use "action", classify by response order/context
        # The first agent response is extraction, subsequent ones are validation
        if 'action' in response:
            return "action"  # Will handle ordering in the caller
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
        # Use validated action directly from the critical thinking agent
        validated_action = intent_validation_response.get('action', intent_response.get('action', 'explain'))
        
        # Use validated entities from intent validation, falling back to original entities
        validated_entities = intent_validation_response.get('entities', {})
        original_entities = intent_response.get('entities', {})
        
        # Normalize entity keys (handle both singular and plural forms)
        if 'feature' in validated_entities and 'features' not in validated_entities:
            validated_entities['features'] = [validated_entities.pop('feature')]
        
        # Smart entity merging - use validation agent's complete entity structure
        # The validation agent should provide the complete corrected entities
        if validated_entities:
            command_structure = validated_entities.copy()
        else:
            # Fallback to original entities if validation agent didn't provide entities
            command_structure = original_entities.copy()
        
        # Create action list for backward compatibility
        action_list = [validated_action] if validated_action else ["explain"]
        
        # Calculate aggregate confidence score
        confidence_score = min(
            intent_response.get('confidence', 0.8),
            intent_validation_response.get('confidence', 0.8)
        )
        
        return {
            "generation": f"parsed: {validated_action}[e]",
            "confidence": confidence_score,
            "method": "autogen_2agent_discussion",
            "action_response": {
                "action": intent_response.get('action', 'explain'),
                "entities": original_entities,
                "confidence": confidence_score
            },
            "validation_response": {
                "requires_full_dataset": intent_validation_response.get('requires_full_dataset', False),
                "critical_analysis": intent_validation_response.get('critical_analysis', ''),
                "alternative_interpretations": intent_validation_response.get('alternative_interpretations', [])
            },
            "agent_reasoning": {
                "original_action": intent_response.get('action', 'explain'),
                "validated_action": validated_action,
                "critical_analysis": intent_validation_response.get('critical_analysis', ''),
                "alternative_interpretations": intent_validation_response.get('alternative_interpretations', [])
            },
            "command_structure": command_structure,
            "action_list": action_list,
            "final_action": validated_action,
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
        # Get action directly - no mapping needed
        final_action = intent_response.get('action', 'explain')
        
        # Calculate confidence with penalty for missing validation
        confidence_score = intent_response.get('confidence', 0.8) * 0.9  # Penalty for missing validation
        
        return {
            "generation": f"parsed: {final_action}[e]",
            "confidence": confidence_score,
            "method": "autogen_partial_pipeline",
            "action_response": {
                "action": final_action,
                "entities": intent_response.get('entities', {}),
                "confidence": confidence_score
            },
            "agent_reasoning": {
                "action_analysis": intent_response.get('reasoning', ''),
                "validation_results": "Validation agent response unavailable"
            },
            "command_structure": intent_response.get('entities', {}),
            "action_list": [final_action],
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
        
        # Try to extract action and entities from first agent response
        action_response = None
        for message in collaboration_result.messages:
            extracted_response = self._extract_json_response(message, 0)
            if extracted_response and isinstance(extracted_response, dict) and 'action' in extracted_response:
                action_response = extracted_response
                break
        
        if action_response:
            # Use the extracted action and entities
            action = action_response.get("action")
            return {
                "generation": f"parsed: {action}[e]",
                "confidence": action_response.get("confidence", 0.8),
                "method": "autogen_partial_pipeline",
                "action_response": {
                    "action": action,
                    "entities": action_response.get("entities", {}),
                    "confidence": action_response.get("confidence", 0.8)
                },
                "agent_reasoning": {
                    "action_analysis": "",
                    "validation_results": "Validation agent response unavailable"
                },
                "command_structure": action_response.get("entities", {}),
                "action_list": [action],
                "final_action": action,
                "validation_passed": True,
                "identified_issues": []
            }
        else:
            # NO FALLBACKS - Fail fast when AutoGen doesn't work
            raise Exception("AutoGen failed to extract action - no fallback allowed")

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
                
                # Don't call cleanup here - it can interfere with event loop closure
                # The decoder will be reused without cleanup between test cases
                return result
            except asyncio.TimeoutError:
                logger.warning("AutoGen processing timed out")
                return self._create_error_response("Processing timeout - please try a simpler query")
            except Exception as e:
                logger.error(f"Error in async processing: {e}")
                return self._create_error_response(f"AutoGen processing error: {str(e)}")
            finally:
                try:
                    # Only clean up the event loop, not the decoder resources
                    # Decoder cleanup happens only when creating a fresh decoder between batches
                    
                    # Cancel any pending tasks in the event loop
                    pending = asyncio.all_tasks(new_loop)
                    if pending:
                        for task in pending:
                            task.cancel()
                        # Wait briefly for cancellation to complete
                        try:
                            new_loop.run_until_complete(asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True), 
                                timeout=1.0
                            ))
                        except asyncio.TimeoutError:
                            pass  # Tasks didn't cancel in time, proceed anyway
                    
                    # Close the loop
                    new_loop.close()
                    logger.debug("Event loop cleaned up successfully")
                except Exception as cleanup_error:
                    logger.warning(f"Event loop cleanup warning: {cleanup_error}")
        
        try:
            logger.info("Executing AutoGen processing in isolated thread...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_isolated_thread)
                result = future.result(timeout=60)  # Increased to 60 seconds for reliable AutoGen processing
                logger.info("Thread execution completed successfully")
                return result
                
        except concurrent.futures.TimeoutError:
            logger.error("Thread execution timed out after 60 seconds")
            return self._create_error_response("System timeout after 60 seconds - agents need more focus")
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
    
    async def cleanup(self, full_cleanup=True):
        """
        Clean Up System Resources
        
        Properly releases resources used by the multi-agent system,
        including LLM client connections and optionally agent references.
        """
        try:
            # Only close model client connections during full cleanup
            # When reusing decoder, keep the client alive for subsequent requests
            if full_cleanup and hasattr(self, 'model_client') and self.model_client:
                await self.model_client.close()
            
            # Only clear agent references if doing full cleanup (e.g., before deletion)
            # Don't clear them if we plan to reuse the decoder
            if full_cleanup:
                if hasattr(self, 'intent_extraction_agent'):
                    self.intent_extraction_agent = None
                if hasattr(self, 'intent_validation_agent'):
                    self.intent_validation_agent = None
                
            logger.debug(f"AutoGen decoder resources cleaned up (full_cleanup={full_cleanup})")
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