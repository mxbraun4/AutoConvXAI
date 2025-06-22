"""AutoGen Multi-Agent Decoder for TalkToModel.

This module uses AutoGen's multi-agent framework with specialized agents for:
- Intent extraction and understanding
- Action planning and syntax generation  
- Validation and error correction
- Response coordination

Each agent specializes in one aspect for better accuracy and maintainability.
"""
import os
import json
import asyncio
from typing import Dict, Any, Optional

# Import AutoGen components with compatibility for newer versions
try:
    # Try new AutoGen structure (v0.4+)
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    try:
        # Try older AutoGen structure
        from autogen.agentchat.agents import AssistantAgent
        from autogen.agentchat.teams import RoundRobinGroupChat
        from autogen.models.openai import OpenAIChatCompletionClient
        AUTOGEN_AVAILABLE = True
    except ImportError:
        # AutoGen not available - raise clear error
        AUTOGEN_AVAILABLE = False
        AssistantAgent = None
        RoundRobinGroupChat = None
        OpenAIChatCompletionClient = None


class AutoGenDecoder:
    """Multi-agent decoder using AutoGen framework."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o",
                 max_rounds: int = 3):
        """Initialize AutoGen multi-agent decoder.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use for all agents
            max_rounds: Maximum conversation rounds between agents
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen packages not available. Please install with: "
                "pip install autogen-agentchat>=0.4.0 autogen-core>=0.4.0 autogen-ext>=0.4.0"
            )
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.max_rounds = max_rounds
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        # Initialize the model client
        self.model_client = OpenAIChatCompletionClient(
            model=self.model,
            api_key=self.api_key,
        )
        
        # Initialize specialized agents
        self._setup_agents()
    
    def _validate_and_fix_action(self, action_syntax: str) -> str:
        """Validate and fix only critical action syntax errors, letting the smart dispatcher handle complexity."""
        if not action_syntax or not isinstance(action_syntax, str):
            return "explain"
        
        action_syntax = action_syntax.strip().lower()
        parts = action_syntax.split()
        
        if not parts:
            return "explain"
        
        # Only fix truly critical errors - let smart dispatcher handle the rest
        if parts[0] == "important" and len(parts) == 1:
            # "important" alone should become "important all"
            return "important all"
        
        # Validate against known actions
        valid_actions = ["filter", "predict", "explain", "important", "score", "show", "change", "mistake", "data"]
        if parts[0] not in valid_actions:
            return "explain"  # Safe fallback
        
        # For everything else, trust the smart dispatcher to handle it
        return action_syntax

    def _setup_agents(self):
        """Set up the specialized agents for different tasks."""
        
        # Intent Extraction Agent
        self.intent_agent = AssistantAgent(
            name="IntentExtractor",
            model_client=self.model_client,
            system_message="""You are an expert at understanding user queries about machine learning models.

Your job is to extract the user's intent and identify key entities from their natural language query.

INTENT TYPES:
- explain: User wants explanation of predictions
- predict: User wants to see predictions
- filter: User wants to filter/explore data
- performance: User wants model performance metrics
- importance: User wants feature importance
- whatif: User wants what-if analysis
- mistakes: User wants to see model errors
- data: User wants data summary/exploration
- casual: Casual conversation/greetings (hi, hello, test, what, who, how are you, thank you, etc.)

ENTITIES TO EXTRACT:
- patient_id: Specific patient/case ID mentioned
- features: Feature names mentioned (age, bmi, glucose, etc.)
- operators: Comparison operators (greater than, less than, equal to)
- values: Specific values mentioned
- explanation_type: lime, shap, or general

RESPONSE FORMAT:
{
  "intent": "intent_type",
  "entities": {
    "patient_id": number_or_null,
    "features": ["feature1", "feature2"],
    "operators": ["greater", "less"],
    "values": [50, 30],
    "explanation_type": "lime/shap/general"
  },
  "confidence": 0.95,
  "reasoning": "brief explanation"
}

Only respond with valid JSON in this format."""
        )
        
        # Action Planning Agent
        self.action_agent = AssistantAgent(
            name="ActionPlanner",
            model_client=self.model_client,
            system_message="""You are an expert at converting user intents into precise action commands.

Based on the intent and entities extracted, generate the correct action syntax.

ACTION SYNTAX:
- filter {feature} {operator} {value} - Filter data
- predict - Show predictions for current data
- predict {id} - Show prediction for specific ID
- explain {id} - Explain prediction for ID
- explain {id} lime - LIME explanation
- explain {id} shap - SHAP explanation
- important all - Show all feature importance rankings
- important topk {number} - Show top N features
- important {feature} - Show importance of specific feature
- score {metric} - Model performance (accuracy, precision, recall, f1, roc, default)
- show {id} - Show data for ID
- change {feature} {operation} {value} - What-if analysis
- mistake - Show model errors
- data - Show data summary

OPERATORS: greater, less, greaterequal, lessequal, equal
OPERATIONS: set, increase, decrease

GUIDELINES:
- Generate natural actions that match user intent intuitively
- For feature importance, default to "important all" if unclear
- When users want predictions with conditions, express their complete intent naturally
- Be flexible and expressive - compound actions are handled intelligently

EXAMPLES:
Intent: explain, entities: {patient_id: 5} → "explain 5"
Intent: predict, entities: {features: ["age"], operators: ["greater"], values: [50]} → "filter age greater 50 predict"
Intent: predict, entities: {features: ["age", "pregnancies"], operators: ["greater", "equal"], values: [50, 0]} → "filter age greater 50 pregnancies equal 0 predict"
Intent: performance → "score accuracy"
Intent: importance, entities: {number: 3} → "important topk 3"
Intent: importance, entities: {} → "important all"

RESPONSE FORMAT:
{
  "action": "filter age greater 50",
  "reasoning": "User wants to filter patients over 50",
  "confidence": 0.95
}

Only respond with valid JSON in this format."""
        )
        
        # Validation Agent
        self.validation_agent = AssistantAgent(
            name="ActionValidator",
            model_client=self.model_client,
            system_message="""You are an expert at validating and correcting action commands for a diabetes prediction model.

DATASET CONTEXT:
- Features: pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age
- Target classes: 0 (no diabetes), 1 (diabetes)
- Valid patient IDs: typically 1-768 (check context for exact range)

VALIDATION RULES:
1. Feature names must match exactly (case-sensitive)
2. Operators must be: greater, less, greaterequal, lessequal, equal
3. Patient IDs must be positive integers
4. Numeric values must be reasonable for medical data
5. Actions must follow correct syntax
6. Validate that "important" has proper specification - suggest "important all" if unclear
7. Accept compound actions - they will be handled by the smart dispatcher

COMMON CORRECTIONS:
- "important" → "important all"
- "BMI" → "bmi"
- "Age" → "age"
- ">" → "greater"
- "<" → "less"
- Missing space between tokens

ACCEPTED ACTION FORMATS:
- important all (show all features)
- important topk 3 (top 3 features)
- important bmi (specific feature)
- filter age greater 50 predict (compound actions are acceptable)
- filter age greater 50 pregnancies equal 0 predict (multi-condition filters are acceptable)

RESPONSE FORMAT:
{
  "valid": true/false,
  "corrected_action": "corrected action if needed",
  "issues": ["list of issues found"],
  "confidence": 0.95
}

If valid, return the original action. If invalid, provide corrected version."""
        )
        

    
    def _build_context_prompt(self, user_query: str, conversation) -> str:
        """Build context about the dataset and current state."""
        context_parts = [f"USER QUERY: {user_query}"]
        
        # Add dataset context
        try:
            dataset = conversation.get_var('dataset')
            if dataset:
                data_size = len(dataset.contents.get('X', []))
                context_parts.append(f"DATASET SIZE: {data_size} patients")
                
                # Get feature info
                cat_features = dataset.contents.get('cat', [])
                num_features = dataset.contents.get('numeric', [])
                all_features = cat_features + num_features
                context_parts.append(f"AVAILABLE FEATURES: {', '.join(all_features)}")
                
                # Get class info
                if hasattr(conversation, 'class_names'):
                    context_parts.append(f"TARGET CLASSES: {', '.join(map(str, conversation.class_names))}")
        except Exception as e:
            context_parts.append("DATASET CONTEXT: Unable to retrieve")
        
        return "\n".join(context_parts)
    
    async def complete(self, user_query: str, conversation, grammar: str = None) -> Dict[str, Any]:
        """Generate completion using multi-agent collaboration.
        
        Args:
            user_query: User's natural language query
            conversation: Conversation object with context
            grammar: Ignored (kept for compatibility)
            
        Returns:
            Dict with action information compatible with existing system
        """
        try:
            # Build context for all agents
            context = self._build_context_prompt(user_query, conversation)
            
            # -------------------------------------------------------------
            # Create the multi-agent team with a termination mechanism.
            # The AutoGen public API changed around v0.6: the keyword
            # formerly called "termination_condition" was renamed to
            # "termination" (and in some nightlies "termination_checker").
            # Using a hard-coded argument therefore breaks with an
            # "unexpected keyword argument 'termination_condition'" error
            # on newer versions.
            #
            # To stay compatible across versions we inspect the signature
            # of `RoundRobinGroupChat` at runtime and supply whichever
            # keyword is accepted. If none of the known keywords are
            # present we simply omit the argument – the group chat then
            # falls back to its internal default termination logic which
            # is usually a sensible safeguard.
            # -------------------------------------------------------------
            # Import termination helper – account for renamed modules across
            # AutoGen releases.
            try:
                from autogen_agentchat.conditions import MaxMessageTermination
            except ModuleNotFoundError:
                # Newer versions moved it to `autogen.stop.conditions`.
                from autogen.stop.conditions import MaxMessageTermination  # type: ignore

            import inspect

            termination = MaxMessageTermination(self.max_rounds)

            # Figure out which keyword the current AutoGen build expects
            rr_sig = inspect.signature(RoundRobinGroupChat)
            param_kwargs = {}
            if "termination_condition" in rr_sig.parameters:
                param_kwargs["termination_condition"] = termination
            elif "termination" in rr_sig.parameters:
                param_kwargs["termination"] = termination
            elif "termination_checker" in rr_sig.parameters:
                param_kwargs["termination_checker"] = termination

            team = RoundRobinGroupChat(
                participants=[
                    self.intent_agent,
                    self.action_agent,
                    self.validation_agent,
                ],
                **param_kwargs,
            )
            
            # Run the multi-agent conversation
            full_prompt = f"{context}\n\nPlease process this query through the full pipeline: intent extraction → action planning → validation → final formatting."
            
            result = await team.run(task=full_prompt)
            
            # Extract responses from each agent in order
            intent_response = None
            action_response = None
            validation_response = None
            
            for msg in result.messages:
                try:
                    if hasattr(msg, 'content') and msg.content and '{' in msg.content:
                        content = msg.content
                        # Get the source of the message
                        source = getattr(msg, 'source', '')
                        
                        if source == 'IntentExtractor' and '"intent":' in content:
                            intent_response = json.loads(content)
                            if intent_response.get('intent') == 'casual':
                                # Handle casual conversation immediately
                                return {
                                    "generation": None,
                                    "direct_response": "Hello! I'm an AI assistant that helps explain machine learning predictions for diabetes risk assessment. I can help you analyze patient data, understand model predictions, and explore feature importance. What would you like to know?",
                                    "method": "autogen_conversation",
                                    "confidence": 0.95
                                }
                        elif source == 'ActionPlanner' and '"action":' in content:
                            action_response = json.loads(content)
                            if action_response.get('action') is None:
                                # No action needed - this is conversational
                                return {
                                    "generation": None,
                                    "direct_response": "Hello! I'm an AI assistant that helps explain machine learning predictions for diabetes risk assessment. I can help you analyze patient data, understand model predictions, and explore feature importance. What would you like to know?",
                                    "method": "autogen_conversation",
                                    "confidence": 0.95
                                }
                        elif source == 'ActionValidator' and '"valid":' in content:
                            validation_response = json.loads(content)
                            
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            # If we have all three responses, build the final result manually
            if intent_response and action_response and validation_response:
                final_action = validation_response.get('corrected_action', action_response.get('action', 'explain'))
                
                # Apply additional programmatic validation
                original_action = final_action
                final_action = self._validate_and_fix_action(final_action)
                
                # Log if action was modified
                if original_action != final_action:
                    print(f"AutoGenDecoder: Fixed action '{original_action}' → '{final_action}'")
                
                return {
                    "generation": f"parsed: {final_action}[e]",
                    "confidence": min(
                        intent_response.get('confidence', 0.8),
                        action_response.get('confidence', 0.8), 
                        validation_response.get('confidence', 0.8)
                    ),
                    "method": "autogen_multi_agent",
                    "agent_reasoning": {
                        "intent": intent_response.get('reasoning', ''),
                        "action": action_response.get('reasoning', ''),
                        "validation": f"Valid: {validation_response.get('valid', False)}, Issues: {validation_response.get('issues', [])}"
                    },
                    "corrected_action": final_action,
                    "valid": validation_response.get('valid', False),
                    "issues": validation_response.get('issues', [])
                }
            
            # If we have at least intent and action responses, use those
            elif intent_response and action_response:
                final_action = action_response.get('action', 'explain')
                
                # Apply additional programmatic validation
                original_action = final_action
                final_action = self._validate_and_fix_action(final_action)
                
                # Log if action was modified
                if original_action != final_action:
                    print(f"AutoGenDecoder: Fixed action '{original_action}' → '{final_action}'")
                
                return {
                    "generation": f"parsed: {final_action}[e]",
                    "confidence": min(
                        intent_response.get('confidence', 0.8),
                        action_response.get('confidence', 0.8)
                    ),
                    "method": "autogen_multi_agent_partial",
                    "agent_reasoning": {
                        "intent": intent_response.get('reasoning', ''),
                        "action": action_response.get('reasoning', ''),
                        "validation": "Validation agent response not available"
                    },
                    "corrected_action": final_action,
                    "valid": True,  # Assume valid if no validation was performed
                    "issues": []
                }
            
            # Fallback: try to extract action from any available messages
            extracted_action = None
            for msg in result.messages:
                try:
                    if hasattr(msg, 'content') and msg.content:
                        content = msg.content
                        
                        # Look for action patterns in the response
                        action_keywords = ['filter', 'predict', 'explain', 'important', 'score', 'show', 'change', 'mistake', 'data']
                        
                        for keyword in action_keywords:
                            if keyword in content.lower():
                                # Try to extract the action line
                                lines = content.split('\n')
                                for line in lines:
                                    if keyword in line.lower() and not line.strip().startswith('#'):
                                        # Clean up the line
                                        action_line = line.strip().strip('"').strip("'")
                                        if action_line:
                                            extracted_action = action_line
                                            break
                                if extracted_action:
                                    break
                        if extracted_action:
                            break
                            
                except (AttributeError, TypeError):
                    continue
            
            if not extracted_action:
                extracted_action = "explain"  # Ultimate fallback
            
            return {
                "generation": f"parsed: {extracted_action}[e]",
                "confidence": 0.70,
                "method": "autogen_multi_agent_fallback",
                "raw_messages": [str(msg.content if hasattr(msg, 'content') else msg) for msg in result.messages[:3]]
            }
            
        except Exception as e:
            # Error handling - return safe fallback
            return {
                "generation": f"parsed: explain[e]",
                "error": str(e),
                "confidence": 0.1,
                "method": "autogen_error"
            }
    
    def complete_sync(self, user_query: str, conversation, grammar: str = None) -> Dict[str, Any]:
        """Synchronous wrapper for the async complete method."""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (common in Flask), we need to handle this differently
                import concurrent.futures
                import threading
                
                # Run in a separate thread to avoid "RuntimeError: cannot be called from a running event loop"
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.complete(user_query, conversation, grammar))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=30)  # 30 second timeout
            else:
                # Loop exists but not running, use it directly
                return loop.run_until_complete(self.complete(user_query, conversation, grammar))
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.complete(user_query, conversation, grammar))
            finally:
                loop.close()
        except Exception as e:
            # Fallback to error response
            return {
                "generation": f"parsed: explain[e]",
                "error": f"Async/sync conversion error: {str(e)}",
                "confidence": 0.1,
                "method": "autogen_sync_error"
            }
    
    async def close(self):
        """Clean up resources."""
        try:
            await self.model_client.close()
        except:
            pass


# Integration functions for compatibility
def create_autogen_decoder(**kwargs) -> AutoGenDecoder:
    """Factory function to create AutoGen decoder."""
    return AutoGenDecoder(**kwargs)


def get_autogen_predict_func(api_key: str = None, model: str = "gpt-4o"):
    """Get prediction function compatible with existing decoder interface."""
    decoder = AutoGenDecoder(api_key=api_key, model=model)
    
    def predict_func(prompt: str, grammar: str = None, conversation=None):
        # Extract actual user query from prompt if needed
        if "Query:" in prompt:
            user_query = prompt.split("Query:")[-1].strip()
        else:
            user_query = prompt
            
        return decoder.complete_sync(user_query, conversation, grammar)
    
    return predict_func 