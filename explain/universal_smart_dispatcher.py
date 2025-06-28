"""Universal Smart Action Dispatcher

This replaces the complex regex-based smart_action_dispatcher.py with a clean
universal approach that leverages AutoGen for command extraction and
the universal command parser for execution.

This eliminates micromanagement while maintaining full generalizability.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class UniversalSmartDispatcher:
    """
    Universal smart dispatcher using AutoGen + Universal Command Parser.
    
    This replaces 1000+ lines of regex patterns with a clean, universal approach
    that works with any feature combination through AutoGen intelligence.
    """
    
    def __init__(self, api_key: str):
        """Initialize universal smart dispatcher."""
        self.api_key = api_key
        
        # Initialize AutoGen decoder for command extraction
        try:
            from explain.autogen_decoder import AutoGenDecoder
            self.autogen_decoder = AutoGenDecoder(api_key=api_key)
            logger.info("✅ Universal Smart Dispatcher initialized with AutoGen")
        except ImportError as e:
            logger.error(f"AutoGen not available: {e}")
            raise ImportError("AutoGen is required for Universal Smart Dispatcher")
        
        # Initialize universal command parser
        from explain.universal_command_parser import create_universal_parser
        self.universal_parser = create_universal_parser()
        
    def dispatch(self, command: str, conversation, actions_map: Dict, 
                 available_features: List[str] = None) -> Tuple[str, int]:
        """
        Universal dispatch using AutoGen + Universal Command Parser.
        
        This method replaces complex regex patterns with intelligent AutoGen
        extraction followed by universal command parsing.
        
        Args:
            command: User query or command
            conversation: Conversation context
            actions_map: Available action functions
            available_features: Available dataset features
            
        Returns:
            Tuple of (response, status_code)
        """
        try:
            logger.info(f"Universal dispatch processing: {command}")
            
            # Step 1: Use AutoGen to extract structured command
            completion = self.autogen_decoder.complete_sync(command, conversation)
            
            if not completion:
                logger.warning("AutoGen failed to extract command structure")
                return self._fallback_execution(command, conversation, actions_map)
            
            # Step 2: Check for direct conversational response
            if completion.get("direct_response"):
                return completion["direct_response"], 1
            
            # Step 3: Extract action list using universal parser
            action_list = completion.get('action_list', [])
            
            if not action_list:
                # Try to parse from command structure
                command_structure = completion.get('command_structure')
                if command_structure:
                    action_list = self.universal_parser.parse_command(command_structure)
                    logger.info(f"Parsed command structure: {action_list}")
            
            if not action_list:
                # Fallback parsing from final_action
                final_action = completion.get('final_action', command)
                action_list = [final_action] if final_action else [command]
            
            # Step 4: Execute actions using universal approach
            return self._execute_universal_actions(action_list, conversation, actions_map)
            
        except Exception as e:
            logger.error(f"Error in universal dispatch: {e}")
            return self._fallback_execution(command, conversation, actions_map)
    
    def _execute_universal_actions(self, action_list: List[str], conversation, actions_map: Dict) -> Tuple[str, int]:
        """Execute actions using universal approach."""
        try:
            from explain.action import run_action
            
            responses = []
            
            for action in action_list:
                logger.info(f"Executing universal action: {action}")
                
                try:
                    # Execute action
                    response = run_action(conversation, None, action)
                    
                    if response and response.strip():
                        responses.append(response)
                        logger.info(f"Action '{action}' executed successfully")
                    else:
                        logger.warning(f"Action '{action}' returned empty response")
                        
                except Exception as action_error:
                    logger.error(f"Error executing action '{action}': {action_error}")
                    # Continue with other actions
                    continue
            
            # Return the last meaningful response
            if responses:
                final_response = responses[-1]
                logger.info(f"Universal execution completed: {len(responses)} actions executed")
                return final_response, 1
            else:
                logger.warning("No successful action executions")
                return "I processed your request but couldn't generate a response.", 0
                
        except Exception as e:
            logger.error(f"Error in universal action execution: {e}")
            return f"Error executing actions: {str(e)}", 0
    
    def _fallback_execution(self, command: str, conversation, actions_map: Dict) -> Tuple[str, int]:
        """Fallback execution for when universal approach fails."""
        try:
            from explain.action import run_action
            
            logger.info(f"Fallback execution for: {command}")
            
            # Try direct execution
            response = run_action(conversation, None, command)
            
            if response:
                return response, 1
            else:
                return "I couldn't understand your request. Please try rephrasing.", 0
                
        except Exception as e:
            logger.error(f"Error in fallback execution: {e}")
            return f"Error processing request: {str(e)}", 0
    
    def validate_command_template(self, command: str) -> Tuple[bool, str, Dict]:
        """
        Universal command validation.
        
        This replaces complex template validation with AutoGen intelligence.
        """
        try:
            # Use AutoGen to validate and understand the command
            completion = self.autogen_decoder.complete_sync(command, None)
            
            if completion and completion.get('confidence', 0) > 0.5:
                return True, "Valid command", completion
            else:
                return False, "Command not understood", {}
                
        except Exception as e:
            logger.error(f"Error in command validation: {e}")
            return False, f"Validation error: {str(e)}", {}
    
    def parse_compound_command(self, command: str, available_features: List[str] = None) -> List[str]:
        """
        Universal compound command parsing.
        
        This replaces complex regex patterns with AutoGen + Universal Parser.
        """
        try:
            # Use AutoGen to parse compound command
            completion = self.autogen_decoder.complete_sync(command, None)
            
            # Extract action list
            action_list = completion.get('action_list', [])
            
            if not action_list:
                command_structure = completion.get('command_structure')
                if command_structure:
                    action_list = self.universal_parser.parse_command(command_structure)
            
            if not action_list:
                # Fallback to single command
                final_action = completion.get('final_action', command)
                action_list = [final_action] if final_action else [command]
            
            logger.info(f"Parsed compound command into: {action_list}")
            return action_list
            
        except Exception as e:
            logger.error(f"Error parsing compound command: {e}")
            return [command]  # Fallback to original command


def get_universal_smart_dispatcher(api_key: str) -> UniversalSmartDispatcher:
    """Factory function to create universal smart dispatcher."""
    return UniversalSmartDispatcher(api_key)


# For backward compatibility, create an alias
def get_smart_dispatcher(api_key: str) -> UniversalSmartDispatcher:
    """Backward compatibility alias for the universal smart dispatcher."""
    return get_universal_smart_dispatcher(api_key) 