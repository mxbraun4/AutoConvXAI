"""Universal Command Parser for AutoGen Response Processing

This module provides a unified parser for converting AutoGen agent responses
into actionable command lists that can be executed by the action dispatcher.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class UniversalCommandParser:
    """
    Universal parser for AutoGen agent responses.
    
    This parser extracts structured commands from AutoGen agent responses
    and converts them into actionable command lists for execution.
    """
    
    def __init__(self):
        """Initialize the universal parser."""
        self.supported_actions = [
            'data', 'predict', 'explain', 'important', 'score', 
            'filter', 'show', 'change', 'mistake'
        ]
    
    def parse_autogen_response(self, action_response: Dict[str, Any]) -> List[str]:
        """
        Parse AutoGen action response into command list.
        
        Args:
            action_response: Response from AutoGen action planning agent
            
        Returns:
            List of command strings for execution
        """
        try:
            # Extract action from response
            action = action_response.get('action', '')
            if not action:
                logger.warning("No action found in AutoGen response")
                return ['explain']  # Safe default
            
            # Split action into command components
            command_parts = action.strip().split()
            if not command_parts:
                logger.warning("Empty action after splitting")
                return ['explain']
            
            # Validate primary action
            primary_action = command_parts[0].lower()
            if primary_action not in self.supported_actions:
                logger.warning(f"Unsupported action: {primary_action}")
                return ['explain']
            
            # Return command list
            return command_parts
            
        except Exception as e:
            logger.error(f"Error parsing AutoGen response: {e}")
            return ['explain']  # Safe fallback
    
    def extract_command_structure(self, action_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured command information from AutoGen response.
        
        Args:
            action_response: Response from AutoGen action planning agent
            
        Returns:
            Structured command information
        """
        try:
            command_parts = self.parse_autogen_response(action_response)
            
            return {
                'primary_action': command_parts[0] if command_parts else 'explain',
                'modifiers': command_parts[1:] if len(command_parts) > 1 else [],
                'entities': action_response.get('entities', {}),
                'full_command': ' '.join(command_parts)
            }
            
        except Exception as e:
            logger.error(f"Error extracting command structure: {e}")
            return {
                'primary_action': 'explain',
                'modifiers': [],
                'entities': {},
                'full_command': 'explain'
            }


def create_universal_parser() -> UniversalCommandParser:
    """
    Factory function to create a universal parser instance.
    
    Returns:
        Configured UniversalCommandParser instance
    """
    return UniversalCommandParser()