"""Smart Action Dispatcher for AutoGen-generated actions.

This dispatcher intelligently parses compound actions and breaks them down
into executable steps, making the system more flexible and generalizable.
"""
import re
from typing import List, Tuple, Dict, Any


class SmartActionDispatcher:
    """Intelligent action dispatcher that handles compound and natural actions."""
    
    def __init__(self):
        # Define action patterns and their handlers
        self.action_patterns = {
            'filter': self._parse_filter_action,
            'predict': self._parse_predict_action,
            'explain': self._parse_explain_action,
            'important': self._parse_important_action,
            'score': self._parse_score_action,
            'show': self._parse_show_action,
            'change': self._parse_change_action,
            'mistake': self._parse_mistake_action,
            'data': self._parse_data_action,
        }
        
    def parse_and_execute(self, action_text: str, conversation, action_module) -> Tuple[str, int]:
        """Parse natural action text and execute the appropriate actions.
        
        Args:
            action_text: Natural action text from AutoGen
            conversation: Conversation object
            action_module: The action module containing action functions
            
        Returns:
            Tuple of (response, status)
        """
        if not action_text or not isinstance(action_text, str):
            return self._execute_simple_action(['explain'], conversation, action_module)
        
        # Clean and tokenize the action
        tokens = self._tokenize_action(action_text)
        
        if not tokens:
            return self._execute_simple_action(['explain'], conversation, action_module)
        
        # Detect the primary action type
        primary_action = self._detect_primary_action(tokens)
        
        if primary_action not in self.action_patterns:
            return self._execute_simple_action(['explain'], conversation, action_module)
        
        # Parse the action using the appropriate parser
        try:
            action_steps = self.action_patterns[primary_action](tokens)
            
            # Execute the action steps
            return self._execute_action_sequence(action_steps, conversation, action_module)
            
        except Exception as e:
            print(f"SmartActionDispatcher: Error parsing action '{action_text}': {e}")
            return self._execute_simple_action(['explain'], conversation, action_module)
    
    def _tokenize_action(self, action_text: str) -> List[str]:
        """Clean and tokenize action text."""
        # Remove common prefixes
        action_text = re.sub(r'^parsed:\s*', '', action_text.strip())
        action_text = re.sub(r'\[e\]$', '', action_text.strip())
        
        # Split into tokens and clean
        tokens = action_text.lower().split()
        return [token.strip() for token in tokens if token.strip()]
    
    def _detect_primary_action(self, tokens: List[str]) -> str:
        """Detect the primary action type from tokens."""
        for token in tokens:
            if token in self.action_patterns:
                return token
        return 'explain'  # Default fallback
    
    def _parse_filter_action(self, tokens: List[str]) -> List[List[str]]:
        """Parse filter actions, including compound ones and multi-condition filters."""
        action_steps = []
        
        if 'filter' not in tokens:
            return action_steps
        
        # Find filter start
        filter_idx = tokens.index('filter')
        i = filter_idx
        
        # Extract all filter conditions
        while i < len(tokens) - 3:
            if i == filter_idx or (i > filter_idx and tokens[i] in ['filter'] + list(self.action_patterns.keys())):
                if tokens[i] != 'filter' and i > filter_idx:
                    break  # We've hit another action
                    
                if tokens[i] == 'filter':
                    i += 1  # Skip the 'filter' keyword
                
                # Extract filter components
                if i + 2 < len(tokens):
                    feature = tokens[i]
                    operator = tokens[i + 1]
                    
                    # Handle two-word operators
                    if (i + 3 < len(tokens) and 
                        operator in ['greater', 'less'] and 
                        tokens[i + 2] == 'equal'):
                        operator = f"{operator}equal"
                        if i + 3 < len(tokens):
                            value = tokens[i + 3]
                            i += 4
                        else:
                            break
                    else:
                        if i + 2 < len(tokens):
                            value = tokens[i + 2]
                            i += 3
                        else:
                            break
                    
                    # Create the filter action
                    filter_action = ['filter', feature, operator, value]
                    action_steps.append(filter_action)
                else:
                    break
            else:
                i += 1
        
        # Check for follow-up actions after all filters
        if i < len(tokens):
            remaining_tokens = tokens[i:]
            if 'predict' in remaining_tokens:
                action_steps.append(['predict'])
            elif 'explain' in remaining_tokens:
                # Look for ID after explain
                explain_idx = remaining_tokens.index('explain')
                if len(remaining_tokens) > explain_idx + 1:
                    try:
                        patient_id = int(remaining_tokens[explain_idx + 1])
                        action_steps.append(['explain', str(patient_id)])
                    except ValueError:
                        action_steps.append(['explain'])
                else:
                    action_steps.append(['explain'])
        
        return action_steps
    
    def _parse_predict_action(self, tokens: List[str]) -> List[List[str]]:
        """Parse predict actions."""
        if 'predict' not in tokens:
            return []
        
        predict_idx = tokens.index('predict')
        
        # Check if there's an ID specified
        if len(tokens) > predict_idx + 1:
            try:
                patient_id = int(tokens[predict_idx + 1])
                return [['predict', str(patient_id)]]
            except ValueError:
                pass
        
        return [['predict']]
    
    def _parse_explain_action(self, tokens: List[str]) -> List[List[str]]:
        """Parse explain actions."""
        if 'explain' not in tokens:
            return []
        
        explain_idx = tokens.index('explain')
        
        # Check if there's an ID specified
        if len(tokens) > explain_idx + 1:
            try:
                patient_id = int(tokens[explain_idx + 1])
                return [['explain', str(patient_id)]]
            except ValueError:
                pass
        
        # Check for explanation type (lime, shap)
        if 'lime' in tokens:
            if len(tokens) > explain_idx + 1:
                try:
                    patient_id = int(tokens[explain_idx + 1])
                    return [['explain', str(patient_id), 'lime']]
                except ValueError:
                    pass
            return [['explain', 'lime']]
        elif 'shap' in tokens:
            if len(tokens) > explain_idx + 1:
                try:
                    patient_id = int(tokens[explain_idx + 1])
                    return [['explain', str(patient_id), 'shap']]
                except ValueError:
                    pass
            return [['explain', 'shap']]
        
        return [['explain']]
    
    def _parse_important_action(self, tokens: List[str]) -> List[List[str]]:
        """Parse importance actions."""
        if 'important' not in tokens:
            return []
        
        important_idx = tokens.index('important')
        
        # Check for specific patterns
        if len(tokens) > important_idx + 1:
            next_token = tokens[important_idx + 1]
            
            if next_token == 'all':
                return [['important', 'all']]
            elif next_token == 'topk' and len(tokens) > important_idx + 2:
                try:
                    k = int(tokens[important_idx + 2])
                    return [['important', 'topk', str(k)]]
                except ValueError:
                    return [['important', 'all']]
            else:
                # Assume it's a feature name
                return [['important', next_token]]
        
        # Default to showing all importance
        return [['important', 'all']]
    
    def _parse_score_action(self, tokens: List[str]) -> List[List[str]]:
        """Parse score/performance actions."""
        if 'score' not in tokens:
            return []
        
        score_idx = tokens.index('score')
        
        # Check for specific metric
        if len(tokens) > score_idx + 1:
            metric = tokens[score_idx + 1]
            valid_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc', 'default']
            if metric in valid_metrics:
                return [['score', metric]]
        
        return [['score', 'default']]
    
    def _parse_show_action(self, tokens: List[str]) -> List[List[str]]:
        """Parse show data actions."""
        if 'show' not in tokens:
            return []
        
        show_idx = tokens.index('show')
        
        # Check if there's an ID specified
        if len(tokens) > show_idx + 1:
            try:
                patient_id = int(tokens[show_idx + 1])
                return [['show', str(patient_id)]]
            except ValueError:
                pass
        
        return [['show']]
    
    def _parse_change_action(self, tokens: List[str]) -> List[List[str]]:
        """Parse what-if change actions."""
        if 'change' not in tokens:
            return []
        
        change_idx = tokens.index('change')
        
        # Expect: change feature operation value
        if len(tokens) >= change_idx + 4:
            feature = tokens[change_idx + 1]
            operation = tokens[change_idx + 2]
            value = tokens[change_idx + 3]
            
            valid_operations = ['set', 'increase', 'decrease']
            if operation in valid_operations:
                return [['change', feature, operation, value]]
        
        return []
    
    def _parse_mistake_action(self, tokens: List[str]) -> List[List[str]]:
        """Parse mistake analysis actions."""
        if 'mistake' in tokens:
            return [['mistake']]
        return []
    
    def _parse_data_action(self, tokens: List[str]) -> List[List[str]]:
        """Parse data summary actions."""
        if 'data' in tokens:
            return [['data']]
        return []
    
    def _execute_action_sequence(self, action_steps: List[List[str]], conversation, action_module) -> Tuple[str, int]:
        """Execute a sequence of actions."""
        if not action_steps:
            return self._execute_simple_action(['explain'], conversation, action_module)
        
        # Handle sequences properly
        response = ""
        status = 1
        
        for action in action_steps:
            action_response, action_status = self._execute_simple_action(action, conversation, action_module)
            
            # Accumulate responses
            if action_response:
                if response:
                    response += "<br><br>"
                response += action_response
            
            # Stop on error
            if action_status == 0:
                status = 0
                break
        
        return response, status
    
    def _execute_simple_action(self, action_tokens: List[str], conversation, action_module) -> Tuple[str, int]:
        """Execute a simple action using the existing action system."""
        if not action_tokens:
            action_tokens = ['explain']
        
        action_name = action_tokens[0]
        
        # Get the action function from the action module
        if hasattr(action_module, 'actions') and action_name in action_module.actions:
            action_func = action_module.actions[action_name]
            
            # Call the action function with the conversation and tokens
            try:
                # Find the index where this action starts (usually 0 for simple dispatch)
                action_index = 0
                return action_func(conversation, action_tokens, action_index)
            except Exception as e:
                print(f"SmartActionDispatcher: Error executing action '{action_name}': {e}")
                # Fallback to explain
                if 'explain' in action_module.actions:
                    return action_module.actions['explain'](conversation, ['explain'], 0)
                return "I encountered an error processing your request.", 0
        else:
            print(f"SmartActionDispatcher: Unknown action '{action_name}'")
            # Fallback to explain if available
            if hasattr(action_module, 'actions') and 'explain' in action_module.actions:
                return action_module.actions['explain'](conversation, ['explain'], 0)
            return "I don't understand that request.", 0


# Global instance for easy access
smart_dispatcher = SmartActionDispatcher() 