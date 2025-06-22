"""Simple intent recognition decoder inspired by LLMCheckup.

This identifies user intent and routes ML queries to existing action system,
while handling conversational queries directly.
"""
import os
from typing import Dict, Any
import openai


class SimpleIntentDecoder:
    """Simple decoder that routes ML queries to actions, handles conversation directly."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model
    
    def complete(self, user_query: str, conversation) -> Dict[str, Any]:
        """Use LLM to intelligently determine intent and route appropriately."""
        
        # Build context about available actions
        context = self._build_context(conversation)
        
        system_prompt = f"""You are an intelligent intent classifier for a diabetes prediction model interface.

{context}

Available Actions (with required parameters):
- important all: Show all important features ranked by importance
- important [feature]: Show importance of specific feature (e.g., "important BMI")
- score default: Show model performance metrics (accuracy, precision, recall)
- mistake typical: Show cases where model made typical errors
- mistake sample: Show sample of model mistakes
- predict: Make prediction for patient data in query
- show: Display data examples from dataset
- explain features: Explain how model makes predictions using features
- explain lime: Show LIME explanations for predictions
- data: Show dataset overview and summary

User Query: "{user_query}"

Analyze the query and decide:
1. If casual conversation (greetings, thanks, general chat) → respond: "CONVERSATION"
2. If ML analysis needed → respond with full action + parameter (e.g., "important all", "score default", "mistakes typical")

Examples:
- "what features are important?" → "important all"
- "how important is BMI?" → "important BMI" 
- "how well does the model perform?" → "score default"
- "what mistakes does it make?" → "mistake typical"
- "make a prediction" → "predict"
- "explain how it works" → "explain features"

Only respond with "CONVERSATION" or a complete action+parameter. Be smart about understanding user intent."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=50
            )
            
            intent = response.choices[0].message.content.strip().upper()
            
            if intent == "CONVERSATION":
                return self._handle_conversation(user_query, conversation)
            else:
                # Route to action system
                action = intent.lower()
                return {
                    "generation": f"parsed: {action}[e]",
                    "method": "llm_intent_action",
                    "detected_intent": action
                }
                
        except Exception as e:
            # Fallback - handle as conversation
            return self._handle_conversation(user_query, conversation)
    

    
    def _handle_conversation(self, query: str, conversation) -> Dict[str, Any]:
        """Handle conversational queries directly."""
        context = self._build_context(conversation)
        
        system_prompt = f"""You are a helpful AI assistant for a diabetes prediction model interface.

{context}

The user is having a casual conversation with you. Respond naturally and helpfully. If they greet you, greet them back and offer to help with the diabetes model. If they ask for help, explain what kinds of questions you can answer about the model and data.

Keep responses friendly but focused on your role as an ML explanation assistant."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return {
                "direct_response": response.choices[0].message.content,
                "method": "simple_intent_conversation"
            }
            
        except Exception as e:
            return {
                "direct_response": "Hello! I'm here to help you understand the diabetes prediction model. What would you like to know?",
                "method": "simple_intent_fallback"
            }
    

    
    def _build_context(self, conversation) -> str:
        """Build simple context about the model."""
        try:
            dataset = conversation.get_var('dataset')
            if dataset and hasattr(dataset, 'contents'):
                data_info = dataset.contents
                size = len(data_info.get('X', []))
                return f"You're helping with a diabetes prediction model trained on {size} patients."
            return "You're helping with a diabetes prediction model."
        except:
            return "You're helping with a diabetes prediction model." 