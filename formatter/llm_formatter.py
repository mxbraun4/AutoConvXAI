"""Response formatting components for converting structured results to natural language."""
import logging
import openai

logger = logging.getLogger(__name__)

class LLMFormatter:
    """Intelligent LLM-based formatter that creates question-tailored responses"""
    
    def __init__(self, api_key, model='gpt-4o-mini'):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def format_response(self, user_query, action_name, action_result, autogen_intent, conversation=None):
        """Format action result into natural, question-tailored language using AutoGen-detected intent"""
        
        # Extract context about the conversation
        context_info = self._build_context(conversation)
        
        # Create intelligent formatting prompt based on action type and AutoGen intent
        format_prompt = self._create_intelligent_prompt(
            user_query, action_name, action_result, autogen_intent, context_info, conversation
        )
        
        try:
            # Use direct OpenAI API call for formatting
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data scientist assistant. Provide clear, concise responses based only on the provided data. Be conversational but brief (2-3 sentences max). Never speculate or add information not in the data."},
                    {"role": "user", "content": format_prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            formatted = response.choices[0].message.content.strip()
            
            # Log short responses for debugging but don't fallback
            if not formatted or len(formatted.strip()) < 10:
                logger.warning(f"LLM formatter returned short response: {formatted}")
                
            return formatted
            
        except Exception as e:
            logger.error(f"Error in LLM formatting: {e}")
            # Re-raise the error instead of using fallback
            raise
    
    def _build_context(self, conversation):
        """Extract relevant context from conversation"""
        if not conversation:
            return {}
            
        context = {}
        
        # Dataset context
        try:
            dataset = conversation.get_var('dataset')
            if dataset:
                context['dataset_size'] = len(dataset.contents.get('X', []))
                context['features'] = list(dataset.contents.get('X', {}).columns) if hasattr(dataset.contents.get('X', {}), 'columns') else []
        except:
            pass
            
        # NEW: Enhanced filtering context using filter state
        try:
            if hasattr(conversation, 'get_filter_context_for_response'):
                filter_context = conversation.get_filter_context_for_response()
                context['filter_context'] = filter_context
                
                if filter_context['type'] == 'query_filter':
                    context['communication_style'] = 'query_applied_filter'
                    context['filter_description'] = filter_context['description']
                    context['original_size'] = filter_context['original_size']
                    context['current_size'] = filter_context['filtered_size']
                elif filter_context['type'] == 'inherited_filter':
                    context['communication_style'] = 'inherited_filter'
                    context['filter_description'] = filter_context['description']
                    context['original_size'] = filter_context['original_size']
                    context['current_size'] = filter_context['filtered_size']
                else:
                    context['communication_style'] = 'no_filter'
                    context['original_size'] = filter_context['original_size']
                    context['current_size'] = filter_context['filtered_size']
            else:
                # Fallback to old logic
                if hasattr(conversation, 'temp_dataset') and conversation.temp_dataset:
                    temp_size = len(conversation.temp_dataset.contents.get('X', []))
                    full_size = context.get('dataset_size', temp_size)
                    if temp_size < full_size:
                        context['filtered'] = True
                        context['filtered_size'] = temp_size
                        context['filter_description'] = getattr(conversation, 'parse_operation', [])
                    else:
                        context['filtered'] = False
                        context['using_full_dataset'] = True
                else:
                    context['filtered'] = False
                    context['using_full_dataset'] = True
        except:
            pass
            
        # Model context
        try:
            if hasattr(conversation, 'class_names'):
                context['class_names'] = conversation.class_names
        except:
            pass
            
        return context
    
    def _create_intelligent_prompt(self, user_query, action_name, action_result, autogen_intent, context_info, conversation=None):
        """Create an intelligent prompt tailored to the specific action and AutoGen-detected intent"""
        
        # Build context string with clear filtering explanation
        context_str = ""
        
        # SKIP dataset context for conceptual queries and specific instance scenarios
        if autogen_intent in ['predict', 'whatif', 'define', 'model', 'about']:
            context_str = "CONCEPTUAL/SCENARIO FOCUS: Ignore dataset context and filtering information. Focus only on the specific concept, instance, or hypothetical scenario without mentioning dataset size or filters."
        else:
            # NEW: Use enhanced filter context for clearer communication
            communication_style = context_info.get('communication_style', 'unknown')
            
            if communication_style == 'query_applied_filter':
                # This query applied a new filter - communicate as "X out of Y total"
                original_size = context_info['original_size']
                current_size = context_info['current_size']
                filter_desc = context_info['filter_description']
                context_str += f"QUERY FILTER APPLIED: User asked about {filter_desc}. Found {current_size} instances out of {original_size} total that match this criteria. "
                context_str += f"COMMUNICATION STYLE: Say 'Of the {original_size} total instances, {current_size} have {filter_desc}' NOT 'In the filtered dataset of {current_size} instances'. "
                
            elif communication_style == 'inherited_filter':
                # Data was already filtered - communicate about the subset
                original_size = context_info['original_size']
                current_size = context_info['current_size']
                filter_desc = context_info['filter_description']
                context_str += f"INHERITED FILTER: Dataset was already filtered to {current_size} instances (from {original_size} total) where {filter_desc}. "
                context_str += f"COMMUNICATION STYLE: Say 'In the filtered subset of {current_size} instances' or 'Among the {current_size} instances where {filter_desc}'. "
                
            elif communication_style == 'no_filter':
                # No filtering applied
                total_size = context_info['current_size']
                context_str += f"NO FILTER: Using complete dataset ({total_size} instances). "
                context_str += f"COMMUNICATION STYLE: Say 'Of the {total_size} total instances' or 'In the complete dataset'. "
            
            else:
                # Fallback to old logic
                if context_info.get('filtered'):
                    filter_size = context_info['filtered_size']
                    total_size = context_info.get('dataset_size', filter_size)
                    
                    # Determine what kind of filtering happened by looking at the user query and results
                    if filter_size == total_size:
                        # Full dataset was used after reset
                        context_str += f"Dataset: Complete dataset ({total_size} instances). "
                    else:
                        # Data was filtered
                        context_str += f"Dataset: Filtered to {filter_size} instances from {total_size} total. "
                        
                        # Add percentage clarification if needed
                        if hasattr(action_result, '__getitem__') and isinstance(action_result, (dict, tuple)):
                            result_data = action_result[0] if isinstance(action_result, tuple) else action_result
                            if isinstance(result_data, dict):
                                request_type = result_data.get('request_type')
                                if request_type == 'class_distribution':
                                    context_str += f"Percentages refer to the {filter_size} filtered instances. "
                elif context_info.get('using_full_dataset'):
                    total_size = context_info['dataset_size']
                    context_str += f"Dataset: Complete dataset ({total_size} instances). "
                
        # Only show class context for dataset-level queries, not single patient queries or filter operations
        if context_info.get('class_names') and autogen_intent not in ['show', 'explain', 'predict', 'filter']:
            class_list = ', '.join(context_info['class_names'].values())
            context_str += f"Possible classes: {class_list}. "
        
        # Add recent conversation context for comparative questions
        if conversation and hasattr(conversation, 'conversation_turns') and len(conversation.conversation_turns) > 0:
            recent_turn = conversation.conversation_turns[-1]
            if recent_turn.get('action_name') == 'score':
                context_str += f"Previous result: {recent_turn.get('response', '')} "
        
        # Handle followup intent specially - use conversational approach
        if autogen_intent == 'followup':
            return self._create_followup_prompt(user_query, action_result, context_info, conversation)
        
        # Create tailored prompt based on action and intent
        base_prompt = f"""
You are an expert data scientist explaining machine learning results to users. 

USER QUESTION: "{user_query}"
ACTION PERFORMED: {action_name}
AUTOGEN INTENT: {autogen_intent}
CONTEXT: {context_str}

RAW RESULTS:
{action_result}

INTERPRETING STRUCTURED DATA:
- If result has type="feature_importance" and top_k field: User asked for top N features, mention the exact number
- If result has type="single_prediction": Focus on the prediction class and confidence
- If result has type="data_summary": Describe the dataset characteristics
- If result is a dictionary with structured fields: Extract the key information, don't show raw dict format
- If result is a tuple: Use the first element which contains the structured data

TASK: Transform these raw results into a brief, natural response (2-3 sentences max) that:

1. DIRECTLY ANSWERS the user's specific question with factual data only
2. Uses plain language (avoid technical jargon)
3. States only what is shown in the raw results
4. Never speculates, assumes, or adds context not in the data
5. Be conversational but concise
6. CRITICAL: If data is filtered, be clear about what subset is being analyzed
7. CRITICAL: For percentages, always specify they are percentages of the filtered subset, not the original dataset

"""
        
        # Add intent-specific instructions based on AutoGen intent
        if autogen_intent == 'count' or autogen_intent == 'data':
            base_prompt += """
COUNT/DATA INTENT: The user wants to know HOW MANY. Be direct and clear:
- ALWAYS use simple language: "There are X instances with [condition]" or "Of the Y total instances, X have [condition]"
- NEVER say "In the filtered dataset" - this confuses users
- Just state the facts: numbers and what they represent
- Skip technical implementation details about filtering
- Be concise - one clear sentence with the count
"""
        elif autogen_intent == 'casual':
            base_prompt += """
CASUAL INTENT: The user is making conversational comments or observations. Respond naturally:
- Acknowledge their observation appropriately
- Provide relevant context if helpful
- Keep responses brief and conversational
- Don't introduce new statistics unless directly relevant to their comment
"""
        elif autogen_intent == 'explain':
            base_prompt += """
EXPLAIN INTENT: The user wants to understand WHY something happened. Focus on:
- Clear causal explanations
- The most influential factors
- Use phrases like "because", "due to", "the main reason"
- Make it feel like you're walking them through the reasoning
"""
        elif autogen_intent == 'predict':
            base_prompt += """
PREDICT INTENT: The user wants to know WHAT WOULD HAPPEN. Focus on:
- Clear prediction outcomes for the SPECIFIC instance provided
- Confidence levels if available
- What factors drive the prediction
- Use phrases like "would result in", "likely to", "expected outcome"
- NEVER mention dataset size or total instances - focus only on the specific prediction
- DO NOT reference the broader dataset context
"""
        elif autogen_intent == 'important':
            base_prompt += """
IMPORTANCE INTENT: The user wants to know WHAT MATTERS MOST. Focus on:
- Ranking and prioritization of features
- Relative importance and impact on outcomes
- If it's a "top N" request, clearly state the number requested
- Use phrases like "most critical", "key factors", "strongest influence"
- For top-k requests: "The top [N] most important features are: [list]"
- Always explain WHY these features are important
"""
        elif autogen_intent == 'performance':
            base_prompt += """
PERFORMANCE INTENT: The user wants to know HOW WELL the model works. Focus on:
- Accuracy and reliability
- Strengths and limitations
- Practical implications
- Use phrases like "performs well", "accurate in", "reliable for"
"""
        elif autogen_intent == 'whatif':
            base_prompt += """
WHAT-IF INTENT: The user wants to explore hypothetical scenarios. Focus on:
- CLEAR BEFORE/AFTER COMPARISONS showing the original prediction vs new prediction
- Whether the prediction CHANGED or STAYED THE SAME
- Use specific language like "the prediction would CHANGE from X to Y" or "would FLIP from X to Y"
- If no change: "the prediction would REMAIN the same (X)"
- Impact of changes on predictions and confidence levels
- Practical implications of modifications
- NEVER mention dataset size or existing instances with those values
- DO NOT reference broader dataset statistics - focus only on the hypothetical change
- Use phrases like "if we changed X to Y, the prediction would change from A to B"
"""
        elif autogen_intent == 'counterfactual':
            base_prompt += """
COUNTERFACTUAL INTENT: The user wants alternative scenarios to change the outcome. Focus on:
- What specific changes would flip the prediction
- Clear feature modifications needed
- Practical actionable changes
- Alternative pathways to different outcomes
- Use phrases like "to change the outcome", "alternatively", "if instead", "would need to modify"
"""
        elif autogen_intent == 'mistakes':
            base_prompt += """
MISTAKES INTENT: The user wants to understand model errors. Focus on:
- Specific cases where model was wrong
- Patterns in mistakes
- Why errors occurred
- Use phrases like "incorrectly predicted", "failed to recognize", "got wrong because"
"""
        elif autogen_intent == 'confidence':
            base_prompt += """
CONFIDENCE INTENT: The user wants certainty information. Focus on:
- Probability scores and what they mean
- Model certainty levels
- Reliability of predictions
- Use phrases like "confident that", "probability of", "certain about"
"""
        elif autogen_intent == 'interactions':
            base_prompt += """
INTERACTIONS INTENT: The user wants to understand feature relationships. Focus on:
- How features work together
- Combined effects
- Synergistic relationships
- Use phrases like "work together", "combined effect", "interact to"
"""
        elif autogen_intent == 'statistics':
            base_prompt += """
STATISTICS INTENT: The user wants numerical summaries. Focus on:
- Clear statistical descriptions
- Meaningful comparisons
- Distribution characteristics
- Use phrases like "on average", "typically", "ranges from"
"""
        elif autogen_intent == 'define':
            base_prompt += """
DEFINE INTENT: The user wants explanations of terms. Focus on:
- Clear, simple definitions
- Practical context
- Why it matters for health/diabetes
- Use phrases like "refers to", "means", "is important because"
"""
        elif autogen_intent == 'about':
            base_prompt += """
ABOUT INTENT: The user wants system information. Focus on:
- Capabilities and features
- What you can help with
- Encouraging exploration
- Use phrases like "I can help you", "my capabilities include", "you can ask me"
"""
        
        base_prompt += """
RESPONSE (2-3 sentences max, factual only, no speculation):"""
        
        return base_prompt
    
    def _create_followup_prompt(self, user_query, action_result, context_info, conversation):
        """Create a specialized prompt for follow-up questions using conversation context."""
        
        # Extract context about recent results for analytical follow-ups
        context_str = ""
        
        # Get model vs ground truth context if available
        try:
            if conversation:
                dataset = conversation.get_var('dataset')
                model = conversation.get_var('model') 
                
                if dataset and model and hasattr(dataset, 'contents'):
                    y_true = dataset.contents.get('y', [])
                    X = dataset.contents.get('X', [])
                    
                    if y_true and X:
                        ground_truth_positive = sum(y_true)
                        total_instances = len(y_true)
                        
                        predictions = model.predict(X)
                        predicted_positive = sum(predictions)
                        
                        gt_percentage = (ground_truth_positive / total_instances) * 100
                        pred_percentage = (predicted_positive / total_instances) * 100
                        
                        context_str = f"""
RECENT CONVERSATION CONTEXT:
- Ground truth: {ground_truth_positive} positive cases ({gt_percentage:.1f}%)
- Model predictions: {predicted_positive} positive cases ({pred_percentage:.1f}%)
- Total instances: {total_instances}
"""
                        
                        # Add prediction bias context
                        if predicted_positive < ground_truth_positive:
                            context_str += "- Model underpredicts (conservative)\n"
                        elif predicted_positive > ground_truth_positive:
                            context_str += "- Model overpredicts (aggressive)\n"
                        else:
                            context_str += "- Model predictions match ground truth\n"
        except Exception as e:
            logger.warning(f"Could not extract context for followup: {e}")
        
        followup_prompt = f"""
You are a data scientist having a conversation with a user. They just asked a follow-up question.

USER FOLLOW-UP QUESTION: "{user_query}"
{context_str}

FOLLOW-UP RESPONSE FROM SYSTEM:
{action_result}

TASK: The system already provided a good analytical response. Simply return it as-is, or if it's a tuple, extract the meaningful text response. Be conversational and direct.

GUIDELINES:
- If the response is already a clear answer, return it directly
- If it's a tuple, extract the text portion
- Keep it conversational (2-3 sentences max)
- Use the context numbers if they help explain the answer
- Don't add unnecessary explanations or summaries
"""
        
        return followup_prompt