# State Tracking Fix for TalkToModel

## Problem Description

The TalkToModel system had a critical state tracking issue where filtered context from one query would persist into subsequent queries. For example:

1. User asks: "what is your prediction for instance 2"
   - System filters dataset to instance 2
2. User asks: "what is the overall performance of the model"
   - System incorrectly continues to reference instance 2 instead of the full dataset

## Root Cause

The issue occurred because:
- The `conversation.temp_dataset` persists between queries
- Once filtered (e.g., to a specific instance), it remains filtered
- Subsequent queries operate on the filtered data instead of the full dataset
- There was no mechanism to detect when users switch context

## Solution Overview

The fix implements a **multi-layered context awareness system** within the AutoGen architecture:

### 1. Enhanced Context Building
- Added context tracking to `_build_contextual_prompt()` in `autogen_decoder.py`
- Provides agents with information about current filtering state
- Includes previous query context to help detect context switches

### 2. Intent Agent Enhancement
- Updated the Intent Extraction agent prompt to detect context switches
- Added `context_reset` flag to entity extraction
- Identifies keywords like "overall", "global", "entire" as context switch indicators

### 3. Action Planning Updates
- Modified Action Planning agent to respect context reset flags
- When `context_reset: true`, generates actions for full dataset
- Avoids prepending filters from previous context

### 4. Smart Dispatcher Heuristics
- Added context switch detection in `smart_action_dispatcher.py`
- Detects when queries ask about general performance after specific instances
- Automatically rebuilds temp dataset when context switch detected

### 5. Enhanced Logic Integration
- Updated `enhanced_logic.py` to check for context reset flags
- Rebuilds temp dataset before processing when context switch detected

## Technical Details

### Code Changes

1. **autogen_decoder.py**:
   - Enhanced `_build_contextual_prompt()` to include filtering state
   - Updated intent extraction prompt with context awareness
   - Modified action planning prompt to handle context resets
   - Added `intent_response` to output for context reset detection

2. **smart_action_dispatcher.py**:
   - Added context switch detection logic in `dispatch()`
   - Checks for keywords and dataset filtering state
   - Automatically rebuilds temp dataset when needed

3. **enhanced_logic.py**:
   - Added context reset check in `update_state()`
   - Rebuilds temp dataset when agents detect context switch

### Context Switch Detection

The system detects context switches through:

1. **Keyword Analysis**: 
   - "overall", "total", "global", "entire", "whole", "complete"
   - "general", "all data", "full dataset", "model performance"

2. **State Analysis**:
   - Checks if temp dataset is filtered to single instance
   - Detects performance queries without instance references
   - Identifies topic changes from specific to general

3. **Multi-Agent Consensus**:
   - Intent agent identifies context switch need
   - Action planner respects context reset flag
   - Smart dispatcher provides additional validation

## Testing

Run the test script to verify the fix:

```bash
python test_state_tracking_fix.py
```

This demonstrates:
- Instance-specific query followed by general query
- Filtered predictions followed by overall importance
- Proper context switching behavior

## Benefits

1. **Improved User Experience**: Users can naturally switch between specific and general queries
2. **Accurate Results**: Performance metrics and analyses correctly reflect the intended scope
3. **Generalizable Solution**: Works across all action types and query patterns
4. **Maintains Context When Appropriate**: Still preserves filtering when users intend to continue with filtered data

## Example Interactions

### Before Fix:
```
User: what is your prediction for instance 2
Bot: [Shows prediction for instance 2]

User: what is the overall performance of the model
Bot: [WRONG: Shows performance for instance 2 only]
```

### After Fix:
```
User: what is your prediction for instance 2
Bot: [Shows prediction for instance 2]

User: what is the overall performance of the model  
Bot: [CORRECT: Shows performance for entire dataset (768 instances)]
```

## Future Enhancements

1. **Explicit Context Management**: Add commands like "clear filters" or "reset context"
2. **Context Confirmation**: Ask users when context switch intent is ambiguous
3. **Multi-Turn Context**: Track context across longer conversation threads
4. **Context Visualization**: Show current filtering state in UI

## Implementation Philosophy

This fix follows the principle of **implicit context management** - the system intelligently infers when users want to switch context without requiring explicit commands. This creates a more natural conversational experience while maintaining the flexibility of the AutoGen multi-agent architecture. 