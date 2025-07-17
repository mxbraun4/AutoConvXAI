
EVALUATION SETUP INSTRUCTIONS
=============================

1. INSTALL DEPENDENCIES:
   pip install autogen-agentchat[openai]

2. SET API KEY:
   export OPENAI_API_KEY=your_openai_api_key_here

3. RUN EVALUATION:
   python parsing_accuracy/autogen_evaluator.py

4. REVIEW RESULTS:
   - Check parsing_accuracy/evaluation_results.json
   - Compare intent accuracy with old system (was ~76%)
   - Analyze which types of queries work better/worse

5. EXPECTED METRICS TO TRACK:
   - Intent accuracy (% of intents correctly identified)
   - Entity accuracy (% of entities correctly extracted)
   - Overall accuracy (% of complete matches)
   - Performance by intent type (filter, explain, important, etc.)

6. COMPARISON WITH OLD SYSTEM:
   - Old system: ~76% accuracy
   - Goal: Match or exceed this with better entity extraction
   - Focus areas: Complex filtering, counterfactual queries

7. ANALYSIS AREAS:
   - Which question types are hardest to parse?
   - Are there patterns in misclassifications?
   - How does performance vary by intent type?
   - What improvements could be made to prompts?

The conversion has prepared 191 test cases covering:
- explain: 55 cases (filtered explanations)
- filter: 87 cases (data filtering)
- likelihood: 29 cases (prediction confidence)
- important: 9 cases (feature importance)
- counterfactual: 11 cases (what-if analysis)
