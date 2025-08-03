"""Contains a function returning a dict mapping the key word for actions to the function.

This function is used to generate a dictionary of all the actions and the corresponding function.
This functionality is used later on to determine the set of allowable operations and what functions
to run when parsing the grammar.
"""
from explainability.actions.define import define_operation
from explainability.actions.explanation import explain_operation
from explainability.actions.feature_stats import feature_stats
from explainability.actions.important import important_operation
from explainability.actions.filter import filter_operation
from explainability.actions.followup import followup_operation
from explainability.actions.interaction_effects import measure_interaction_effects
from explainability.actions.labels import show_labels_operation
from explainability.actions.mistakes import show_mistakes_operation
from explainability.actions.self import self_operation
from explainability.actions.predict import predict_operation
from explainability.actions.score import score_operation
from explainability.actions.what_if import what_if_operation
from explainability.actions.counterfactual import counterfactual_operation


def get_all_action_functions_map():
    """Gets a dictionary mapping all the names of the actions in the parse tree to their functions."""
    actions = {
        'interact': measure_interaction_effects,
        'filter': filter_operation,
        'explain': explain_operation,
        'predict': predict_operation,
        'followup': followup_operation,
        'important': important_operation,
        'self': self_operation,
        'score': score_operation,
        'label': show_labels_operation,
        'mistake': show_mistakes_operation,
        'statistic': feature_stats,
        'define': define_operation,
        'counterfactual': counterfactual_operation,
        'whatif': what_if_operation
    }
    return actions
