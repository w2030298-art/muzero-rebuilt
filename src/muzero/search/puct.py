"""PUCT (Predictor + UCT) scoring formula for MCTS child selection.

Implements the exact formula from the MuZero paper:
    pb_c = log((parent_N + c_base + 1) / c_base) + c_init
    pb_c *= sqrt(parent_N) / (child_N + 1)
    score = Q + pb_c * P
"""

from __future__ import annotations

import math


def puct_score(
    parent_visit_count: int,
    child_visit_count: int,
    child_prior: float,
    child_value: float,
    pb_c_base: float,
    pb_c_init: float,
) -> float:
    """Compute the PUCT score for a child node.

    Args:
        parent_visit_count: Total visits to the parent node.
        child_visit_count: Visits to this child node.
        child_prior: Prior probability from the policy network.
        child_value: Mean action-value Q(s, a) for this child.
        pb_c_base: Base constant for exploration coefficient.
        pb_c_init: Initial constant for exploration coefficient.

    Returns:
        PUCT score: higher = more promising for selection.
    """
    pb_c = math.log((parent_visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent_visit_count) / (child_visit_count + 1)
    return child_value + pb_c * child_prior
