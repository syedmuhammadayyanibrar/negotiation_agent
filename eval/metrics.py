from typing import Dict, List, Optional
from pydantic import BaseModel


class NegotiationMetrics(BaseModel):
    negotiation_id: str
    fairness_score: float
    efficiency_score: float
    nash_bargaining_distance: float
    utility_gini: float
    outcome: str
    rounds_taken: int
    via_coalition: bool


def jains_fairness(allocation: Dict[str, float]) -> float:
    """
    Jain's fairness index. Range [1/n, 1.0].
    1.0 = perfectly equal, 1/n = maximally unequal.
    """
    values = list(allocation.values())
    n = len(values)
    if n == 0:
        return 0.0
    numerator = sum(values) ** 2
    denominator = n * sum(v ** 2 for v in values)
    return numerator / denominator if denominator > 0 else 0.0


def efficiency_score(rounds_taken: int, max_rounds: int) -> float:
    """1.0 = agreed on round 1, 0.0 = used all rounds."""
    return 1.0 - (rounds_taken / max_rounds)


def gini_coefficient(utilities: List[float]) -> float:
    """
    Gini coefficient over agent utilities.
    0.0 = perfect equality, 1.0 = maximum inequality.
    """
    n = len(utilities)
    if n == 0:
        return 0.0
    utilities_sorted = sorted(utilities)
    cumulative = sum(
        (2 * (i + 1) - n - 1) * v
        for i, v in enumerate(utilities_sorted)
    )
    total = sum(utilities_sorted)
    return cumulative / (n * total) if total > 0 else 0.0


def nash_bargaining_distance(
    allocation: Dict[str, float],
    utility_functions: Dict[str, callable],
    reservation_values: Dict[str, float],
) -> float:
    """
    Distance from Nash bargaining solution.
    Nash solution maximizes product of (utility - reservation_value).
    Lower = closer to optimal.
    """
    # Compute actual product
    actual_product = 1.0
    for agent_id, util_fn in utility_functions.items():
        u = util_fn(allocation)
        r = reservation_values.get(agent_id, 0.0)
        actual_product *= max(0.0, u - r)

    # Approximate Nash solution via equal split (simplified baseline)
    n = len(utility_functions)
    equal_alloc = {aid: 1.0 / n for aid in utility_functions}
    nash_product = 1.0
    for agent_id, util_fn in utility_functions.items():
        u = util_fn(equal_alloc)
        r = reservation_values.get(agent_id, 0.0)
        nash_product *= max(0.0, u - r)

    if nash_product == 0:
        return 1.0
    return abs(nash_product - actual_product) / nash_product


def compute_all_metrics(
    negotiation_id: str,
    allocation: Dict[str, float],
    utility_functions: Dict[str, callable],
    reservation_values: Dict[str, float],
    rounds_taken: int,
    max_rounds: int,
    outcome: str,
    via_coalition: bool = False,
) -> NegotiationMetrics:
    utilities = {aid: fn(allocation) for aid, fn in utility_functions.items()}
    return NegotiationMetrics(
        negotiation_id=negotiation_id,
        fairness_score=round(jains_fairness(allocation), 4),
        efficiency_score=round(efficiency_score(rounds_taken, max_rounds), 4),
        nash_bargaining_distance=round(
            nash_bargaining_distance(allocation, utility_functions, reservation_values), 4
        ),
        utility_gini=round(gini_coefficient(list(utilities.values())), 4),
        outcome=outcome,
        rounds_taken=rounds_taken,
        via_coalition=via_coalition,
    )
