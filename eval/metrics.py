from typing import Dict, List, Optional, Callable
from pydantic import BaseModel


class NegotiationMetrics(BaseModel):
    negotiation_id: str
    outcome: str
    rounds_taken: int
    via_coalition: bool

    # ── Fairness & distribution ──────────────────────────────────────────────
    fairness_score: float           # Jain's fairness index [1/n, 1.0]
    efficiency_score: float         # 1.0 = agreed round 1, 0.0 = max rounds
    nash_bargaining_distance: float # Lower = closer to Nash optimum
    utility_gini: float             # Gini over per-agent utilities [0, 1]
    pareto_efficient: bool          # True if no agent can do better without harming another

    # ── Dynamics ─────────────────────────────────────────────────────────────
    concession_speed: float         # Average total concession per round per agent

    # ── Trust & honesty ───────────────────────────────────────────────────────
    trust_calibration_score: Optional[float] = None  # Correlation of inflation vs. actual lying
    lying_cost: Optional[float] = None               # Utility difference: honest - liar outcome
    avg_hint_error: Optional[float] = None           # Mean |stated hint - actual utility|


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
    """
    BUG FIX: was `1 - rounds_taken/max_rounds` which gave 0.9 for round 1.
    Now: 1.0 = agreed on round 1, 0.0 = used all rounds.
    """
    if max_rounds <= 1:
        return 1.0
    return round(max(0.0, 1.0 - (rounds_taken - 1) / (max_rounds - 1)), 4)


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
    utility_functions: Dict[str, Callable],
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


def pareto_efficiency_check(
    allocation: Dict[str, float],
    utility_functions: Dict[str, Callable],
    total_resource: float,
    steps: int = 50,
) -> bool:
    """
    Checks if the allocation is approximately Pareto-efficient.
    Tests whether any alternative allocation dominates (all agents weakly better,
    at least one strictly better). Approximate via grid search.
    Only implemented for 2-agent case.
    """
    agent_ids = list(utility_functions.keys())
    if len(agent_ids) != 2:
        return True  # Skip for n>2, assume efficient

    current_utils = {aid: utility_functions[aid](allocation) for aid in agent_ids}

    for i in range(steps + 1):
        share_0 = (i / steps) * total_resource
        share_1 = total_resource - share_0
        alt_alloc = {agent_ids[0]: share_0, agent_ids[1]: share_1}
        alt_utils = {aid: utility_functions[aid](alt_alloc) for aid in agent_ids}

        # Domination check: all >= current, at least one strictly >
        all_weakly_better = all(alt_utils[aid] >= current_utils[aid] for aid in agent_ids)
        one_strictly_better = any(alt_utils[aid] > current_utils[aid] + 1e-4 for aid in agent_ids)

        if all_weakly_better and one_strictly_better:
            return False  # A dominating allocation exists → not Pareto efficient

    return True


def concession_speed_metric(round_traces: List[Dict]) -> float:
    """
    Average concession per round per agent.
    Computed from round_traces if available.
    """
    if not round_traces:
        return 0.0

    concessions_per_agent: Dict[str, float] = {}
    rounds_per_agent: Dict[str, int] = {}

    for trace in round_traces:
        if trace.get("action") in ("COUNTER", "REJECT+COUNTER"):
            agent = trace["agent"]
            concessions_per_agent[agent] = trace.get("concessions_made", 0.0)
            rounds_per_agent[agent] = rounds_per_agent.get(agent, 0) + 1

    if not concessions_per_agent:
        return 0.0

    rates = []
    for agent_id, total_concession in concessions_per_agent.items():
        rounds = rounds_per_agent.get(agent_id, 1)
        rates.append(total_concession / max(rounds, 1))

    return round(sum(rates) / len(rates), 4)


def trust_calibration_score(round_traces: List[Dict]) -> Optional[float]:
    """
    Measures how well the trust score (hint_inflation_score) tracks actual behavior.
    Computes Pearson correlation between trust_score and (hint - utility) per round.
    Returns None if insufficient data.
    """
    pairs = []
    for trace in round_traces:
        trust = trace.get("trust_score")
        hint = trace.get("hint")
        utility = trace.get("utility")
        if trust is not None and hint is not None and utility is not None:
            inflation = hint - utility  # positive = agent overstated (lied)
            pairs.append((trust, inflation))

    if len(pairs) < 3:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5

    if std_x < 1e-9 or std_y < 1e-9:
        return None

    return round(cov / (std_x * std_y), 4)


def avg_hint_error_metric(round_traces: List[Dict]) -> Optional[float]:
    """
    Mean absolute error between stated hint and actual utility across all rounds.
    Captures average magnitude of deception (or understatement).
    """
    errors = []
    for trace in round_traces:
        hint = trace.get("hint")
        utility = trace.get("utility")
        if hint is not None and utility is not None:
            errors.append(abs(hint - utility))

    if not errors:
        return None
    return round(sum(errors) / len(errors), 4)


def compute_all_metrics(
    negotiation_id: str,
    allocation: Dict[str, float],
    utility_functions: Dict[str, Callable],
    reservation_values: Dict[str, float],
    rounds_taken: int,
    max_rounds: int,
    outcome: str,
    via_coalition: bool = False,
    total_resource: float = 1.0,
    round_traces: Optional[List[Dict]] = None,
    honest_agent_id: Optional[str] = None,
    liar_agent_id: Optional[str] = None,
) -> NegotiationMetrics:
    traces = round_traces or []
    utilities = {aid: fn(allocation) for aid, fn in utility_functions.items()}

    # Compute lying cost: utility difference between honest and lying agent
    lying_cost = None
    if honest_agent_id and liar_agent_id and honest_agent_id in utilities and liar_agent_id in utilities:
        lying_cost = round(utilities[honest_agent_id] - utilities[liar_agent_id], 4)

    return NegotiationMetrics(
        negotiation_id=negotiation_id,
        outcome=outcome,
        rounds_taken=rounds_taken,
        via_coalition=via_coalition,
        fairness_score=round(jains_fairness(allocation), 4),
        efficiency_score=efficiency_score(rounds_taken, max_rounds),
        nash_bargaining_distance=round(
            nash_bargaining_distance(allocation, utility_functions, reservation_values), 4
        ),
        utility_gini=round(gini_coefficient(list(utilities.values())), 4),
        pareto_efficient=pareto_efficiency_check(allocation, utility_functions, total_resource),
        concession_speed=concession_speed_metric(traces),
        trust_calibration_score=trust_calibration_score(traces),
        lying_cost=lying_cost,
        avg_hint_error=avg_hint_error_metric(traces),
    )
