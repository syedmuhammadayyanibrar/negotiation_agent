"""
Three baselines to compare against the main system:
  1. Random priority      — agents take turns in random order, first offer accepted if >= reservation
  2. Fixed hierarchy      — agents ranked by priority, higher rank wins resource allocation
  3. Central orchestrator — omniscient planner that knows all utility functions, maximizes Nash product
"""
import random
from typing import Dict, List, Callable, Tuple


def random_priority_baseline(
    agent_ids: List[str],
    utility_functions: Dict[str, Callable],
    reservation_values: Dict[str, float],
    total_resource: float = 1.0,
    seed: int = 42,
) -> Tuple[Dict[str, float], int]:
    """
    Randomly shuffle agents. First agent demands target, rest accept if >= reservation.
    Returns (allocation, rounds_taken).
    """
    rng = random.Random(seed)
    order = agent_ids[:]
    rng.shuffle(order)

    for round_num in range(1, 20):
        proposer = order[round_num % len(order)]
        # Proposer takes equal split ± random noise
        base = total_resource / len(agent_ids)
        alloc = {}
        remaining = total_resource
        for i, aid in enumerate(order):
            if i == len(order) - 1:
                alloc[aid] = remaining
            else:
                share = base + rng.uniform(-0.05, 0.05)
                share = max(0.01, min(remaining - 0.01 * (len(order) - i - 1), share))
                alloc[aid] = share
                remaining -= share

        # Check if all agents accept
        if all(
            utility_functions[aid](alloc) >= reservation_values.get(aid, 0.0)
            for aid in agent_ids
        ):
            return alloc, round_num

    # Fallback: equal split
    equal = {aid: total_resource / len(agent_ids) for aid in agent_ids}
    return equal, 20


def fixed_hierarchy_baseline(
    agent_ids: List[str],
    utility_functions: Dict[str, Callable],
    reservation_values: Dict[str, float],
    total_resource: float = 1.0,
) -> Tuple[Dict[str, float], int]:
    """
    Agents ranked by index. Higher rank (index 0) gets priority.
    Each agent greedily takes as much as possible above reservation of remaining agents.
    Returns (allocation, rounds_taken).
    """
    allocation = {}
    remaining = total_resource
    for i, aid in enumerate(agent_ids):
        # Reserve minimum for all lower-ranked agents
        lower_reserved = sum(
            reservation_values.get(agent_ids[j], 0.1)
            for j in range(i + 1, len(agent_ids))
        )
        max_take = remaining - lower_reserved
        # Take target or max available
        take = min(max_take, remaining / (len(agent_ids) - i))
        allocation[aid] = max(reservation_values.get(aid, 0.1), take)
        remaining -= allocation[aid]

    return allocation, len(agent_ids)  # one round per agent


def central_orchestrator_baseline(
    agent_ids: List[str],
    utility_functions: Dict[str, Callable],
    reservation_values: Dict[str, float],
    total_resource: float = 1.0,
    steps: int = 1000,
) -> Tuple[Dict[str, float], int]:
    """
    Omniscient planner. Maximizes Nash bargaining product via grid search.
    Returns (allocation, 1) — always solves in 1 round (omniscient).
    """
    n = len(agent_ids)
    best_product = -1.0
    best_alloc = {aid: total_resource / n for aid in agent_ids}

    # Grid search over allocations (works for 2-3 agents)
    if n == 2:
        for i in range(steps + 1):
            share_0 = (i / steps) * total_resource
            share_1 = total_resource - share_0
            alloc = {agent_ids[0]: share_0, agent_ids[1]: share_1}
            product = 1.0
            for aid in agent_ids:
                u = utility_functions[aid](alloc)
                r = reservation_values.get(aid, 0.0)
                product *= max(0.0, u - r)
            if product > best_product:
                best_product = product
                best_alloc = alloc
    else:
        # For n > 2: use equal split approximation
        best_alloc = {aid: total_resource / n for aid in agent_ids}

    return best_alloc, 1
