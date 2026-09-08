"""
Three baselines + tit-for-tat to compare against the main system:
  1. Random priority      — agents take turns in random order, first offer accepted if >= reservation
  2. Fixed hierarchy      — agents ranked by priority, higher rank wins resource allocation
  3. Central orchestrator — omniscient planner that knows all utility functions, maximizes Nash product
  4. Tit-for-tat          — mirrors counterpart's last concession rate
"""
import random
from typing import Dict, List, Callable, Tuple


def random_priority_baseline(
    agent_ids: List[str],
    utility_functions: Dict[str, Callable],
    reservation_values: Dict[str, float],
    total_resource: float = 1.0,
    seed: int = 42,
    max_rounds: int = 20,  # FIX: was hardcoded 20, now configurable
) -> Tuple[Dict[str, float], int]:
    """
    Randomly shuffle agents. First agent demands target, rest accept if >= reservation.
    Returns (allocation, rounds_taken).
    """
    rng = random.Random(seed)
    order = agent_ids[:]
    rng.shuffle(order)

    for round_num in range(1, max_rounds + 1):
        # Proposer takes equal split ± random noise
        base = total_resource / len(agent_ids)
        alloc = {}
        remaining = total_resource
        for i, aid in enumerate(order):
            if i == len(order) - 1:
                alloc[aid] = round(remaining, 6)
            else:
                share = base + rng.uniform(-0.05, 0.05)
                share = max(0.01, min(remaining - 0.01 * (len(order) - i - 1), share))
                alloc[aid] = round(share, 6)
                remaining -= share

        # Check if all agents accept
        if all(
            utility_functions[aid](alloc) >= reservation_values.get(aid, 0.0)
            for aid in agent_ids
        ):
            return alloc, round_num

    # Fallback: equal split
    equal = {aid: round(total_resource / len(agent_ids), 6) for aid in agent_ids}
    return equal, max_rounds


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
        allocation[aid] = round(max(reservation_values.get(aid, 0.1), take), 6)
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
                best_alloc = {k: round(v, 6) for k, v in alloc.items()}
    else:
        # For n > 2: use equal split approximation
        best_alloc = {aid: round(total_resource / n, 6) for aid in agent_ids}

    return best_alloc, 1


def tit_for_tat_baseline(
    agent_ids: List[str],
    utility_functions: Dict[str, Callable],
    reservation_values: Dict[str, float],
    total_resource: float = 1.0,
    initial_ask_fraction: float = 0.6,
    max_rounds: int = 20,
) -> Tuple[Dict[str, float], int]:
    """
    Tit-for-tat: each agent mirrors the counterpart's last concession rate.
    Agents start with initial_ask_fraction of the resource, then concede
    at the same rate their counterpart conceded in the previous round.

    This baseline tests whether cooperative reciprocity leads to faster convergence
    vs random priority or fixed hierarchy.

    Returns (allocation, rounds_taken).
    """
    if len(agent_ids) != 2:
        # TFT only defined for bilateral negotiation — fallback to equal split
        equal = {aid: round(total_resource / len(agent_ids), 6) for aid in agent_ids}
        return equal, 1

    a0, a1 = agent_ids[0], agent_ids[1]
    ask = {
        a0: initial_ask_fraction * total_resource,
        a1: initial_ask_fraction * total_resource,
    }
    prev_ask = dict(ask)

    for round_num in range(1, max_rounds + 1):
        # Each agent mirrors the opponent's concession from last round
        if round_num > 1:
            concession_by_a0 = max(0.0, prev_ask[a0] - ask[a0])
            concession_by_a1 = max(0.0, prev_ask[a1] - ask[a1])
        else:
            # Round 1: no history, use fixed starting concession of 0.05
            concession_by_a0 = 0.05 * total_resource
            concession_by_a1 = 0.05 * total_resource

        prev_ask = dict(ask)

        # Each agent reduces their ask by the amount their counterpart conceded
        ask[a0] = max(reservation_values.get(a0, 0.0), ask[a0] - concession_by_a1)
        ask[a1] = max(reservation_values.get(a1, 0.0), ask[a1] - concession_by_a0)

        # Build allocation: each agent gets what they currently ask, constrained by total
        # Split remainder proportionally if asks sum to more than total
        total_ask = ask[a0] + ask[a1]
        if total_ask <= total_resource + 1e-6:
            alloc = {a0: round(ask[a0], 6), a1: round(total_resource - ask[a0], 6)}
        else:
            # Scale down proportionally
            scale = total_resource / total_ask
            alloc = {aid: round(ask[aid] * scale, 6) for aid in agent_ids}

        # Accept if both agents get >= their reservation value
        if all(
            utility_functions[aid](alloc) >= reservation_values.get(aid, 0.0)
            for aid in agent_ids
        ):
            return alloc, round_num

    # Deadlock fallback: equal split
    equal = {aid: round(total_resource / len(agent_ids), 6) for aid in agent_ids}
    return equal, max_rounds
