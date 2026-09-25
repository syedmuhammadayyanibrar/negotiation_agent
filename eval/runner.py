"""
Standalone evaluation runner for the negotiation agent system.

Runs 4 canonical research scenarios directly against the Orchestrator
(no FastAPI server needed). Saves JSON reports to research_output/.

Usage:
    python -m eval.runner
    python -m eval.runner --scenarios honest_honest inflate_inflate
    python -m eval.runner --max-rounds 15 --agents agent_a agent_b agent_c
"""
import asyncio
import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is on path when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protocol.channel import Channel
from graph.orchestrator import Orchestrator
from core.agent import Agent
from eval.harness import EvalHarness
from eval.metrics import compute_all_metrics


# ── Canonical enterprise B2B & game-theoretic research scenarios ─────────────

SCENARIOS = {
    # ── Enterprise B2B Commercial & Procurement Scenarios ─────────────────────
    "b2b_procurement_buyer_vendor": {
        "description": "Enterprise B2B Procurement: Enterprise Buyer vs SaaS Vendor negotiating commercial contract & SLA terms",
        "hint_strategies": ["adaptive", "inflate"],
        "rq": "B2B-1",
        "resource_type": "contract_commercial_terms",
    },
    "enterprise_sla_settlement": {
        "description": "Multi-Vendor SLA Dispute: Cloud Provider vs Enterprise Client negotiating SLA credit & liability terms",
        "hint_strategies": ["honest", "adaptive"],
        "rq": "B2B-2",
        "resource_type": "sla_liability_settlement",
    },
    # ── Canonical Game-Theoretic Baselines ───────────────────────────────────
    "honest_honest": {
        "description": "Both agents honest (Trust calibration baseline)",
        "hint_strategies": ["honest", "honest"],
        "rq": "RQ1",
        "resource_type": "commercial_deal_split",
    },
    "inflate_inflate": {
        "description": "Both agents inflate claims (Deception dynamics under competitive bidding)",
        "hint_strategies": ["inflate", "inflate"],
        "rq": "RQ3",
        "resource_type": "commercial_deal_split",
    },
    "honest_vs_inflate": {
        "description": "Honest buyer vs. inflating vendor (Adversarial concession asymmetry)",
        "hint_strategies": ["honest", "inflate"],
        "rq": "RQ3",
        "resource_type": "commercial_deal_split",
    },
    "adaptive_vs_honest": {
        "description": "Adaptive buyer vs. honest vendor (Mature behavioral trust profile)",
        "hint_strategies": ["adaptive", "honest"],
        "rq": "RQ2",
        "resource_type": "commercial_deal_split",
    },
}


def make_agents(
    agent_ids: List[str],
    hint_strategies: List[str],
    negotiation_id: str,
    reservation_values: Dict[str, float],
    target_values: Dict[str, float],
    max_rounds: int,
) -> List[Agent]:
    return [
        Agent(
            agent_id=aid,
            negotiation_id=negotiation_id,
            role=aid,
            utility_function=lambda x, a=aid: x.get(a, 0.0),
            reservation_value=reservation_values.get(aid, 0.3),
            target_value=target_values.get(aid, 0.6),
            max_rounds=max_rounds,
            hint_strategy=hint_strategies[i % len(hint_strategies)],
        )
        for i, aid in enumerate(agent_ids)
    ]


async def run_scenario(
    scenario_name: str,
    scenario_cfg: Dict[str, Any],
    agent_ids: List[str],
    reservation_values: Dict[str, float],
    target_values: Dict[str, float],
    max_rounds: int,
    resource_type: str = "commercial_contract_terms",
    total_resource: float = 1.0,
) -> Dict[str, Any]:
    active_resource = scenario_cfg.get("resource_type", resource_type)
    negotiation_id = f"eval_{scenario_name}_{datetime.now().strftime('%H%M%S')}"
    channel = Channel()
    orchestrator = Orchestrator(
        agent_ids=agent_ids,
        negotiation_id=negotiation_id,
        resource_type=active_resource,
        total_resource=total_resource,
        channel=channel,
    )
    orchestrator.setup()

    hint_strategies = scenario_cfg["hint_strategies"]
    agents = make_agents(
        agent_ids=agent_ids,
        hint_strategies=hint_strategies,
        negotiation_id=negotiation_id,
        reservation_values=reservation_values,
        target_values=target_values,
        max_rounds=max_rounds,
    )

    print(f"\n[RUNNER] ━━ Scenario: {scenario_name} ━━")
    print(f"[RUNNER]    {scenario_cfg['description']}")
    print(f"[RUNNER]    Agents: {', '.join(f'{aid}({s})' for aid, s in zip(agent_ids, hint_strategies))}")

    try:
        result = await orchestrator.run_negotiation(agents=agents, max_rounds=max_rounds)
    except Exception as e:
        print(f"[RUNNER] ❌ Exception in scenario {scenario_name}: {e}")
        result = {"outcome": "breakdown", "reason": str(e), "round_traces": []}

    # Build hint_strategies map for harness
    strategies_map = {aid: hint_strategies[i % len(hint_strategies)] for i, aid in enumerate(agent_ids)}
    utility_fns = {a.agent_id: a.utility_function for a in agents}
    res_vals = {a.agent_id: a.reservation_value for a in agents}

    harness = EvalHarness(
        agent_ids=agent_ids,
        utility_functions=utility_fns,
        reservation_values=res_vals,
        total_resource=total_resource,
        max_rounds=max_rounds,
        hint_strategies=strategies_map,
    )

    report = harness.compare(result, negotiation_id)

    # Annotate with scenario metadata
    report["scenario"] = {
        "name": scenario_name,
        "description": scenario_cfg["description"],
        "rq": scenario_cfg.get("rq"),
        "hint_strategies": strategies_map,
        "negotiation_id": negotiation_id,
    }

    harness.print_report(report)
    return report


def _print_summary_table(all_reports: Dict[str, Dict[str, Any]]) -> None:
    """Print a cross-scenario summary comparison table."""
    print("\n" + "═" * 80)
    print("  CROSS-SCENARIO SUMMARY")
    print("═" * 80)

    header = f"  {'Scenario':<25} {'Outcome':<12} {'Rounds':>7} {'Fairness':>9} {'Efficiency':>11} {'Pareto':>8} {'TrustCal':>10}"
    print(header)
    print("  " + "─" * 78)

    for name, report in all_reports.items():
        s = report.get("system")
        if s:
            pareto_icon = "✓" if s.get("pareto_efficient") else "✗"
            tc = s.get("trust_calibration_score")
            tc_str = f"{tc:.3f}" if tc is not None else "  N/A"
            print(
                f"  {name:<25} {s['outcome']:<12} {s['rounds_taken']:>7} "
                f"{s['fairness_score']:>9.4f} {s['efficiency_score']:>11.4f} "
                f"{pareto_icon:>8} {tc_str:>10}"
            )
        else:
            print(f"  {name:<25} {'BREAKDOWN':<12}")

    print("═" * 80 + "\n")


async def main(args: argparse.Namespace) -> None:
    agent_ids = args.agents
    max_rounds = args.max_rounds
    total_resource = args.total_resource

    # Default reservation / target values
    n = len(agent_ids)
    reservation_values = {aid: 0.3 for aid in agent_ids}
    target_values = {aid: round(0.55 + 0.05 * (i % 2), 4) for i, aid in enumerate(agent_ids)}

    # Filter scenarios
    selected = args.scenarios if args.scenarios else list(SCENARIOS.keys())
    selected_scenarios = {k: SCENARIOS[k] for k in selected if k in SCENARIOS}

    if not selected_scenarios:
        print(f"[RUNNER] ❌ No valid scenarios in: {args.scenarios}")
        print(f"[RUNNER]    Available: {list(SCENARIOS.keys())}")
        return

    print("\n" + "═" * 80)
    print("  NEGOTIATION AGENT — EVALUATION RUNNER")
    print(f"  Agents: {agent_ids}  |  Max rounds: {max_rounds}  |  Resource: {total_resource}")
    print("═" * 80)

    all_reports: Dict[str, Dict[str, Any]] = {}

    for scenario_name, scenario_cfg in selected_scenarios.items():
        report = await run_scenario(
            scenario_name=scenario_name,
            scenario_cfg=scenario_cfg,
            agent_ids=agent_ids,
            reservation_values=reservation_values,
            target_values=target_values,
            max_rounds=max_rounds,
            total_resource=total_resource,
        )
        all_reports[scenario_name] = report

    _print_summary_table(all_reports)

    # Save to research_output/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent.parent / "research_output"
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / f"eval_{timestamp}.json"

    with open(out_file, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"[RUNNER] ✅ Report saved to {out_file}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone negotiation agent eval runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  honest_honest      Both agents honest (RQ1 baseline)
  inflate_inflate    Both inflate hints (RQ3)
  honest_vs_inflate  Honest vs inflater (RQ3 lying cost)
  adaptive_vs_honest Adaptive vs honest (RQ2)

Examples:
  python -m eval.runner
  python -m eval.runner --scenarios honest_honest honest_vs_inflate
  python -m eval.runner --agents alpha beta gamma --max-rounds 15
        """,
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS.keys()),
        default=None,
        help="Scenarios to run (default: all 4)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["agent_a", "agent_b"],
        help="Agent IDs (default: agent_a agent_b)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Max negotiation rounds per scenario (default: 10)",
    )
    parser.add_argument(
        "--total-resource",
        type=float,
        default=1.0,
        help="Total divisible resource (default: 1.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Load .env if dotenv available
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"[RUNNER] Loaded .env from {env_path}")
    except ImportError:
        pass

    if not os.environ.get("GROQ_API_KEY"):
        print("[RUNNER] ⚠️  GROQ_API_KEY not set — LLM policy calls will fail.")

    args = parse_args()
    asyncio.run(main(args))
