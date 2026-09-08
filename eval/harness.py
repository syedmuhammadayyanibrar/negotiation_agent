"""
Evaluation harness: runs the main system and all 4 baselines,
computes all metrics, and produces a comparison report.
"""
import json
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional
from eval.metrics import compute_all_metrics, NegotiationMetrics
from eval.baselines import (
    random_priority_baseline,
    fixed_hierarchy_baseline,
    central_orchestrator_baseline,
    tit_for_tat_baseline,
)


class EvalHarness:
    def __init__(
        self,
        agent_ids: List[str],
        utility_functions: Dict[str, Callable],
        reservation_values: Dict[str, float],
        total_resource: float = 1.0,
        max_rounds: int = 10,
        hint_strategies: Optional[Dict[str, str]] = None,
    ):
        self.agent_ids = agent_ids
        self.utility_functions = utility_functions
        self.reservation_values = reservation_values
        self.total_resource = total_resource
        self.max_rounds = max_rounds
        self.hint_strategies = hint_strategies or {aid: "honest" for aid in agent_ids}

    def _honest_agent(self) -> Optional[str]:
        """Return the ID of the first honest agent (for lying_cost computation)."""
        for aid, strat in self.hint_strategies.items():
            if strat == "honest":
                return aid
        return None

    def _liar_agent(self) -> Optional[str]:
        """Return the ID of the first inflating/deflating agent."""
        for aid, strat in self.hint_strategies.items():
            if strat in ("inflate", "deflate", "adaptive"):
                return aid
        return None

    def run_baselines(self) -> Dict[str, NegotiationMetrics]:
        results = {}

        # Baseline 1: Random priority
        alloc, rounds = random_priority_baseline(
            self.agent_ids, self.utility_functions,
            self.reservation_values, self.total_resource,
            max_rounds=self.max_rounds,
        )
        results["random_priority"] = compute_all_metrics(
            negotiation_id="baseline_random",
            allocation=alloc,
            utility_functions=self.utility_functions,
            reservation_values=self.reservation_values,
            rounds_taken=rounds,
            max_rounds=self.max_rounds,
            outcome="agreement",
            total_resource=self.total_resource,
        )

        # Baseline 2: Fixed hierarchy
        alloc, rounds = fixed_hierarchy_baseline(
            self.agent_ids, self.utility_functions,
            self.reservation_values, self.total_resource,
        )
        results["fixed_hierarchy"] = compute_all_metrics(
            negotiation_id="baseline_hierarchy",
            allocation=alloc,
            utility_functions=self.utility_functions,
            reservation_values=self.reservation_values,
            rounds_taken=rounds,
            max_rounds=self.max_rounds,
            outcome="agreement",
            total_resource=self.total_resource,
        )

        # Baseline 3: Central orchestrator
        alloc, rounds = central_orchestrator_baseline(
            self.agent_ids, self.utility_functions,
            self.reservation_values, self.total_resource,
        )
        results["central_orchestrator"] = compute_all_metrics(
            negotiation_id="baseline_central",
            allocation=alloc,
            utility_functions=self.utility_functions,
            reservation_values=self.reservation_values,
            rounds_taken=rounds,
            max_rounds=self.max_rounds,
            outcome="agreement",
            total_resource=self.total_resource,
        )

        # Baseline 4: Tit-for-tat
        alloc, rounds = tit_for_tat_baseline(
            self.agent_ids, self.utility_functions,
            self.reservation_values, self.total_resource,
            max_rounds=self.max_rounds,
        )
        results["tit_for_tat"] = compute_all_metrics(
            negotiation_id="baseline_tft",
            allocation=alloc,
            utility_functions=self.utility_functions,
            reservation_values=self.reservation_values,
            rounds_taken=rounds,
            max_rounds=self.max_rounds,
            outcome="agreement",
            total_resource=self.total_resource,
        )

        return results

    def compare(
        self,
        system_result: Dict[str, Any],
        system_negotiation_id: str,
    ) -> Dict[str, Any]:
        """
        Compare system result against all baselines.
        Returns a structured comparison report.
        """
        round_traces = system_result.get("round_traces", [])

        if system_result.get("outcome") == "breakdown":
            system_metrics = None
        else:
            system_metrics = compute_all_metrics(
                negotiation_id=system_negotiation_id,
                allocation=system_result.get("allocation", {}),
                utility_functions=self.utility_functions,
                reservation_values=self.reservation_values,
                rounds_taken=system_result.get("rounds_taken", self.max_rounds),
                max_rounds=self.max_rounds,
                outcome=system_result.get("outcome", "breakdown"),
                via_coalition=system_result.get("via_coalition", False),
                total_resource=self.total_resource,
                round_traces=round_traces,
                honest_agent_id=self._honest_agent(),
                liar_agent_id=self._liar_agent(),
            )

        baseline_metrics = self.run_baselines()

        report = {
            "system": system_metrics.model_dump() if system_metrics else None,
            "baselines": {k: v.model_dump() for k, v in baseline_metrics.items()},
            "comparison": {},
            "round_traces": round_traces,
        }

        if system_metrics:
            for baseline_name, bm in baseline_metrics.items():
                report["comparison"][baseline_name] = {
                    "fairness_delta": round(system_metrics.fairness_score - bm.fairness_score, 4),
                    "efficiency_delta": round(system_metrics.efficiency_score - bm.efficiency_score, 4),
                    "gini_delta": round(system_metrics.utility_gini - bm.utility_gini, 4),
                    "nash_delta": round(
                        system_metrics.nash_bargaining_distance - bm.nash_bargaining_distance, 4
                    ),
                }

        return report

    def run_scenario_suite(
        self,
        scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run all 4 canonical research scenarios via baselines and aggregate.
        (For LLM agent results, use the runner.py CLI which invokes the orchestrator.)

        Scenarios include different hint_strategy pairings for research questions:
          - honest×honest  (RQ1 trust calibration baseline)
          - inflate×inflate (RQ3 does lying pay when both lie?)
          - honest vs inflate (RQ3 lying cost)
          - adaptive vs honest (RQ2 does mature trust improve outcomes?)

        Returns dict: scenario_name -> NegotiationMetrics
        """
        if scenarios is None:
            scenarios = [
                {"name": "honest_honest",     "strategies": {aid: "honest"   for aid in self.agent_ids}},
                {"name": "inflate_inflate",   "strategies": {aid: "inflate"  for aid in self.agent_ids}},
                {"name": "honest_vs_inflate", "strategies": {
                    self.agent_ids[0]: "honest",
                    **{aid: "inflate" for aid in self.agent_ids[1:]},
                }},
                {"name": "adaptive_vs_honest", "strategies": {
                    self.agent_ids[0]: "adaptive",
                    **{aid: "honest" for aid in self.agent_ids[1:]},
                }},
            ]

        suite_results: Dict[str, Any] = {}
        for scenario in scenarios:
            harness = EvalHarness(
                agent_ids=self.agent_ids,
                utility_functions=self.utility_functions,
                reservation_values=self.reservation_values,
                total_resource=self.total_resource,
                max_rounds=self.max_rounds,
                hint_strategies=scenario["strategies"],
            )
            baselines = harness.run_baselines()
            suite_results[scenario["name"]] = {
                b: m.model_dump() for b, m in baselines.items()
            }

        return suite_results

    def export_report(self, report: Dict[str, Any], path: str) -> None:
        """Save the full evaluation report to a JSON file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[EVAL] Report saved to {out_path}")

    def print_report(self, report: Dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print("  NEGOTIATION EVALUATION REPORT")
        print("=" * 70)

        if report["system"]:
            s = report["system"]
            print(f"\n{'SYSTEM RESULT':─<70}")
            print(f"  Outcome:            {s['outcome']}")
            print(f"  Rounds taken:       {s['rounds_taken']}")
            print(f"  Via coalition:      {s['via_coalition']}")
            print(f"  Fairness (Jain's):  {s['fairness_score']:.4f}")
            print(f"  Efficiency:         {s['efficiency_score']:.4f}")
            print(f"  Gini (utilities):   {s['utility_gini']:.4f}")
            print(f"  Nash distance:      {s['nash_bargaining_distance']:.4f}")
            print(f"  Pareto-efficient:   {s['pareto_efficient']}")
            print(f"  Concession speed:   {s['concession_speed']:.4f}")
            if s.get("trust_calibration_score") is not None:
                print(f"  Trust calibration:  {s['trust_calibration_score']:.4f}  (Pearson ρ, higher=better)")
            if s.get("lying_cost") is not None:
                sign = "+" if s["lying_cost"] >= 0 else ""
                print(f"  Lying cost:         {sign}{s['lying_cost']:.4f}  (honest_utility - liar_utility)")
            if s.get("avg_hint_error") is not None:
                print(f"  Avg hint error:     {s['avg_hint_error']:.4f}  (mean |hint - utility|)")
        else:
            print("\n  SYSTEM RESULT: ❌ BREAKDOWN (no agreement reached)")

        print(f"\n{'BASELINE COMPARISON':─<70}")
        header = f"  {'Baseline':<25} {'Fairness':>8} {'Efficiency':>10} {'Gini':>7} {'Nash':>8} {'Pareto':>8}"
        print(header)
        print("  " + "-" * 68)

        for name, bm in report.get("baselines", {}).items():
            pareto_icon = "✓" if bm.get("pareto_efficient") else "✗"
            print(
                f"  {name:<25} {bm['fairness_score']:>8.4f} {bm['efficiency_score']:>10.4f} "
                f"{bm['utility_gini']:>7.4f} {bm['nash_bargaining_distance']:>8.4f} {pareto_icon:>8}"
            )

        if report.get("system") and report.get("comparison"):
            print(f"\n{'DELTAS (system − baseline; + = system better)':─<70}")
            for name, delta in report["comparison"].items():
                signs = {k: ("+" if v >= 0 else "") for k, v in delta.items()}
                # For Gini and Nash, lower is better, so flip sign interpretation
                better_metrics = []
                worse_metrics = []
                for k, v in delta.items():
                    if k in ("gini_delta", "nash_delta"):
                        if v < 0:
                            better_metrics.append(k)
                        elif v > 0:
                            worse_metrics.append(k)
                    else:
                        if v > 0:
                            better_metrics.append(k)
                        elif v < 0:
                            worse_metrics.append(k)
                print(f"\n  vs {name}:")
                print(f"    Better on: {better_metrics if better_metrics else 'none'}")
                print(f"    Worse on:  {worse_metrics if worse_metrics else 'none'}")
                for metric, val in delta.items():
                    print(f"    {metric:<25}: {signs[metric]}{val:.4f}")

        print("=" * 70 + "\n")
