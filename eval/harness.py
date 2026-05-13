"""
Evaluation harness: runs the main system and all 3 baselines,
computes all metrics, and produces a comparison report.
"""
from typing import Dict, List, Callable, Any
from eval.metrics import compute_all_metrics, NegotiationMetrics
from eval.baselines import (
    random_priority_baseline,
    fixed_hierarchy_baseline,
    central_orchestrator_baseline,
)


class EvalHarness:
    def __init__(
        self,
        agent_ids: List[str],
        utility_functions: Dict[str, Callable],
        reservation_values: Dict[str, float],
        total_resource: float = 1.0,
        max_rounds: int = 10,
    ):
        self.agent_ids = agent_ids
        self.utility_functions = utility_functions
        self.reservation_values = reservation_values
        self.total_resource = total_resource
        self.max_rounds = max_rounds

    def run_baselines(self) -> Dict[str, NegotiationMetrics]:
        results = {}

        # Baseline 1: Random priority
        alloc, rounds = random_priority_baseline(
            self.agent_ids, self.utility_functions,
            self.reservation_values, self.total_resource
        )
        results["random_priority"] = compute_all_metrics(
            negotiation_id="baseline_random",
            allocation=alloc,
            utility_functions=self.utility_functions,
            reservation_values=self.reservation_values,
            rounds_taken=rounds,
            max_rounds=self.max_rounds,
            outcome="agreement",
        )

        # Baseline 2: Fixed hierarchy
        alloc, rounds = fixed_hierarchy_baseline(
            self.agent_ids, self.utility_functions,
            self.reservation_values, self.total_resource
        )
        results["fixed_hierarchy"] = compute_all_metrics(
            negotiation_id="baseline_hierarchy",
            allocation=alloc,
            utility_functions=self.utility_functions,
            reservation_values=self.reservation_values,
            rounds_taken=rounds,
            max_rounds=self.max_rounds,
            outcome="agreement",
        )

        # Baseline 3: Central orchestrator
        alloc, rounds = central_orchestrator_baseline(
            self.agent_ids, self.utility_functions,
            self.reservation_values, self.total_resource
        )
        results["central_orchestrator"] = compute_all_metrics(
            negotiation_id="baseline_central",
            allocation=alloc,
            utility_functions=self.utility_functions,
            reservation_values=self.reservation_values,
            rounds_taken=rounds,
            max_rounds=self.max_rounds,
            outcome="agreement",
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
            )

        baseline_metrics = self.run_baselines()

        report = {
            "system": system_metrics.model_dump() if system_metrics else None,
            "baselines": {k: v.model_dump() for k, v in baseline_metrics.items()},
            "comparison": {},
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

    def print_report(self, report: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("NEGOTIATION EVALUATION REPORT")
        print("=" * 60)

        if report["system"]:
            s = report["system"]
            print(f"\nSYSTEM RESULT:")
            print(f"  Outcome:    {s['outcome']}")
            print(f"  Rounds:     {s['rounds_taken']}")
            print(f"  Fairness:   {s['fairness_score']:.4f}  (Jain's index)")
            print(f"  Efficiency: {s['efficiency_score']:.4f}")
            print(f"  Gini:       {s['utility_gini']:.4f}")
            print(f"  Nash dist:  {s['nash_bargaining_distance']:.4f}")
        else:
            print("\nSYSTEM RESULT: BREAKDOWN (no agreement reached)")

        print("\nBASELINE COMPARISON:")
        for name, delta in report["comparison"].items():
            better = [k for k, v in delta.items() if v > 0]
            worse = [k for k, v in delta.items() if v < 0]
            print(f"  vs {name}:")
            print(f"    Better on: {better if better else 'none'}")
            print(f"    Worse on:  {worse if worse else 'none'}")
            for metric, val in delta.items():
                sign = "+" if val >= 0 else ""
                print(f"    {metric}: {sign}{val:.4f}")

        print("=" * 60 + "\n")
