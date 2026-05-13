"""
Research experiment runner — with parameter randomization for genuine variance.
Each run gets slightly different reservation values, targets, and pressure
so results are statistically meaningful.
"""

import asyncio
import csv
import random
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

API_URL = "http://127.0.0.1:8000/negotiate"
RUNS_PER_EXPERIMENT = 20
DELAY_BETWEEN_CALLS = 2.5
OUTPUT_DIR = Path("research_output")
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(None)  # true randomness across runs


def jitter(base: float, low: float, high: float) -> float:
    """Add random noise within bounds."""
    return round(min(high, max(low, base + random.uniform(-0.08, 0.08))), 2)


def make_params(experiment_id: str, run_id: int) -> Dict:
    """
    Generate slightly varied parameters per run.
    Each experiment has a core setup but reservation/target values vary
    so agents face genuinely different negotiation landscapes.
    """
    r = random.Random(run_id * 31 + hash(experiment_id) % 997)  # reproducible per run

    if experiment_id == "RQ1_honest_vs_honest":
        res_a = r.uniform(0.25, 0.38)
        res_b = r.uniform(0.25, 0.38)
        tgt_a = r.uniform(0.52, 0.62)
        tgt_b = r.uniform(0.48, 0.58)
        return {
            "resource_type": "compute_budget",
            "total_resource": 1.0,
            "max_rounds": r.randint(8, 12),
            "hint_strategies": {"agent_a": "honest", "agent_b": "honest"},
            "reservation_values": {"agent_a": round(res_a, 2), "agent_b": round(res_b, 2)},
            "target_values": {"agent_a": round(tgt_a, 2), "agent_b": round(tgt_b, 2)},
        }

    elif experiment_id == "RQ2_honest_vs_inflate":
        res_a = r.uniform(0.25, 0.38)
        res_b = r.uniform(0.25, 0.38)
        tgt_a = r.uniform(0.52, 0.62)
        tgt_b = r.uniform(0.50, 0.62)
        return {
            "resource_type": "compute_budget",
            "total_resource": 1.0,
            "max_rounds": r.randint(8, 12),
            "hint_strategies": {"agent_a": "honest", "agent_b": "inflate"},
            "reservation_values": {"agent_a": round(res_a, 2), "agent_b": round(res_b, 2)},
            "target_values": {"agent_a": round(tgt_a, 2), "agent_b": round(tgt_b, 2)},
        }

    elif experiment_id == "RQ3_deadlock_coalition":
        # High reservation values — guaranteed deadlock zone
        res_a = r.uniform(0.42, 0.52)
        res_b = r.uniform(0.42, 0.52)
        tgt_a = r.uniform(0.65, 0.75)
        tgt_b = r.uniform(0.65, 0.75)
        return {
            "resource_type": "compute_budget",
            "total_resource": 1.0,
            "max_rounds": r.randint(12, 18),
            "hint_strategies": {"agent_a": "honest", "agent_b": "honest"},
            "reservation_values": {"agent_a": round(res_a, 2), "agent_b": round(res_b, 2)},
            "target_values": {"agent_a": round(tgt_a, 2), "agent_b": round(tgt_b, 2)},
        }

    elif experiment_id == "RQ4_deadline_pressure":
        res_a = r.uniform(0.25, 0.35)
        res_b = r.uniform(0.25, 0.35)
        tgt_a = r.uniform(0.55, 0.65)
        tgt_b = r.uniform(0.50, 0.60)
        # Vary deadline — some tight (5-6), some moderate (7-9)
        max_rounds = r.randint(5, 9)
        return {
            "resource_type": "compute_budget",
            "total_resource": 1.0,
            "max_rounds": max_rounds,
            "hint_strategies": {"agent_a": "honest", "agent_b": "honest"},
            "reservation_values": {"agent_a": round(res_a, 2), "agent_b": round(res_b, 2)},
            "target_values": {"agent_a": round(tgt_a, 2), "agent_b": round(tgt_b, 2)},
        }

    elif experiment_id == "RQ5_adaptive_vs_inflate":
        res_a = r.uniform(0.25, 0.38)
        res_b = r.uniform(0.25, 0.38)
        tgt_a = r.uniform(0.55, 0.65)
        tgt_b = r.uniform(0.55, 0.65)
        return {
            "resource_type": "compute_budget",
            "total_resource": 1.0,
            "max_rounds": r.randint(10, 14),
            "hint_strategies": {"agent_a": "adaptive", "agent_b": "inflate"},
            "reservation_values": {"agent_a": round(res_a, 2), "agent_b": round(res_b, 2)},
            "target_values": {"agent_a": round(tgt_a, 2), "agent_b": round(tgt_b, 2)},
        }

    return {}


EXPERIMENTS = [
    {
        "id": "RQ1_honest_vs_honest",
        "label": "RQ1 — Bilateral: Both Honest",
        "description": "Baseline bilateral negotiation with varied parameters. Tests agreement rate and fairness under honest communication.",
    },
    {
        "id": "RQ2_honest_vs_inflate",
        "label": "RQ2 — Trust: Honest vs Inflater",
        "description": "One agent inflates utility hints. Tests whether deception yields better outcomes.",
    },
    {
        "id": "RQ3_deadlock_coalition",
        "label": "RQ3 — Coalition: Genuine Deadlock",
        "description": "High reservation values guarantee bilateral deadlock. Tests coalition formation reliability.",
    },
    {
        "id": "RQ4_deadline_pressure",
        "label": "RQ4 — Deadline Pressure",
        "description": "Varied round budgets from tight (5) to moderate (9). Tests minimum viable negotiation window.",
    },
    {
        "id": "RQ5_adaptive_vs_inflate",
        "label": "RQ5 — Adaptive Strategy vs Inflater",
        "description": "Adaptive hint strategy against inflater. Tests whether strategic hint adaptation improves outcomes.",
    },
]


async def run_single(client, experiment, run_id):
    negotiation_id = f"{experiment['id']}_run{run_id:03d}"
    params = make_params(experiment["id"], run_id)
    payload = {
        "negotiation_id": negotiation_id,
        "agent_ids": ["agent_a", "agent_b"],
        **params,
    }

    try:
        response = await client.post(API_URL, json=payload, timeout=120.0)
        data = response.json()
        result = data.get("result", {})
        eval_report = result.get("eval_report", {})
        system = eval_report.get("system", {})
        comparison = eval_report.get("comparison", {})

        row = {
            "experiment_id": experiment["id"],
            "run_id": run_id,
            "negotiation_id": negotiation_id,
            "outcome": result.get("outcome", "unknown"),
            "via_coalition": result.get("via_coalition", False),
            "rounds_taken": result.get("rounds_taken", -1),
            "allocation_agent_a": result.get("allocation", {}).get("agent_a", 0),
            "allocation_agent_b": result.get("allocation", {}).get("agent_b", 0),
            "fairness_score": system.get("fairness_score", 0),
            "efficiency_score": system.get("efficiency_score", 0),
            "utility_gini": system.get("utility_gini", 0),
            "nash_distance": system.get("nash_bargaining_distance", 0),
            "fairness_vs_random": comparison.get("random_priority", {}).get("fairness_delta", 0),
            "efficiency_vs_random": comparison.get("random_priority", {}).get("efficiency_delta", 0),
            "fairness_vs_central": comparison.get("central_orchestrator", {}).get("fairness_delta", 0),
            "efficiency_vs_central": comparison.get("central_orchestrator", {}).get("efficiency_delta", 0),
            "hint_strategy_a": params.get("hint_strategies", {}).get("agent_a"),
            "hint_strategy_b": params.get("hint_strategies", {}).get("agent_b"),
            "max_rounds": params.get("max_rounds"),
            "reservation_a": params.get("reservation_values", {}).get("agent_a"),
            "reservation_b": params.get("reservation_values", {}).get("agent_b"),
            "target_a": params.get("target_values", {}).get("agent_a"),
            "target_b": params.get("target_values", {}).get("agent_b"),
        }
        print(f"  Run {run_id:02d} | {row['outcome']:12s} | rounds={row['rounds_taken']:2d} | "
              f"fairness={row['fairness_score']:.3f} | alloc_a={row['allocation_agent_a']:.2f} | "
              f"max_r={params.get('max_rounds')} res=({params['reservation_values']['agent_a']:.2f},{params['reservation_values']['agent_b']:.2f})")
        return row

    except Exception as e:
        print(f"  Run {run_id:02d} FAILED: {e}")
        return {
            "experiment_id": experiment["id"],
            "run_id": run_id,
            "outcome": "error",
            "error": str(e),
        }


async def run_all():
    all_results = []
    async with httpx.AsyncClient() as client:
        for exp in EXPERIMENTS:
            print(f"\n{'='*65}")
            print(f"  {exp['label']}")
            print(f"  {exp['description']}")
            print(f"{'='*65}")
            for run_id in range(1, RUNS_PER_EXPERIMENT + 1):
                row = await run_single(client, exp, run_id)
                all_results.append(row)
                await asyncio.sleep(DELAY_BETWEEN_CALLS)
    return all_results


def save_csv(results):
    valid = [r for r in results if "error" not in r]
    path = OUTPUT_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if valid:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(valid[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(valid)
    print(f"\nCSV saved: {path}")
    return path


def make_charts(results):
    groups = {}
    for r in results:
        eid = r.get("experiment_id", "")
        if r.get("outcome") not in ("error", "unknown", None):
            groups.setdefault(eid, []).append(r)

    exp_ids = [e["id"] for e in EXPERIMENTS if e["id"] in groups]
    labels = [e["label"].split("—")[1].strip() for e in EXPERIMENTS if e["id"] in exp_ids]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # Chart 1 — metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("System Performance Across Experiments (mean ± std)", fontsize=13, fontweight="bold")

    for ax, (metric, title, ylim) in zip(axes, [
        ("fairness_score", "Fairness (Jain's Index)", (0, 1.1)),
        ("efficiency_score", "Efficiency Score", (0, 1.1)),
        ("rounds_taken", "Rounds to Resolution", (0, None)),
    ]):
        means, stds = [], []
        for eid in exp_ids:
            vals = [r[metric] for r in groups[eid] if r.get(metric) is not None and r.get(metric) != 0 or metric == "fairness_score"]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        bars = ax.bar(range(len(exp_ids)), means, yerr=stds, capsize=5,
                      color=colors[:len(exp_ids)], alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(exp_ids)))
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        if ylim[1]:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "chart_performance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 2 — outcome distribution
    fig, axes = plt.subplots(1, len(exp_ids), figsize=(4.5 * len(exp_ids), 4))
    if len(exp_ids) == 1:
        axes = [axes]
    fig.suptitle("Outcome Distribution per Experiment", fontsize=13, fontweight="bold")
    outcome_colors = {"agreement": "#55A868", "coalition": "#4C72B0", "breakdown": "#C44E52", "unknown": "#999"}
    for ax, eid, label in zip(axes, exp_ids, labels):
        outcomes = [r.get("outcome") for r in groups.get(eid, [])]
        counts = {}
        for o in set(outcomes):
            counts[o] = outcomes.count(o)
        ax.pie(counts.values(), labels=[f"{k}\n({v})" for k,v in counts.items()],
               autopct="%1.0f%%",
               colors=[outcome_colors.get(k, "#999") for k in counts.keys()])
        ax.set_title(label, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "chart_outcomes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 3 — vs baselines
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("System vs Baselines (positive = outperforms baseline)", fontsize=12, fontweight="bold")
    for ax, (metric, title) in zip(axes, [
        ("fairness_vs_random", "Fairness delta vs Random Priority"),
        ("efficiency_vs_central", "Efficiency delta vs Central Orchestrator"),
    ]):
        means, stds = [], []
        for eid in exp_ids:
            vals = [r[metric] for r in groups[eid] if r.get(metric) is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        bar_colors = ["#55A868" if m >= 0 else "#C44E52" for m in means]
        ax.bar(range(len(exp_ids)), means, yerr=stds, capsize=5, color=bar_colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(exp_ids)))
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "chart_vs_baselines.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 4 — allocation distribution (box plot)
    fig, ax = plt.subplots(figsize=(12, 5))
    alloc_data = []
    alloc_labels = []
    for eid in exp_ids:
        vals = [r["allocation_agent_a"] for r in groups.get(eid, [])
                if r.get("allocation_agent_a") and r.get("outcome") != "breakdown"]
        if vals:
            alloc_data.append(vals)
            alloc_labels.append(next(e["label"].split("—")[1].strip() for e in EXPERIMENTS if e["id"] == eid))
    if alloc_data:
        bp = ax.boxplot(alloc_data, labels=alloc_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Equal split (0.5)")
        ax.set_title("Agent A Allocation Distribution", fontsize=12, fontweight="bold")
        ax.set_ylabel("Allocation to Agent A")
        ax.legend()
        plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "chart_allocation.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Charts saved to research_output/")


def make_report(results):
    groups = {}
    for r in results:
        eid = r.get("experiment_id", "")
        if r.get("outcome") not in ("error",):
            groups.setdefault(eid, []).append(r)

    path = OUTPUT_DIR / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    lines = [
        "# Negotiation Experiment Results",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Runs per experiment: {RUNS_PER_EXPERIMENT} | Total: {len(results)}\n",
        "---\n",
    ]

    for exp in EXPERIMENTS:
        eid = exp["id"]
        rows = groups.get(eid, [])
        if not rows:
            continue

        outcomes = [r["outcome"] for r in rows]
        n = len(outcomes)
        agreement_r = outcomes.count("agreement") / n
        coalition_r = outcomes.count("coalition") / n
        breakdown_r = outcomes.count("breakdown") / n

        fairness = [r["fairness_score"] for r in rows if r.get("fairness_score") and r["outcome"] != "breakdown"]
        efficiency = [r["efficiency_score"] for r in rows if r.get("efficiency_score") and r["outcome"] != "breakdown"]
        rounds = [r["rounds_taken"] for r in rows if r.get("rounds_taken", -1) > 0]
        alloc_a = [r["allocation_agent_a"] for r in rows if r.get("allocation_agent_a") and r["outcome"] != "breakdown"]

        lines += [
            f"## {exp['label']}",
            f"\n_{exp['description']}_\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| N runs | {n} |",
            f"| Agreement rate | {agreement_r:.1%} |",
            f"| Coalition rate | {coalition_r:.1%} |",
            f"| Breakdown rate | {breakdown_r:.1%} |",
            f"| Mean fairness (Jain) | {np.mean(fairness):.4f} ± {np.std(fairness):.4f} |" if fairness else "| Mean fairness | N/A |",
            f"| Mean efficiency | {np.mean(efficiency):.4f} ± {np.std(efficiency):.4f} |" if efficiency else "| Mean efficiency | N/A |",
            f"| Mean rounds | {np.mean(rounds):.2f} ± {np.std(rounds):.2f} |" if rounds else "| Mean rounds | N/A |",
            f"| Mean alloc agent_a | {np.mean(alloc_a):.4f} ± {np.std(alloc_a):.4f} |" if alloc_a else "| Mean alloc | N/A |",
            "\n",
        ]

    lines += [
        "---\n## Baseline Comparison Summary\n",
        "| Experiment | Fairness vs Random (mean±std) | Efficiency vs Central (mean±std) |",
        "|------------|-------------------------------|----------------------------------|",
    ]
    for exp in EXPERIMENTS:
        rows = groups.get(exp["id"], [])
        fvr = [r["fairness_vs_random"] for r in rows if "fairness_vs_random" in r]
        evc = [r["efficiency_vs_central"] for r in rows if "efficiency_vs_central" in r]
        label = exp["label"].split("—")[1].strip()
        fvr_str = f"{np.mean(fvr):+.4f}±{np.std(fvr):.4f}" if fvr else "N/A"
        evc_str = f"{np.mean(evc):+.4f}±{np.std(evc):.4f}" if evc else "N/A"
        lines.append(f"| {label} | {fvr_str} | {evc_str} |")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {path}")


async def main():
    print("=" * 65)
    print("NEGOTIATION RESEARCH EXPERIMENT RUNNER v2 — WITH VARIANCE")
    print(f"Experiments: {len(EXPERIMENTS)} x {RUNS_PER_EXPERIMENT} runs each")
    print(f"Total negotiations: {len(EXPERIMENTS) * RUNS_PER_EXPERIMENT}")
    print(f"Est. API calls: ~{len(EXPERIMENTS) * RUNS_PER_EXPERIMENT * 16}")
    print(f"Est. time: ~{len(EXPERIMENTS) * RUNS_PER_EXPERIMENT * (DELAY_BETWEEN_CALLS + 6) / 60:.0f} min")
    print("=" * 65)

    results = await run_all()
    print("\nSaving outputs...")
    save_csv(results)
    make_charts(results)
    make_report(results)
    print("\nDone. Check research_output/ folder.")


if __name__ == "__main__":
    asyncio.run(main())
