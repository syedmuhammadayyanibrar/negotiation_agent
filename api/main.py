import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from protocol.channel import Channel
from graph.orchestrator import Orchestrator
from eval.harness import EvalHarness
from memory.shared import SharedMemory
from core.agent import Agent
from datetime import datetime

app = FastAPI(
    title="Decentralized Multi-Agent Negotiation API",
    description=(
        "Multi-agent negotiation system with game-theoretic protocols, "
        "emergent trust mechanisms, and coalition formation."
    ),
    version="2.0.0",
)

shared_memory = SharedMemory(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    db=0,
)


class NegotiationRequest(BaseModel):
    negotiation_id: str
    agent_ids: List[str]
    resource_type: str
    total_resource: float
    max_rounds: int = 10
    hint_strategies: Optional[Dict[str, str]] = None   # agent_id -> strategy
    reservation_values: Optional[Dict[str, float]] = None
    target_values: Optional[Dict[str, float]] = None


def _build_agents(req: NegotiationRequest) -> List[Agent]:
    return [
        Agent(
            agent_id=aid,
            negotiation_id=req.negotiation_id,
            role=aid,
            utility_function=lambda x, a=aid: x.get(a, 0.0),  # linear utility
            reservation_value=(req.reservation_values or {}).get(aid, 0.3),
            target_value=(req.target_values or {}).get(aid, 0.6),
            max_rounds=req.max_rounds,
            hint_strategy=(req.hint_strategies or {}).get(aid, "honest"),
        )
        for aid in req.agent_ids
    ]


@app.post("/negotiate")
async def start_negotiation(
    req: NegotiationRequest,
    traces: bool = Query(default=False, description="Include per-round traces in response"),
) -> Dict[str, Any]:
    channel = Channel()
    orchestrator = Orchestrator(
        agent_ids=req.agent_ids,
        negotiation_id=req.negotiation_id,
        resource_type=req.resource_type,
        total_resource=req.total_resource,
        channel=channel,
    )
    orchestrator.setup()

    agents = _build_agents(req)
    try:
        result = await orchestrator.run_negotiation(agents=agents, max_rounds=req.max_rounds)
        print(f"[MAIN] result: {result}")
    except Exception as e:
        print(f"[MAIN] EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        result = {"outcome": "breakdown", "reason": str(e), "round_traces": []}

    print(f"[MAIN] Negotiation complete: {result}")

    # Strip traces from storage object (keep full version for response)
    result_for_storage = {k: v for k, v in result.items() if k != "round_traces"}
    shared_memory.store_agreement(req.negotiation_id, result_for_storage)

    # Run eval if agreement reached
    if result.get("outcome") != "breakdown":
        utility_fns = {a.agent_id: a.utility_function for a in agents}
        reservation_vals = {a.agent_id: a.reservation_value for a in agents}
        strategies = {a.agent_id: a.hint_strategy for a in agents}

        harness = EvalHarness(
            agent_ids=req.agent_ids,
            utility_functions=utility_fns,
            reservation_values=reservation_vals,
            total_resource=req.total_resource,
            max_rounds=req.max_rounds,
            hint_strategies=strategies,
        )
        report = harness.compare(result, req.negotiation_id)

        # Optionally strip round_traces from report to reduce response size
        if not traces:
            report.pop("round_traces", None)
            result.pop("round_traces", None)

        result["eval_report"] = report

    elif not traces:
        result.pop("round_traces", None)

    return {"status": "complete", "negotiation_id": req.negotiation_id, "result": result}


@app.get("/negotiation/{negotiation_id}")
async def get_negotiation(negotiation_id: str) -> Dict[str, Any]:
    agreement = shared_memory.get_agreement(negotiation_id)
    if not agreement:
        raise HTTPException(status_code=404, detail="Negotiation not found")
    return {"negotiation_id": negotiation_id, "result": agreement}


@app.post("/eval/run")
async def run_eval_suite(
    req: NegotiationRequest,
) -> Dict[str, Any]:
    """
    Run all 4 canonical eval scenarios for the given agent setup and return
    aggregated baseline comparison across scenarios.
    Uses baseline algorithms only (no LLM calls) — fast and deterministic.
    """
    utility_fns = {
        aid: (lambda x, a=aid: x.get(a, 0.0))
        for aid in req.agent_ids
    }
    reservation_vals = {
        aid: (req.reservation_values or {}).get(aid, 0.3)
        for aid in req.agent_ids
    }

    harness = EvalHarness(
        agent_ids=req.agent_ids,
        utility_functions=utility_fns,
        reservation_values=reservation_vals,
        total_resource=req.total_resource,
        max_rounds=req.max_rounds,
    )

    suite = harness.run_scenario_suite()
    return {
        "negotiation_id": req.negotiation_id,
        "resource_type": req.resource_type,
        "agent_ids": req.agent_ids,
        "scenario_suite": suite,
    }


@app.post("/eval/compare")
async def compare_negotiations(
    negotiation_id_a: str,
    negotiation_id_b: str,
) -> Dict[str, Any]:
    """
    Compare two stored negotiation outcomes head-to-head.
    Returns the stored results for both and delta on key metrics.
    """
    result_a = shared_memory.get_agreement(negotiation_id_a)
    result_b = shared_memory.get_agreement(negotiation_id_b)

    if not result_a:
        raise HTTPException(status_code=404, detail=f"Negotiation {negotiation_id_a} not found")
    if not result_b:
        raise HTTPException(status_code=404, detail=f"Negotiation {negotiation_id_b} not found")

    # Extract top-level eval metrics if present
    def get_metrics(result: Dict) -> Optional[Dict]:
        report = result.get("eval_report", {})
        return report.get("system")

    metrics_a = get_metrics(result_a)
    metrics_b = get_metrics(result_b)

    comparison = {}
    if metrics_a and metrics_b:
        for key in ("fairness_score", "efficiency_score", "utility_gini", "nash_bargaining_distance"):
            if key in metrics_a and key in metrics_b:
                comparison[f"{key}_delta_a_minus_b"] = round(metrics_a[key] - metrics_b[key], 4)

    return {
        negotiation_id_a: {"result": result_a, "metrics": metrics_a},
        negotiation_id_b: {"result": result_b, "metrics": metrics_b},
        "comparison": comparison,
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "2.0.0"}
