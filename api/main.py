import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from protocol.channel import Channel
from graph.orchestrator import Orchestrator
from eval.harness import EvalHarness
from memory.shared import SharedMemory
from core.agent import Agent

app = FastAPI(title="Decentralized Multi-Agent Negotiation API")

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
    hint_strategies: Optional[Dict[str, str]] = None  # agent_id -> strategy
    reservation_values: Optional[Dict[str, float]] = None
    target_values: Optional[Dict[str, float]] = None


@app.post("/negotiate")
async def start_negotiation(req: NegotiationRequest) -> Dict[str, Any]:
    channel = Channel()
    orchestrator = Orchestrator(
        agent_ids=req.agent_ids,
        negotiation_id=req.negotiation_id,
        resource_type=req.resource_type,
        total_resource=req.total_resource,
        channel=channel,
    )
    orchestrator.setup()

    agents = [
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
    try:
        result = await orchestrator.run_negotiation(agents=agents, max_rounds=req.max_rounds)
        print(f"[MAIN] result: {result}")
    except Exception as e:
        print(f"[MAIN] EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        result = {"outcome": "breakdown", "reason": str(e)}
    print(f"[MAIN] Negotiation complete: {result}")

    # Store outcome in Redis
    shared_memory.store_agreement(req.negotiation_id, result)

    # Run eval if agreement reached
    if result.get("outcome") != "breakdown":
        utility_fns = {a.agent_id: a.utility_function for a in agents}
        reservation_vals = {a.agent_id: a.reservation_value for a in agents}
        harness = EvalHarness(
            agent_ids=req.agent_ids,
            utility_functions=utility_fns,
            reservation_values=reservation_vals,
            total_resource=req.total_resource,
            max_rounds=req.max_rounds,
        )
        report = harness.compare(result, req.negotiation_id)
        result["eval_report"] = report

    return {"status": "complete", "negotiation_id": req.negotiation_id, "result": result}


@app.get("/negotiation/{negotiation_id}")
async def get_negotiation(negotiation_id: str) -> Dict[str, Any]:
    agreement = shared_memory.get_agreement(negotiation_id)
    if not agreement:
        raise HTTPException(status_code=404, detail="Negotiation not found")
    return {"negotiation_id": negotiation_id, "result": agreement}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
