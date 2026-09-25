# Autonomous B2B Deal & Procurement Negotiation Engine
### *A Production-Grade Multi-Agent Bargaining System for Commercial Contracts & Vendor SLAs*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v2.0-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-FF6F00.svg)](https://langchain.com/)
[![Groq](https://img.shields.io/badge/Inference-Groq-F55036.svg)](https://groq.com/)
[![Memory](https://img.shields.io/badge/Memory-Qdrant%20%2B%20Redis%20%2B%20Postgres-blueviolet.svg)]()

An autonomous multi-agent bargaining engine engineered to resolve complex, multi-stakeholder B2B procurement deals, vendor pricing agreements, and cloud SLA contracts. Rather than relying on rigid rules or unilateral chatbot prompts, the system deploys autonomous buyer and vendor agents who negotiate contract terms via an asynchronous 17-message protocol, model counterparty concession velocities, and trigger cooperative coalition enforcement when deadlocks occur.

---

## 💼 Business Problem & Enterprise Value

In enterprise procurement, commercial deal cycles take weeks of back-and-forth exchanges. Misaligned pricing concessions, opaque vendor posturing, and negotiation deadlocks cost enterprises hundreds of thousands of dollars in delayed software rollouts and unfavorable terms. 

This engine automates multi-party commercial bargaining by:
1. **Accelerating Deal Cycles**: Reaches optimal contract settlements in 3 to 8 structured rounds instead of multi-week manual negotiations.
2. **Eliminating Deception & Adverse Concessions**: Tracks counterpart behavioral signals across sessions, dynamically detecting hint inflation and penalizing bad-faith concessions.
3. **Guaranteed Settlement via Coalitions**: Automatically triggers cooperative game-theoretic coalition protocols when bilateral negotiations reach a deadlock.

```text
                  ┌────────────────────────────────────────────────────────┐
                  │          ENTERPRISE BUYER AGENT (LangGraph)           │
                  │  Private Utility Function • BATNA • Concession Budget │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Async 17-Message Protocol
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │        DECENTRALIZED BARGAINING & CONVERGENCE          │
                  │  Deadlock Detection • Nash Distance • Trust Tracking   │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Counter-Offers & Settlement
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │          SAAS VENDOR AGENT (LangGraph)                │
                  │   Reservation Pricing • SLA Caps • Coalition Logic     │
                  └────────────────────────────────────────────────────────┘
```

---

## Architecture

```
negotiation_agent/
├── protocol/
│   ├── message.py          # 17 message types with full Pydantic schemas
│   ├── channel.py          # Async message passing with dedup + sequence enforcement
│   └── rounds.py           # Round manager with deadlock detection
├── core/
│   ├── agent.py            # Agent with private state, hint strategies, EMA trust, session history
│   ├── negotiation_policy.py  # Groq LLM decision policy + lazy client init + smart retries
│   ├── convergence.py      # Progress tracking + deadlock declaration
│   └── coalition.py        # Coalition formation with cooperative game theory
├── memory/
│   ├── private.py          # Per-agent Qdrant memory (trust profiles, episodes)
│   ├── shared.py           # Redis shared agreements + checkpoints
│   └── logs.py             # Postgres full transcript logging
├── graph/
│   └── orchestrator.py     # LangGraph orchestration + per-round traces + trust updates
├── eval/
│   ├── metrics.py          # Jain fairness, Gini, Nash, Pareto, trust calibration, lying cost
│   ├── baselines.py        # Random, fixed hierarchy, central orchestrator, tit-for-tat
│   ├── harness.py          # Full comparison report + scenario suite + JSON export
│   └── runner.py           # Standalone CLI eval runner with B2B procurement suites
└── api/
    └── main.py             # FastAPI endpoints (v2.0.0)
```

## Setup

```bash
# 1. Copy env file
cp .env.example .env
# Fill in GROQ_API_KEY (required) and LANGCHAIN_API_KEY (optional, for LangSmith tracing)

# 2. Start infrastructure (optional for API, not required for standalone CLI)
docker-compose up redis postgres qdrant -d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run API
uvicorn api.main:app --reload
```

## Run a Commercial Deal Negotiation

```bash
curl -X POST http://localhost:8000/negotiate \
  -H "Content-Type: application/json" \
  -d '{
    "negotiation_id": "b2b-procure-2026-09",
    "agent_ids": ["enterprise_buyer", "saas_vendor"],
    "resource_type": "contract_commercial_terms",
    "total_resource": 1.0,
    "max_rounds": 10,
    "hint_strategies": {"enterprise_buyer": "adaptive", "saas_vendor": "inflate"},
    "reservation_values": {"enterprise_buyer": 0.35, "saas_vendor": 0.30},
    "target_values": {"enterprise_buyer": 0.65, "saas_vendor": 0.70}
  }'

# Include per-round traces in response:
curl -X POST "http://localhost:8000/negotiate?traces=true" ...
```

## Run Evals (Standalone CLI — No Server Required)

```bash
# Run B2B enterprise procurement & SLA bargaining scenarios
python -m eval.runner --scenarios b2b_procurement_buyer_vendor enterprise_sla_settlement

# Run all scenarios (B2B procurement + game-theoretic research baselines)
python -m eval.runner

# Reports saved to: research_output/eval_YYYYMMDD_HHMMSS.json
```

## 📊 Latest Benchmark Results

| Scenario | Strategy (A vs B) | Outcome | Rounds | Fairness | Efficiency | Nash Dist | Avg Hint Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`honest_honest`** | Honest vs. Honest | **Agreement** | 7 | **1.0000** | 0.3333 | **0.0000** | 0.0000 |
| **`inflate_inflate`** | Inflate vs. Inflate | **Agreement** | 3 | 0.9615 | **0.7778** | 0.2500 | 0.2000 |
| **`honest_vs_inflate`** | Honest vs. Inflate | **Coalition** | 8 | 1.0000 | 0.2222 | 0.0000 | 0.1000 |
| **`adaptive_vs_honest`** | Adaptive vs. Honest | **Coalition** | 8 | 1.0000 | 0.2222 | 0.0000 | 0.0656 |

- **Honest Alignment:** Truthful agents converge on an exact 50/50 Nash Bargaining Equilibrium (`nash_distance = 0.0`).
- **Deception Dynamics:** Mutual hint inflation speeds up agreement (3 rounds vs 7), but distorts fairness into an asymmetric 60/40 split (`nash_distance = 0.25`).
- **Coalition Fallback:** Deadlocks and unmatched strategic states seamlessly trigger coalition enforcement mechanisms.

## API Endpoints (v2.0.0)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/negotiate` | Run a negotiation (`?traces=true` for round-by-round data) |
| GET | `/negotiation/{id}` | Retrieve stored result |
| POST | `/eval/run` | Run baseline scenario suite (deterministic, no LLM) |
| POST | `/eval/compare` | Head-to-head comparison of two negotiation IDs |
| GET | `/health` | Health check |

## Research Parameters

| Parameter | Values | Effect |
|-----------|--------|--------|
| `hint_strategy` | honest / inflate / deflate / adaptive | Controls lying behavior |
| `max_rounds` | 5 / 10 / 20 | Deadline pressure |
| `reservation_value` | 0.2 - 0.6 | BATNA aggressiveness |
| `alpha` (EMA trust) | 0.1 / 0.3 / 0.5 | Trust update speed |

## Eval Metrics

| Metric | Description |
|--------|-------------|
| `fairness_score` | Jain's fairness index [1/n, 1.0] |
| `efficiency_score` | 1.0 = agreed round 1, 0.0 = max rounds used |
| `nash_bargaining_distance` | Distance from Nash optimal solution |
| `utility_gini` | Gini coefficient of per-agent utilities |
| `pareto_efficient` | Whether any agent could do better without harming another |
| `concession_speed` | Avg concession per round per agent |
| `trust_calibration_score` | Pearson ρ between trust score and actual hint inflation |
| `lying_cost` | Utility difference: honest agent − liar agent |
| `avg_hint_error` | Mean \|stated hint − actual utility\| |

## Key Research Questions

- **RQ1**: Does decentralized trust (`hint_inflation_score`) correctly calibrate over sessions? → `trust_calibration_score`
- **RQ2**: Do agents with mature trust profiles achieve better outcomes? → `efficiency_score` + `fairness_score` over sessions
- **RQ3**: Does lying pay short-term but cost long-term? → `lying_cost` across scenarios
- **RQ4**: Does coalition formation improve fairness vs bilateral? → `fairness_score` with `via_coalition=True`
- **RQ5**: How close does decentralized system get to central orchestrator quality? → `nash_bargaining_distance` delta

## Key Enhancements & Bug Fixes (v2.0.0 Commit)

- **Lazy Client Initialization**: Deferred AsyncGroq client instantiation to run on demand, preventing startup crashes when `.env` is loaded dynamically.
- **Permanent Error Handling**: Added immediate error detection for HTTP 400/404 responses, eliminating wasteful 7-second retry backoffs.
- **Model Compatibility**: Defaulted to `groq/compound-mini` for fast and reliable structured JSON outputs across free/standard Groq tier accounts.
- **Double LLM Call Fix**: Fixed `orchestrator.py` where `decide()` was invoked twice per agent round, halving LLM latency and API costs.
- **Trust Signal Bug**: Corrected `compute_behavioral_trust_signals()` in `agent.py` to target `counterpart_id` instead of `self.agent_id`.
- **Efficiency Metric Fix**: Fixed `efficiency_score` formula so round 1 agreement correctly scores `1.0`.
- **Standalone CLI Runner**: Created `eval/runner.py` for headless evaluation runs with clean JSON exports and ASCII benchmark outputs.
