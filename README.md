# Decentralized Multi-Agent Negotiation System

A production-grade agentic AI system where multiple LLM-powered agents negotiate resource allocation without any central authority, using game-theoretic protocols, emergent trust mechanisms, and coalition formation.

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
│   └── runner.py           # Standalone CLI eval runner (no server needed)
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

## Run a Negotiation

```bash
curl -X POST http://localhost:8000/negotiate \
  -H "Content-Type: application/json" \
  -d '{
    "negotiation_id": "test-001",
    "agent_ids": ["agent_a", "agent_b"],
    "resource_type": "compute_budget",
    "total_resource": 1.0,
    "max_rounds": 10,
    "hint_strategies": {"agent_a": "honest", "agent_b": "inflate"},
    "reservation_values": {"agent_a": 0.3, "agent_b": 0.25},
    "target_values": {"agent_a": 0.6, "agent_b": 0.65}
  }'

# Include per-round traces in response:
curl -X POST "http://localhost:8000/negotiate?traces=true" ...
```

## Run Evals (Standalone CLI — No Server Required)

```bash
# Run all 4 canonical research scenarios
python -m eval.runner

# Run specific scenarios
python -m eval.runner --scenarios honest_honest honest_vs_inflate

# Custom agents and round count
python -m eval.runner --agents alpha beta gamma --max-rounds 15

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
