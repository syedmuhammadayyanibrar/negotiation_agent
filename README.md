# Decentralized Multi-Agent Negotiation System

A production-grade agentic AI system where multiple LLM-powered agents negotiate 
resource allocation without any central authority, using game-theoretic protocols,
emergent trust mechanisms, and coalition formation.

## Architecture

```
negotiation_agent/
├── protocol/
│   ├── message.py          # 17 message types with full Pydantic schemas
│   ├── channel.py          # Async message passing with dedup + sequence enforcement
│   └── rounds.py           # Round manager with deadlock detection
├── core/
│   ├── agent.py            # Agent with private state, hint strategies, trust modeling
│   ├── negotiation_policy.py  # Groq LLaMA 70B powered decision making
│   ├── convergence.py      # Progress tracking + deadlock declaration
│   └── coalition.py        # Coalition formation with cooperative game theory
├── memory/
│   ├── private.py          # Per-agent Qdrant memory (trust profiles, episodes)
│   ├── shared.py           # Redis shared agreements + checkpoints
│   └── logs.py             # Postgres full transcript logging
├── graph/
│   └── orchestrator.py     # LangGraph orchestration
├── eval/
│   ├── metrics.py          # Jain fairness, Gini, Nash bargaining distance
│   ├── baselines.py        # Random, fixed hierarchy, central orchestrator
│   └── harness.py          # Full comparison report
└── api/
    └── main.py             # FastAPI endpoints
```

## Setup

```bash
# 1. Copy env file
cp .env.example .env
# Fill in GROQ_API_KEY and LANGCHAIN_API_KEY

# 2. Start infrastructure
docker-compose up redis postgres qdrant -d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run API
uvicorn api.main:app --reload
```

## Run a negotiation

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
```

## Research Parameters

| Parameter | Values | Effect |
|-----------|--------|--------|
| `hint_strategy` | honest / inflate / deflate / adaptive | Controls lying behavior |
| `max_rounds` | 5 / 10 / 20 | Deadline pressure |
| `reservation_value` | 0.2 - 0.6 | BATNA aggressiveness |
| `alpha` (EMA) | 0.1 / 0.3 / 0.5 | Trust update speed |

## Key Research Questions

- **RQ1**: Does decentralized trust (hint_inflation_score) correctly calibrate over sessions?
- **RQ2**: Do agents with mature trust profiles achieve better outcomes?  
- **RQ3**: Does lying pay short-term but cost long-term?
- **RQ4**: Does coalition formation improve fairness vs bilateral?
- **RQ5**: How close does decentralized system get to central orchestrator quality?
