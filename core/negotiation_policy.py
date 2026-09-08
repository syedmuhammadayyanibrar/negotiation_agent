import os
import json
import asyncio
from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel
from groq import AsyncGroq
from groq import GroqError

try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        """No-op fallback if langsmith not installed / key not set."""
        def decorator(fn):
            return fn
        return decorator

MODEL = "groq/compound-mini"
_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    """Lazy-initialize the Groq client so .env can be loaded before first use."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise GroqError(
                "GROQ_API_KEY not set. Add it to your .env file or set the environment variable."
            )
        _client = AsyncGroq(api_key=api_key)
    return _client

# Only enable LangSmith tracing when the API key is present
_LANGSMITH_ENABLED = bool(os.environ.get("LANGCHAIN_API_KEY"))


class PolicyAction(Enum):
    OFFER = "OFFER"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    COUNTER = "COUNTER"
    PROPOSE_COALITION = "PROPOSE_COALITION"


class PolicyOutput(BaseModel):
    action: PolicyAction
    reasoning: str
    proposed_allocation: Optional[Dict[str, float]] = None


def _build_prompt(agent, offer) -> str:
    trust = agent.get_trust_profile(offer.sender_agent_id)
    inflation = trust.get("hint_inflation_score", 0.3)
    reliability = trust.get("reliability", "none")
    sessions_seen = trust.get("sessions_observed", 0)
    utility = agent.evaluate_offer(offer)
    pressure = agent.deadline_pressure
    concession_rate = agent.concession_rate_per_round

    # Trust-adjusted effective minimum: if counterpart lies a lot, discount their hints
    trust_discount = 0.0
    if inflation > 0.6:
        trust_discount = 0.05  # raise effective minimum when counterpart is a known liar
    effective_min = round(agent.reservation_value + trust_discount, 4)

    # Concession trend context
    if concession_rate > 0.08:
        concession_ctx = "making large concessions (aggressive)"
    elif concession_rate > 0.03:
        concession_ctx = "making moderate concessions"
    else:
        concession_ctx = "holding firm (minimal concessions)"

    # Coalition context
    coalition_hint = ""
    if agent.rounds_without_progress >= 3:
        coalition_hint = "\n- Consider PROPOSE_COALITION: negotiation is stalling."

    return f"""You are a rational negotiation agent maximizing your own utility.

CURRENT SITUATION:
- Offer from {offer.sender_agent_id}: {offer.proposed_allocation}
- Your utility for this offer: {utility:.3f}
- Your minimum acceptable: {agent.reservation_value} (effective min w/ trust discount: {effective_min})
- Your target: {agent.target_value}
- Round: {offer.round_number}/{agent.max_rounds} (deadline pressure: {pressure:.2f})
- Your current ask: {agent.my_current_allocation_ask:.4f}
- Concession budget remaining: {agent.remaining_concession_budget():.4f}
- Your concession behavior: {concession_ctx}

COUNTERPART TRUST PROFILE ({offer.sender_agent_id}):
- Hint inflation score: {inflation:.2f} (0=honest, 1=known liar)
- Reliability: {reliability} (based on {sessions_seen} sessions)
- Their offer hint: {getattr(offer, 'offer_utility_hint', 'N/A')}
{"- WARNING: counterpart has high inflation score — their hints are likely inflated." if inflation > 0.6 else ""}

DECISION RULES:
- ACCEPT: if utility >= target, OR (utility >= effective_min AND pressure > 0.65)
- COUNTER: if utility < target AND concession budget remaining > 0 — propose a new allocation
- REJECT+COUNTER: if utility < effective_min — always include a counter proposal
- PROPOSE_COALITION: if negotiation is deeply stalled (3+ rounds without progress){coalition_hint}

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown:
{{"action":"ACCEPT|COUNTER|REJECT|PROPOSE_COALITION","reasoning":"one sentence","proposed_allocation":{{"agent_id":0.0}} or null}}"""


async def _call_llm_with_retry(prompt: str, agent_id: str, max_retries: int = 3) -> str:
    """Call Groq LLM with exponential backoff retry on transient errors.
    Permanent errors (404 model not found, 400 bad request) raise immediately.
    """
    from groq import NotFoundError, BadRequestError
    for attempt in range(max_retries):
        try:
            response = await _get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a negotiation agent. "
                            "Respond ONLY with valid JSON. No markdown, no explanation outside JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except (NotFoundError, BadRequestError) as e:
            # Permanent errors — no point retrying
            print(f"[POLICY] Permanent error for {agent_id}: {str(e)[:120]}")
            raise
        except Exception as e:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"[POLICY] API error for {agent_id} (attempt {attempt + 1}/{max_retries}): {str(e)[:80]} — retrying in {wait}s")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
            else:
                raise


def _strip_markdown(raw: str) -> str:
    """Strip ``` fences that some models add despite instructions."""
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove opening fence line and closing fence
        inner = "\n".join(lines[1:])
        if inner.endswith("```"):
            inner = inner[: inner.rfind("```")]
        return inner.strip()
    return raw


def _make_decide(enable_tracing: bool):
    """Factory that returns decide() with or without @traceable."""

    async def _decide_impl(agent, offer) -> PolicyOutput:
        prompt = _build_prompt(agent, offer)
        try:
            raw = await _call_llm_with_retry(prompt, agent.agent_id)
            raw = _strip_markdown(raw)
            parsed = json.loads(raw)
            print(
                f"[POLICY] {agent.agent_id} -> {parsed.get('action')} | "
                f"{parsed.get('reasoning', '')[:70]}"
            )
            return PolicyOutput(**parsed)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[POLICY] Parse error for {agent.agent_id}: {e}")
            return PolicyOutput(
                action=PolicyAction.COUNTER,
                reasoning="Parse error — defaulting to COUNTER",
                proposed_allocation=None,
            )
        except Exception as e:
            print(f"[POLICY] Fatal error for {agent.agent_id}: {str(e)[:100]}")
            return PolicyOutput(
                action=PolicyAction.COUNTER,
                reasoning="Fatal error — defaulting to COUNTER",
                proposed_allocation=None,
            )

    if enable_tracing:
        return traceable(
            name="negotiation_policy_decide",
            run_type="llm",
            tags=["negotiation", "policy"],
        )(_decide_impl)

    return _decide_impl


decide = _make_decide(enable_tracing=_LANGSMITH_ENABLED)
