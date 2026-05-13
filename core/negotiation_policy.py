import os
import json
from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel
from groq import AsyncGroq

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


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
    utility = agent.evaluate_offer(offer)
    pressure = agent.deadline_pressure

    # Compact prompt — under 150 tokens
    return f"""Negotiation agent. Decide action.

Offer from {offer.sender_agent_id}: {offer.proposed_allocation}
Your utility: {utility:.2f} | Min acceptable: {agent.reservation_value} | Target: {agent.target_value}
Round: {offer.round_number}/{agent.max_rounds} | Deadline pressure: {pressure:.2f}
Counterpart trust — hint inflation: {inflation:.2f} (0=honest, 1=liar)
Concessions made: {agent.concessions_made:.2f} / budget {agent.concession_budget:.2f}

Rules:
- ACCEPT if utility >= target OR (pressure > 0.8 AND utility >= min)
- COUNTER if utility < target AND budget remaining
- REJECT+COUNTER if utility < min
- PROPOSE_COALITION if stuck > 3 rounds

Respond ONLY valid JSON:
{{"action":"ACCEPT|COUNTER|REJECT|PROPOSE_COALITION","reasoning":"brief","proposed_allocation":{{"agent_id":0.0}} or null}}"""


async def decide(agent, offer) -> PolicyOutput:
    prompt = _build_prompt(agent, offer)

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a negotiation agent. Respond ONLY with valid JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        print(f"[POLICY] {agent.agent_id} -> {parsed.get('action')} | {parsed.get('reasoning','')[:60]}")
        return PolicyOutput(**parsed)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[POLICY] Parse error for {agent.agent_id}: {e}")
        return PolicyOutput(
            action=PolicyAction.COUNTER,
            reasoning=f"Parse error — defaulting to COUNTER",
            proposed_allocation=None,
        )
    except Exception as e:
        print(f"[POLICY] API error for {agent.agent_id}: {str(e)[:100]}")
        return PolicyOutput(
            action=PolicyAction.COUNTER,
            reasoning=f"API error — defaulting to COUNTER",
            proposed_allocation=None,
        )
