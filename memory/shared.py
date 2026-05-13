import redis
import json
from typing import Dict, Any, Optional


class SharedMemory:
    def __init__(self, host: str, port: int, db: int):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def _agreement_key(self, negotiation_id: str) -> str:
        return f"agreement:{negotiation_id}"

    def _checkpoint_key(self, agent_id: str, negotiation_id: str) -> str:
        return f"checkpoint:{negotiation_id}:{agent_id}"

    def store_agreement(self, negotiation_id: str, agreement_data: Dict[str, Any]) -> None:
        self.client.set(self._agreement_key(negotiation_id), json.dumps(agreement_data))

    def get_agreement(self, negotiation_id: str) -> Optional[Dict[str, Any]]:
        raw = self.client.get(self._agreement_key(negotiation_id))
        return json.loads(raw) if raw else None

    def store_checkpoint(self, agent_id: str, negotiation_id: str, state_data: Dict[str, Any]) -> None:
        self.client.set(self._checkpoint_key(agent_id, negotiation_id), json.dumps(state_data))

    def get_checkpoint(self, agent_id: str, negotiation_id: str) -> Optional[Dict[str, Any]]:
        raw = self.client.get(self._checkpoint_key(agent_id, negotiation_id))
        return json.loads(raw) if raw else None

    def delete_negotiation(self, negotiation_id: str) -> None:
        keys = self.client.keys(f"*:{negotiation_id}*")
        if keys:
            self.client.delete(*keys)
