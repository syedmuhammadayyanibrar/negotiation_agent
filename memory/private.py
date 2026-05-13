import hashlib
import os
from typing import Dict, Any, List, Optional
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, Filter,
    FieldCondition, MatchValue
)

VECTOR_DIM = 128
DEFAULT_TRUST_PROFILE = {
    "hint_inflation_score": 0.3,
    "sessions_observed": 0,
    "concession_velocity_history": [],
    "rejection_contradictions": 0,
    "acceptance_timing_history": [],
    "allocation_proxy_floors": [],
    "reliability": "none",
}


def _stable_id(key: str) -> int:
    """Stable, collision-resistant integer ID from a string key."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) >> 64


def _deterministic_vector(text: str) -> List[float]:
    """Deterministic placeholder vector until sentence-transformers is integrated."""
    import hashlib
    seed_bytes = hashlib.sha256(text.encode()).digest()
    floats = []
    for i in range(0, VECTOR_DIM * 4, 4):
        chunk = seed_bytes[i % len(seed_bytes): (i % len(seed_bytes)) + 4]
        val = int.from_bytes(chunk.ljust(4, b"\x00"), "big") / (2**32)
        floats.append(val)
    return floats[:VECTOR_DIM]


class PrivateMemory:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    # ── Trust profiles ────────────────────────────────────────────────────────

    def store_trust_profile(self, counterpart_id: str, profile_data: Dict[str, Any]) -> None:
        profile_data["counterpart_id"] = counterpart_id
        sessions = profile_data.get("sessions_observed", 0)
        if sessions == 0:
            reliability = "none"
        elif sessions < 3:
            reliability = "low"
        elif sessions < 6:
            reliability = "medium"
        else:
            reliability = "high"
        profile_data["reliability"] = reliability

        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=_stable_id(f"trust:{counterpart_id}"),
                vector=_deterministic_vector(f"trust_profile_{counterpart_id}"),
                payload=profile_data,
            )],
        )

    def get_trust_profile(self, counterpart_id: str) -> Dict[str, Any]:
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[_stable_id(f"trust:{counterpart_id}")],
        )
        return result[0].payload if result else dict(DEFAULT_TRUST_PROFILE)

    def update_hint_inflation(
        self,
        counterpart_id: str,
        hint_error: float,
        alpha: float = 0.3,
    ) -> Dict[str, Any]:
        """EMA update of hint_inflation_score after a completed session."""
        profile = self.get_trust_profile(counterpart_id)
        old_score = profile.get("hint_inflation_score", 0.3)
        new_score = alpha * hint_error + (1 - alpha) * old_score
        profile["hint_inflation_score"] = round(new_score, 4)
        profile["sessions_observed"] = profile.get("sessions_observed", 0) + 1
        self.store_trust_profile(counterpart_id, profile)
        return profile

    # ── Episode memories ──────────────────────────────────────────────────────

    def store_episode(self, episode_data: Dict[str, Any]) -> None:
        description = (
            f"negotiation with {episode_data.get('counterpart_id')} "
            f"outcome {episode_data.get('outcome')} "
            f"rounds {episode_data.get('rounds_taken')}"
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=_stable_id(str(uuid4())),
                vector=_deterministic_vector(description),
                payload=episode_data,
            )],
        )

    def search_episodes(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=_deterministic_vector(query),
            limit=limit,
        )
        return [r.payload for r in results]

    # ── Behavioral patterns ───────────────────────────────────────────────────

    def store_behavioral_pattern(
        self,
        counterpart_id: str,
        session_n: int,
        pattern_data: Dict[str, Any],
    ) -> None:
        pattern_data["counterpart_id"] = counterpart_id
        pattern_data["session_n"] = session_n
        description = pattern_data.get("description", f"behavior of {counterpart_id} session {session_n}")
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=_stable_id(f"pattern:{counterpart_id}:{session_n}"),
                vector=_deterministic_vector(description),
                payload=pattern_data,
            )],
        )

    def search_behavioral_patterns(
        self, counterpart_id: str, limit: int = 2
    ) -> List[Dict[str, Any]]:
        query = f"behavior patterns of agent {counterpart_id}"
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=_deterministic_vector(query),
            limit=limit,
        )
        return [r.payload for r in results if r.payload.get("counterpart_id") == counterpart_id]
