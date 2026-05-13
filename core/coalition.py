import asyncio
from enum import Enum
from typing import List, Dict, Set, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict
from protocol.channel import Channel
from protocol.message import (
    CoalitionProposalMessage, CoalitionLockedMessage,
    NegotiationCompleteMessage, MessageType
)


class CoalitionStatus(Enum):
    PENDING = "PENDING"
    LOCKED = "LOCKED"
    FAILED = "FAILED"


class CoalitionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    coalition_id: UUID = None
    proposer_id: str
    proposed_members: List[str]
    proposed_allocation: Dict[str, float]
    accepted_by: List[str] = []
    rejected_by: List[str] = []
    status: CoalitionStatus = CoalitionStatus.PENDING
    rounds_remaining: int = 2
    round_number: int
    negotiation_id: str
    resource_type: str
    total_resource: float

    def model_post_init(self, __context):
        if self.coalition_id is None:
            self.coalition_id = uuid4()


class CoalitionManager:
    def __init__(self, state: CoalitionState, channel: Channel):
        self.state = state
        self.channel = channel

    def record_vote(self, agent_id: str, accepted: bool) -> None:
        if accepted:
            if agent_id not in self.state.accepted_by:
                self.state.accepted_by.append(agent_id)
        else:
            if agent_id not in self.state.rejected_by:
                self.state.rejected_by.append(agent_id)

    def check_outcome(self) -> CoalitionStatus:
        # Any rejection fails the coalition immediately
        if self.state.rejected_by:
            self.state.status = CoalitionStatus.FAILED
            return self.state.status

        # All non-proposer members must accept
        required = set(self.state.proposed_members) - {self.state.proposer_id}
        if required.issubset(set(self.state.accepted_by)):
            self.state.status = CoalitionStatus.LOCKED
            return self.state.status

        self.state.status = CoalitionStatus.PENDING
        return self.state.status

    async def propose(self) -> None:
        """Broadcast coalition proposal to all proposed members."""
        tasks = [
            self.channel.send(
                CoalitionProposalMessage(
                    sender_agent_id=self.state.proposer_id,
                    recipient_agent_id=agent_id,
                    negotiation_id=self.state.negotiation_id,
                    coalition_id=self.state.coalition_id,
                    proposed_members=self.state.proposed_members,
                    proposed_allocation=self.state.proposed_allocation,
                    round_number=self.state.round_number,
                    sequence_number=0,
                    expires_after_rounds=self.state.rounds_remaining,
                    resource_type=self.state.resource_type,
                    total_resource=self.state.total_resource,
                )
            )
            for agent_id in self.state.proposed_members
            if agent_id != self.state.proposer_id
        ]
        await asyncio.gather(*tasks)

    async def lock_coalition(self) -> None:
        """Broadcast COALITION_LOCKED then NEGOTIATION_COMPLETE."""
        lock_msg = CoalitionLockedMessage(
            sender_agent_id="coalition_manager",
            recipient_agent_id="broadcast",
            negotiation_id=self.state.negotiation_id,
            round_number=self.state.round_number,
            sequence_number=0,
            coalition_id=self.state.coalition_id,
            coalition_members=self.state.proposed_members,
            final_allocation=self.state.proposed_allocation,
        )
        await self.channel.broadcast(lock_msg)

        complete_msg = NegotiationCompleteMessage(
            sender_agent_id="coalition_manager",
            recipient_agent_id="broadcast",
            negotiation_id=self.state.negotiation_id,
            round_number=self.state.round_number,
            sequence_number=0,
            agreed_allocation=self.state.proposed_allocation,
            members=self.state.proposed_members,
            rounds_taken=self.state.round_number,
            via_coalition=True,
        )
        await self.channel.broadcast(complete_msg)

    def compute_coalition_utility(
        self,
        agent_id: str,
        utility_function,
    ) -> float:
        """Utility agent would get from the proposed coalition allocation."""
        return utility_function(self.state.proposed_allocation)

    def is_worth_joining(
        self,
        agent_id: str,
        utility_function,
        bilateral_best: float,
    ) -> bool:
        """
        Cooperative game theory check:
        join coalition if coalition utility > best bilateral outcome achievable.
        """
        coalition_util = self.compute_coalition_utility(agent_id, utility_function)
        return coalition_util > bilateral_best
