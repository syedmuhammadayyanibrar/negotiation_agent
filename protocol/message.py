from enum import Enum
from typing import Dict, Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from uuid import UUID, uuid4


class MessageType(Enum):
    OFFER = "OFFER"
    COUNTER_OFFER = "COUNTER_OFFER"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    WITHDRAW = "WITHDRAW"
    COALITION_PROPOSAL = "COALITION_PROPOSAL"
    COALITION_ACCEPT = "COALITION_ACCEPT"
    COALITION_REJECT = "COALITION_REJECT"
    COALITION_WITHDRAW = "COALITION_WITHDRAW"
    COALITION_LOCKED = "COALITION_LOCKED"
    DEADLOCK_DECLARED = "DEADLOCK_DECLARED"
    TIMEOUT_NOTICE = "TIMEOUT_NOTICE"
    HEALTH_PING = "HEALTH_PING"
    HEALTH_ACK = "HEALTH_ACK"
    CHECKPOINT = "CHECKPOINT"
    NEGOTIATION_COMPLETE = "NEGOTIATION_COMPLETE"
    NEGOTIATION_FAILED = "NEGOTIATION_FAILED"


class AgentStatus(Enum):
    ACTIVE = "ACTIVE"
    UNRESPONSIVE = "UNRESPONSIVE"
    CRASHED = "CRASHED"


class BaseMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    message_id: UUID = Field(default_factory=uuid4)
    message_type: MessageType
    sender_agent_id: str
    recipient_agent_id: str
    negotiation_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    round_number: int
    sequence_number: int

    @classmethod
    def from_dict(cls, data: dict) -> "BaseMessage":
        message_type = MessageType(data["message_type"])
        target_class = MESSAGE_REGISTRY[message_type]
        return target_class(**data)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class OfferMessage(BaseMessage):
    message_type: MessageType = MessageType.OFFER
    proposed_allocation: Dict[str, float]
    resource_type: str
    total_resource: float
    offer_utility_hint: Optional[float] = None
    concession_signal: bool = False
    expires_after_rounds: int = 3


class CounterOfferMessage(BaseMessage):
    message_type: MessageType = MessageType.COUNTER_OFFER
    in_response_to_message_id: UUID
    proposed_allocation: Dict[str, float]
    resource_type: str
    total_resource: float
    offer_utility_hint: Optional[float] = None
    concession_signal: bool = False
    expires_after_rounds: int = 3


class AcceptMessage(BaseMessage):
    message_type: MessageType = MessageType.ACCEPT
    accepted_offer_id: UUID


class RejectMessage(BaseMessage):
    message_type: MessageType = MessageType.REJECT
    rejected_message_id: UUID
    rejection_reason: Optional[str] = None


class WithdrawMessage(BaseMessage):
    message_type: MessageType = MessageType.WITHDRAW
    withdrawn_offer_id: UUID


class CoalitionProposalMessage(BaseMessage):
    message_type: MessageType = MessageType.COALITION_PROPOSAL
    coalition_id: UUID = Field(default_factory=uuid4)
    proposed_members: List[str]
    proposed_allocation: Dict[str, float]
    expires_after_rounds: int = 2
    resource_type: str
    total_resource: float


class CoalitionAcceptMessage(BaseMessage):
    message_type: MessageType = MessageType.COALITION_ACCEPT
    accepted_coalition_id: UUID


class CoalitionRejectMessage(BaseMessage):
    message_type: MessageType = MessageType.COALITION_REJECT
    rejected_coalition_id: UUID
    rejection_reason: Optional[str] = None


class CoalitionWithdrawMessage(BaseMessage):
    message_type: MessageType = MessageType.COALITION_WITHDRAW
    withdrawn_coalition_id: UUID


class CoalitionLockedMessage(BaseMessage):
    message_type: MessageType = MessageType.COALITION_LOCKED
    coalition_id: UUID
    coalition_members: List[str]
    final_allocation: Dict[str, float]


class DeadlockDeclaredMessage(BaseMessage):
    message_type: MessageType = MessageType.DEADLOCK_DECLARED
    deadlock_members: List[str]
    n_rounds_stalled: int


class TimeoutNoticeMessage(BaseMessage):
    message_type: MessageType = MessageType.TIMEOUT_NOTICE
    timedout_offer_id: UUID


class HealthPingMessage(BaseMessage):
    message_type: MessageType = MessageType.HEALTH_PING


class HealthAckMessage(BaseMessage):
    message_type: MessageType = MessageType.HEALTH_ACK
    agent_status: AgentStatus


class CheckpointMessage(BaseMessage):
    message_type: MessageType = MessageType.CHECKPOINT
    checkpoint_data: Dict[str, Any]


class NegotiationCompleteMessage(BaseMessage):
    message_type: MessageType = MessageType.NEGOTIATION_COMPLETE
    agreed_allocation: Dict[str, float]
    members: List[str]
    rounds_taken: int
    via_coalition: bool = False


class NegotiationFailedMessage(BaseMessage):
    message_type: MessageType = MessageType.NEGOTIATION_FAILED
    members: List[str]
    failure_reason: str


MESSAGE_REGISTRY = {
    MessageType.OFFER: OfferMessage,
    MessageType.COUNTER_OFFER: CounterOfferMessage,
    MessageType.ACCEPT: AcceptMessage,
    MessageType.REJECT: RejectMessage,
    MessageType.WITHDRAW: WithdrawMessage,
    MessageType.COALITION_PROPOSAL: CoalitionProposalMessage,
    MessageType.COALITION_ACCEPT: CoalitionAcceptMessage,
    MessageType.COALITION_REJECT: CoalitionRejectMessage,
    MessageType.COALITION_WITHDRAW: CoalitionWithdrawMessage,
    MessageType.COALITION_LOCKED: CoalitionLockedMessage,
    MessageType.DEADLOCK_DECLARED: DeadlockDeclaredMessage,
    MessageType.TIMEOUT_NOTICE: TimeoutNoticeMessage,
    MessageType.HEALTH_PING: HealthPingMessage,
    MessageType.HEALTH_ACK: HealthAckMessage,
    MessageType.CHECKPOINT: CheckpointMessage,
    MessageType.NEGOTIATION_COMPLETE: NegotiationCompleteMessage,
    MessageType.NEGOTIATION_FAILED: NegotiationFailedMessage,
}


def encode_channel(sender_id: str, recipient_id: str) -> str:
    return f"{sender_id}->{recipient_id}"
