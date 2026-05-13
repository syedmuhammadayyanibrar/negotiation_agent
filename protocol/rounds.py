import asyncio
from enum import Enum
from typing import Dict, List, Set, Optional
from pydantic import BaseModel, ConfigDict
from protocol.message import (
    BaseMessage, MessageType, HealthPingMessage,
    DeadlockDeclaredMessage, AgentStatus
)
from protocol.channel import Channel


class AgentStates(Enum):
    ACTIVE = "ACTIVE"
    UNRESPONSIVE = "UNRESPONSIVE"
    CRASHED = "CRASHED"


class RoundState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    round_number: int
    negotiation_id: str
    moved_agents: List[str] = []   # agents that changed position this round
    rounds_without_progress: int = 0
    agent_states: Dict[str, AgentStates]


class RoundManager:
    def __init__(self, channel: Channel, state: RoundState):
        self.channel = channel
        self.state = state

    def record_position_change(self, agent_id: str) -> None:
        """Call when agent sends OFFER or COUNTER_OFFER (not any message)."""
        if agent_id not in self.state.moved_agents:
            self.state.moved_agents.append(agent_id)

    def advance_round(self) -> None:
        self.state.round_number += 1
        if not self.state.moved_agents:
            self.state.rounds_without_progress += 1
        else:
            self.state.rounds_without_progress = 0
        self.state.moved_agents = []

    def check_deadlock(self, threshold: int) -> bool:
        return self.state.rounds_without_progress >= threshold

    async def ping_agent(self, agent_id: str) -> None:
        message = HealthPingMessage(
            sender_agent_id="round_manager",
            recipient_agent_id=agent_id,
            negotiation_id=self.state.negotiation_id,
            round_number=self.state.round_number,
            sequence_number=0,
        )
        await self.channel.send(message)

    async def check_agent_health(
        self, agent_id: str, timeout_seconds: float = 5.0
    ) -> AgentStates:
        """Ping agent and wait for ACK. Returns ACTIVE or CRASHED."""
        await self.ping_agent(agent_id)
        try:
            # Wait up to timeout_seconds for any message from that agent
            response = await asyncio.wait_for(
                self.channel.receive(agent_id), timeout=timeout_seconds
            )
            if response.message_type == MessageType.HEALTH_ACK:
                self.state.agent_states[agent_id] = AgentStates.ACTIVE
                return AgentStates.ACTIVE
        except asyncio.TimeoutError:
            pass
        self.state.agent_states[agent_id] = AgentStates.CRASHED
        return AgentStates.CRASHED

    async def declare_deadlock(self, involved_agents: List[str]) -> None:
        message = DeadlockDeclaredMessage(
            sender_agent_id="round_manager",
            recipient_agent_id="broadcast",
            negotiation_id=self.state.negotiation_id,
            round_number=self.state.round_number,
            sequence_number=0,
            deadlock_members=involved_agents,
            n_rounds_stalled=self.state.rounds_without_progress,
        )
        await self.channel.broadcast(message)
