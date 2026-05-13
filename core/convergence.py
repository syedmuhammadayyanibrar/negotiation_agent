from typing import Dict, List
from pydantic import BaseModel
from protocol.channel import Channel
from protocol.message import DeadlockDeclaredMessage


class ConvergenceState(BaseModel):
    progress_scores: List[float] = []
    previous_allocations: Dict[str, Dict[str, float]] = {}
    threshold: float = 0.05   # minimum progress per round to not count as stalled
    round_number: int = 0
    negotiation_id: str
    stall_rounds: int = 0
    stall_limit: int = 3      # rounds without progress before deadlock declared


class ConvergenceChecker:
    def __init__(self, state: ConvergenceState, channel: Channel):
        self.state = state
        self.channel = channel

    def _compute_progress(
        self,
        current_allocations: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Progress = average absolute change in allocation per agent.
        0.0 = no movement, 1.0 = complete reversal.
        """
        if not self.state.previous_allocations:
            return 1.0  # first round, always progress

        total_delta = 0.0
        count = 0
        for agent_id, curr_alloc in current_allocations.items():
            prev_alloc = self.state.previous_allocations.get(agent_id, {})
            for resource, value in curr_alloc.items():
                prev_value = prev_alloc.get(resource, value)
                total_delta += abs(value - prev_value)
                count += 1

        return total_delta / count if count > 0 else 0.0

    async def check_and_escalate(
        self,
        current_allocations: Dict[str, Dict[str, float]],
        involved_agents: List[str],
        deadlock_threshold: int = 3,
    ) -> bool:
        """
        Returns True if deadlock declared, False otherwise.
        Broadcasts DEADLOCK_DECLARED if threshold crossed.
        """
        self.state.round_number += 1
        progress = self._compute_progress(current_allocations)
        self.state.progress_scores.append(progress)
        self.state.previous_allocations = current_allocations

        if progress < self.state.threshold:
            self.state.stall_rounds += 1
        else:
            self.state.stall_rounds = 0

        if self.state.stall_rounds >= deadlock_threshold:
            await self._declare_deadlock(involved_agents)
            return True
        return False

    async def _declare_deadlock(self, involved_agents: List[str]) -> None:
        msg = DeadlockDeclaredMessage(
            sender_agent_id="convergence_checker",
            recipient_agent_id="broadcast",
            negotiation_id=self.state.negotiation_id,
            round_number=self.state.round_number,
            sequence_number=0,
            deadlock_members=involved_agents,
            n_rounds_stalled=self.state.stall_rounds,
        )
        await self.channel.broadcast(msg)

    def is_converging(self) -> bool:
        """True if recent trend shows improvement."""
        if len(self.state.progress_scores) < 3:
            return True
        recent = self.state.progress_scores[-3:]
        return any(p >= self.state.threshold for p in recent)
