import asyncio
from typing import List, Dict, Any, TypedDict, Optional
from langsmith import traceable
from langgraph.graph import StateGraph, END
from protocol.channel import Channel
from protocol.rounds import RoundManager, RoundState, AgentStates
from protocol.message import (
    NegotiationFailedMessage, NegotiationCompleteMessage,
    MessageType, OfferMessage, CounterOfferMessage,
    AcceptMessage, RejectMessage
)
from core.agent import Agent
from core.convergence import ConvergenceChecker, ConvergenceState
from core.negotiation_policy import decide, PolicyAction
from core.coalition import CoalitionManager, CoalitionState
from uuid import uuid4


class NegotiationState(TypedDict):
    negotiation_id: str
    round_number: int
    agent_ids: List[str]
    current_allocations: Dict[str, Dict[str, float]]
    status: str
    result: Dict[str, Any]


class Orchestrator:
    def __init__(
        self,
        agent_ids: List[str],
        negotiation_id: str,
        resource_type: str,
        total_resource: float,
        channel: Channel,
    ):
        self.agent_ids = agent_ids
        self.negotiation_id = negotiation_id
        self.resource_type = resource_type
        self.total_resource = total_resource
        self.channel = channel
        self.round_manager: Optional[RoundManager] = None
        self.convergence_checker: Optional[ConvergenceChecker] = None

    def setup(self) -> None:
        for agent_id in self.agent_ids:
            self.channel.register_agent(agent_id)

        round_state = RoundState(
            round_number=0,
            negotiation_id=self.negotiation_id,
            moved_agents=[],
            rounds_without_progress=0,
            agent_states={aid: AgentStates.ACTIVE for aid in self.agent_ids},
        )
        self.round_manager = RoundManager(channel=self.channel, state=round_state)

        convergence_state = ConvergenceState(
            negotiation_id=self.negotiation_id,
            stall_limit=5,      # was 3
            threshold=0.01,     # was 0.02
        )
        self.convergence_checker = ConvergenceChecker(
            state=convergence_state, channel=self.channel
        )

    async def run_negotiation(
        self, agents: List[Agent], max_rounds: int = 10
    ) -> Dict[str, Any]:
        print(f"[ORCH] run_negotiation called with {len(agents)} agents")
        agents_by_id = {a.agent_id: a for a in agents}
        result: Dict[str, Any] = {}

        # Track latest offer per agent pair: (sender, recipient) -> OfferMessage
        active_offers: Dict[str, Any] = {}
        seq_counters: Dict[str, int] = {aid: 0 for aid in self.agent_ids}

        # ── Bootstrap: each agent sends opening offer to next agent ──────────
        for i, agent in enumerate(agents):
            counterpart_id = self.agent_ids[(i + 1) % len(self.agent_ids)]
            alloc = {
                agent.agent_id: agent.target_value,
                counterpart_id: round(self.total_resource - agent.target_value, 4),
            }
            true_utility = agent.utility_function(alloc)
            hint = agent.compute_hint(true_utility)
            seq_counters[agent.agent_id] += 1

            opening_offer = OfferMessage(
                sender_agent_id=agent.agent_id,
                recipient_agent_id=counterpart_id,
                negotiation_id=self.negotiation_id,
                round_number=0,
                sequence_number=seq_counters[agent.agent_id],
                proposed_allocation=alloc,
                resource_type=self.resource_type,
                total_resource=self.total_resource,
                offer_utility_hint=hint,
                concession_signal=False,
            )
            await self.channel.send(opening_offer)
            active_offers[f"{agent.agent_id}->{counterpart_id}"] = opening_offer

        # ── Main negotiation loop ─────────────────────────────────────────────
        current_allocations: Dict[str, Dict[str, float]] = {
            a.agent_id: {a.agent_id: a.target_value} for a in agents
        }

        for round_num in range(1, max_rounds + 1):
            self.round_manager.advance_round()
            round_had_agreement = False

            # Each agent processes ONE message from their inbox this round
            for agent in agents:
                agent.current_round = round_num
                msg = self.channel.receive_nowait(agent.agent_id)
                print(f"[ORCH] Round {round_num} Agent {agent.agent_id} msg: {msg.message_type if msg else None}")
                

                if msg is None:
                    continue

                # Skip non-negotiation messages
                if msg.message_type not in (
                    MessageType.OFFER,
                    MessageType.COUNTER_OFFER,
                    MessageType.ACCEPT,
                    MessageType.REJECT,
                ):
                    continue

                # Handle ACCEPT from counterpart
                if msg.message_type == MessageType.ACCEPT:
                    offer_key = f"{agent.agent_id}->{msg.sender_agent_id}"
                    accepted_offer = active_offers.get(offer_key)
                    if accepted_offer:
                        agreed_allocation = accepted_offer.proposed_allocation
                        agent.record_outcome("agreement", agreed_allocation, round_num)
                        complete_msg = NegotiationCompleteMessage(
                            sender_agent_id=agent.agent_id,
                            recipient_agent_id="broadcast",
                            negotiation_id=self.negotiation_id,
                            round_number=round_num,
                            sequence_number=0,
                            agreed_allocation=agreed_allocation,
                            members=self.agent_ids,
                            rounds_taken=round_num,
                            via_coalition=False,
                        )
                        await self.channel.broadcast(complete_msg)
                        result = {
                            "outcome": "agreement",
                            "allocation": agreed_allocation,
                            "rounds_taken": round_num,
                            "via_coalition": False,
                        }
                        round_had_agreement = True
                        break

                # Handle OFFER or COUNTER_OFFER — agent must decide
                if msg.message_type in (MessageType.OFFER, MessageType.COUNTER_OFFER):
                    agent.update_counterpart_model(msg)
                    print(f"[ORCH] Calling decide for {agent.agent_id}")
                    policy_out = await decide(agent=agent, offer=msg)
                    # Get policy decision from Gemini
                    policy_out = await decide(agent=agent, offer=msg)

                    if policy_out.action == PolicyAction.ACCEPT:
                        # Send ACCEPT back
                        seq_counters[agent.agent_id] += 1
                        accept_msg = AcceptMessage(
                            sender_agent_id=agent.agent_id,
                            recipient_agent_id=msg.sender_agent_id,
                            negotiation_id=self.negotiation_id,
                            round_number=round_num,
                            sequence_number=seq_counters[agent.agent_id],
                            accepted_offer_id=msg.message_id,
                        )
                        await self.channel.send(accept_msg)

                        # Record agreement from this agent's side
                        agreed_allocation = msg.proposed_allocation
                        agent.record_outcome("agreement", agreed_allocation, round_num)
                        result = {
                            "outcome": "agreement",
                            "allocation": agreed_allocation,
                            "rounds_taken": round_num,
                            "via_coalition": False,
                        }

                        # Broadcast complete
                        complete_msg = NegotiationCompleteMessage(
                            sender_agent_id=agent.agent_id,
                            recipient_agent_id="broadcast",
                            negotiation_id=self.negotiation_id,
                            round_number=round_num,
                            sequence_number=0,
                            agreed_allocation=agreed_allocation,
                            members=self.agent_ids,
                            rounds_taken=round_num,
                            via_coalition=False,
                        )
                        await self.channel.broadcast(complete_msg)
                        round_had_agreement = True
                        break

                    elif policy_out.action in (PolicyAction.COUNTER, PolicyAction.OFFER):
                        # Build counter offer
                        if policy_out.proposed_allocation:
                            new_alloc = policy_out.proposed_allocation
                        else:
                            # Step down by 5% from current ask
                            new_ask = max(
                                agent.reservation_value,
                                agent.my_current_allocation_ask - 0.05,
                            )
                            new_alloc = {
                                agent.agent_id: round(new_ask, 4),
                                msg.sender_agent_id: round(self.total_resource - new_ask, 4),
                            }

                        agent.record_concession(new_alloc.get(agent.agent_id, agent.my_current_allocation_ask))
                        self.round_manager.record_position_change(agent.agent_id)
                        current_allocations[agent.agent_id] = new_alloc

                        true_utility = agent.utility_function(new_alloc)
                        seq_counters[agent.agent_id] += 1

                        counter = CounterOfferMessage(
                            sender_agent_id=agent.agent_id,
                            recipient_agent_id=msg.sender_agent_id,
                            negotiation_id=self.negotiation_id,
                            round_number=round_num,
                            sequence_number=seq_counters[agent.agent_id],
                            in_response_to_message_id=msg.message_id,
                            proposed_allocation=new_alloc,
                            resource_type=self.resource_type,
                            total_resource=self.total_resource,
                            offer_utility_hint=agent.compute_hint(true_utility),
                            concession_signal=True,
                        )
                        await self.channel.send(counter)
                        offer_key = f"{agent.agent_id}->{msg.sender_agent_id}"
                        active_offers[offer_key] = counter

                    elif policy_out.action == PolicyAction.REJECT:
                        # On reject, always send a counter offer — never go silent
                        new_ask = max(
                            agent.reservation_value,
                            agent.my_current_allocation_ask - 0.05,
                        )
                        new_alloc = {
                            agent.agent_id: round(new_ask, 4),
                            msg.sender_agent_id: round(self.total_resource - new_ask, 4),
                        }
                        agent.record_concession(new_alloc.get(agent.agent_id, agent.my_current_allocation_ask))
                        self.round_manager.record_position_change(agent.agent_id)
                        current_allocations[agent.agent_id] = new_alloc
                        true_utility = agent.utility_function(new_alloc)
                        seq_counters[agent.agent_id] += 1
                        counter = CounterOfferMessage(
                            sender_agent_id=agent.agent_id,
                            recipient_agent_id=msg.sender_agent_id,
                            negotiation_id=self.negotiation_id,
                            round_number=round_num,
                            sequence_number=seq_counters[agent.agent_id],
                            in_response_to_message_id=msg.message_id,
                            proposed_allocation=new_alloc,
                            resource_type=self.resource_type,
                            total_resource=self.total_resource,
                            offer_utility_hint=agent.compute_hint(true_utility),
                            concession_signal=True,
                        )
                        await self.channel.send(counter)
                        offer_key = f"{agent.agent_id}->{msg.sender_agent_id}"
                        active_offers[offer_key] = counter
                        agent.counterpart_rejection_history.append(
                            msg.proposed_allocation.get(msg.sender_agent_id, 0)
                        )

            if round_had_agreement:
                break

            # Check convergence only after minimum 3 rounds of actual negotiation
            if round_num >= 5:
                deadlocked = await self.convergence_checker.check_and_escalate(
                    current_allocations=current_allocations,
                    involved_agents=self.agent_ids,
                    deadlock_threshold=3,
                )

                if deadlocked:
                    coalition_result = await self._try_coalition(
                        agents=agents,
                        round_num=round_num,
                        seq_counters=seq_counters,
                    )
                    if coalition_result:
                        result = coalition_result
                    else:
                        await self.channel.broadcast(
                            NegotiationFailedMessage(
                                sender_agent_id="orchestrator",
                                recipient_agent_id="broadcast",
                                negotiation_id=self.negotiation_id,
                                round_number=round_num,
                                sequence_number=0,
                                members=self.agent_ids,
                                failure_reason="Deadlock — coalition formation failed",
                            )
                        )
                        result = {"outcome": "breakdown", "round": round_num}
                    break

        else:
            # Max rounds exhausted without agreement
            await self.channel.broadcast(
                NegotiationFailedMessage(
                    sender_agent_id="orchestrator",
                    recipient_agent_id="broadcast",
                    negotiation_id=self.negotiation_id,
                    round_number=max_rounds,
                    sequence_number=0,
                    members=self.agent_ids,
                    failure_reason="Max rounds reached without agreement",
                )
            )
            result = {"outcome": "breakdown", "reason": "max_rounds", "round": max_rounds}

        return result

    async def _try_coalition(
        self,
        agents: List[Agent],
        round_num: int,
        seq_counters: Dict[str, int],
    ) -> Dict[str, Any]:
        equal_alloc = {
            aid: round(self.total_resource / len(self.agent_ids), 4)
            for aid in self.agent_ids
        }
        proposer = agents[0]

        coalition_state = CoalitionState(
            proposer_id=proposer.agent_id,
            proposed_members=self.agent_ids,
            proposed_allocation=equal_alloc,
            round_number=round_num,
            negotiation_id=self.negotiation_id,
            resource_type=self.resource_type,
            total_resource=self.total_resource,
        )
        manager = CoalitionManager(state=coalition_state, channel=self.channel)
        await manager.propose()

        # Each non-proposer agent votes based on whether coalition beats their reservation
        for agent in agents[1:]:
            coalition_util = agent.utility_function(equal_alloc)
            accepts = coalition_util >= agent.reservation_value
            manager.record_vote(agent.agent_id, accepts)

        from core.coalition import CoalitionStatus
        outcome = manager.check_outcome()
        if outcome == CoalitionStatus.LOCKED:
            await manager.lock_coalition()
            return {
                "outcome": "coalition",
                "allocation": equal_alloc,
                "rounds_taken": round_num,
                "via_coalition": True,
            }
        return {}
