from typing import Callable, Dict, List, Any, Optional
from protocol.message import OfferMessage, CounterOfferMessage


class Agent:
    def __init__(
        self,
        agent_id: str,
        negotiation_id: str,
        role: str,
        utility_function: Callable[[Dict[str, float]], float],
        reservation_value: float,
        target_value: float,
        max_rounds: int = 10,
        hint_strategy: str = "honest",  # honest | inflate | deflate | adaptive
        trust_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        # ── Identity ────────────────────────────────────────────────────────
        self.agent_id = agent_id
        self.negotiation_id = negotiation_id
        self.role = role
        self.utility_function = utility_function
        self.reservation_value = reservation_value
        self.target_value = target_value
        self.max_rounds = max_rounds
        self.hint_strategy = hint_strategy

        # ── Position tracking ────────────────────────────────────────────────
        self.opening_ask: float = target_value
        self.my_current_allocation_ask: float = target_value
        self.concession_budget: float = target_value - reservation_value
        self.concessions_made: float = 0.0
        self._concession_history: List[float] = []  # per-round concession deltas

        # ── Round tracking ───────────────────────────────────────────────────
        self.current_round: int = 0
        self.rounds_without_progress: int = 0
        self.last_progress_round: int = 0

        # ── Offer history ────────────────────────────────────────────────────
        self.offer_history: List[OfferMessage] = []
        self.current_offers: Dict[str, OfferMessage] = {}  # counterpart_id -> latest offer

        # ── Counterpart modeling ─────────────────────────────────────────────
        self.counterpart_offers_received: List = []
        self.counterpart_concession_history: List[float] = []
        self.counterpart_rejection_history: List[float] = []
        self.inferred_counterpart_utility: float = 0.5

        # ── Trust ────────────────────────────────────────────────────────────
        self.trust_profiles: Dict[str, Dict[str, Any]] = trust_profiles or {}

        # ── Hint tracking ────────────────────────────────────────────────────
        self.hint_sent: Optional[float] = None
        self.true_utility_for_current_offer: Optional[float] = None

        # ── Coalition ────────────────────────────────────────────────────────
        self.in_coalition: bool = False
        self.coalition_id: Optional[str] = None
        self.coalition_members: List[str] = []
        self.pending_coalition_proposals: List = []

        # ── Outcome ──────────────────────────────────────────────────────────
        self.final_agreement: Optional[Dict] = None
        self.final_utility_achieved: Optional[float] = None
        self.negotiation_outcome: Optional[str] = None
        self.rounds_to_agreement: Optional[int] = None

        # ── Multi-session history (for RQ1/RQ3 — trust calibration over time) ──
        self.session_outcomes: List[Dict[str, Any]] = []

    # ── Core evaluation ──────────────────────────────────────────────────────

    def evaluate_offer(self, offer: OfferMessage) -> float:
        return self.utility_function(offer.proposed_allocation)

    def should_accept(self, offer: OfferMessage) -> bool:
        return self.evaluate_offer(offer) >= self.reservation_value

    @property
    def deadline_pressure(self) -> float:
        """0.0 = round 1, 1.0 = final round."""
        return self.current_round / max(self.max_rounds, 1)

    @property
    def concession_rate_per_round(self) -> float:
        """Average concession made per round so far. 0 if no rounds elapsed."""
        if not self._concession_history:
            return 0.0
        return sum(self._concession_history) / len(self._concession_history)

    # ── Hint strategy ────────────────────────────────────────────────────────

    def compute_hint(self, true_utility: float) -> float:
        if self.hint_strategy == "honest":
            return true_utility
        elif self.hint_strategy == "inflate":
            return min(1.0, true_utility + 0.2)
        elif self.hint_strategy == "deflate":
            return max(0.0, true_utility - 0.2)
        elif self.hint_strategy == "adaptive":
            # deflate early, honest near deadline
            if self.deadline_pressure > 0.7:
                return true_utility
            return max(0.0, true_utility - 0.15)
        return true_utility

    # ── Position tracking ────────────────────────────────────────────────────

    def record_concession(self, new_ask: float) -> None:
        delta = self.my_current_allocation_ask - new_ask
        if delta > 0:
            self.concessions_made += delta
            self._concession_history.append(delta)
            self.my_current_allocation_ask = new_ask

    def remaining_concession_budget(self) -> float:
        return max(0.0, self.concession_budget - self.concessions_made)

    # ── Counterpart modeling ─────────────────────────────────────────────────

    def update_counterpart_model(self, offer: OfferMessage) -> None:
        self.counterpart_offers_received.append(offer)
        if len(self.counterpart_offers_received) > 1:
            prev = self.counterpart_offers_received[-2]
            prev_ask = prev.proposed_allocation.get(offer.sender_agent_id, 0)
            curr_ask = offer.proposed_allocation.get(offer.sender_agent_id, 0)
            concession = prev_ask - curr_ask
            self.counterpart_concession_history.append(concession)

    def compute_behavioral_trust_signals(self, counterpart_id: str) -> Dict[str, float]:
        """
        Compute normalized trust signals for updating hint_inflation_score.
        Returns dict with keys: velocity_signal, rejection_signal, timing_signal
        All values in [0, 1] where 1 = strong evidence of lying.

        BUG FIX: was incorrectly passing self.agent_id — now correctly uses
        counterpart_id to observe the counterpart's concession behavior.
        """
        # Velocity signal: fast concession = suspicious (counterpart had more room)
        if len(self.counterpart_concession_history) == 0:
            velocity_signal = 0.5  # no data
        else:
            avg_concession = sum(self.counterpart_concession_history) / len(self.counterpart_concession_history)
            # Normalize: >0.1 per round is fast
            velocity_signal = min(1.0, avg_concession / 0.1)

        # Rejection contradiction: did they reject a better offer?
        rejection_signal = 1.0 if len(self.counterpart_rejection_history) > 0 else 0.0

        # Timing signal: accepted early = had more room than hinted
        if self.rounds_to_agreement and self.max_rounds:
            timing_signal = 1.0 - (self.rounds_to_agreement / self.max_rounds)
        else:
            timing_signal = 0.5

        return {
            "velocity_signal": round(velocity_signal, 4),
            "rejection_signal": round(rejection_signal, 4),
            "timing_signal": round(timing_signal, 4),
        }

    def compute_hint_error(self, stated_hint: float, accepted_allocation_value: float) -> float:
        """
        Compute how much the stated hint understated true utility.
        Uses behavioral signals + allocation proxy floor.
        """
        signals = self.compute_behavioral_trust_signals(self.agent_id)
        behavioral_estimate = (
            0.40 * signals["velocity_signal"]
            + 0.35 * signals["rejection_signal"]
            + 0.25 * signals["timing_signal"]
        )
        # Allocation proxy: accepted allocation is a floor on true utility
        inferred_utility = max(accepted_allocation_value, behavioral_estimate)
        return round(inferred_utility - stated_hint, 4)

    # ── Trust ────────────────────────────────────────────────────────────────

    def get_trust_profile(self, counterpart_id: str) -> Dict[str, Any]:
        return self.trust_profiles.get(
            counterpart_id,
            {
                "hint_inflation_score": 0.3,
                "sessions_observed": 0,
                "reliability": "none",
            },
        )

    def update_trust_profile(self, counterpart_id: str, profile_data: Dict[str, Any]) -> None:
        self.trust_profiles[counterpart_id] = profile_data

    def update_hint_inflation_score(
        self,
        counterpart_id: str,
        hint_error: float,
        alpha: float = 0.3,
    ) -> None:
        """
        EMA update of hint_inflation_score for a counterpart.
        hint_error > 0 → counterpart overstated utility → increase inflation score.
        hint_error < 0 → counterpart understated (deflated) → decrease score.
        alpha: EMA weight for new observation (0.1 = slow update, 0.5 = fast update).
        """
        profile = self.get_trust_profile(counterpart_id)
        current_score = profile.get("hint_inflation_score", 0.3)
        # Normalize hint_error to [0, 1] — clip extreme values
        normalized_error = max(0.0, min(1.0, (hint_error + 0.5)))
        new_score = round((1 - alpha) * current_score + alpha * normalized_error, 4)

        sessions = profile.get("sessions_observed", 0) + 1
        reliability = (
            "high" if sessions >= 5 and new_score < 0.25
            else "low" if sessions >= 5 and new_score > 0.65
            else "medium" if sessions >= 3
            else "none"
        )

        self.trust_profiles[counterpart_id] = {
            **profile,
            "hint_inflation_score": new_score,
            "sessions_observed": sessions,
            "reliability": reliability,
            "last_hint_error": hint_error,
        }

    # ── Outcome recording ────────────────────────────────────────────────────

    def record_outcome(
        self,
        outcome: str,  # "agreement" | "coalition" | "breakdown"
        final_allocation: Optional[Dict] = None,
        rounds_taken: Optional[int] = None,
    ) -> None:
        self.negotiation_outcome = outcome
        self.final_agreement = final_allocation
        self.rounds_to_agreement = rounds_taken
        if final_allocation:
            self.final_utility_achieved = self.utility_function(final_allocation)

        # Append to session history for multi-session research tracking
        self.session_outcomes.append({
            "negotiation_id": self.negotiation_id,
            "outcome": outcome,
            "final_allocation": final_allocation,
            "rounds_taken": rounds_taken,
            "final_utility": self.final_utility_achieved,
            "hint_strategy": self.hint_strategy,
            "concession_rate": self.concession_rate_per_round,
        })

    def to_checkpoint(self) -> Dict[str, Any]:
        """Serializable snapshot of key state for Redis checkpoint."""
        return {
            "agent_id": self.agent_id,
            "negotiation_id": self.negotiation_id,
            "current_round": self.current_round,
            "my_current_allocation_ask": self.my_current_allocation_ask,
            "concessions_made": self.concessions_made,
            "concession_rate_per_round": self.concession_rate_per_round,
            "in_coalition": self.in_coalition,
            "negotiation_outcome": self.negotiation_outcome,
            "final_utility_achieved": self.final_utility_achieved,
            "hint_strategy": self.hint_strategy,
            "trust_profiles": self.trust_profiles,
        }
