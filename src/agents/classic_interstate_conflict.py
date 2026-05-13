"""Classic rule-based agent for the Fearon Bargaining Grid.

Implements the closed-form decision rules from Fearon (1995) §pp.387 and
§pp.394:
    Responder threshold:  x* = 1 - perceived_P(self wins) + self.war_cost
    Proposer demand:      x  = round_to_bucket(estimated_opp_threshold)

Signals are honest unless `model.bluffing_enabled` is True and the agent is
weak (capability < 0.5), in which case it signals STRONG with probability
`model.bluffing_rate`. Strong agents are always honest.
"""

from typing import Optional

from src.agents.base import AgentDecision, BaseAgent

# --- Constants (keep in sync with src/models/interstate_conflict.py) ---

STRONG = "STRONG"
MODERATE = "MODERATE"
WEAK = "WEAK"
SIGNAL_TO_CAP = {STRONG: 0.75, MODERATE: 0.50, WEAK: 0.25}
DEMAND_BUCKETS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _truthful_signal(capability: float) -> str:
    if capability > 0.66:
        return STRONG
    if capability > 0.33:
        return MODERATE
    return WEAK


def _round_to_bucket(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return min(DEMAND_BUCKETS, key=lambda b: abs(b - x))


class ClassicInterstateConflictAgent(BaseAgent):
    """Rule-based Fearon bargaining agent."""

    def __init__(self, model, true_capability: float, true_war_cost: float):
        super().__init__(model, agent_type="classic_interstate_conflict")
        self.true_capability = true_capability
        self.true_war_cost = true_war_cost
        self.signal: Optional[str] = None
        # neighbor_unique_id -> demand x (filled when this agent is proposer)
        self.demands: dict = {}
        # neighbor_unique_id -> threshold x* (filled when this agent is responder)
        self.thresholds: dict = {}
        self.payoff = 0.0

    def get_local_observation(self) -> dict:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        return {
            "my_true_capability": self.true_capability,
            "my_true_war_cost": self.true_war_cost,
            "my_signal": self.signal,
            "neighbors": [
                {
                    "id": n.unique_id,
                    "pos": n.pos,
                    "signal": getattr(n, "signal", None),
                }
                for n in neighbors
            ],
        }

    # --- Phase 1: signal broadcast ---

    def broadcast_signal(self):
        if not self.model.bluffing_enabled:
            self.signal = _truthful_signal(self.true_capability)
            return
        if (self.true_capability < 0.5
                and self.random.random() < self.model.bluffing_rate):
            self.signal = STRONG
        else:
            self.signal = _truthful_signal(self.true_capability)

    # --- Phase 2: per-neighbor demand or threshold ---

    def decide(self):
        self.demands = {}
        self.thresholds = {}
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)

        for neighbor in neighbors:
            proposer_pos = self.model.proposer_pos_of(self.pos, neighbor.pos)
            if proposer_pos == self.pos:
                self.demands[neighbor.unique_id] = self._compute_demand(neighbor)
            else:
                self.thresholds[neighbor.unique_id] = self._compute_threshold(neighbor)

        decision = AgentDecision(
            observed_state_summary=(
                f"cap={self.true_capability:.2f}, "
                f"war_cost={self.true_war_cost:.2f}, signal={self.signal}"
            ),
            beliefs={
                "demands_to_neighbors": dict(self.demands),
                "thresholds_to_neighbors": dict(self.thresholds),
            },
            action=self.signal,
            confidence=1.0,
            short_rationale=(
                "Fearon-rational baseline: honest signal (unless bluffing knob "
                "fires); demand up to estimated opponent threshold; accept iff "
                "incoming demand <= own reservation."
            ),
        )
        self.record_decision(decision)

    # --- Decision rule helpers ---

    def _compute_threshold(self, opponent) -> float:
        """Fearon §pp.387: x* = 1 - perceived_P(self wins) + self.war_cost."""
        opp_signal = getattr(opponent, "signal", MODERATE)
        perceived_cap_opp = SIGNAL_TO_CAP.get(opp_signal, 0.5)
        perceived_p_self_wins = self.true_capability / (
            self.true_capability + perceived_cap_opp
        )
        return 1.0 - perceived_p_self_wins + self.true_war_cost

    def _compute_demand(self, opponent) -> float:
        """Fearon §pp.394: x = 1 - cap_self / (cap_self + perceived_cap_opp)
                              + E[war_cost], rounded to nearest bucket.
        """
        opp_signal = getattr(opponent, "signal", MODERATE)
        perceived_cap_opp = SIGNAL_TO_CAP.get(opp_signal, 0.5)
        estimated_opp_threshold = (
            1.0
            - self.true_capability / (self.true_capability + perceived_cap_opp)
            + self.model.E_war_cost
        )
        return _round_to_bucket(estimated_opp_threshold)
