"""Classic rule-based juror for the Jury Deliberation model.

Implements a status-weighted DeGroot (1974) opinion update:
    new_belief = alpha * my_belief + (1 - alpha) * weighted_avg(neighbors' beliefs)
    weight_i  = social_status_i / sum(social_status_j)

The classic juror does NOT read the testimony text. Its initial belief is
0.5 + Gaussian noise; its argument is simply (vote, confidence, status) with
no natural-language content.
"""

from typing import Optional

from src.agents.base import AgentDecision, BaseAgent

# Constants (keep in sync with src/models/jury_deliberation.py)
GUILTY = "GUILTY"
NOT_GUILTY = "NOT_GUILTY"


def _vote_of(belief: float) -> str:
    return GUILTY if belief > 0.5 else NOT_GUILTY


def _confidence_of(belief: float) -> float:
    return abs(belief - 0.5) * 2.0


class ClassicJurorAgent(BaseAgent):
    """Status-weighted DeGroot juror with noise-based belief initialization."""

    def __init__(self, model, social_status: float, noise_sigma: float = 0.15):
        super().__init__(model, agent_type="classic_juror")
        self.social_status = social_status
        self.noise_sigma = noise_sigma
        self.belief: float = 0.5  # placeholder; set in initialize_belief()
        self.current_argument: Optional[dict] = None
        # Buffer filled in listen() and consumed in update_belief().
        self._observations: list = []

    # --- Required by BaseAgent ---

    def get_local_observation(self) -> dict:
        return {
            "my_belief": self.belief,
            "my_social_status": self.social_status,
            "my_vote": _vote_of(self.belief),
            "neighbors": list(self._observations),
        }

    # --- Round 0: form initial belief ---

    def initialize_belief(self):
        """Classic agent does not read testimony; uses 0.5 + Gaussian noise."""
        noise = self.random.gauss(0.0, self.noise_sigma)
        self.belief = max(0.0, min(1.0, 0.5 + noise))

    # --- Phase 1: speak ---

    def speak(self):
        """Emit (vote, confidence, status) tuple as this round's argument."""
        self.current_argument = {
            "vote": _vote_of(self.belief),
            "confidence": _confidence_of(self.belief),
            "social_status": self.social_status,
        }

    # --- Phase 2: listen ---

    def listen(self):
        """Snapshot every other juror's (belief, status) before any update happens."""
        self._observations = []
        for a in self.model.agents:
            if a is self:
                continue
            if not hasattr(a, "belief") or not hasattr(a, "social_status"):
                continue
            self._observations.append({
                "id": a.unique_id,
                "belief": float(a.belief),
                "social_status": float(a.social_status),
                "vote": _vote_of(a.belief),
                "confidence": _confidence_of(a.belief),
                "argument": getattr(a, "current_argument", None),
            })

    # --- Phase 3: update ---

    def update_belief(self):
        """Apply status-weighted DeGroot update.

        Uses the snapshot taken in listen(), so all agents update from the
        same pre-update belief state regardless of iteration order.
        """
        old_belief = self.belief
        alpha = self.model.mixing_alpha

        if not self._observations:
            return

        total_status = sum(o["social_status"] for o in self._observations)
        if total_status <= 0:
            weighted_avg = 0.5
        else:
            weighted_avg = sum(
                o["belief"] * o["social_status"] for o in self._observations
            ) / total_status

        new_belief = alpha * old_belief + (1.0 - alpha) * weighted_avg
        self.belief = max(0.0, min(1.0, new_belief))

        decision = AgentDecision(
            observed_state_summary=(
                f"belief={old_belief:.2f}, status={self.social_status:.2f}, "
                f"vote={_vote_of(old_belief)}"
            ),
            beliefs={
                "n_neighbors": len(self._observations),
                "weighted_avg_neighbor_belief": float(weighted_avg),
                "alpha": float(alpha),
                "old_belief": float(old_belief),
                "new_belief": float(self.belief),
            },
            action=_vote_of(self.belief),
            confidence=_confidence_of(self.belief),
            short_rationale=(
                f"DeGroot status-weighted update: {alpha:.2f} * self + "
                f"{1 - alpha:.2f} * status-weighted neighbors."
            ),
        )
        self.record_decision(decision)
