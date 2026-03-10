"""Classic rule-based agent for the Prisoner's Dilemma Grid.

Implements the canonical strategy: adopt the action of the neighbor
(or self) that received the highest payoff in the previous round.
This replicates standard ABM behavior for baseline comparison.
"""

from typing import Optional

from src.agents.base import AgentDecision, BaseAgent

# Actions
COOPERATE = "COOPERATE"
DEFECT = "DEFECT"


class ClassicPDAgent(BaseAgent):
    """Rule-based PD agent using best-neighbor imitation."""

    def __init__(self, model, initial_action: str = COOPERATE):
        super().__init__(model, agent_type="classic")
        self.action = initial_action
        self.next_action: Optional[str] = None
        self.payoff = 0.0

    def get_local_observation(self) -> dict:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        neighbor_actions = [n.action for n in neighbors
                           if hasattr(n, "action")]
        neighbor_payoffs = [n.payoff for n in neighbors
                            if hasattr(n, "payoff")]
        return {
            "my_action": self.action,
            "my_payoff": self.payoff,
            "neighbor_actions": neighbor_actions,
            "neighbor_payoffs": neighbor_payoffs,
            "cooperation_rate": (
                neighbor_actions.count(COOPERATE) / len(neighbor_actions)
                if neighbor_actions else 0.0
            ),
        }

    def decide(self):
        """Determine next action by imitating the highest-payoff neighbor."""
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=True)

        # Find the agent (self or neighbor) with the highest payoff
        best_agent = max(
            [a for a in neighbors if hasattr(a, "payoff")],
            key=lambda a: a.payoff,
            default=self,
        )
        self.next_action = best_agent.action

        obs = self.get_local_observation()
        decision = AgentDecision(
            observed_state_summary=(
                f"Payoff={self.payoff:.1f}, "
                f"neighborhood cooperation={obs['cooperation_rate']:.2f}"
            ),
            beliefs={
                "expected_neighbor_behavior": (
                    "mostly cooperate"
                    if obs["cooperation_rate"] > 0.5
                    else "mostly defect"
                ),
                "risk_assessment": "low" if self.payoff >= 3 else "high",
            },
            action=self.next_action,
            confidence=1.0,
            short_rationale=(
                f"Imitating best-performing neighbor "
                f"(payoff={best_agent.payoff:.1f})."
            ),
        )
        self.record_decision(decision)

    def advance(self):
        """Apply the decided action."""
        if self.next_action is not None:
            self.action = self.next_action
            self.next_action = None
