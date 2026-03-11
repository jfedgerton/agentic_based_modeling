"""Classic rule-based agents for the Epstein Civil Violence model.

Implements the canonical Epstein (2002) model:
- Citizens decide to rebel if grievance - risk > threshold
- Cops arrest active rebels in their neighborhood

This replicates the standard ABM behavior for baseline comparison.
"""

import math
from typing import Optional

from src.agents.base import AgentDecision, BaseAgent

QUIET = "QUIET"
ACTIVE = "ACTIVE"
JAILED = "JAILED"


class ClassicCitizenAgent(BaseAgent):
    """Rule-based citizen in the Civil Violence model."""

    def __init__(self, model, hardship: float, regime_legitimacy: float,
                 risk_aversion: float, threshold: float = 0.1,
                 vision: int = 7):
        super().__init__(model, agent_type="classic_citizen")
        self.hardship = hardship
        self.regime_legitimacy = regime_legitimacy
        self.risk_aversion = risk_aversion
        self.threshold = threshold
        self.vision = vision
        self.state = QUIET
        self.jail_term = 0

    @property
    def grievance(self) -> float:
        """Perceived grievance = hardship * (1 - legitimacy)."""
        return self.hardship * (1.0 - self.regime_legitimacy)

    def _estimated_arrest_prob(self) -> float:
        """Estimate arrest probability from local cop/active ratio."""
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.vision)
        cops_nearby = sum(1 for n in neighbors
                         if isinstance(n, ClassicCopAgent))
        actives_nearby = sum(1 for n in neighbors
                             if isinstance(n, ClassicCitizenAgent)
                             and n.state == ACTIVE)
        # Include self as potential active
        actives_nearby_plus = 1 + actives_nearby
        if cops_nearby == 0:
            return 0.0
        return 1.0 - math.exp(-2.3 * cops_nearby / actives_nearby_plus)

    def get_local_observation(self) -> dict:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.vision)
        cops = sum(1 for n in neighbors if isinstance(n, ClassicCopAgent))
        actives = sum(1 for n in neighbors
                      if isinstance(n, ClassicCitizenAgent)
                      and n.state == ACTIVE)
        quiets = sum(1 for n in neighbors
                     if isinstance(n, ClassicCitizenAgent)
                     and n.state == QUIET)
        return {
            "grievance": self.grievance,
            "hardship": self.hardship,
            "legitimacy": self.regime_legitimacy,
            "risk_aversion": self.risk_aversion,
            "cops_nearby": cops,
            "actives_nearby": actives,
            "quiets_nearby": quiets,
            "arrest_prob": self._estimated_arrest_prob(),
            "state": self.state,
            "jail_term": self.jail_term,
        }

    def step(self):
        """Citizen decision rule and movement."""
        # Handle jail
        if self.jail_term > 0:
            self.jail_term -= 1
            if self.jail_term == 0:
                self.state = QUIET
            return

        # Compute net risk
        arrest_prob = self._estimated_arrest_prob()
        net_risk = self.risk_aversion * arrest_prob

        # Decision rule: rebel if grievance - net_risk > threshold
        if self.grievance - net_risk > self.threshold:
            new_state = ACTIVE
        else:
            new_state = QUIET

        self.state = new_state

        decision = AgentDecision(
            observed_state_summary=(
                f"Grievance={self.grievance:.2f}, "
                f"arrest_prob={arrest_prob:.2f}"
            ),
            beliefs={
                "expected_neighbor_behavior": "uncertain",
                "risk_assessment": (
                    "high" if arrest_prob > 0.5 else "low"
                ),
            },
            action=new_state,
            confidence=1.0,
            short_rationale=(
                f"Grievance ({self.grievance:.2f}) - "
                f"risk ({net_risk:.2f}) "
                f"{'>' if new_state == ACTIVE else '<='} "
                f"threshold ({self.threshold})."
            ),
        )
        self.record_decision(decision)

        # Move to random empty cell in vision
        if self.model.movement:
            self._move()

    def _move(self):
        """Move to a random empty cell within vision."""
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision)
        empty_cells = [c for c in neighborhood if self.model.grid.is_cell_empty(c)]
        if empty_cells:
            new_pos = self.random.choice(empty_cells)
            self.model.grid.move_agent(self, new_pos)


class ClassicCopAgent(BaseAgent):
    """Rule-based cop in the Civil Violence model."""

    def __init__(self, model, vision: int = 7):
        super().__init__(model, agent_type="classic_cop")
        self.vision = vision

    def get_local_observation(self) -> dict:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.vision)
        actives = [n for n in neighbors
                   if self._is_active_citizen(n)]
        return {
            "actives_nearby": len(actives),
        }

    @staticmethod
    def _is_citizen(agent) -> bool:
        """Citizen-like agents expose the Civil Violence state attributes."""
        return hasattr(agent, "state") and hasattr(agent, "jail_term")

    @classmethod
    def _is_active_citizen(cls, agent) -> bool:
        return cls._is_citizen(agent) and agent.state == ACTIVE

    def step(self):
        """Arrest a random active citizen in vision, then move."""
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.vision)
        actives = [n for n in neighbors
                   if self._is_active_citizen(n)]

        if actives:
            target = self.random.choice(actives)
            target.state = JAILED
            target.jail_term = self.random.randint(1, self.model.max_jail_term)

        # Move to random empty cell
        if self.model.movement:
            neighborhood = self.model.grid.get_neighborhood(
                self.pos, moore=True, radius=self.vision)
            empty_cells = [c for c in neighborhood
                           if self.model.grid.is_cell_empty(c)]
            if empty_cells:
                new_pos = self.random.choice(empty_cells)
                self.model.grid.move_agent(self, new_pos)
