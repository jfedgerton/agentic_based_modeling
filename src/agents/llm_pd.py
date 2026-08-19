"""LLM-based agent for the Prisoner's Dilemma Grid.

Supports three input modes — Calculator, Reasoner, Role-player — whose
prompt templates live in :mod:`src.prompts.pd_prompts`. The LLM emits
different surface action labels per mode (``A``/``B`` for Calculator,
``COOPERATE``/``DEFECT`` for Reasoner, ``OPEN``/``CLOSE`` for Role-player);
this agent translates them into the canonical ``COOPERATE``/``DEFECT``
action that the simulator uses.
"""

import json
from typing import Literal, Optional

from src.agents.base import AgentDecision, BaseAgent, parse_llm_response
from src.llm.provider import LLMProvider
from src.prompts import get_prompt

COOPERATE = "COOPERATE"
DEFECT = "DEFECT"

Mode = Literal["c" \
"alculator", "reasoner", "roleplayer"]

# Surface labels the LLM is asked to output in each mode.
_SURFACE_LABELS: dict[str, list[str]] = {
    "calculator": ["A", "B"],
    "reasoner": [COOPERATE, DEFECT],
    "roleplayer": ["OPEN", "CLOSE"],
}

# Surface (LLM output) → canonical (simulator state).
_SURFACE_TO_CANONICAL: dict[str, dict[str, str]] = {
    "calculator": {"A": COOPERATE, "B": DEFECT},
    "reasoner": {COOPERATE: COOPERATE, DEFECT: DEFECT},
    "roleplayer": {"OPEN": COOPERATE, "CLOSE": DEFECT},
}

# Canonical → surface, used when echoing prior actions back in the prompt.
_CANONICAL_TO_SURFACE: dict[str, dict[str, str]] = {
    mode: {can: surf for surf, can in mapping.items()}
    for mode, mapping in _SURFACE_TO_CANONICAL.items()
}


class LLMPDAgent(BaseAgent):
    """LLM-driven PD agent supporting Calculator / Reasoner / Role-player modes."""

    def __init__(self, model, llm_provider: LLMProvider, mode: Mode,
                 initial_action: str = COOPERATE):
        super().__init__(model, agent_type="llm")
        if mode not in _SURFACE_LABELS:
            raise ValueError(
                f"unknown mode: {mode!r}. Expected one of {list(_SURFACE_LABELS)}"
            )
        self.mode = mode
        self.llm = llm_provider
        self.action = initial_action
        self.next_action: Optional[str] = None
        self.payoff = 0.0

    def get_local_observation(self) -> dict:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        neighbor_actions = [n.action for n in neighbors
                            if hasattr(n, "action")]

        # Find the action of the neighbor with the highest payoff (canonical).
        # Returns "none" when no payoff history exists (e.g. step 1).
        neighbors_with_payoff = [n for n in neighbors
                                  if hasattr(n, "action") and hasattr(n, "payoff")]
        if neighbors_with_payoff:
            best_neighbor = max(neighbors_with_payoff, key=lambda n: n.payoff)
            best_neighbor_strategy = best_neighbor.action  # canonical
        else:
            best_neighbor_strategy = "none"

        return {
            "my_action": self.action,  # canonical; mapped to surface in _build_prompt
            "my_payoff": self.payoff,
            "num_neighbors": len(neighbor_actions),
            "coop_count": neighbor_actions.count(COOPERATE),
            "defect_count": neighbor_actions.count(DEFECT),
            "coop_rate": (
                neighbor_actions.count(COOPERATE) / len(neighbor_actions)
                if neighbor_actions else 0.0
            ),
            "strategy_of_neighbor_with_highest_payoff": best_neighbor_strategy,
            "x": self.pos[0] if self.pos else 0,
            "y": self.pos[1] if self.pos else 0,
        }

    def _build_prompt(self, obs: dict) -> str:
        # Render canonical labels as surface labels for this mode so the prompt
        # uses the same vocabulary it asks the LLM to produce.
        surface_obs = dict(obs)
        surface_obs["my_action"] = _CANONICAL_TO_SURFACE[self.mode][self.action]
        raw_best = obs["strategy_of_neighbor_with_highest_payoff"]
        surface_obs["strategy_of_neighbor_with_highest_payoff"] = (
            "none" if raw_best == "none"
            else _CANONICAL_TO_SURFACE[self.mode][raw_best]
        )
        template = get_prompt("pd", self.mode)
        return template.format(**surface_obs)

    def decide(self):
        """Query the LLM for a decision."""
        obs = self.get_local_observation()
        prompt = self._build_prompt(obs)

        raw_response = self.llm.query(prompt)

        try:
            decision = parse_llm_response(raw_response, _SURFACE_LABELS[self.mode])
            # Translate the LLM's surface label to the canonical action so
            # downstream code and analytics never see mode-specific vocab.
            canonical = _SURFACE_TO_CANONICAL[self.mode][decision.action]
            decision.action = canonical
            self.next_action = canonical
        except (ValueError, json.JSONDecodeError) as e:
            if hasattr(self.model, "logger") and self.model.logger:
                self.model.logger.log_parse_failure(
                    step=self.model.schedule_step if hasattr(self.model, "schedule_step") else -1,
                    agent_id=self.unique_id,
                    raw_response=raw_response,
                    error=str(e),
                )
            decision = AgentDecision(
                observed_state_summary="Parse failure, using fallback.",
                beliefs={},
                action=self.action,
                confidence=0.0,
                short_rationale=f"LLM parse failure: {e}",
            )
            self.next_action = self.action

        if hasattr(self.model, "logger") and self.model.logger:
            self.model.logger.log_llm_interaction(
                step=self.model.schedule_step if hasattr(self.model, "schedule_step") else -1,
                agent_id=self.unique_id,
                prompt=prompt,
                raw_response=raw_response,
                parsed=decision.to_dict(),
            )

        self.record_decision(decision)

    def advance(self):
        """Apply the decided action."""
        if self.next_action is not None:
            self.action = self.next_action
            self.next_action = None
