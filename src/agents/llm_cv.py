"""LLM-based citizen agent for the Epstein Civil Violence model.

Supports three input modes — Calculator, Reasoner, Role-player — whose
prompt templates live in :mod:`src.prompts.cv_prompts`. The LLM emits
different surface action labels per mode (``A``/``B`` for Calculator,
``QUIET``/``ACTIVE`` for Reasoner, ``STAY_HOME``/``JOIN_PROTEST`` for
Role-player); this agent translates them into the canonical ``QUIET`` /
``ACTIVE`` state used by the simulator.

The prompt language (``en`` / ``zh``) is read from the model
(:attr:`CivilViolenceModel.language`), since it is a run-level IV shared
by all agents in a simulation.
"""

import json
import math
from typing import Literal, Optional

from src.agents.base import AgentDecision, BaseAgent, parse_llm_response
from src.agents.classic_cv import ClassicCopAgent, QUIET, ACTIVE, JAILED
from src.llm.provider import LLMProvider
from src.prompts import get_prompt

Mode = Literal["calculator", "reasoner", "roleplayer"]

# Surface labels the LLM is asked to output for each mode.
_SURFACE_LABELS: dict[str, list[str]] = {
    "calculator": ["A", "B"],
    "reasoner": [QUIET, ACTIVE],
    "roleplayer": ["STAY_HOME", "JOIN_PROTEST"],
}

# Surface (LLM output) → canonical (simulator state).
_SURFACE_TO_CANONICAL: dict[str, dict[str, str]] = {
    "calculator": {"A": QUIET, "B": ACTIVE},
    "reasoner": {QUIET: QUIET, ACTIVE: ACTIVE},
    "roleplayer": {"STAY_HOME": QUIET, "JOIN_PROTEST": ACTIVE},
}

# How to render the agent's current state in the prompt, per mode.
# Includes JAILED since the simulator may show it during the no-op jail step.
_STATE_RENDER: dict[str, dict[str, str]] = {
    "calculator": {QUIET: "0", ACTIVE: "1", JAILED: "2"},
    "reasoner": {QUIET: "QUIET", ACTIVE: "ACTIVE", JAILED: "JAILED"},
    "roleplayer": {QUIET: "STAY_HOME", ACTIVE: "PROTESTING", JAILED: "DETAINED"},
}


class LLMCitizenAgent(BaseAgent):
    """LLM-driven citizen in the Civil Violence model.

    On LLM parse failure, falls back to the classical Epstein rule
    (``rebel if grievance − risk_aversion × arrest_prob > threshold``).
    """

    def __init__(self, model, llm_provider: LLMProvider, mode: Mode,
                 hardship: float, regime_legitimacy: float,
                 risk_aversion: float, threshold: float = 0.1,
                 vision: int = 7):
        super().__init__(model, agent_type="llm_citizen")
        if mode not in _SURFACE_LABELS:
            raise ValueError(
                f"unknown mode: {mode!r}. Expected one of {list(_SURFACE_LABELS)}"
            )
        self.mode = mode
        self.llm = llm_provider
        self.hardship = hardship
        self.regime_legitimacy = regime_legitimacy
        self.risk_aversion = risk_aversion
        self.threshold = threshold
        self.vision = vision
        self.state = QUIET
        self.jail_term = 0

    @property
    def grievance(self) -> float:
        return self.hardship * (1.0 - self.regime_legitimacy)

    def _estimated_arrest_prob(self) -> float:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.vision)
        cops_nearby = sum(1 for n in neighbors
                          if isinstance(n, ClassicCopAgent)
                          or getattr(n, "agent_type", "") == "classic_cop")
        actives_nearby = sum(
            1 for n in neighbors
            if hasattr(n, "state") and n.state == ACTIVE
        )
        actives_nearby_plus = 1 + actives_nearby
        if cops_nearby == 0:
            return 0.0
        return 1.0 - math.exp(-2.3 * cops_nearby / actives_nearby_plus)

    def get_local_observation(self) -> dict:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.vision)
        cops = sum(1 for n in neighbors
                   if isinstance(n, ClassicCopAgent)
                   or getattr(n, "agent_type", "") == "classic_cop")
        actives = sum(1 for n in neighbors
                      if hasattr(n, "state") and n.state == ACTIVE)
        quiets = sum(1 for n in neighbors
                     if hasattr(n, "state") and n.state == QUIET)
        return {
            "grievance": self.grievance,
            "hardship": self.hardship,
            "legitimacy": self.regime_legitimacy,
            "risk_aversion": self.risk_aversion,
            "cops_nearby": cops,
            "actives_nearby": actives,
            "quiets_nearby": quiets,
            "arrest_prob": self._estimated_arrest_prob(),
            "state": self.state,  # canonical; rendered per-mode in _build_prompt
            "max_jail_term": self.model.max_jail_term,
            "jail_term": self.jail_term,
            "vision": self.vision,
            "vision_diameter": 2 * self.vision + 1,
            "observable_cells": (2 * self.vision + 1) ** 2 - 1,
        }

    def _build_prompt(self, obs: dict) -> str:
        # Render state as the surface label for this mode.
        rendered_obs = dict(obs)
        rendered_obs["state"] = _STATE_RENDER[self.mode][self.state]
        template = get_prompt("cv", self.mode, language=self.model.language)
        return template.format(**rendered_obs)

    def step(self):
        """LLM-driven citizen decision and movement."""
        if self.jail_term > 0:
            self.jail_term -= 1
            if self.jail_term == 0:
                self.state = QUIET
            return

        obs = self.get_local_observation()
        prompt = self._build_prompt(obs)
        raw_response = self.llm.query(prompt)

        try:
            decision = parse_llm_response(raw_response, _SURFACE_LABELS[self.mode])
            canonical = _SURFACE_TO_CANONICAL[self.mode][decision.action]
            decision.action = canonical
            self.state = canonical
        except (ValueError, json.JSONDecodeError) as e:
            if hasattr(self.model, "logger") and self.model.logger:
                self.model.logger.log_parse_failure(
                    step=getattr(self.model, "schedule_step", -1),
                    agent_id=self.unique_id,
                    raw_response=raw_response,
                    error=str(e),
                )
            # Fallback: classical Epstein rule
            arrest_prob = self._estimated_arrest_prob()
            net_risk = self.risk_aversion * arrest_prob
            if self.grievance - net_risk > self.threshold:
                self.state = ACTIVE
            else:
                self.state = QUIET
            decision = AgentDecision(
                observed_state_summary="Parse failure, using classical fallback.",
                beliefs={},
                action=self.state,
                confidence=0.0,
                short_rationale=f"LLM parse failure: {e}",
            )

        if hasattr(self.model, "logger") and self.model.logger:
            self.model.logger.log_llm_interaction(
                step=getattr(self.model, "schedule_step", -1),
                agent_id=self.unique_id,
                prompt=prompt,
                raw_response=raw_response,
                parsed=decision.to_dict(),
            )

        self.record_decision(decision)

        # Move
        if self.model.movement:
            neighborhood = self.model.grid.get_neighborhood(
                self.pos, moore=True, radius=self.vision)
            empty_cells = [c for c in neighborhood
                           if self.model.grid.is_cell_empty(c)]
            if empty_cells:
                new_pos = self.random.choice(empty_cells)
                self.model.grid.move_agent(self, new_pos)
