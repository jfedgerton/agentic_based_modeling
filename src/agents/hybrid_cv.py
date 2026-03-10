"""Hybrid citizen agent for the Epstein Civil Violence model.

LLM interprets the environment and provides a recommendation,
but the final decision is constrained by the classical rule structure.
"""

import json
import math
from typing import Optional

from src.agents.base import AgentDecision, BaseAgent, parse_llm_response
from src.agents.classic_cv import QUIET, ACTIVE, JAILED
from src.llm.provider import LLMProvider

VALID_ACTIONS = [QUIET, ACTIVE]

HYBRID_CV_PROMPT = """You are advising a citizen in a political simulation.
Analyze the situation and recommend whether the citizen should remain QUIET or become ACTIVE (rebel).

The citizen's situation:
- Hardship: {hardship:.2f} (0=none, 1=extreme)
- Regime legitimacy: {legitimacy:.2f}
- Grievance: {grievance:.2f}
- Risk aversion: {risk_aversion:.2f}
- Cops nearby: {cops_nearby}
- Active rebels nearby: {actives_nearby}
- Quiet citizens nearby: {quiets_nearby}
- Estimated arrest probability: {arrest_prob:.2f}
- Classical rule would choose: {rule_action}

Respond with ONLY a JSON object:
{{
  "observed_state_summary": "<what you observe>",
  "beliefs": {{
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  }},
  "action": "QUIET" or "ACTIVE",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}"""


class HybridCitizenAgent(BaseAgent):
    """Hybrid citizen: LLM advises, classical rules constrain."""

    def __init__(self, model, llm_provider: LLMProvider,
                 hardship: float, regime_legitimacy: float,
                 risk_aversion: float, threshold: float = 0.1,
                 vision: int = 7, override_threshold: float = 0.8):
        super().__init__(model, agent_type="hybrid_citizen")
        self.llm = llm_provider
        self.hardship = hardship
        self.regime_legitimacy = regime_legitimacy
        self.risk_aversion = risk_aversion
        self.threshold = threshold
        self.vision = vision
        self.override_threshold = override_threshold
        self.state = QUIET
        self.jail_term = 0

    @property
    def grievance(self) -> float:
        return self.hardship * (1.0 - self.regime_legitimacy)

    def _estimated_arrest_prob(self) -> float:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.vision)
        from src.agents.classic_cv import ClassicCopAgent
        cops_nearby = sum(1 for n in neighbors
                         if isinstance(n, (ClassicCopAgent,))
                         or getattr(n, 'agent_type', '') == 'classic_cop')
        actives_nearby = sum(
            1 for n in neighbors
            if hasattr(n, 'state') and n.state == ACTIVE
        )
        actives_nearby_plus = 1 + actives_nearby
        if cops_nearby == 0:
            return 0.0
        return 1.0 - math.exp(-2.3 * cops_nearby / actives_nearby_plus)

    def _get_rule_action(self) -> str:
        """Classical Epstein rule."""
        arrest_prob = self._estimated_arrest_prob()
        net_risk = self.risk_aversion * arrest_prob
        if self.grievance - net_risk > self.threshold:
            return ACTIVE
        return QUIET

    def get_local_observation(self) -> dict:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.vision)
        from src.agents.classic_cv import ClassicCopAgent
        cops = sum(1 for n in neighbors
                   if isinstance(n, (ClassicCopAgent,))
                   or getattr(n, 'agent_type', '') == 'classic_cop')
        actives = sum(1 for n in neighbors
                      if hasattr(n, 'state') and n.state == ACTIVE)
        quiets = sum(1 for n in neighbors
                     if hasattr(n, 'state') and n.state == QUIET)
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
        }

    def step(self):
        if self.jail_term > 0:
            self.jail_term -= 1
            if self.jail_term == 0:
                self.state = QUIET
            return

        obs = self.get_local_observation()
        rule_action = self._get_rule_action()

        prompt = HYBRID_CV_PROMPT.format(rule_action=rule_action, **obs)
        raw_response = self.llm.query(prompt)

        try:
            llm_decision = parse_llm_response(raw_response, VALID_ACTIONS)
            llm_action = llm_decision.action
            llm_confidence = llm_decision.confidence
            llm_rationale = llm_decision.short_rationale
        except (ValueError, json.JSONDecodeError) as e:
            if hasattr(self.model, "logger") and self.model.logger:
                self.model.logger.log_parse_failure(
                    step=getattr(self.model, "schedule_step", -1),
                    agent_id=self.unique_id,
                    raw_response=raw_response,
                    error=str(e),
                )
            llm_action = rule_action
            llm_confidence = 0.0
            llm_rationale = f"Parse failure: {e}"

        if llm_confidence >= self.override_threshold:
            self.state = llm_action
            source = "llm"
        else:
            self.state = rule_action
            source = "rule"

        decision = AgentDecision(
            observed_state_summary=(
                f"Rule: {rule_action}, LLM: {llm_action}."
            ),
            beliefs={
                "expected_neighbor_behavior": "uncertain",
                "risk_assessment": (
                    "high" if obs["arrest_prob"] > 0.5 else "low"
                ),
                "decision_source": source,
                "llm_confidence": llm_confidence,
            },
            action=self.state,
            confidence=llm_confidence if source == "llm" else 1.0,
            short_rationale=f"Source: {source}. LLM: {llm_rationale}",
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

        if self.model.movement:
            neighborhood = self.model.grid.get_neighborhood(
                self.pos, moore=True, radius=self.vision)
            empty_cells = [c for c in neighborhood
                           if self.model.grid.is_cell_empty(c)]
            if empty_cells:
                new_pos = self.random.choice(empty_cells)
                self.model.grid.move_agent(self, new_pos)
