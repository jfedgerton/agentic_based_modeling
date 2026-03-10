"""LLM-based citizen agent for the Epstein Civil Violence model.

Uses an LLM to decide whether to remain QUIET or become ACTIVE,
based on the same local information available to the classic agent.
"""

import json
from typing import Optional

from src.agents.base import AgentDecision, BaseAgent, parse_llm_response
from src.agents.classic_cv import ClassicCitizenAgent, QUIET, ACTIVE, JAILED
from src.llm.provider import LLMProvider

VALID_ACTIONS = [QUIET, ACTIVE]

CV_SYSTEM_PROMPT = """You are a citizen in a society with a political regime.
You have a level of hardship and perceive the regime's legitimacy.
Each round, you decide whether to remain QUIET or become ACTIVE (rebel).

If you rebel (ACTIVE), you may be arrested by nearby cops and jailed.
Your decision factors:
- Grievance = hardship * (1 - regime_legitimacy)
- Risk = estimated probability of arrest if you rebel
- More cops nearby = higher arrest risk
- More active rebels nearby = lower individual arrest risk (safety in numbers)

You must respond with ONLY a JSON object:
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

CV_DECISION_PROMPT = """Your current situation:
- Your hardship level: {hardship:.2f} (0=none, 1=extreme)
- Regime legitimacy: {legitimacy:.2f} (0=illegitimate, 1=fully legitimate)
- Your grievance: {grievance:.2f}
- Your risk aversion: {risk_aversion:.2f}
- Cops visible nearby: {cops_nearby}
- Active rebels nearby: {actives_nearby}
- Quiet citizens nearby: {quiets_nearby}
- Estimated arrest probability: {arrest_prob:.2f}
- Your current state: {state}

Choose your action for this round."""


class LLMCitizenAgent(BaseAgent):
    """LLM-driven citizen in the Civil Violence model."""

    def __init__(self, model, llm_provider: LLMProvider,
                 hardship: float, regime_legitimacy: float,
                 risk_aversion: float, threshold: float = 0.1,
                 vision: int = 7):
        super().__init__(model, agent_type="llm_citizen")
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
        import math
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
            "jail_term": self.jail_term,
        }

    def step(self):
        """LLM-driven citizen decision and movement."""
        if self.jail_term > 0:
            self.jail_term -= 1
            if self.jail_term == 0:
                self.state = QUIET
            return

        obs = self.get_local_observation()
        prompt = f"{CV_SYSTEM_PROMPT}\n\n{CV_DECISION_PROMPT.format(**obs)}"

        raw_response = self.llm.query(prompt)

        try:
            decision = parse_llm_response(raw_response, VALID_ACTIONS)
            self.state = decision.action
        except (ValueError, json.JSONDecodeError) as e:
            if hasattr(self.model, "logger") and self.model.logger:
                self.model.logger.log_parse_failure(
                    step=getattr(self.model, "schedule_step", -1),
                    agent_id=self.unique_id,
                    raw_response=raw_response,
                    error=str(e),
                )
            # Fallback: use classical rule
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
