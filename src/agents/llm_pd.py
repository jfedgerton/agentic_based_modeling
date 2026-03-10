"""LLM-based agent for the Prisoner's Dilemma Grid.

Uses an LLM to choose between COOPERATE and DEFECT based on the
same local information available to the classic agent.
"""

import json
from typing import Optional

from src.agents.base import AgentDecision, BaseAgent, parse_llm_response
from src.llm.provider import LLMProvider

COOPERATE = "COOPERATE"
DEFECT = "DEFECT"
VALID_ACTIONS = [COOPERATE, DEFECT]

PD_SYSTEM_PROMPT = """You are an agent in a spatial Prisoner's Dilemma game on a grid.
Each round, you choose to COOPERATE or DEFECT.
You play against all your immediate neighbors simultaneously.

Payoff structure:
- Both cooperate: {cc} each
- You cooperate, they defect: {cd} for you, {dc} for them
- You defect, they cooperate: {dc} for you, {cd} for them
- Both defect: {dd} each

Your goal is to maximize your total payoff over time.

You must respond with ONLY a JSON object in this exact format:
{{
  "observed_state_summary": "<brief description of what you observe>",
  "beliefs": {{
    "expected_neighbor_behavior": "<your expectation>",
    "risk_assessment": "<low/medium/high>"
  }},
  "action": "COOPERATE" or "DEFECT",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}"""

PD_DECISION_PROMPT = """Current situation:
- Your previous action: {my_action}
- Your previous payoff: {my_payoff}
- Number of neighbors: {num_neighbors}
- Neighbors who cooperated: {coop_count} ({coop_rate:.0%})
- Neighbors who defected: {defect_count}
- Your position: ({x}, {y})

Choose your action for this round."""


class LLMPDAgent(BaseAgent):
    """LLM-driven PD agent."""

    def __init__(self, model, llm_provider: LLMProvider,
                 initial_action: str = COOPERATE):
        super().__init__(model, agent_type="llm")
        self.llm = llm_provider
        self.action = initial_action
        self.next_action: Optional[str] = None
        self.payoff = 0.0

    def get_local_observation(self) -> dict:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        neighbor_actions = [n.action for n in neighbors
                           if hasattr(n, "action")]
        return {
            "my_action": self.action,
            "my_payoff": self.payoff,
            "num_neighbors": len(neighbor_actions),
            "coop_count": neighbor_actions.count(COOPERATE),
            "defect_count": neighbor_actions.count(DEFECT),
            "coop_rate": (
                neighbor_actions.count(COOPERATE) / len(neighbor_actions)
                if neighbor_actions else 0.0
            ),
            "x": self.pos[0] if self.pos else 0,
            "y": self.pos[1] if self.pos else 0,
        }

    def _build_prompt(self, obs: dict) -> str:
        payoffs = self.model.payoff_matrix
        system = PD_SYSTEM_PROMPT.format(
            cc=payoffs["CC"], cd=payoffs["CD"],
            dc=payoffs["DC"], dd=payoffs["DD"],
        )
        decision = PD_DECISION_PROMPT.format(**obs)
        return f"{system}\n\n{decision}"

    def decide(self):
        """Query the LLM for a decision."""
        obs = self.get_local_observation()
        prompt = self._build_prompt(obs)

        raw_response = self.llm.query(prompt)

        try:
            decision = parse_llm_response(raw_response, VALID_ACTIONS)
            self.next_action = decision.action
        except (ValueError, json.JSONDecodeError) as e:
            # Log parse failure, fall back to previous action
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

        # Log the LLM interaction
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
