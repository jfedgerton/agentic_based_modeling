"""Hybrid agent for the Prisoner's Dilemma Grid.

The LLM interprets the local environment and ranks possible actions,
but the final decision is constrained by the rule structure:
the agent selects among the LLM's top-ranked actions only if they
are consistent with a bounded rationality filter.
"""

import json
from typing import Optional

from src.agents.base import AgentDecision, BaseAgent, parse_llm_response
from src.llm.provider import LLMProvider

COOPERATE = "COOPERATE"
DEFECT = "DEFECT"
VALID_ACTIONS = [COOPERATE, DEFECT]

HYBRID_PROMPT = """You are an advisor for an agent in a spatial Prisoner's Dilemma game.
Analyze the situation and recommend an action. The agent will consider your
recommendation but may override it based on payoff rules.

Payoff structure:
- Both cooperate: {cc} each
- You cooperate, they defect: {cd} for you, {dc} for them
- You defect, they cooperate: {dc} for you, {cd} for them
- Both defect: {dd} each

Current situation:
- Agent's previous action: {my_action}
- Agent's previous payoff: {my_payoff}
- Neighbors who cooperated: {coop_count}/{num_neighbors} ({coop_rate:.0%})
- Best neighbor payoff: {best_neighbor_payoff}
- Best neighbor action: {best_neighbor_action}

Respond with ONLY a JSON object:
{{
  "observed_state_summary": "<what you observe>",
  "beliefs": {{
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  }},
  "action": "COOPERATE" or "DEFECT",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}"""


class HybridPDAgent(BaseAgent):
    """Hybrid PD agent: LLM advises, rules constrain."""

    def __init__(self, model, llm_provider: LLMProvider,
                 initial_action: str = COOPERATE,
                 override_threshold: float = 0.8):
        super().__init__(model, agent_type="hybrid")
        self.llm = llm_provider
        self.action = initial_action
        self.next_action: Optional[str] = None
        self.payoff = 0.0
        self.override_threshold = override_threshold

    def get_local_observation(self) -> dict:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        pd_neighbors = [n for n in neighbors if hasattr(n, "payoff")]
        neighbor_actions = [n.action for n in pd_neighbors]
        best_neighbor = max(pd_neighbors, key=lambda a: a.payoff,
                            default=None)
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
            "best_neighbor_payoff": best_neighbor.payoff if best_neighbor else 0.0,
            "best_neighbor_action": best_neighbor.action if best_neighbor else COOPERATE,
        }

    def _get_rule_action(self, obs: dict) -> str:
        """Classical rule: imitate best neighbor."""
        return obs["best_neighbor_action"]

    def decide(self):
        """Get LLM recommendation, apply rule-based constraint."""
        obs = self.get_local_observation()
        payoffs = self.model.payoff_matrix

        prompt = HYBRID_PROMPT.format(
            cc=payoffs["CC"], cd=payoffs["CD"],
            dc=payoffs["DC"], dd=payoffs["DD"],
            **obs,
        )

        raw_response = self.llm.query(prompt)
        rule_action = self._get_rule_action(obs)

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
            llm_rationale = f"Parse failure, using rule: {e}"

        # Hybrid logic: use LLM recommendation only if confidence is high
        # enough; otherwise fall back to the classical rule
        if llm_confidence >= self.override_threshold:
            self.next_action = llm_action
            source = "llm"
        else:
            self.next_action = rule_action
            source = "rule"

        decision = AgentDecision(
            observed_state_summary=f"Rule suggests {rule_action}, LLM suggests {llm_action}.",
            beliefs={
                "expected_neighbor_behavior": (
                    "mostly cooperate" if obs["coop_rate"] > 0.5
                    else "mostly defect"
                ),
                "risk_assessment": "low" if self.payoff >= 3 else "high",
                "decision_source": source,
                "llm_confidence": llm_confidence,
            },
            action=self.next_action,
            confidence=llm_confidence if source == "llm" else 1.0,
            short_rationale=(
                f"Source: {source}. "
                f"LLM: {llm_rationale}"
            ),
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

    def advance(self):
        """Apply the decided action."""
        if self.next_action is not None:
            self.action = self.next_action
            self.next_action = None
