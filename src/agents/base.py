"""Base agent class and shared data structures.

All agent architectures (classic, LLM) inherit from BaseAgent,
ensuring a uniform interface for decision-making across models.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from mesa import Agent


@dataclass
class AgentDecision:
    """Structured decision output from any agent architecture.

    Mirrors the required JSON output format for LLM agents, but is used
    uniformly across all agent types for consistent logging and analysis.
    """
    observed_state_summary: str
    beliefs: dict
    action: str
    confidence: float
    short_rationale: str

    def to_dict(self) -> dict:
        return {
            "observed_state_summary": self.observed_state_summary,
            "beliefs": self.beliefs,
            "action": self.action,
            "confidence": self.confidence,
            "short_rationale": self.short_rationale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentDecision":
        return cls(
            observed_state_summary=d.get("observed_state_summary", ""),
            beliefs=d.get("beliefs", {}),
            action=d.get("action", ""),
            confidence=d.get("confidence", 0.0),
            short_rationale=d.get("short_rationale", ""),
        )


class BaseAgent(Agent):
    """Base class for all benchmark agents.

    Provides common infrastructure for decision logging and
    architecture-agnostic interfaces.
    """

    def __init__(self, model, agent_type: str = "base"):
        super().__init__(model)
        self.agent_type = agent_type
        self.decision_history: list[AgentDecision] = []
        self.last_decision: Optional[AgentDecision] = None

    def record_decision(self, decision: AgentDecision):
        """Store a decision for logging and analysis."""
        self.last_decision = decision
        self.decision_history.append(decision)

    def get_local_observation(self) -> dict:
        """Return the local information available to this agent.

        Subclasses must implement this. The observation must contain
        only information that a classical agent would have access to,
        ensuring fair comparison across architectures.
        """
        raise NotImplementedError


def parse_llm_response(raw_response: str, valid_actions: list[str]) -> AgentDecision:
    """Parse an LLM response string into an AgentDecision.

    Attempts to extract valid JSON from the response. If the action
    is not in valid_actions, raises ValueError.
    """
    # Try to find JSON in the response
    text = raw_response.strip()

    # Handle cases where the LLM wraps JSON in markdown code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    data = json.loads(text)

    action = data.get("action", "").upper()
    if action not in valid_actions:
        raise ValueError(
            f"Invalid action '{action}'. Valid actions: {valid_actions}")

    return AgentDecision.from_dict({**data, "action": action})
