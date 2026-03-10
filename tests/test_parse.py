"""Tests for LLM response parsing."""

import json
import pytest

from src.agents.base import parse_llm_response, AgentDecision


class TestParseLLMResponse:
    def test_valid_json(self):
        response = json.dumps({
            "observed_state_summary": "I see neighbors",
            "beliefs": {"expected_neighbor_behavior": "cooperate",
                        "risk_assessment": "low"},
            "action": "COOPERATE",
            "confidence": 0.8,
            "short_rationale": "Neighbors are cooperating.",
        })
        decision = parse_llm_response(response, ["COOPERATE", "DEFECT"])
        assert decision.action == "COOPERATE"
        assert decision.confidence == 0.8

    def test_markdown_wrapped_json(self):
        response = '```json\n{"action": "DEFECT", "confidence": 0.5, "observed_state_summary": "x", "beliefs": {}, "short_rationale": "y"}\n```'
        decision = parse_llm_response(response, ["COOPERATE", "DEFECT"])
        assert decision.action == "DEFECT"

    def test_lowercase_action_normalized(self):
        response = json.dumps({
            "action": "cooperate",
            "confidence": 0.5,
            "observed_state_summary": "",
            "beliefs": {},
            "short_rationale": "",
        })
        decision = parse_llm_response(response, ["COOPERATE", "DEFECT"])
        assert decision.action == "COOPERATE"

    def test_invalid_action_raises(self):
        response = json.dumps({"action": "INVALID", "confidence": 0.5,
                                "observed_state_summary": "", "beliefs": {},
                                "short_rationale": ""})
        with pytest.raises(ValueError, match="Invalid action"):
            parse_llm_response(response, ["COOPERATE", "DEFECT"])

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_response("not json at all", ["COOPERATE", "DEFECT"])


class TestAgentDecision:
    def test_roundtrip(self):
        d = AgentDecision(
            observed_state_summary="test",
            beliefs={"key": "value"},
            action="COOPERATE",
            confidence=0.75,
            short_rationale="reason",
        )
        d2 = AgentDecision.from_dict(d.to_dict())
        assert d2.action == d.action
        assert d2.confidence == d.confidence
