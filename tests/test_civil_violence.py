"""Tests for the Civil Violence model."""

import pytest

from src.models.civil_violence import CivilViolenceModel
from src.llm.provider import MockProvider


class TestCivilViolenceClassic:
    def test_model_creation(self):
        model = CivilViolenceModel(
            width=10, height=10, citizen_density=0.5,
            cop_density=0.04, agent_type="classic", seed=42,
        )
        agents = list(model.agents)
        assert len(agents) > 0

    def test_step_runs(self):
        model = CivilViolenceModel(
            width=10, height=10, citizen_density=0.5,
            cop_density=0.04, agent_type="classic", seed=42,
        )
        model.step()
        model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert len(data) == 2

    def test_rebellion_rate_bounded(self):
        model = CivilViolenceModel(
            width=10, height=10, citizen_density=0.7,
            cop_density=0.04, agent_type="classic", seed=42,
        )
        for _ in range(5):
            model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert all(0.0 <= r <= 1.0 for r in data["rebellion_rate"])


class TestCivilViolenceLLM:
    def test_llm_model_runs(self):
        import json
        provider = MockProvider(game="cv", mode="reasoner", rate_limit_delay=0)
        # Set up mock responses for civil violence
        provider.set_response("citizen", json.dumps({
            "observed_state_summary": "Mock citizen observation",
            "beliefs": {"expected_neighbor_behavior": "quiet",
                        "risk_assessment": "high"},
            "action": "QUIET",
            "confidence": 0.6,
            "short_rationale": "Too risky to rebel.",
        }))
        model = CivilViolenceModel(
            width=5, height=5, citizen_density=0.5,
            cop_density=0.04, agent_type="llm", mode="reasoner",
            llm_provider=provider, seed=42,
        )
        model.step()


    def test_cops_can_arrest_llm_citizens(self):
        import json
        from src.agents.classic_cv import ClassicCopAgent, JAILED

        provider = MockProvider(game="cv", mode="reasoner", rate_limit_delay=0)
        # Override default to force ACTIVE so we can test that cops arrest them.
        # The substring must appear in the CV Reasoner prompt body.
        provider.set_response("Choosing ACTIVE", json.dumps({
            "observed_state_summary": "Mock citizen observation",
            "beliefs": {"expected_neighbor_behavior": "active",
                        "risk_assessment": "low"},
            "action": "ACTIVE",
            "confidence": 0.9,
            "short_rationale": "I will rebel.",
        }))

        model = CivilViolenceModel(
            width=5, height=5, citizen_density=0.3, cop_density=0.3,
            citizen_vision=7, cop_vision=7, movement=False,
            agent_type="llm", mode="reasoner",
            llm_provider=provider, seed=42,
        )

        model.step()

        jailed_count = sum(
            1 for a in model.agents
            if hasattr(a, "state") and a.state == JAILED
        )
        assert jailed_count > 0

        for cop in [a for a in model.agents if isinstance(a, ClassicCopAgent)]:
            obs = cop.get_local_observation()
            assert "actives_nearby" in obs

    def test_requires_provider(self):
        with pytest.raises(ValueError, match="LLM provider required"):
            CivilViolenceModel(
                width=5, height=5, agent_type="llm", seed=42,
            )
