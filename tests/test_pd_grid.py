"""Tests for the Prisoner's Dilemma Grid model."""

import pytest

from src.models.pd_grid import PDGridModel
from src.llm.provider import MockProvider


class TestPDGridClassic:
    def test_model_creation(self):
        model = PDGridModel(width=5, height=5, agent_type="classic", seed=42)
        assert len(list(model.agents)) == 25

    def test_step_runs(self):
        model = PDGridModel(width=5, height=5, agent_type="classic", seed=42)
        model.step()
        model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert len(data) == 2

    def test_cooperation_rate_bounded(self):
        model = PDGridModel(width=10, height=10, agent_type="classic", seed=42)
        for _ in range(10):
            model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert all(0.0 <= r <= 1.0 for r in data["cooperation_rate"])

    def test_payoff_computation(self):
        model = PDGridModel(width=3, height=3, agent_type="classic",
                            initial_cooperation_prob=1.0, seed=42)
        model._compute_payoffs()
        # All cooperate: each agent gets 3 * num_neighbors
        for agent in model.agents:
            neighbors = model.grid.get_neighbors(
                agent.pos, moore=True, include_center=False)
            expected = 3 * len(neighbors)
            assert agent.payoff == expected


class TestPDGridLLM:
    def test_llm_model_creation(self):
        provider = MockProvider(rate_limit_delay=0)
        model = PDGridModel(width=3, height=3, agent_type="llm",
                            llm_provider=provider, seed=42)
        assert len(list(model.agents)) == 9

    def test_llm_step_runs(self):
        provider = MockProvider(rate_limit_delay=0)
        model = PDGridModel(width=3, height=3, agent_type="llm",
                            llm_provider=provider, seed=42)
        model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert len(data) == 1

    def test_requires_provider(self):
        with pytest.raises(ValueError, match="LLM provider required"):
            PDGridModel(width=3, height=3, agent_type="llm", seed=42)


class TestPDGridHybrid:
    def test_hybrid_model_creation(self):
        provider = MockProvider(rate_limit_delay=0)
        model = PDGridModel(width=3, height=3, agent_type="hybrid",
                            llm_provider=provider, seed=42)
        assert len(list(model.agents)) == 9

    def test_hybrid_step_runs(self):
        provider = MockProvider(rate_limit_delay=0)
        model = PDGridModel(width=3, height=3, agent_type="hybrid",
                            llm_provider=provider, seed=42)
        model.step()
        model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert len(data) == 2
