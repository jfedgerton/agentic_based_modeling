"""Tests for the Fearon Bargaining Grid model."""

import pytest

from src.models.interstate_conflict import (
    InterstateConflictModel,
    truthful_signal,
)
from src.llm.provider import MockProvider


class TestInterstateConflictClassic:
    def test_model_creation(self):
        model = InterstateConflictModel(width=5, height=5, agent_type="classic", seed=42)
        assert len(list(model.agents)) == 25

    def test_step_runs(self):
        model = InterstateConflictModel(width=5, height=5, agent_type="classic", seed=42)
        model.step()
        model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert len(data) == 2

    def test_bluffing_disabled_forces_truthful_signals(self):
        model = InterstateConflictModel(width=5, height=5, agent_type="classic",
                                bluffing_enabled=False, seed=42)
        model.step()
        for agent in model.agents:
            assert agent.signal == truthful_signal(agent.true_capability)

    def test_proposer_alternates_with_step_parity(self):
        model = InterstateConflictModel(width=5, height=5, agent_type="classic", seed=42)
        a, b = (1, 2), (3, 4)  # a is lex-lower than b
        model.schedule_step = 0
        assert model.proposer_pos_of(a, b) == a
        model.schedule_step = 1
        assert model.proposer_pos_of(a, b) == b


class TestInterstateConflictLLM:
    def test_llm_model_creation(self):
        provider = MockProvider(rate_limit_delay=0)
        model = InterstateConflictModel(width=3, height=3, agent_type="llm",
                                llm_provider=provider, seed=42)
        assert len(list(model.agents)) == 9

    def test_llm_step_runs(self):
        provider = MockProvider(rate_limit_delay=0)
        model = InterstateConflictModel(width=3, height=3, agent_type="llm",
                                llm_provider=provider, seed=42)
        model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert len(data) == 1

    def test_requires_provider(self):
        with pytest.raises(ValueError, match="LLM provider required"):
            InterstateConflictModel(width=3, height=3, agent_type="llm", seed=42)


class TestInterstateConflictHybrid:
    def test_hybrid_model_creation_and_step(self):
        provider = MockProvider(rate_limit_delay=0)
        model = InterstateConflictModel(width=3, height=3, agent_type="hybrid",
                                llm_provider=provider, seed=42)
        assert len(list(model.agents)) == 9
        model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert len(data) == 1
