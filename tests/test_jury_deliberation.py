"""Tests for the Jury Deliberation model."""

import pytest

from src.models.jury_deliberation import (
    JuryDeliberationModel, GUILTY, NOT_GUILTY, HUNG,
)
from src.llm.provider import MockProvider


class TestJuryDeliberationClassic:
    def test_model_creation(self):
        model = JuryDeliberationModel(
            n_jurors=12, agent_type="classic", ground_truth=GUILTY, seed=42,
        )
        assert len(list(model.agents)) == 12
        # Initial beliefs should all be in [0, 1].
        for a in model.agents:
            assert 0.0 <= a.belief <= 1.0

    def test_step_runs_and_belief_bounded(self):
        model = JuryDeliberationModel(
            n_jurors=12, max_rounds=3, agent_type="classic",
            ground_truth=GUILTY, seed=42,
        )
        model.step()
        model.step()
        for a in model.agents:
            assert 0.0 <= a.belief <= 1.0
        data = model.datacollector.get_model_vars_dataframe()
        # Round 0 collection + 2 step collections = 3 rows.
        assert len(data) == 3

    def test_termination_sets_verdict(self):
        # With max_rounds=10 and 12 classic jurors, deliberation typically
        # converges. Run until terminated.
        model = JuryDeliberationModel(
            n_jurors=12, max_rounds=10, agent_type="classic",
            ground_truth=GUILTY, seed=42,
        )
        for _ in range(10):
            model.step()
            if model.is_terminated:
                break
        assert model.is_terminated
        assert model.verdict in (GUILTY, NOT_GUILTY, HUNG)

    def test_status_permutation_shuffles_labels(self):
        model_a = JuryDeliberationModel(
            n_jurors=12, agent_type="classic",
            ground_truth=GUILTY, status_permutation=False, seed=42,
        )
        model_b = JuryDeliberationModel(
            n_jurors=12, agent_type="classic",
            ground_truth=GUILTY, status_permutation=True, seed=42,
        )
        statuses_a = sorted(a.social_status for a in model_a.agents)
        statuses_b = sorted(a.social_status for a in model_b.agents)
        # Multisets of status values should match (permutation is a shuffle,
        # not a regeneration).
        for sa, sb in zip(statuses_a, statuses_b):
            assert abs(sa - sb) < 1e-9
        # But the ordering relative to juror id should differ for at least
        # one juror (otherwise permutation was a no-op).
        ordered_a = [a.social_status for a in model_a.agents]
        ordered_b = [a.social_status for a in model_b.agents]
        assert ordered_a != ordered_b


class TestJuryDeliberationLLM:
    def test_llm_model_creation(self):
        provider = MockProvider(rate_limit_delay=0)
        model = JuryDeliberationModel(
            n_jurors=12, agent_type="llm", llm_provider=provider,
            ground_truth=GUILTY, seed=42,
        )
        assert len(list(model.agents)) == 12

    def test_llm_step_runs(self):
        # Mock provider returns PD-style JSON, which will fail jury parsing
        # and fall back to classic behavior; the run should still complete.
        provider = MockProvider(rate_limit_delay=0)
        model = JuryDeliberationModel(
            n_jurors=12, max_rounds=2, agent_type="llm",
            llm_provider=provider, ground_truth=GUILTY, seed=42,
        )
        model.step()
        for a in model.agents:
            assert 0.0 <= a.belief <= 1.0

    def test_requires_provider(self):
        with pytest.raises(ValueError, match="LLM provider required"):
            JuryDeliberationModel(
                n_jurors=12, agent_type="llm",
                ground_truth=GUILTY, seed=42,
            )


class TestJuryDeliberationHybrid:
    def test_hybrid_model_creation_and_step(self):
        provider = MockProvider(rate_limit_delay=0)
        model = JuryDeliberationModel(
            n_jurors=12, max_rounds=2, agent_type="hybrid",
            llm_provider=provider, ground_truth=GUILTY, seed=42,
        )
        assert len(list(model.agents)) == 12
        model.step()
        for a in model.agents:
            assert 0.0 <= a.belief <= 1.0
