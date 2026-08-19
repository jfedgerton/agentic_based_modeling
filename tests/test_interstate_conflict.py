"""Tests for the Fearon Bargaining Grid model."""

import pandas as pd
import pytest

from src.models.interstate_conflict import (
    InterstateConflictModel,
    truthful_signal,
)
from src.llm.provider import MockProvider
from src.utils.agent_panel import (
    IC_DYAD_SCHEMA,
    AgentPanelWriter,
    flatten_dyad_outcome,
)


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
        provider = MockProvider(game="ic", mode="reasoner", rate_limit_delay=0)
        model = InterstateConflictModel(width=3, height=3, agent_type="llm",
                                mode="reasoner", llm_provider=provider, seed=42)
        assert len(list(model.agents)) == 9

    def test_llm_step_runs(self):
        provider = MockProvider(game="ic", mode="reasoner", rate_limit_delay=0)
        model = InterstateConflictModel(width=3, height=3, agent_type="llm",
                                mode="reasoner", llm_provider=provider, seed=42)
        model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert len(data) == 1

    def test_requires_provider(self):
        with pytest.raises(ValueError, match="LLM provider required"):
            InterstateConflictModel(width=3, height=3, agent_type="llm", seed=42)


class TestDyadPanel:
    @staticmethod
    def _run(tmp_path, width=5, height=5, steps=3):
        writer = AgentPanelWriter(tmp_path / "dyads", schema=IC_DYAD_SCHEMA)
        model = InterstateConflictModel(width=width, height=height,
                                        agent_type="classic",
                                        dyad_writer=writer, seed=42)
        for _ in range(steps):
            model.step()
        writer.close()
        return model, pd.read_parquet(writer.path)

    def test_panel_is_off_by_default(self):
        model = InterstateConflictModel(width=5, height=5,
                                        agent_type="classic", seed=42)
        model.step()
        assert model.dyads.enabled is False

    def test_every_dyad_of_every_step_is_written(self, tmp_path):
        model, dyads = self._run(tmp_path, width=5, height=5, steps=3)
        # A torus grid gives every cell 8 Moore neighbours; each undirected
        # dyad is resolved once, so 25 * 8 / 2 per step.
        assert (dyads.groupby("step").size() == 100).all()
        assert len(dyads) == model._dyads_this_step * 3

    def test_positions_are_split_into_columns(self, tmp_path):
        _, dyads = self._run(tmp_path, steps=2)
        for col in ("proposer_x", "proposer_y", "responder_x", "responder_y"):
            assert dyads[col].notna().all()
        assert "proposer_pos" not in dyads.columns

    def test_war_rows_have_a_winner_and_settlements_do_not(self, tmp_path):
        _, dyads = self._run(tmp_path, steps=3)
        assert dyads.loc[dyads["war"], "winner_id"].notna().all()
        assert dyads.loc[~dyads["war"], "winner_id"].isna().all()

    def test_settlement_payoffs_split_the_pie(self, tmp_path):
        _, dyads = self._run(tmp_path, steps=3)
        settled = dyads[~dyads["war"]]
        total = settled["proposer_payoff"] + settled["responder_payoff"]
        assert (total - 1.0).abs().max() < 1e-9

    def test_flatten_preserves_every_non_position_field(self):
        outcome = {
            "step": 7, "proposer_id": 3, "responder_id": 4,
            "proposer_pos": (1, 2), "responder_pos": (3, 4),
            "demand": 0.5, "war": False,
        }
        row = flatten_dyad_outcome(outcome)
        assert row["proposer_x"] == 1 and row["proposer_y"] == 2
        assert row["responder_x"] == 3 and row["responder_y"] == 4
        assert row["demand"] == 0.5 and row["war"] is False
        assert "proposer_pos" not in row
