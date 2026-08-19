"""Tests for the Prisoner's Dilemma Grid model."""

import pandas as pd
import pytest

from src.models.pd_grid import PDGridModel
from src.llm.provider import MockProvider
from src.utils.agent_panel import PD_PANEL_SCHEMA, AgentPanelWriter


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
        provider = MockProvider(game="pd", mode="reasoner", rate_limit_delay=0)
        model = PDGridModel(width=3, height=3, agent_type="llm", mode="reasoner",
                            llm_provider=provider, seed=42)
        assert len(list(model.agents)) == 9

    def test_llm_step_runs(self):
        provider = MockProvider(game="pd", mode="reasoner", rate_limit_delay=0)
        model = PDGridModel(width=3, height=3, agent_type="llm", mode="reasoner",
                            llm_provider=provider, seed=42)
        model.step()
        data = model.datacollector.get_model_vars_dataframe()
        assert len(data) == 1

    def test_requires_provider(self):
        with pytest.raises(ValueError, match="LLM provider required"):
            PDGridModel(width=3, height=3, agent_type="llm", seed=42)


class TestPDAgentPanel:
    @staticmethod
    def _run(tmp_path, width=5, height=5, steps=4):
        panel = AgentPanelWriter(tmp_path / "panel", schema=PD_PANEL_SCHEMA)
        model = PDGridModel(width=width, height=height, agent_type="classic",
                            panel_writer=panel, seed=42)
        for _ in range(steps):
            model.step()
        panel.close()
        return model, pd.read_parquet(panel.path)

    def test_panel_is_off_by_default(self):
        model = PDGridModel(width=5, height=5, agent_type="classic", seed=42)
        model.step()
        assert model.panel.enabled is False

    def test_one_row_per_agent_per_step(self, tmp_path):
        _, panel = self._run(tmp_path, width=5, height=5, steps=4)
        assert len(panel) == 25 * 4
        assert (panel.groupby("step").size() == 25).all()

    def test_agents_never_move(self, tmp_path):
        """PD places agents once; a moving agent would break the id-to-cell map."""
        _, panel = self._run(tmp_path, steps=4)
        assert (panel.groupby("agent_id")[["x", "y"]].nunique() == 1).all().all()

    def test_unique_id_maps_to_cell(self, tmp_path):
        """Agents are created row-major, so id determines the cell exactly."""
        height = 5
        _, panel = self._run(tmp_path, width=5, height=height, steps=1)
        first = panel.drop_duplicates("agent_id")
        assert (first["x"] == (first["agent_id"] - 1) // height).all()
        assert (first["y"] == (first["agent_id"] - 1) % height).all()

    def test_action_before_matches_previous_action(self, tmp_path):
        """decide() records the action it is replacing, advance() applies it."""
        _, panel = self._run(tmp_path, steps=4)
        p = panel.sort_values(["agent_id", "step"])
        prev_action = p.groupby("agent_id")["action"].shift(1)
        comparable = prev_action.notna()
        assert comparable.sum() > 0
        assert (p.loc[comparable, "action_before"] == prev_action[comparable]).all()

    def test_neighbour_counts_sum_to_neighbour_total(self, tmp_path):
        _, panel = self._run(tmp_path, steps=2)
        assert (panel["coop_count"] + panel["defect_count"]
                == panel["num_neighbors"]).all()
