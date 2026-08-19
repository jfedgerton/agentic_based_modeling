"""Tests for the Civil Violence model."""

import pandas as pd
import pytest

from src.models.civil_violence import CivilViolenceModel
from src.llm.provider import MockProvider
from src.utils.agent_panel import (
    CV_ARREST_SCHEMA,
    CV_PANEL_SCHEMA,
    AgentPanelWriter,
)


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


def _run_with_panel(tmp_path, steps=5, **kwargs):
    """Run a small classic model with panel collection on, return the frames."""
    panel = AgentPanelWriter(tmp_path / "panel", schema=CV_PANEL_SCHEMA)
    arrests = AgentPanelWriter(tmp_path / "arrests", schema=CV_ARREST_SCHEMA)
    model = CivilViolenceModel(
        width=10, height=10, citizen_density=0.5, cop_density=0.1,
        agent_type="classic", panel_writer=panel, arrest_writer=arrests,
        seed=42, **kwargs,
    )
    for _ in range(steps):
        model.step()
    panel.close()
    arrests.close()
    return model, pd.read_parquet(panel.path)


class TestAgentPanel:
    def test_panel_is_off_by_default(self):
        model = CivilViolenceModel(
            width=10, height=10, citizen_density=0.5,
            cop_density=0.04, agent_type="classic", seed=42,
        )
        model.step()
        assert model.panel.enabled is False
        # Agent reporters are collected only alongside the panel.
        assert model.datacollector.agent_reporters == {}

    def test_records_one_row_per_deciding_citizen_per_step(self, tmp_path):
        model, panel = _run_with_panel(tmp_path, steps=5)

        citizens = [a for a in model.agents if hasattr(a, "hardship")]
        # Jailed citizens skip their step, so rows never exceed the citizen
        # count and the shortfall equals the number of jailed-agent steps.
        assert panel["agent_id"].nunique() <= len(citizens)
        assert set(panel["step"]) == {1, 2, 3, 4, 5}
        assert (panel.groupby("step").size() <= len(citizens)).all()

    def test_decision_position_precedes_the_move(self, tmp_path):
        _, panel = _run_with_panel(tmp_path, steps=5)

        p = panel.sort_values(["agent_id", "step"])
        prev_x = p.groupby("agent_id")["moved_to_x"].shift(1)
        prev_y = p.groupby("agent_id")["moved_to_y"].shift(1)
        consecutive = p.groupby("agent_id")["step"].diff() == 1
        comparable = consecutive & prev_x.notna()

        assert comparable.sum() > 0
        assert (p.loc[comparable, "x"] == prev_x[comparable]).all()
        assert (p.loc[comparable, "y"] == prev_y[comparable]).all()

    def test_static_grid_keeps_agents_in_place(self, tmp_path):
        """With movement off, decision and post-move positions coincide."""
        _, panel = _run_with_panel(tmp_path, steps=3, movement=False)

        assert (panel["x"] == panel["moved_to_x"]).all()
        assert (panel["y"] == panel["moved_to_y"]).all()

    def test_activation_index_is_a_permutation_of_all_agents(self, tmp_path):
        model, panel = _run_with_panel(tmp_path, steps=3)

        n_agents = len(list(model.agents))
        per_step = panel.groupby("step")["activation_idx"]
        assert per_step.max().max() < n_agents
        # Indices identify position in the shuffled order, so no repeats.
        assert (per_step.nunique() == per_step.size()).all()

    def test_panel_does_not_enter_the_logger_records(self, tmp_path):
        """Panel rows must stay out of ExperimentLogger's in-memory list."""
        from src.utils.logging import ExperimentLogger

        logger = ExperimentLogger("t", "r", log_dir=str(tmp_path / "logs"))
        panel = AgentPanelWriter(tmp_path / "panel", schema=CV_PANEL_SCHEMA)
        model = CivilViolenceModel(
            width=10, height=10, citizen_density=0.5, cop_density=0.1,
            agent_type="classic", logger=logger, panel_writer=panel, seed=42,
        )
        for _ in range(3):
            model.step()
        panel.close()

        # One record per step, not one per agent per step.
        assert len(logger._records) == 3
