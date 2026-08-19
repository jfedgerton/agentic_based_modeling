"""Tests for panel wiring in ExperimentRunner.

Covers the path a fresh run takes — the one used for conditions that have no
cached responses yet, where there is no original output to compare against.
"""

import pandas as pd
import pytest

from src.experiments.runner import ExperimentRunner

# Small grids and few steps: these exercise wiring, not model behaviour.
GRIDS = {
    "pd_grid": {"width": 4, "height": 4},
    "civil_violence": {"width": 6, "height": 6, "citizen_density": 0.5,
                       "cop_density": 0.1},
    "interstate_conflict": {"width": 4, "height": 4},
}


def _runner(tmp_path, arm="classic", panel=True, num_steps=2, **cv_overrides):
    grids = {game: dict(params) for game, params in GRIDS.items()}
    grids["civil_violence"].update(cv_overrides)
    overrides = {
        "experiment": {
            "name": "panel_test",
            "log_dir": str(tmp_path / "logs"),
            "output_dir": str(tmp_path / "out"),
            "num_steps": num_steps,
        },
        "agent": {"type": arm},
        "llm": {"provider": "mock"},
        "logging": {"log_agent_panel": panel},
        **grids,
    }
    return ExperimentRunner(config_overrides=overrides)


EXPECTED_TABLES = {
    "pd_grid": ["agent_panel.parquet"],
    "civil_violence": ["agent_panel.parquet", "arrests.parquet"],
    "interstate_conflict": ["dyads.parquet"],
}


@pytest.mark.parametrize("model_name", sorted(EXPECTED_TABLES))
def test_panel_files_land_in_the_run_directory(tmp_path, model_name):
    runner = _runner(tmp_path)
    result = getattr(runner, f"run_{model_name}")(seed=1)

    log_dir = tmp_path / "logs" / "panel_test" / result["run_id"]
    # An empty table writes no file, so only assert on the panel that must
    # have rows; arrests may legitimately be empty in two steps.
    primary = EXPECTED_TABLES[model_name][0]
    assert (log_dir / primary).exists()
    assert len(pd.read_parquet(log_dir / primary)) > 0


@pytest.mark.parametrize("model_name", sorted(EXPECTED_TABLES))
def test_panel_stats_are_reported_in_the_result(tmp_path, model_name):
    runner = _runner(tmp_path)
    result = getattr(runner, f"run_{model_name}")(seed=1)

    assert "panel_stats" in result
    rows = {arg: s["rows"] for arg, s in result["panel_stats"].items()}
    assert sum(rows.values()) > 0


def test_disabling_the_flag_writes_no_panel(tmp_path):
    runner = _runner(tmp_path, panel=False)
    result = runner.run_civil_violence(seed=1)

    log_dir = tmp_path / "logs" / "panel_test" / result["run_id"]
    assert not (log_dir / "agent_panel.parquet").exists()
    assert "panel_stats" not in result
    # The pre-panel outputs are still produced.
    assert (log_dir / "time_series.csv").exists()
    assert (log_dir / "summary.json").exists()


def test_panel_row_count_matches_steps_and_agents(tmp_path):
    runner = _runner(tmp_path, num_steps=3)
    result = runner.run_pd_grid(seed=1)

    log_dir = tmp_path / "logs" / "panel_test" / result["run_id"]
    panel = pd.read_parquet(log_dir / "agent_panel.parquet")
    # PD has one agent per cell and none are ever skipped.
    assert len(panel) == 4 * 4 * 3
    assert result["panel_stats"]["panel_writer"]["rows"] == len(panel)


def test_panel_aggregates_back_to_the_model_time_series(tmp_path):
    """The invariant a fresh run is verified by: with no original to diff
    against, the panel must reproduce the rate recorded beside it.

    Uses low legitimacy and heavy policing so arrests actually happen — the
    naive count matches only on arrest-free runs, which is exactly the case
    this test must not be lulled by.
    """
    runner = _runner(tmp_path, num_steps=6, legitimacy=0.05, cop_density=0.3)
    result = runner.run_civil_violence(seed=1)

    log_dir = tmp_path / "logs" / "panel_test" / result["run_id"]
    panel = pd.read_parquet(log_dir / "agent_panel.parquet")
    arrests = pd.read_parquet(log_dir / "arrests.parquet")
    ts = pd.read_csv(log_dir / "time_series.csv", index_col=0)

    assert len(arrests) > 0, "配置本应产生逮捕，否则这个测试形同虚设"

    steps = range(1, len(ts) + 1)
    n_citizens = panel["agent_id"].nunique()
    active = (panel[panel["action"] == "ACTIVE"]
              .groupby("step").size().reindex(steps, fill_value=0))

    # A citizen that chose ACTIVE and was then jailed by a cop activating
    # later in the same step reads as JAILED when the rate is measured.
    jailed_after_acting = (
        arrests.merge(panel[["step", "agent_id"]],
                      left_on=["step", "target_id"],
                      right_on=["step", "agent_id"], how="inner")
        .groupby("step").size().reindex(steps, fill_value=0)
    )
    assert jailed_after_acting.sum() > 0, "本测试要覆盖的正是这种情况"

    derived = ((active - jailed_after_acting) / n_citizens).to_numpy()
    assert derived == pytest.approx(ts["rebellion_rate"].to_numpy())


def test_llm_arm_records_confidence_and_parse_status(tmp_path):
    runner = _runner(tmp_path, arm="reasoner")
    result = runner.run_civil_violence(seed=1)

    log_dir = tmp_path / "logs" / "panel_test" / result["run_id"]
    panel = pd.read_parquet(log_dir / "agent_panel.parquet")
    assert panel["confidence"].notna().all()
    assert panel["parse_failed"].notna().all()
