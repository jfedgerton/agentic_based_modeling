"""Tests for the experiment logging framework."""

import json
import tempfile

from src.utils.logging import ExperimentLogger


class TestExperimentLogger:
    def test_logger_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                experiment_name="test_exp",
                run_id="run_001",
                log_dir=tmpdir,
            )
            assert logger.log_dir.exists()

    def test_log_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                experiment_name="test_exp",
                run_id="run_001",
                log_dir=tmpdir,
                config={"key": "value"},
            )
            config_path = logger.log_dir / "config.json"
            assert config_path.exists()
            with open(config_path) as f:
                data = json.load(f)
            assert data["key"] == "value"

    def test_finalize_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                experiment_name="test_exp",
                run_id="run_001",
                log_dir=tmpdir,
            )
            logger.log_step(1, {"metric": 0.5})
            logger.log_agent_decision(
                step=1, agent_id="a1", agent_type="classic",
                action="COOPERATE",
            )
            summary = logger.finalize()
            assert summary["total_records"] == 2
            assert (logger.log_dir / "summary.json").exists()
            assert (logger.log_dir / "records.jsonl").exists()

    def test_parse_failure_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                experiment_name="test_exp",
                run_id="run_001",
                log_dir=tmpdir,
            )
            logger.log_parse_failure(
                step=1, agent_id="a1",
                raw_response="bad json", error="JSONDecodeError",
            )
            summary = logger.finalize()
            assert summary["total_parse_failures"] == 1
            assert (logger.log_dir / "parse_failures.jsonl").exists()
