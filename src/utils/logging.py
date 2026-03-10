"""Experiment logging framework.

Records all aspects of a simulation run: configuration, agent decisions,
LLM interactions, parse failures, and runtime metrics.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class ExperimentLogger:
    """Structured logger for experiment runs.

    Writes both human-readable logs and structured JSON records.
    """

    def __init__(self, experiment_name: str, run_id: str,
                 log_dir: str = "logs", config: Optional[dict] = None):
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.log_dir = Path(log_dir) / experiment_name / run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()
        self._records = []
        self._parse_failures = []
        self._prompts = []

        # Set up Python logger
        self._logger = logging.getLogger(f"abm.{experiment_name}.{run_id}")
        self._logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(self.log_dir / "run.log")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"))
        self._logger.addHandler(handler)

        # Log initial config
        if config:
            self.log_config(config)

    def log_config(self, config: dict):
        """Record the full experiment configuration."""
        self._write_json("config.json", config)
        self._logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    def log_step(self, step: int, data: dict):
        """Record aggregate data for a simulation step."""
        record = {"step": step, "timestamp": time.time(), **data}
        self._records.append(record)
        self._logger.debug(f"Step {step}: {data}")

    def log_agent_decision(self, step: int, agent_id: Any,
                           agent_type: str, action: str,
                           rationale: Optional[str] = None,
                           confidence: Optional[float] = None,
                           beliefs: Optional[dict] = None):
        """Record an individual agent decision."""
        record = {
            "step": step,
            "agent_id": str(agent_id),
            "agent_type": agent_type,
            "action": action,
            "rationale": rationale,
            "confidence": confidence,
            "beliefs": beliefs,
            "timestamp": time.time(),
        }
        self._records.append(record)

    def log_llm_interaction(self, step: int, agent_id: Any,
                            prompt: str, raw_response: str,
                            parsed: Optional[dict] = None,
                            cached: bool = False):
        """Record an LLM prompt-response pair."""
        record = {
            "step": step,
            "agent_id": str(agent_id),
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed": parsed,
            "cached": cached,
            "timestamp": time.time(),
        }
        self._prompts.append(record)

    def log_parse_failure(self, step: int, agent_id: Any,
                          raw_response: str, error: str):
        """Record a failed LLM response parse."""
        record = {
            "step": step,
            "agent_id": str(agent_id),
            "raw_response": raw_response,
            "error": error,
            "timestamp": time.time(),
        }
        self._parse_failures.append(record)
        self._logger.warning(
            f"Parse failure at step {step}, agent {agent_id}: {error}")

    def log_metric(self, name: str, value: Any, step: Optional[int] = None):
        """Record a named metric."""
        record = {
            "metric": name,
            "value": value,
            "step": step,
            "timestamp": time.time(),
        }
        self._records.append(record)

    def finalize(self, llm_stats: Optional[dict] = None):
        """Write all accumulated records to disk and compute runtime."""
        elapsed = time.time() - self._start_time

        summary = {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "runtime_seconds": elapsed,
            "total_records": len(self._records),
            "total_llm_interactions": len(self._prompts),
            "total_parse_failures": len(self._parse_failures),
            "llm_stats": llm_stats,
            "finalized_at": datetime.now().isoformat(),
        }

        self._write_json("summary.json", summary)
        self._write_jsonl("records.jsonl", self._records)
        self._write_jsonl("llm_interactions.jsonl", self._prompts)
        if self._parse_failures:
            self._write_jsonl("parse_failures.jsonl", self._parse_failures)

        self._logger.info(f"Run complete. Runtime: {elapsed:.2f}s")
        self._logger.info(f"Records: {len(self._records)}, "
                          f"LLM calls: {len(self._prompts)}, "
                          f"Parse failures: {len(self._parse_failures)}")
        return summary

    def _write_json(self, filename: str, data: Any):
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _write_jsonl(self, filename: str, records: list):
        path = self.log_dir / filename
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
