"""Evaluation and metrics computation for benchmark results.

Computes emergent behavior metrics, benchmark statistics,
and interpretability measures across experimental runs.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class BenchmarkEvaluator:
    """Compute and compare metrics across experimental runs."""

    def __init__(self, results: list[dict]):
        self.results = results
        self.df = pd.DataFrame(results)

    @classmethod
    def from_file(cls, path: str) -> "BenchmarkEvaluator":
        with open(path, "r") as f:
            results = json.load(f)
        return cls(results)

    # -- PD Grid metrics --

    def pd_cooperation_summary(self) -> pd.DataFrame:
        """Summarize cooperation rates by agent type."""
        pd_runs = self.df[self.df["model"] == "pd_grid"]
        if pd_runs.empty:
            return pd.DataFrame()
        return pd_runs.groupby("agent_type").agg(
            mean_coop=("mean_cooperation_rate", "mean"),
            std_coop=("mean_cooperation_rate", "std"),
            final_coop_mean=("final_cooperation_rate", "mean"),
            final_coop_std=("final_cooperation_rate", "std"),
            mean_clustering=("final_spatial_clustering", "mean"),
            mean_runtime=("runtime_seconds", "mean"),
            n_runs=("seed", "count"),
        ).round(4)

    # -- Civil Violence metrics --

    def cv_rebellion_summary(self) -> pd.DataFrame:
        """Summarize rebellion rates by agent type."""
        cv_runs = self.df[self.df["model"] == "civil_violence"]
        if cv_runs.empty:
            return pd.DataFrame()
        return cv_runs.groupby("agent_type").agg(
            mean_rebellion=("mean_rebellion_rate", "mean"),
            std_rebellion=("mean_rebellion_rate", "std"),
            final_rebellion_mean=("final_rebellion_rate", "mean"),
            mean_jailed=("mean_jailed_rate", "mean"),
            mean_runtime=("runtime_seconds", "mean"),
            n_runs=("seed", "count"),
        ).round(4)

    # -- Interstate Conflict metrics --

    def ic_war_summary(self) -> pd.DataFrame:
        """Summarize war / bargaining outcomes by agent type."""
        ic_runs = self.df[self.df["model"] == "interstate_conflict"]
        if ic_runs.empty:
            return pd.DataFrame()
        return ic_runs.groupby("agent_type").agg(
            mean_war_freq=("mean_war_frequency", "mean"),
            std_war_freq=("mean_war_frequency", "std"),
            final_war_freq=("final_war_frequency", "mean"),
            mean_payoff=("mean_payoff", "mean"),
            mean_welfare_loss=("mean_welfare_loss", "mean"),
            mean_bluff_rate=("mean_bluff_rate", "mean"),
            mean_bluff_success_rate=("mean_bluff_success_rate", "mean"),
            mean_runtime=("runtime_seconds", "mean"),
            n_runs=("seed", "count"),
        ).round(4)

    # -- Cross-architecture comparison --

    def runtime_comparison(self) -> pd.DataFrame:
        """Compare runtime across agent types."""
        return self.df.groupby(["model", "agent_type"]).agg(
            mean_runtime=("runtime_seconds", "mean"),
            std_runtime=("runtime_seconds", "std"),
            n_runs=("seed", "count"),
        ).round(4)

    def seed_sensitivity(self, metric: str) -> pd.DataFrame:
        """Measure variance across seeds for a given metric."""
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in results.")
        return self.df.groupby(["model", "agent_type"]).agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            cv=(metric, lambda x: x.std() / x.mean() if x.mean() != 0 else 0),
            min=(metric, "min"),
            max=(metric, "max"),
        ).round(4)

    # -- LLM-specific metrics --

    # Arms that invoke the LLM (i.e. everything except classic).
    _LLM_ARMS = ("calculator", "reasoner", "roleplayer")

    def llm_cost_summary(self) -> pd.DataFrame:
        """Summarize LLM usage and estimated costs across LLM arms."""
        llm_runs = self.df[self.df["agent_type"].isin(self._LLM_ARMS)]
        if llm_runs.empty:
            return pd.DataFrame()

        rows = []
        for _, run in llm_runs.iterrows():
            stats = run.get("llm_stats", {})
            if stats:
                rows.append({
                    "model": run["model"],
                    "agent_type": run["agent_type"],
                    "seed": run["seed"],
                    "api_calls": stats.get("call_count", 0),
                    "prompt_tokens": stats.get("total_prompt_tokens", 0),
                    "completion_tokens": stats.get("total_completion_tokens", 0),
                })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def malformed_response_rate(self, log_dirs: Optional[list[str]] = None) -> dict:
        """Compute malformed response rate from parse failure logs."""
        if log_dirs is None:
            log_dirs = [r.get("log_dir", "") for r in self.results
                        if r.get("log_dir")]

        total_interactions = 0
        total_failures = 0

        for log_dir in log_dirs:
            log_path = Path(log_dir)
            interactions_file = log_path / "llm_interactions.jsonl"
            failures_file = log_path / "parse_failures.jsonl"

            if interactions_file.exists():
                with open(interactions_file) as f:
                    total_interactions += sum(1 for _ in f)

            if failures_file.exists():
                with open(failures_file) as f:
                    total_failures += sum(1 for _ in f)

        rate = total_failures / total_interactions if total_interactions > 0 else 0.0
        return {
            "total_interactions": total_interactions,
            "total_failures": total_failures,
            "malformed_rate": round(rate, 4),
        }

    # -- Rationale analysis --

    @staticmethod
    def analyze_rationales(log_dir: str,
                           keywords: Optional[list[str]] = None) -> dict:
        """Analyze rationale themes from LLM interaction logs."""
        if keywords is None:
            keywords = [
                "cooperat", "defect", "trust", "risk", "punish",
                "reward", "retaliat", "griev", "fear", "safe",
                "rebel", "protest", "arrest", "jail",
            ]

        interactions_file = Path(log_dir) / "llm_interactions.jsonl"
        if not interactions_file.exists():
            return {}

        theme_counts = Counter()
        action_theme = {}  # action -> Counter of themes
        total = 0

        with open(interactions_file) as f:
            for line in f:
                record = json.loads(line)
                parsed = record.get("parsed", {})
                rationale = parsed.get("short_rationale", "").lower()
                action = parsed.get("action", "UNKNOWN")

                if not rationale:
                    continue
                total += 1

                if action not in action_theme:
                    action_theme[action] = Counter()

                for kw in keywords:
                    if kw in rationale:
                        theme_counts[kw] += 1
                        action_theme[action][kw] += 1

        return {
            "total_rationales": total,
            "theme_frequencies": dict(theme_counts),
            "themes_by_action": {
                a: dict(c) for a, c in action_theme.items()
            },
        }

    def generate_summary_table(self) -> str:
        """Generate a formatted text summary of all results."""
        lines = ["=" * 60, "BENCHMARK RESULTS SUMMARY", "=" * 60]

        pd_summary = self.pd_cooperation_summary()
        if not pd_summary.empty:
            lines.append("\n--- Prisoner's Dilemma Grid ---")
            lines.append(pd_summary.to_string())

        cv_summary = self.cv_rebellion_summary()
        if not cv_summary.empty:
            lines.append("\n--- Civil Violence ---")
            lines.append(cv_summary.to_string())

        ic_summary = self.ic_war_summary()
        if not ic_summary.empty:
            lines.append("\n--- Interstate Conflict ---")
            lines.append(ic_summary.to_string())

        runtime = self.runtime_comparison()
        if not runtime.empty:
            lines.append("\n--- Runtime Comparison ---")
            lines.append(runtime.to_string())

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
