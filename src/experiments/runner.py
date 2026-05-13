"""Experiment runner for benchmark simulations.

Orchestrates running models across architectures, seeds, and
configurations. Manages logging, data collection, and output.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

from src.llm.cache import PromptCache
from src.llm.provider import get_provider, LLMProvider
from src.models.pd_grid import PDGridModel
from src.models.civil_violence import CivilViolenceModel
from src.models.interstate_conflict import InterstateConflictModel
from src.models.jury_deliberation import JuryDeliberationModel
from src.utils.config import load_config
from src.utils.logging import ExperimentLogger


class ExperimentRunner:
    """Run benchmark experiments across model configurations."""

    def __init__(self, config_path: str = "config/default.yaml",
                 config_overrides: Optional[dict] = None):
        self.config = load_config(config_path, config_overrides)
        self.results = []

    def _setup_llm(self) -> Optional[LLMProvider]:
        """Create LLM provider from config if needed."""
        agent_type = self.config["agent"]["type"]
        if agent_type in ("llm", "hybrid"):
            llm_config = self.config["llm"]
            cache = None
            if llm_config.get("cache_enabled", True):
                cache = PromptCache(
                    cache_dir=llm_config.get("cache_dir", ".llm_cache"),
                    enabled=True,
                )
            return get_provider(
                llm_config["provider"],
                cache=cache,
                model=llm_config.get("model", "gpt-5.4-mini"),
                temperature=llm_config.get("temperature", 0.0),
                max_tokens=llm_config.get("max_tokens", 256),
                rate_limit_delay=llm_config.get("rate_limit_delay", 0.1),
            )
        return None

    def run_pd_grid(self, seed: int, num_steps: Optional[int] = None) -> dict:
        """Run a single PD Grid simulation."""
        num_steps = num_steps or self.config["experiment"]["num_steps"]
        run_id = f"pd_{self.config['agent']['type']}_seed{seed}_{uuid.uuid4().hex[:8]}"

        logger = ExperimentLogger(
            experiment_name=self.config["experiment"]["name"],
            run_id=run_id,
            log_dir=self.config["experiment"]["log_dir"],
            config=self.config,
        )

        llm_provider = self._setup_llm()
        pd_config = self.config["pd_grid"]

        model = PDGridModel(
            width=pd_config["width"],
            height=pd_config["height"],
            agent_type=self.config["agent"]["type"],
            initial_cooperation_prob=pd_config["initial_cooperation_prob"],
            payoff_matrix=pd_config["payoff_matrix"],
            llm_provider=llm_provider,
            logger=logger,
            seed=seed,
        )

        start_time = time.time()
        for step in range(num_steps):
            model.step()
        elapsed = time.time() - start_time

        # Collect results
        model_data = model.datacollector.get_model_vars_dataframe()
        result = {
            "model": "pd_grid",
            "agent_type": self.config["agent"]["type"],
            "seed": seed,
            "num_steps": num_steps,
            "runtime_seconds": elapsed,
            "final_cooperation_rate": model_data["cooperation_rate"].iloc[-1],
            "mean_cooperation_rate": model_data["cooperation_rate"].mean(),
            "final_spatial_clustering": model_data["spatial_clustering"].iloc[-1],
            "final_mean_payoff": model_data["mean_payoff"].iloc[-1],
            "run_id": run_id,
        }

        # Add LLM stats if applicable
        llm_stats = None
        if llm_provider:
            llm_stats = llm_provider.usage_stats
            result["llm_stats"] = llm_stats
            if llm_provider.cache:
                result["cache_stats"] = llm_provider.cache.stats

        summary = logger.finalize(llm_stats=llm_stats)
        result["log_dir"] = str(logger.log_dir)

        # Save time series
        model_data.to_csv(logger.log_dir / "time_series.csv")

        self.results.append(result)
        return result

    def run_civil_violence(self, seed: int,
                           num_steps: Optional[int] = None) -> dict:
        """Run a single Civil Violence simulation."""
        num_steps = num_steps or self.config["experiment"]["num_steps"]
        run_id = f"cv_{self.config['agent']['type']}_seed{seed}_{uuid.uuid4().hex[:8]}"

        logger = ExperimentLogger(
            experiment_name=self.config["experiment"]["name"],
            run_id=run_id,
            log_dir=self.config["experiment"]["log_dir"],
            config=self.config,
        )

        llm_provider = self._setup_llm()
        cv_config = self.config["civil_violence"]

        model = CivilViolenceModel(
            width=cv_config["width"],
            height=cv_config["height"],
            citizen_density=cv_config["citizen_density"],
            cop_density=cv_config["cop_density"],
            citizen_vision=cv_config["citizen_vision"],
            cop_vision=cv_config["cop_vision"],
            legitimacy=cv_config["legitimacy"],
            max_jail_term=cv_config["max_jail_term"],
            movement=cv_config.get("movement", True),
            agent_type=self.config["agent"]["type"],
            llm_provider=llm_provider,
            logger=logger,
            seed=seed,
        )

        start_time = time.time()
        for step in range(num_steps):
            model.step()
        elapsed = time.time() - start_time

        model_data = model.datacollector.get_model_vars_dataframe()
        result = {
            "model": "civil_violence",
            "agent_type": self.config["agent"]["type"],
            "seed": seed,
            "num_steps": num_steps,
            "runtime_seconds": elapsed,
            "mean_rebellion_rate": model_data["rebellion_rate"].mean(),
            "final_rebellion_rate": model_data["rebellion_rate"].iloc[-1],
            "mean_jailed_rate": model_data["jailed_rate"].mean(),
            "run_id": run_id,
        }

        llm_stats = None
        if llm_provider:
            llm_stats = llm_provider.usage_stats
            result["llm_stats"] = llm_stats
            if llm_provider.cache:
                result["cache_stats"] = llm_provider.cache.stats

        logger.finalize(llm_stats=llm_stats)
        result["log_dir"] = str(logger.log_dir)
        model_data.to_csv(logger.log_dir / "time_series.csv")

        self.results.append(result)
        return result

    def run_interstate_conflict(self, seed: int,
                                num_steps: Optional[int] = None) -> dict:
        """Run a single Interstate Conflict simulation."""
        num_steps = num_steps or self.config["experiment"]["num_steps"]
        run_id = f"interstate_conflict_{self.config['agent']['type']}_seed{seed}_{uuid.uuid4().hex[:8]}"

        logger = ExperimentLogger(
            experiment_name=self.config["experiment"]["name"],
            run_id=run_id,
            log_dir=self.config["experiment"]["log_dir"],
            config=self.config,
        )

        llm_provider = self._setup_llm()
        f_config = self.config["interstate_conflict"]

        model = InterstateConflictModel(
            width=f_config["width"],
            height=f_config["height"],
            agent_type=self.config["agent"]["type"],
            bluffing_enabled=f_config.get("bluffing_enabled", True),
            bluffing_rate=f_config.get("bluffing_rate", 0.3),
            capability_dist=tuple(f_config.get("capability_dist", [2.0, 2.0])),
            war_cost_dist=tuple(f_config.get("war_cost_dist", [2.0, 5.0])),
            capability_range=tuple(f_config.get("capability_range", [0.1, 0.9])),
            war_cost_range=tuple(f_config.get("war_cost_range", [0.1, 0.5])),
            llm_provider=llm_provider,
            logger=logger,
            seed=seed,
        )

        start_time = time.time()
        for step in range(num_steps):
            model.step()
        elapsed = time.time() - start_time

        model_data = model.datacollector.get_model_vars_dataframe()
        result = {
            "model": "interstate_conflict",
            "agent_type": self.config["agent"]["type"],
            "seed": seed,
            "num_steps": num_steps,
            "runtime_seconds": elapsed,
            "mean_war_frequency": model_data["war_frequency"].mean(),
            "final_war_frequency": model_data["war_frequency"].iloc[-1],
            "mean_welfare_loss": model_data["welfare_loss"].mean(),
            "mean_payoff": model_data["mean_payoff"].mean(),
            "mean_bluff_rate": model_data["bluff_rate"].mean(),
            "mean_bluff_success_rate": model_data["bluff_success_rate"].mean(),
            "run_id": run_id,
        }

        llm_stats = None
        if llm_provider:
            llm_stats = llm_provider.usage_stats
            result["llm_stats"] = llm_stats
            if llm_provider.cache:
                result["cache_stats"] = llm_provider.cache.stats

        logger.finalize(llm_stats=llm_stats)
        result["log_dir"] = str(logger.log_dir)
        model_data.to_csv(logger.log_dir / "time_series.csv")

        self.results.append(result)
        return result

    def run_jury_deliberation(self, seed: int,
                              num_steps: Optional[int] = None) -> dict:
        """Run a single Jury Deliberation simulation (terminates early on verdict)."""
        j_config = self.config["jury_deliberation"]
        max_rounds = num_steps if num_steps is not None else j_config["max_rounds"]
        run_id = f"jury_{self.config['agent']['type']}_seed{seed}_{uuid.uuid4().hex[:8]}"

        logger = ExperimentLogger(
            experiment_name=self.config["experiment"]["name"],
            run_id=run_id,
            log_dir=self.config["experiment"]["log_dir"],
            config=self.config,
        )

        llm_provider = self._setup_llm()

        model = JuryDeliberationModel(
            n_jurors=j_config["n_jurors"],
            max_rounds=max_rounds,
            mixing_alpha=j_config.get("mixing_alpha", 0.3),
            classic_noise_sigma=j_config.get("classic_noise_sigma", 0.15),
            agent_type=self.config["agent"]["type"],
            case_description=j_config.get("case_description", ""),
            testimony=j_config.get("testimony", ""),
            ground_truth=j_config.get("ground_truth", "GUILTY"),
            status_permutation=j_config.get("status_permutation", False),
            llm_provider=llm_provider,
            logger=logger,
            seed=seed,
        )

        start_time = time.time()
        for _ in range(max_rounds):
            if model.is_terminated:
                break
            model.step()
        elapsed = time.time() - start_time

        model_data = model.datacollector.get_model_vars_dataframe()
        result = {
            "model": "jury_deliberation",
            "agent_type": self.config["agent"]["type"],
            "seed": seed,
            "max_rounds": max_rounds,
            "runtime_seconds": elapsed,
            "verdict": model.verdict,
            "accuracy": model.accuracy,
            "hung": model.verdict == "HUNG",
            "ground_truth": model.ground_truth,
            "round_0_accuracy": model.round_0_individual_accuracy,
            "round_0_majority_verdict": model.round_0_majority_verdict,
            "rounds_to_verdict": model.schedule_step,
            "status_permutation": j_config.get("status_permutation", False),
            "run_id": run_id,
        }

        llm_stats = None
        if llm_provider:
            llm_stats = llm_provider.usage_stats
            result["llm_stats"] = llm_stats
            if llm_provider.cache:
                result["cache_stats"] = llm_provider.cache.stats

        logger.finalize(llm_stats=llm_stats)
        result["log_dir"] = str(logger.log_dir)
        model_data.to_csv(logger.log_dir / "time_series.csv")

        self.results.append(result)
        return result

    def run_benchmark(self, model_name: str = "pd_grid",
                      seeds: Optional[list[int]] = None,
                      num_steps: Optional[int] = None) -> list[dict]:
        """Run a full benchmark across multiple seeds."""
        if seeds is None:
            base_seed = self.config["experiment"]["random_seed"]
            n_reps = self.config["experiment"]["num_replications"]
            seeds = [base_seed + i for i in range(n_reps)]

        run_func = {
            "pd_grid": self.run_pd_grid,
            "civil_violence": self.run_civil_violence,
            "interstate_conflict": self.run_interstate_conflict,
            "jury_deliberation": self.run_jury_deliberation,
        }

        if model_name not in run_func:
            raise ValueError(f"Unknown model: {model_name}. "
                             f"Available: {list(run_func.keys())}")

        results = []
        for seed in seeds:
            result = run_func[model_name](seed=seed, num_steps=num_steps)
            results.append(result)

        return results

    def save_results(self, output_path: Optional[str] = None):
        """Save all accumulated results to a JSON file."""
        if not self.results:
            return
        output_dir = Path(self.config["experiment"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            output_path = output_dir / f"results_{uuid.uuid4().hex[:8]}.json"
        else:
            output_path = Path(output_path)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        return output_path
