"""Experiment runner for benchmark simulations.

Orchestrates running models across the four-arm benchmark
(classic, calculator, reasoner, roleplayer) × three games (PD, CV, IC) ×
seeds × per-game IVs. Manages logging, data collection, and output.

The ``agent.type`` config value is the arm name (one of the four), which
is translated internally to the underlying ``(agent_type, mode)`` pair
that the model constructors expect.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.llm.cache import PromptCache
from src.llm.provider import get_provider, LLMProvider
from src.models.pd_grid import PDGridModel
from src.models.civil_violence import CivilViolenceModel
from src.models.interstate_conflict import InterstateConflictModel
from src.utils.agent_panel import close_panel_writers, make_panel_writers
from src.utils.config import load_config
from src.utils.logging import ExperimentLogger


# Arm name → (agent_type for model, mode for model).
_ARM_TO_AGENT_TYPE_AND_MODE: dict[str, Tuple[str, Optional[str]]] = {
    "classic": ("classic", None),
    "calculator": ("llm", "calculator"),
    "reasoner": ("llm", "reasoner"),
    "roleplayer": ("llm", "roleplayer"),
}

VALID_ARMS = tuple(_ARM_TO_AGENT_TYPE_AND_MODE.keys())


def _resolve_arm(arm: str) -> Tuple[str, Optional[str]]:
    if arm not in _ARM_TO_AGENT_TYPE_AND_MODE:
        raise ValueError(
            f"Unknown arm: {arm!r}. Expected one of {VALID_ARMS}"
        )
    return _ARM_TO_AGENT_TYPE_AND_MODE[arm]


class ExperimentRunner:
    """Run benchmark experiments across the four-arm × three-game matrix."""

    def __init__(self, config_path: str = "config/default.yaml",
                 config_overrides: Optional[dict] = None):
        self.config = load_config(config_path, config_overrides)
        self.results = []

    @property
    def arm(self) -> str:
        return self.config["agent"]["type"]

    def _setup_llm(self, game: str,
                   signal_form: str = "categorical") -> Optional[LLMProvider]:
        """Create LLM provider from config. Returns None for the classic arm.

        ``game`` and ``signal_form`` are forwarded to MockProvider (which
        requires them); real providers ignore them.
        """
        agent_type, mode = _resolve_arm(self.arm)
        if agent_type != "llm":
            return None

        llm_config = self.config["llm"]
        cache = None
        if llm_config.get("cache_enabled", True):
            cache = PromptCache(
                cache_dir=llm_config.get("cache_dir", ".llm_cache"),
                enabled=True,
            )

        provider_name = llm_config["provider"]
        provider_kwargs = dict(
            cache=cache,
            model=llm_config.get("model", "gpt-5.4-mini"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 256),
            rate_limit_delay=llm_config.get("rate_limit_delay", 0.1),
        )
        if provider_name == "mock":
            provider_kwargs.update(game=game, mode=mode, signal_form=signal_form)
        return get_provider(provider_name, **provider_kwargs)

    def _panel_writers(self, game: str, log_dir) -> dict:
        """Panel writers for this run, or {} when panel logging is off.

        Panels land beside the run's other logs so a run directory stays
        self-contained.
        """
        enabled = self.config.get("logging", {}).get("log_agent_panel", True)
        return make_panel_writers(game, log_dir, enabled=enabled)

    @staticmethod
    def _finalize_panels(writers: dict, result: dict) -> None:
        """Close panel writers and record their row counts in the result."""
        summaries = close_panel_writers(writers)
        if summaries:
            result["panel_stats"] = {
                arg: {"rows": s["rows"], "path": s["path"]}
                for arg, s in summaries.items()
            }

    # ------------------------------------------------------------- PD

    def run_pd_grid(self, seed: int, num_steps: Optional[int] = None) -> dict:
        num_steps = num_steps or self.config["experiment"]["num_steps"]
        agent_type, mode = _resolve_arm(self.arm)
        run_id = f"pd_{self.arm}_seed{seed}_{uuid.uuid4().hex[:8]}"

        logger = ExperimentLogger(
            experiment_name=self.config["experiment"]["name"],
            run_id=run_id,
            log_dir=self.config["experiment"]["log_dir"],
            config=self.config,
        )

        llm_provider = self._setup_llm(game="pd")
        pd_config = self.config["pd_grid"]
        panels = self._panel_writers("pd", logger.log_dir)

        model = PDGridModel(
            width=pd_config["width"],
            height=pd_config["height"],
            agent_type=agent_type,
            mode=mode,
            initial_cooperation_prob=pd_config["initial_cooperation_prob"],
            payoff_matrix=pd_config["payoff_matrix"],
            llm_provider=llm_provider,
            logger=logger,
            seed=seed,
            **panels,
        )

        start_time = time.time()
        for _ in range(num_steps):
            model.step()
        elapsed = time.time() - start_time

        model_data = model.datacollector.get_model_vars_dataframe()
        result = {
            "model": "pd_grid",
            "agent_type": self.arm,
            "seed": seed,
            "num_steps": num_steps,
            "runtime_seconds": elapsed,
            "final_cooperation_rate": model_data["cooperation_rate"].iloc[-1],
            "mean_cooperation_rate": model_data["cooperation_rate"].mean(),
            "final_spatial_clustering": model_data["spatial_clustering"].iloc[-1],
            "final_mean_payoff": model_data["mean_payoff"].iloc[-1],
            "run_id": run_id,
        }

        llm_stats = None
        if llm_provider:
            llm_stats = llm_provider.usage_stats
            result["llm_stats"] = llm_stats
            if llm_provider.cache:
                result["cache_stats"] = llm_provider.cache.stats

        self._finalize_panels(panels, result)
        logger.finalize(llm_stats=llm_stats)
        result["log_dir"] = str(logger.log_dir)
        model_data.to_csv(logger.log_dir / "time_series.csv")

        self.results.append(result)
        return result

    # ------------------------------------------------------------- CV

    def run_civil_violence(self, seed: int,
                           num_steps: Optional[int] = None) -> dict:
        num_steps = num_steps or self.config["experiment"]["num_steps"]
        agent_type, mode = _resolve_arm(self.arm)
        run_id = f"cv_{self.arm}_seed{seed}_{uuid.uuid4().hex[:8]}"

        logger = ExperimentLogger(
            experiment_name=self.config["experiment"]["name"],
            run_id=run_id,
            log_dir=self.config["experiment"]["log_dir"],
            config=self.config,
        )

        cv_config = self.config["civil_violence"]
        # Language is a CV-only IV; passed through to the model.
        language = cv_config.get("language", "en")

        llm_provider = self._setup_llm(game="cv")
        panels = self._panel_writers("cv", logger.log_dir)

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
            agent_type=agent_type,
            mode=mode,
            language=language,
            llm_provider=llm_provider,
            logger=logger,
            seed=seed,
            **panels,
        )

        start_time = time.time()
        for _ in range(num_steps):
            model.step()
        elapsed = time.time() - start_time

        model_data = model.datacollector.get_model_vars_dataframe()
        result = {
            "model": "civil_violence",
            "agent_type": self.arm,
            "language": language,
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

        self._finalize_panels(panels, result)
        logger.finalize(llm_stats=llm_stats)
        result["log_dir"] = str(logger.log_dir)
        model_data.to_csv(logger.log_dir / "time_series.csv")

        self.results.append(result)
        return result

    # ------------------------------------------------------------- IC

    def run_interstate_conflict(self, seed: int,
                                num_steps: Optional[int] = None) -> dict:
        num_steps = num_steps or self.config["experiment"]["num_steps"]
        agent_type, mode = _resolve_arm(self.arm)
        run_id = f"ic_{self.arm}_seed{seed}_{uuid.uuid4().hex[:8]}"

        logger = ExperimentLogger(
            experiment_name=self.config["experiment"]["name"],
            run_id=run_id,
            log_dir=self.config["experiment"]["log_dir"],
            config=self.config,
        )

        ic_config = self.config["interstate_conflict"]
        signal_form = ic_config.get("signal_form", "categorical")

        llm_provider = self._setup_llm(game="ic", signal_form=signal_form)
        panels = self._panel_writers("ic", logger.log_dir)

        model = InterstateConflictModel(
            width=ic_config["width"],
            height=ic_config["height"],
            agent_type=agent_type,
            mode=mode,
            signal_form=signal_form,
            bluffing_enabled=ic_config.get("bluffing_enabled", True),
            bluffing_rate=ic_config.get("bluffing_rate", 0.3),
            capability_dist=tuple(ic_config.get("capability_dist", [2.0, 2.0])),
            war_cost_dist=tuple(ic_config.get("war_cost_dist", [2.0, 5.0])),
            capability_range=tuple(ic_config.get("capability_range", [0.1, 0.9])),
            war_cost_range=tuple(ic_config.get("war_cost_range", [0.1, 0.5])),
            llm_provider=llm_provider,
            logger=logger,
            seed=seed,
            **panels,
        )

        start_time = time.time()
        for _ in range(num_steps):
            model.step()
        elapsed = time.time() - start_time

        model_data = model.datacollector.get_model_vars_dataframe()
        result = {
            "model": "interstate_conflict",
            "agent_type": self.arm,
            "signal_form": signal_form,
            "bluffing_enabled": ic_config.get("bluffing_enabled", True),
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

        self._finalize_panels(panels, result)
        logger.finalize(llm_stats=llm_stats)
        result["log_dir"] = str(logger.log_dir)
        model_data.to_csv(logger.log_dir / "time_series.csv")

        self.results.append(result)
        return result

    # ------------------------------------------------------------- Dispatcher

    def run_benchmark(self, model_name: str = "pd_grid",
                      seeds: Optional[list[int]] = None,
                      num_steps: Optional[int] = None) -> list[dict]:
        """Run a full benchmark across multiple seeds for the configured arm."""
        if seeds is None:
            base_seed = self.config["experiment"]["random_seed"]
            n_reps = self.config["experiment"]["num_replications"]
            seeds = [base_seed + i for i in range(n_reps)]

        run_func = {
            "pd_grid": self.run_pd_grid,
            "civil_violence": self.run_civil_violence,
            "interstate_conflict": self.run_interstate_conflict,
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
