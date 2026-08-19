"""Rebuild and re-run a finished experiment from its own logs.

Every LLM response of a completed run is already on disk in the prompt
cache, keyed by ``(prompt, model, temperature)``. Because the models draw
all randomness from a seeded RNG, re-running with the same seed regenerates
the identical prompt sequence and therefore hits the cache on every call —
reproducing the original trajectory exactly, offline and for free.

That property only holds while the code produces byte-identical prompts in
the same order. :class:`StrictReplayProvider` turns any deviation into an
immediate, loud failure rather than a silent API call: it has no API client
at all, so a diverging replay cannot cost money or quietly fabricate new
responses.

The seed is parsed from the run id, not read from ``config.json`` — the
config records ``experiment.random_seed`` (the base seed), whereas the run
actually executed with the per-replication seed embedded in its directory
name.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

from src.experiments.runner import _resolve_arm
from src.llm.cache import PromptCache
from src.llm.provider import LLMProvider
from src.models.civil_violence import CivilViolenceModel
from src.models.interstate_conflict import InterstateConflictModel
from src.models.pd_grid import PDGridModel
from src.utils.config import load_config  # noqa: F401  (re-exported for callers)

# run_id format: {game}_{arm}_seed{seed}_{uuid8}
RUN_ID_RE = re.compile(
    r"^(?P<game>pd|cv|ic)_"
    r"(?P<arm>classic|calculator|reasoner|roleplayer)_"
    r"seed(?P<seed>\d+)_"
    r"(?P<uid>[0-9a-f]{8})$"
)

GAMES = ("pd", "cv", "ic")


class CacheMissError(RuntimeError):
    """Raised when a replay needs a prompt that is not in the cache.

    Signals that the regenerated trajectory has diverged from the original:
    either the code changed, or the seed/config used to rebuild the model is
    not the one the original run used.
    """


class StrictReplayProvider(LLMProvider):
    """Cache-only LLM provider used for replays.

    Has no API client by construction. A replay that would need to call the
    API is a replay that is already wrong, so the miss is raised instead of
    served.
    """

    def __init__(self, **kwargs):
        # Nothing is ever sent, so the inter-call sleep is pure waste.
        kwargs["rate_limit_delay"] = 0
        super().__init__(**kwargs)
        self.miss_count = 0

    def _call_api(self, prompt: str) -> str:
        self.miss_count += 1
        raise CacheMissError(
            "缓存未命中 —— 轨迹已偏离原 run，重放中止。\n"
            f"--- prompt 开头 ---\n{prompt[:600]}\n--- 结束 ---"
        )


def parse_run_id(run_id: str) -> dict:
    """Split a run directory name into game/arm/seed/uid, or {} if malformed."""
    m = RUN_ID_RE.match(run_id)
    return m.groupdict() if m else {}


def original_step_count(run_dir: Path) -> int:
    """Steps the original run executed, taken from its own time_series.csv.

    Authoritative in a way ``config.json`` is not: the config on disk records
    the configured step count, which may since have been edited.
    """
    ts = Path(run_dir) / "time_series.csv"
    with open(ts) as f:
        return sum(1 for _ in f) - 1  # minus the header row


def make_replay_provider(llm_config: dict) -> Tuple[Optional[LLMProvider],
                                                    Optional[PromptCache]]:
    """Build a cache-only provider from a run's ``llm`` config block."""
    cache = PromptCache(cache_dir=llm_config["cache_dir"], enabled=True)
    provider = StrictReplayProvider(
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config.get("max_tokens", 4096),
        cache=cache,
    )
    return provider, cache


def build_model(game: str, arm: str, seed: int, config: dict,
                provider: Optional[LLMProvider] = None,
                logger=None, **overrides):
    """Construct a model exactly as :class:`ExperimentRunner` would.

    Keyword overrides are forwarded to the model constructor, which is how a
    caller switches on panel collection without touching this function.
    """
    agent_type, mode = _resolve_arm(arm)
    common = dict(agent_type=agent_type, mode=mode, llm_provider=provider,
                  logger=logger, seed=seed, **overrides)

    if game == "cv":
        cv = config["civil_violence"]
        return CivilViolenceModel(
            width=cv["width"], height=cv["height"],
            citizen_density=cv["citizen_density"],
            cop_density=cv["cop_density"],
            citizen_vision=cv["citizen_vision"],
            cop_vision=cv["cop_vision"],
            legitimacy=cv["legitimacy"],
            max_jail_term=cv["max_jail_term"],
            movement=cv.get("movement", True),
            language=cv.get("language", "en"),
            **common,
        )

    if game == "pd":
        pd_cfg = config["pd_grid"]
        return PDGridModel(
            width=pd_cfg["width"], height=pd_cfg["height"],
            initial_cooperation_prob=pd_cfg["initial_cooperation_prob"],
            payoff_matrix=pd_cfg["payoff_matrix"],
            **common,
        )

    if game == "ic":
        ic = config["interstate_conflict"]
        return InterstateConflictModel(
            width=ic["width"], height=ic["height"],
            signal_form=ic["signal_form"],
            bluffing_enabled=ic["bluffing_enabled"],
            bluffing_rate=ic["bluffing_rate"],
            capability_dist=tuple(ic["capability_dist"]),
            war_cost_dist=tuple(ic["war_cost_dist"]),
            capability_range=tuple(ic["capability_range"]),
            war_cost_range=tuple(ic["war_cost_range"]),
            **common,
        )

    raise ValueError(f"unknown game: {game!r}. Expected one of {GAMES}")
