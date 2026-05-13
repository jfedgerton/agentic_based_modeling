"""Jury Deliberation Model.

A 12-juror deliberation model with full-mesh communication and multi-round
dynamics. Jurors hold private continuous beliefs about defendant guilt and
update through synchronous speak -> listen -> update rounds.

Supports classic (status-weighted DeGroot), LLM (natural-language argument),
and hybrid (LLM-quality-weighted) juror architectures. Terminates when the
jury reaches a unanimous verdict (12/12) or when `max_rounds` is exhausted
(hung jury).

Based on DeGroot (1974) opinion pooling and the social-decision-scheme jury
literature (Stasser & Davis 1981; Hastie, Penrod, Pennington 1983).
"""

from typing import Optional

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from src.agents.classic_jury import ClassicJurorAgent
from src.agents.llm_jury import LLMJurorAgent
from src.agents.hybrid_jury import HybridJurorAgent
from src.llm.provider import LLMProvider
from src.utils.logging import ExperimentLogger


# --- Constants ---

GUILTY = "GUILTY"
NOT_GUILTY = "NOT_GUILTY"
HUNG = "HUNG"
VERDICTS = (GUILTY, NOT_GUILTY)


# --- DataCollector reporters (Tier 1) ---

def _mean_belief(model):
    beliefs = [a.belief for a in model.agents if hasattr(a, "belief")]
    if not beliefs:
        return 0.0
    return float(np.mean(beliefs))


def _vote_count_guilty(model):
    return sum(1 for a in model.agents
               if hasattr(a, "belief") and a.belief > 0.5)


def _vote_count_not_guilty(model):
    return sum(1 for a in model.agents
               if hasattr(a, "belief") and a.belief <= 0.5)


def _is_unanimous(model):
    """1 if all jurors vote the same way, else 0."""
    votes = [a.belief > 0.5 for a in model.agents if hasattr(a, "belief")]
    if not votes:
        return 0
    return 1 if (all(votes) or not any(votes)) else 0


def _current_accuracy(model):
    """1.0 if current majority matches ground_truth, else 0.0 (live, even pre-verdict)."""
    if model.ground_truth not in VERDICTS:
        return 0.0
    g = _vote_count_guilty(model)
    ng = _vote_count_not_guilty(model)
    majority = GUILTY if g > ng else NOT_GUILTY
    return 1.0 if majority == model.ground_truth else 0.0


class JuryDeliberationModel(Model):
    """A 12-juror jury deliberating until unanimous or hung."""

    def __init__(self,
                 n_jurors: int = 12,
                 max_rounds: int = 10,
                 mixing_alpha: float = 0.3,
                 classic_noise_sigma: float = 0.15,
                 agent_type: str = "classic",
                 case_description: str = "[TBD: case description placeholder]",
                 testimony: str = "[TBD: testimony placeholder]",
                 ground_truth: str = GUILTY,
                 status_permutation: bool = False,
                 llm_provider: Optional[LLMProvider] = None,
                 logger: Optional[ExperimentLogger] = None,
                 seed: Optional[int] = None):
        super().__init__(seed=seed)

        if ground_truth not in VERDICTS:
            raise ValueError(
                f"ground_truth must be one of {VERDICTS}, got {ground_truth!r}")

        self.n_jurors = n_jurors
        self.max_rounds = max_rounds
        self.mixing_alpha = mixing_alpha
        self.classic_noise_sigma = classic_noise_sigma
        self.agent_type_name = agent_type
        self.case_description = case_description
        self.testimony = testimony
        self.ground_truth = ground_truth
        self.status_permutation = status_permutation
        self.logger = logger
        self.schedule_step = 0

        # Termination state.
        self._verdict: Optional[str] = None
        self._terminated = False

        # Per-round transient buffer (used by logger).
        self._round_arguments_this_step: dict = {}

        # Deterministic numpy RNG for social_status sampling.
        np_rng = np.random.default_rng(seed)

        # Create jurors.
        for i in range(n_jurors):
            social_status = float(np_rng.uniform(0.0, 1.0))

            if agent_type == "classic":
                ClassicJurorAgent(
                    self,
                    social_status=social_status,
                    noise_sigma=classic_noise_sigma,
                )
            elif agent_type == "llm":
                if llm_provider is None:
                    raise ValueError("LLM provider required for llm agents")
                LLMJurorAgent(
                    self,
                    llm_provider=llm_provider,
                    social_status=social_status,
                )
            elif agent_type == "hybrid":
                if llm_provider is None:
                    raise ValueError("LLM provider required for hybrid agents")
                HybridJurorAgent(
                    self,
                    llm_provider=llm_provider,
                    social_status=social_status,
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

        # Status-permutation ablation: shuffle the status labels across jurors
        # AFTER agents are created (so beliefs/initialization use the original
        # values; only the broadcasted "status" tag changes).
        if status_permutation:
            agents_list = list(self.agents)
            statuses = [a.social_status for a in agents_list]
            np_rng.shuffle(statuses)
            for a, s in zip(agents_list, statuses):
                a.social_status = s

        # Round 0: each juror forms its initial belief from the testimony
        # (LLM/Hybrid) or via noise around 0.5 (Classic).
        for agent in self.agents:
            if hasattr(agent, "initialize_belief"):
                agent.initialize_belief()

        # Round 0 snapshot (pre-deliberation baseline; embedded Condorcet).
        self.round_0_majority_verdict = self._current_majority_vote()
        self.round_0_unanimous = bool(_is_unanimous(self))
        self.round_0_individual_accuracy = self._compute_individual_accuracy()

        self.datacollector = DataCollector(
            model_reporters={
                "mean_belief": _mean_belief,
                "votes_guilty": _vote_count_guilty,
                "votes_not_guilty": _vote_count_not_guilty,
                "is_unanimous": _is_unanimous,
                "current_accuracy": _current_accuracy,
            },
            agent_reporters={
                "belief": lambda a: getattr(a, "belief", None),
                "social_status": lambda a: getattr(a, "social_status", None),
                "vote": lambda a: (
                    GUILTY if getattr(a, "belief", 0.5) > 0.5 else NOT_GUILTY
                ),
            },
        )

        # Record Round 0 datapoint.
        self.datacollector.collect(self)

        if self.logger:
            self.logger.log_step(0, {
                "round": 0,
                "phase": "round_0_baseline",
                "votes_guilty": _vote_count_guilty(self),
                "votes_not_guilty": _vote_count_not_guilty(self),
                "is_unanimous": _is_unanimous(self),
                "round_0_accuracy": self.round_0_individual_accuracy,
                "round_0_majority_verdict": self.round_0_majority_verdict,
                "ground_truth": self.ground_truth,
            })

    # --- Helpers ---

    def _current_majority_vote(self) -> str:
        g = _vote_count_guilty(self)
        ng = _vote_count_not_guilty(self)
        return GUILTY if g > ng else NOT_GUILTY

    def _compute_individual_accuracy(self) -> float:
        """Fraction of jurors whose individual vote matches ground_truth."""
        total = 0
        correct = 0
        for a in self.agents:
            if not hasattr(a, "belief"):
                continue
            total += 1
            vote = GUILTY if a.belief > 0.5 else NOT_GUILTY
            if vote == self.ground_truth:
                correct += 1
        return correct / total if total > 0 else 0.0

    def _check_termination(self) -> Optional[str]:
        """Return verdict label if terminated this step, else None."""
        if _is_unanimous(self):
            return self._current_majority_vote()
        if self.schedule_step >= self.max_rounds:
            return HUNG
        return None

    # --- Main step (one deliberation round) ---

    def step(self):
        """One deliberation round: speak -> listen -> update -> check termination."""
        if self._terminated:
            return

        self.schedule_step += 1

        # Phase 1: speak. Each juror produces an argument for this round.
        self._round_arguments_this_step = {}
        for agent in self.agents:
            if hasattr(agent, "speak"):
                agent.speak()
            self._round_arguments_this_step[agent.unique_id] = {
                "argument": getattr(agent, "current_argument", None),
                "belief_before_update": getattr(agent, "belief", None),
                "social_status": getattr(agent, "social_status", None),
                "vote": (
                    GUILTY if getattr(agent, "belief", 0.5) > 0.5 else NOT_GUILTY
                ),
            }

        # Phase 2: listen. Each juror reads the other 11 arguments + status.
        for agent in self.agents:
            if hasattr(agent, "listen"):
                agent.listen()

        # Phase 3: update. All jurors synchronously update belief.
        for agent in self.agents:
            if hasattr(agent, "update_belief"):
                agent.update_belief()

        # Collect data and log.
        self.datacollector.collect(self)

        if self.logger:
            self.logger.log_step(self.schedule_step, {
                "round": self.schedule_step,
                "votes_guilty": _vote_count_guilty(self),
                "votes_not_guilty": _vote_count_not_guilty(self),
                "is_unanimous": _is_unanimous(self),
                "mean_belief": _mean_belief(self),
                "current_accuracy": _current_accuracy(self),
                "arguments_this_round": self._round_arguments_this_step,
            })

        # Termination check.
        verdict = self._check_termination()
        if verdict is not None:
            self._verdict = verdict
            self._terminated = True
            if self.logger:
                self.logger.log_step(self.schedule_step, {
                    "round": self.schedule_step,
                    "phase": "termination",
                    "verdict": verdict,
                    "ground_truth": self.ground_truth,
                    "accuracy": (
                        1.0 if verdict == self.ground_truth else 0.0
                    ) if verdict != HUNG else None,
                    "hung": verdict == HUNG,
                    "round_0_accuracy": self.round_0_individual_accuracy,
                })

    # --- Convenience accessors ---

    @property
    def verdict(self) -> Optional[str]:
        return self._verdict

    @property
    def is_terminated(self) -> bool:
        return self._terminated

    @property
    def accuracy(self) -> Optional[float]:
        """1.0 if verdict matches ground_truth; 0.0 if mismatch; None if hung or not yet terminated."""
        if self._verdict is None or self._verdict == HUNG:
            return None
        return 1.0 if self._verdict == self.ground_truth else 0.0
