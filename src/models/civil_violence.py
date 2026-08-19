"""Epstein Civil Violence Model.

Citizens with heterogeneous grievances decide whether to rebel.
Cops patrol and arrest active rebels. Supports classic and LLM
citizen architectures.

Based on Epstein (2002) "Modeling Civil Violence".
"""

from typing import Optional

import numpy as np
from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from src.agents.classic_cv import ClassicCitizenAgent, ClassicCopAgent
from src.agents.classic_cv import QUIET, ACTIVE, JAILED
from src.agents.llm_cv import LLMCitizenAgent
from src.llm.provider import LLMProvider
from src.utils.logging import ExperimentLogger


def _rebellion_rate(model):
    citizens = [a for a in model.agents if hasattr(a, "state")]
    if not citizens:
        return 0.0
    return sum(1 for c in citizens if c.state == ACTIVE) / len(citizens)


def _quiet_rate(model):
    citizens = [a for a in model.agents if hasattr(a, "state")]
    if not citizens:
        return 0.0
    return sum(1 for c in citizens if c.state == QUIET) / len(citizens)


def _jailed_rate(model):
    citizens = [a for a in model.agents if hasattr(a, "state")]
    if not citizens:
        return 0.0
    return sum(1 for c in citizens if c.state == JAILED) / len(citizens)


def _arrest_count(model):
    """Number of arrests this step (tracked by model)."""
    return getattr(model, "_arrests_this_step", 0)


class CivilViolenceModel(Model):
    """Epstein Civil Violence model."""

    def __init__(self, width: int = 40, height: int = 40,
                 citizen_density: float = 0.7,
                 cop_density: float = 0.04,
                 citizen_vision: int = 7,
                 cop_vision: int = 7,
                 legitimacy: float = 0.82,
                 max_jail_term: int = 30,
                 movement: bool = True,
                 agent_type: str = "classic",
                 mode: Optional[str] = None,
                 language: str = "en",
                 llm_provider: Optional[LLMProvider] = None,
                 logger: Optional[ExperimentLogger] = None,
                 seed: Optional[int] = None):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.legitimacy = legitimacy
        self.max_jail_term = max_jail_term
        self.movement = movement
        self.agent_type_name = agent_type
        self.language = language
        self.logger = logger
        self.schedule_step = 0
        self._arrests_this_step = 0

        self.grid = SingleGrid(width, height, torus=True)

        # Calculate number of citizens and cops
        total_cells = width * height
        n_citizens = int(total_cells * citizen_density)
        n_cops = int(total_cells * cop_density)

        # Get all positions and shuffle
        all_positions = [(x, y) for x in range(width) for y in range(height)]
        self.random.shuffle(all_positions)

        # Place citizens
        for i in range(n_citizens):
            pos = all_positions[i]
            hardship = self.random.random()
            risk_aversion = self.random.random()

            if agent_type == "classic":
                agent = ClassicCitizenAgent(
                    self, hardship=hardship,
                    regime_legitimacy=legitimacy,
                    risk_aversion=risk_aversion,
                    vision=citizen_vision,
                )
            elif agent_type == "llm":
                if llm_provider is None:
                    raise ValueError("LLM provider required for llm agents")
                if mode is None:
                    raise ValueError("mode required for llm agents")
                agent = LLMCitizenAgent(
                    self, llm_provider=llm_provider,
                    mode=mode,
                    hardship=hardship,
                    regime_legitimacy=legitimacy,
                    risk_aversion=risk_aversion,
                    vision=citizen_vision,
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            self.grid.place_agent(agent, pos)

        # Place cops (always classic rule-based)
        for i in range(n_citizens, n_citizens + n_cops):
            pos = all_positions[i]
            agent = ClassicCopAgent(self, vision=cop_vision)
            self.grid.place_agent(agent, pos)

        self.datacollector = DataCollector(
            model_reporters={
                "rebellion_rate": _rebellion_rate,
                "quiet_rate": _quiet_rate,
                "jailed_rate": _jailed_rate,
                "arrests_this_step": _arrest_count,
            },
        )

    def step(self):
        """Execute one step: all agents act."""
        self.schedule_step += 1
        self._arrests_this_step = 0

        # Shuffle and activate agents
        agents = list(self.agents)
        self.random.shuffle(agents)

        for agent in agents:
            if isinstance(agent, ClassicCopAgent):
                # Count arrests before cop acts
                pre_actives = sum(
                    1 for a in self.agents
                    if hasattr(a, "state") and a.state == ACTIVE
                )
                agent.step()
                post_actives = sum(
                    1 for a in self.agents
                    if hasattr(a, "state") and a.state == ACTIVE
                )
                self._arrests_this_step += max(0, pre_actives - post_actives)
            else:
                agent.step()

        self.datacollector.collect(self)

        if self.logger:
            self.logger.log_step(self.schedule_step, {
                "rebellion_rate": _rebellion_rate(self),
                "quiet_rate": _quiet_rate(self),
                "jailed_rate": _jailed_rate(self),
                "arrests": self._arrests_this_step,
            })
