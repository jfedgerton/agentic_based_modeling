"""Prisoner's Dilemma Grid Model.

A spatial iterated Prisoner's Dilemma on a 2D grid where agents play
against all Moore neighbors simultaneously. Supports classic and LLM
agent architectures.

Based on the canonical ABM benchmark (Nowak & May 1992).
"""

import random
from typing import Optional

import numpy as np
from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from src.agents.classic_pd import ClassicPDAgent
from src.agents.llm_pd import LLMPDAgent
from src.llm.provider import LLMProvider
from src.utils.agent_panel import AgentPanelWriter, disabled_writer
from src.utils.logging import ExperimentLogger

COOPERATE = "COOPERATE"
DEFECT = "DEFECT"


def _cooperation_rate(model):
    agents = [a for a in model.agents if hasattr(a, "action")]
    if not agents:
        return 0.0
    return sum(1 for a in agents if a.action == COOPERATE) / len(agents)


def _mean_payoff(model):
    agents = [a for a in model.agents if hasattr(a, "payoff")]
    if not agents:
        return 0.0
    return np.mean([a.payoff for a in agents])


def _spatial_clustering(model):
    """Measure spatial clustering of cooperators (Moran's I approximation).

    Returns fraction of cooperator-cooperator neighbor pairs out of all
    cooperator neighbor pairs. Higher values indicate more clustering.
    """
    coop_coop = 0
    coop_total = 0
    for agent in model.agents:
        if not hasattr(agent, "action") or agent.action != COOPERATE:
            continue
        neighbors = model.grid.get_neighbors(
            agent.pos, moore=True, include_center=False)
        for n in neighbors:
            if hasattr(n, "action"):
                coop_total += 1
                if n.action == COOPERATE:
                    coop_coop += 1
    if coop_total == 0:
        return 0.0
    return coop_coop / coop_total


class PDGridModel(Model):
    """Spatial Prisoner's Dilemma on a grid."""

    def __init__(self, width: int = 20, height: int = 20,
                 agent_type: str = "classic",
                 mode: Optional[str] = None,
                 initial_cooperation_prob: float = 0.5,
                 payoff_matrix: Optional[dict] = None,
                 llm_provider: Optional[LLMProvider] = None,
                 logger: Optional[ExperimentLogger] = None,
                 panel_writer: Optional[AgentPanelWriter] = None,
                 seed: Optional[int] = None):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.agent_type_name = agent_type
        self.logger = logger
        self.schedule_step = 0
        self.panel = panel_writer or disabled_writer()

        self.payoff_matrix = payoff_matrix or {
            "CC": 3, "CD": 0, "DC": 5, "DD": 1,
        }

        self.grid = SingleGrid(width, height, torus=True)

        # Create agents
        for x in range(width):
            for y in range(height):
                initial_action = (
                    COOPERATE if self.random.random() < initial_cooperation_prob
                    else DEFECT
                )

                if agent_type == "classic":
                    agent = ClassicPDAgent(self, initial_action=initial_action)
                elif agent_type == "llm":
                    if llm_provider is None:
                        raise ValueError("LLM provider required for llm agents")
                    if mode is None:
                        raise ValueError("mode required for llm agents")
                    agent = LLMPDAgent(self, llm_provider=llm_provider,
                                       mode=mode,
                                       initial_action=initial_action)
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")

                self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "cooperation_rate": _cooperation_rate,
                "mean_payoff": _mean_payoff,
                "spatial_clustering": _spatial_clustering,
            },
            agent_reporters={
                "action": lambda a: a.action if hasattr(a, "action") else None,
                "payoff": lambda a: a.payoff if hasattr(a, "payoff") else None,
                # PD agents are placed once and never move, so these are
                # constant per agent — recorded anyway so the frame is
                # self-contained rather than requiring the id-to-cell formula.
                "x": lambda a: a.pos[0] if a.pos else None,
                "y": lambda a: a.pos[1] if a.pos else None,
            },
        )

    def _compute_payoffs(self):
        """Calculate payoffs for all agents based on current actions."""
        pm = self.payoff_matrix
        for agent in self.agents:
            if not hasattr(agent, "action"):
                continue
            neighbors = self.grid.get_neighbors(
                agent.pos, moore=True, include_center=False)
            total = 0.0
            for neighbor in neighbors:
                if not hasattr(neighbor, "action"):
                    continue
                key = agent.action[0] + neighbor.action[0]  # e.g. "CC", "CD"
                total += pm[key]
            agent.payoff = total

    def step(self):
        """Execute one step: compute payoffs, decide, advance."""
        self.schedule_step += 1

        # Phase 1: compute payoffs from current actions
        self._compute_payoffs()

        # Phase 2: all agents decide simultaneously
        for agent in self.agents:
            if hasattr(agent, "decide"):
                agent.decide()

        # Phase 3: all agents apply their decisions
        for agent in self.agents:
            if hasattr(agent, "advance"):
                agent.advance()

        # Collect data
        self.datacollector.collect(self)

        # Log step-level metrics
        if self.logger:
            self.logger.log_step(self.schedule_step, {
                "cooperation_rate": _cooperation_rate(self),
                "mean_payoff": _mean_payoff(self),
                "spatial_clustering": _spatial_clustering(self),
            })
