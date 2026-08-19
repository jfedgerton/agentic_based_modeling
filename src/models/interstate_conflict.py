"""Fearon Grid Bargaining Model.

A spatial Fearon bargaining game on a 2D grid. Each cell is a state with
private capability and war_cost. Each step, every state simultaneously
bargains with all 8 Moore neighbors through signal -> demand -> resolve.
Supports classic and LLM agent architectures.

Based on Fearon (1995) "Rationalist Explanations for War", International
Organization 49(3): 379-414.
"""

from typing import Optional, Tuple

import numpy as np
from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from src.agents.classic_interstate_conflict import ClassicInterstateConflictAgent
from src.agents.llm_interstate_conflict import LLMInterstateConflictAgent
from src.llm.provider import LLMProvider
from src.utils.logging import ExperimentLogger


# --- Constants (kept in sync with the *_fearon agent files) ---

STRONG = "STRONG"
MODERATE = "MODERATE"
WEAK = "WEAK"
SIGNALS = (STRONG, MODERATE, WEAK)

SIGNAL_TO_CAP = {STRONG: 0.75, MODERATE: 0.50, WEAK: 0.25}
DEMAND_BUCKETS = (0.1, 0.3, 0.5, 0.7, 0.9)


def truthful_signal(capability: float) -> str:
    """Map a true capability to the honest signal it implies."""
    if capability > 0.66:
        return STRONG
    if capability > 0.33:
        return MODERATE
    return WEAK


def round_to_bucket(x: float) -> float:
    """Round a continuous demand value to the nearest discrete bucket."""
    x = max(0.0, min(1.0, x))
    return min(DEMAND_BUCKETS, key=lambda b: abs(b - x))


# --- DataCollector reporters (Tier 1 metrics) ---

def _war_frequency(model):
    if model._dyads_this_step == 0:
        return 0.0
    return model._wars_this_step / model._dyads_this_step


def _mean_payoff(model):
    agents = [a for a in model.agents if hasattr(a, "payoff")]
    if not agents:
        return 0.0
    return float(np.mean([a.payoff for a in agents]))


def _welfare_loss(model):
    return model._welfare_loss_this_step


def _bluff_rate(model):
    weak = [a for a in model.agents
            if hasattr(a, "true_capability") and a.true_capability < 0.5]
    if not weak:
        return 0.0
    return sum(1 for a in weak if getattr(a, "signal", None) == STRONG) / len(weak)


def _bluff_success_rate(model):
    if model._bluffs_attempted_this_step == 0:
        return 0.0
    return model._bluffs_succeeded_this_step / model._bluffs_attempted_this_step


class InterstateConflictModel(Model):
    """Spatial Fearon bargaining on a grid (Fearon 1995)."""

    def __init__(self, width: int = 20, height: int = 20,
                 agent_type: str = "classic",
                 mode: Optional[str] = None,
                 signal_form: str = "categorical",
                 bluffing_enabled: bool = True,
                 bluffing_rate: float = 0.3,
                 capability_dist: Tuple[float, float] = (2.0, 2.0),
                 war_cost_dist: Tuple[float, float] = (2.0, 5.0),
                 capability_range: Tuple[float, float] = (0.1, 0.9),
                 war_cost_range: Tuple[float, float] = (0.1, 0.5),
                 llm_provider: Optional[LLMProvider] = None,
                 logger: Optional[ExperimentLogger] = None,
                 seed: Optional[int] = None):
        super().__init__(seed=seed)

        # Cross-IV validation
        if signal_form == "free_form_text":
            if mode != "roleplayer":
                raise ValueError(
                    "signal_form='free_form_text' is only valid for mode='roleplayer'"
                )
            if not bluffing_enabled:
                raise ValueError(
                    "signal_form='free_form_text' requires bluffing_enabled=True"
                )

        self.width = width
        self.height = height
        self.agent_type_name = agent_type
        self.signal_form = signal_form
        self.bluffing_enabled = bluffing_enabled
        self.bluffing_rate = bluffing_rate
        self.capability_dist = capability_dist
        self.war_cost_dist = war_cost_dist
        self.capability_range = capability_range
        self.war_cost_range = war_cost_range
        self.logger = logger
        self.schedule_step = 0

        # Step-level counters; reset at the start of each step().
        self._dyads_this_step = 0
        self._wars_this_step = 0
        self._settlements_this_step = 0
        self._welfare_loss_this_step = 0.0
        self._bluffs_attempted_this_step = 0
        self._bluffs_succeeded_this_step = 0
        self._dyad_outcomes_this_step: list = []

        # 1-round history (populated at the end of each step).
        # _dyad_history[(self_id, neighbor_id)] = {"last_signal": str, "last_outcome": str}
        # _agent_history[agent_id] = {"last_signal_self": str, "last_payoff": float}
        self._dyad_history: dict = {}
        self._agent_history: dict = {}

        # Population prior mean for war_cost — used by the proposer rule.
        cost_a, cost_b = war_cost_dist
        cost_min, cost_max = war_cost_range
        self.E_war_cost = cost_min + (cost_max - cost_min) * cost_a / (cost_a + cost_b)

        self.grid = SingleGrid(width, height, torus=True)

        # Deterministic numpy RNG for Beta sampling.
        np_rng = np.random.default_rng(seed)
        cap_min, cap_max = capability_range
        cap_a, cap_b = capability_dist

        for x in range(width):
            for y in range(height):
                capability = cap_min + (cap_max - cap_min) * np_rng.beta(cap_a, cap_b)
                war_cost = cost_min + (cost_max - cost_min) * np_rng.beta(cost_a, cost_b)

                if agent_type == "classic":
                    agent = ClassicInterstateConflictAgent(
                        self,
                        true_capability=capability,
                        true_war_cost=war_cost,
                    )
                elif agent_type == "llm":
                    if llm_provider is None:
                        raise ValueError("LLM provider required for llm agents")
                    if mode is None:
                        raise ValueError("mode required for llm agents")
                    agent = LLMInterstateConflictAgent(
                        self,
                        llm_provider=llm_provider,
                        true_capability=capability,
                        true_war_cost=war_cost,
                        mode=mode,
                    )
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")

                self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "war_frequency": _war_frequency,
                "mean_payoff": _mean_payoff,
                "welfare_loss": _welfare_loss,
                "bluff_rate": _bluff_rate,
                "bluff_success_rate": _bluff_success_rate,
            },
            agent_reporters={
                "signal": lambda a: getattr(a, "signal", None),
                "payoff": lambda a: getattr(a, "payoff", None),
                "true_capability": lambda a: getattr(a, "true_capability", None),
                "true_war_cost": lambda a: getattr(a, "true_war_cost", None),
            },
        )

    # --- Dyad helpers ---

    def proposer_pos_of(self, pos_a: tuple, pos_b: tuple) -> tuple:
        """Return the position of the proposer for dyad (pos_a, pos_b) this step.

        Lex-lower position is the primary cell. On even steps the primary
        proposes; on odd steps the roles swap.
        """
        primary = min(pos_a, pos_b)
        secondary = max(pos_a, pos_b)
        return primary if self.schedule_step % 2 == 0 else secondary

    def _resolve_dyad(self, agent_a, agent_b) -> dict:
        """Resolve a single bargaining dyad, update counters, return outcome."""
        proposer_pos = self.proposer_pos_of(agent_a.pos, agent_b.pos)
        if agent_a.pos == proposer_pos:
            proposer, responder = agent_a, agent_b
        else:
            proposer, responder = agent_b, agent_a

        demand = proposer.demands.get(responder.unique_id, 0.5)
        threshold = responder.thresholds.get(proposer.unique_id, 0.5)

        if demand <= threshold:
            # Peaceful settlement.
            self._settlements_this_step += 1
            proposer_payoff = demand
            responder_payoff = 1.0 - demand
            war_occurred = False
            winner_id = None
        else:
            # War lottery.
            self._wars_this_step += 1
            self._welfare_loss_this_step += (
                proposer.true_war_cost + responder.true_war_cost
            )
            p_proposer_wins = proposer.true_capability / (
                proposer.true_capability + responder.true_capability
            )
            if self.random.random() < p_proposer_wins:
                proposer_payoff = 1.0 - proposer.true_war_cost
                responder_payoff = -responder.true_war_cost
                winner_id = proposer.unique_id
            else:
                proposer_payoff = -proposer.true_war_cost
                responder_payoff = 1.0 - responder.true_war_cost
                winner_id = responder.unique_id
            war_occurred = True

        # Track bluff attempts/successes (weak agent signaling STRONG).
        for agent in (proposer, responder):
            if (agent.true_capability < 0.5
                    and getattr(agent, "signal", None) == STRONG):
                self._bluffs_attempted_this_step += 1
                if not war_occurred:
                    self._bluffs_succeeded_this_step += 1

        proposer.payoff += proposer_payoff
        responder.payoff += responder_payoff

        # Update 1-round dyad history. From each agent's perspective:
        #   last_signal  = the OTHER side's signal this round
        #   last_outcome = the outcome from MY perspective
        if war_occurred:
            prop_outcome = "war_won" if winner_id == proposer.unique_id else "war_lost"
            resp_outcome = "war_won" if winner_id == responder.unique_id else "war_lost"
        else:
            prop_outcome = "settled"
            resp_outcome = "settled"
        self._dyad_history[(proposer.unique_id, responder.unique_id)] = {
            "last_signal": getattr(responder, "signal", None),
            "last_outcome": prop_outcome,
        }
        self._dyad_history[(responder.unique_id, proposer.unique_id)] = {
            "last_signal": getattr(proposer, "signal", None),
            "last_outcome": resp_outcome,
        }

        return {
            "step": self.schedule_step,
            "proposer_id": proposer.unique_id,
            "responder_id": responder.unique_id,
            "proposer_pos": proposer.pos,
            "responder_pos": responder.pos,
            "demand": demand,
            "threshold": threshold,
            "war": war_occurred,
            "winner_id": winner_id,
            "proposer_payoff": proposer_payoff,
            "responder_payoff": responder_payoff,
            "proposer_signal": getattr(proposer, "signal", None),
            "responder_signal": getattr(responder, "signal", None),
            "proposer_true_capability": proposer.true_capability,
            "responder_true_capability": responder.true_capability,
            "proposer_true_war_cost": proposer.true_war_cost,
            "responder_true_war_cost": responder.true_war_cost,
        }

    def step(self):
        """Execute one step: 3 synchronous phases (signal, decide, resolve)."""
        self.schedule_step += 1

        # Reset step-level counters and per-agent payoff accumulator.
        self._dyads_this_step = 0
        self._wars_this_step = 0
        self._settlements_this_step = 0
        self._welfare_loss_this_step = 0.0
        self._bluffs_attempted_this_step = 0
        self._bluffs_succeeded_this_step = 0
        self._dyad_outcomes_this_step = []
        for agent in self.agents:
            agent.payoff = 0.0

        # Phase 1: all agents simultaneously broadcast a signal.
        for agent in self.agents:
            if hasattr(agent, "broadcast_signal"):
                agent.broadcast_signal()

        # Phase 2: all agents simultaneously decide demand/threshold per neighbor.
        for agent in self.agents:
            if hasattr(agent, "decide"):
                agent.decide()

        # Phase 3: resolve every dyad exactly once; accumulate payoffs.
        processed = set()
        for agent in self.agents:
            neighbors = self.grid.get_neighbors(
                agent.pos, moore=True, include_center=False)
            for neighbor in neighbors:
                key = tuple(sorted([agent.unique_id, neighbor.unique_id]))
                if key in processed:
                    continue
                processed.add(key)
                self._dyads_this_step += 1
                outcome = self._resolve_dyad(agent, neighbor)
                self._dyad_outcomes_this_step.append(outcome)

        # Record per-agent 1-round history (signal + total payoff this round).
        for agent in self.agents:
            self._agent_history[agent.unique_id] = {
                "last_signal_self": getattr(agent, "signal", None),
                "last_signal_text": getattr(agent, "signal_text", None),
                "last_payoff": float(getattr(agent, "payoff", 0.0)),
            }

        self.datacollector.collect(self)

        if self.logger:
            self.logger.log_step(self.schedule_step, {
                "war_frequency": _war_frequency(self),
                "mean_payoff": _mean_payoff(self),
                "welfare_loss": self._welfare_loss_this_step,
                "bluff_rate": _bluff_rate(self),
                "bluff_success_rate": _bluff_success_rate(self),
                "wars": self._wars_this_step,
                "settlements": self._settlements_this_step,
                "dyad_outcomes": self._dyad_outcomes_this_step,
            })
