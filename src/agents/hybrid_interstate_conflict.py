"""Hybrid agent for the Fearon Bargaining Grid.

The LLM interprets the environment by estimating each neighbor's true
capability from their broadcast signal. The classic Fearon formula then
uses these LLM-derived estimates to compute the agent's demand or
acceptance threshold per dyad.

This separates "signal interpretation" (LLM) from "decision rule" (formula),
isolating the effect of LLM perception from full LLM strategy choice.
"""

from typing import Optional

from src.agents.base import AgentDecision, BaseAgent
from src.llm.provider import LLMProvider

# Shared constants, helpers, and the signal-phase prompt come from llm_interstate_conflict
# to keep game-rule wording and helper logic in a single source of truth.
from src.agents.llm_interstate_conflict import (
    STRONG, MODERATE, WEAK, VALID_SIGNALS,
    SIGNAL_TO_CAP, DEMAND_BUCKETS,
    GAME_RULES, BLUFFING_HINT, SIGNAL_PROMPT,
    _truthful_signal, _round_to_bucket, _parse_json,
)


INTERPRET_PROMPT = """{game_rules}{bluffing_hint}

You are now in PHASE 2: INTERPRET. You have already broadcast your signal.
For each of your 8 neighbors, output your best estimate of their TRUE
military capability based on their broadcast signal and any other context.

Your private state:
- Your true capability: {capability:.2f}
- Your true war cost: {war_cost:.2f}
- Your signal this round: {self_signal}
- Step: {step}

Your neighbors:
{neighbor_table}

Your job is NOT to choose actions — only to estimate each neighbor's true
capability. A downstream rule will pick demand/threshold based on your
estimates.

Respond with ONLY a JSON object:
{{
  "perceived_capabilities": [
    {{"neighbor_id": <int>, "perceived_capability": <float between 0.1 and 0.9>, "rationale": "<short>"}}
  ],
  "short_rationale": "<1-3 sentence overall reasoning>"
}}"""


class HybridInterstateConflictAgent(BaseAgent):
    """Hybrid Fearon agent: LLM interprets signals, classic formula decides."""

    def __init__(self, model, llm_provider: LLMProvider,
                 true_capability: float, true_war_cost: float):
        super().__init__(model, agent_type="hybrid_interstate_conflict")
        self.llm = llm_provider
        self.true_capability = true_capability
        self.true_war_cost = true_war_cost
        self.signal: Optional[str] = None
        self.demands: dict = {}
        self.thresholds: dict = {}
        self.payoff = 0.0
        self._last_signal_rationale: Optional[str] = None
        self._last_perceived_caps: dict = {}

    def get_local_observation(self) -> dict:
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        return {
            "my_true_capability": self.true_capability,
            "my_true_war_cost": self.true_war_cost,
            "my_signal": self.signal,
            "neighbors": [
                {
                    "id": n.unique_id,
                    "pos": n.pos,
                    "signal": getattr(n, "signal", None),
                }
                for n in neighbors
            ],
        }

    # --- Phase 1: signal broadcast (identical structure to LLM agent) ---

    def broadcast_signal(self):
        if not self.model.bluffing_enabled:
            self.signal = _truthful_signal(self.true_capability)
            self._last_signal_rationale = "Forced honest (bluffing disabled)."
            return

        self_hist = self.model._agent_history.get(
            self.unique_id, {"last_signal_self": None, "last_payoff": 0.0}
        )
        prompt = SIGNAL_PROMPT.format(
            game_rules=GAME_RULES,
            bluffing_hint=BLUFFING_HINT,
            capability=self.true_capability,
            war_cost=self.true_war_cost,
            step=self.model.schedule_step,
            own_previous_signal=self_hist.get("last_signal_self") or "none",
            own_last_round_payoff=float(self_hist.get("last_payoff", 0.0)),
        )
        raw = self.llm.query(prompt)
        parsed = None
        try:
            data = _parse_json(raw)
            sig = str(data.get("signal", "")).upper()
            if sig not in VALID_SIGNALS:
                raise ValueError(f"invalid signal value: {sig!r}")
            self.signal = sig
            self._last_signal_rationale = data.get("short_rationale", "")
            parsed = data
        except Exception as e:
            if self.model.logger:
                self.model.logger.log_parse_failure(
                    step=self.model.schedule_step,
                    agent_id=self.unique_id,
                    raw_response=raw,
                    error=f"signal parse: {e}",
                )
            self.signal = _truthful_signal(self.true_capability)
            self._last_signal_rationale = f"Parse failure fallback: {e}"

        if self.model.logger:
            self.model.logger.log_llm_interaction(
                step=self.model.schedule_step,
                agent_id=self.unique_id,
                prompt=prompt,
                raw_response=raw,
                parsed={
                    "phase": "signal",
                    "signal": self.signal,
                    "rationale": self._last_signal_rationale,
                    "raw_parsed": parsed,
                },
            )

    # --- Phase 2: LLM interprets perceived capability; formula computes ---

    def decide(self):
        self.demands = {}
        self.thresholds = {}
        self._last_perceived_caps = {}

        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        neighbor_info = []
        for n in neighbors:
            proposer_pos = self.model.proposer_pos_of(self.pos, n.pos)
            role = "proposer" if proposer_pos == self.pos else "responder"
            hist = self.model._dyad_history.get(
                (self.unique_id, n.unique_id),
                {"last_signal": None, "last_outcome": "none"},
            )
            neighbor_info.append({
                "agent": n,
                "id": n.unique_id,
                "pos": n.pos,
                "signal": getattr(n, "signal", None),
                "role": role,
                "last_signal": hist.get("last_signal") or "none",
                "last_outcome": hist.get("last_outcome") or "none",
            })

        # Build neighbor table WITHOUT role — LLM only interprets, not decides.
        # Includes 1-round history so LLM can perform reputational reasoning.
        table_lines = [
            "| neighbor_id | position | their_signal | last_signal | last_outcome |",
            "|---|---|---|---|---|",
        ]
        for ni in neighbor_info:
            table_lines.append(
                f"| {ni['id']} | {ni['pos']} | {ni['signal']} | "
                f"{ni['last_signal']} | {ni['last_outcome']} |"
            )
        neighbor_table = "\n".join(table_lines)

        prompt = INTERPRET_PROMPT.format(
            game_rules=GAME_RULES,
            bluffing_hint=BLUFFING_HINT if self.model.bluffing_enabled else "",
            capability=self.true_capability,
            war_cost=self.true_war_cost,
            self_signal=self.signal,
            step=self.model.schedule_step,
            neighbor_table=neighbor_table,
        )
        raw = self.llm.query(prompt)

        perceived_caps: dict = {}
        parsed = None
        try:
            data = _parse_json(raw)
            for entry in data.get("perceived_capabilities", []):
                nid = int(entry["neighbor_id"])
                cap = float(entry.get("perceived_capability", 0.5))
                cap = max(0.1, min(0.9, cap))
                perceived_caps[nid] = cap
            parsed = data
        except Exception as e:
            if self.model.logger:
                self.model.logger.log_parse_failure(
                    step=self.model.schedule_step,
                    agent_id=self.unique_id,
                    raw_response=raw,
                    error=f"interpret parse: {e}",
                )

        # Fill any missing perceived caps with the signal-to-cap fallback.
        for ni in neighbor_info:
            if ni["id"] not in perceived_caps:
                perceived_caps[ni["id"]] = SIGNAL_TO_CAP.get(ni["signal"], 0.5)
        self._last_perceived_caps = perceived_caps

        # Apply the classic Fearon formula with LLM-derived perceived caps.
        for ni in neighbor_info:
            nid = ni["id"]
            opp_cap = perceived_caps[nid]
            if ni["role"] == "proposer":
                estimated_threshold = (
                    1.0
                    - self.true_capability / (self.true_capability + opp_cap)
                    + self.model.E_war_cost
                )
                self.demands[nid] = _round_to_bucket(estimated_threshold)
            else:
                p_self_wins = self.true_capability / (
                    self.true_capability + opp_cap
                )
                self.thresholds[nid] = 1.0 - p_self_wins + self.true_war_cost

        if self.model.logger:
            self.model.logger.log_llm_interaction(
                step=self.model.schedule_step,
                agent_id=self.unique_id,
                prompt=prompt,
                raw_response=raw,
                parsed={
                    "phase": "interpret",
                    "perceived_capabilities": dict(perceived_caps),
                    "demands": dict(self.demands),
                    "thresholds": dict(self.thresholds),
                    "raw_parsed": parsed,
                },
            )

        decision = AgentDecision(
            observed_state_summary=(
                f"cap={self.true_capability:.2f}, "
                f"war_cost={self.true_war_cost:.2f}, signal={self.signal}"
            ),
            beliefs={
                "perceived_capabilities": dict(perceived_caps),
                "demands_to_neighbors": dict(self.demands),
                "thresholds_to_neighbors": dict(self.thresholds),
                "signal_rationale": self._last_signal_rationale,
            },
            action=self.signal,
            confidence=1.0 if parsed is not None else 0.0,
            short_rationale=(
                "Hybrid: LLM-estimated perceived capabilities; Fearon "
                "formula computes demand/threshold from those estimates."
                + (
                    f" Interpret rationale: {parsed.get('short_rationale', '')}"
                    if parsed is not None else " (interpret parse failure)"
                )
            ),
        )
        self.record_decision(decision)
