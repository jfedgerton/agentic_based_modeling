"""LLM-based agent for the Fearon Bargaining Grid.

Uses the LLM provider twice per step:
- Phase 1 (broadcast_signal): chooses the round's signal.
- Phase 2 (decide): chooses demand or threshold for each of 8 neighbors.

Falls back to the Fearon classic closed-form rule on parse failure.
"""

import json
from typing import Optional

from src.agents.base import AgentDecision, BaseAgent
from src.llm.provider import LLMProvider

# --- Constants (keep in sync with src/models/interstate_conflict.py and classic_interstate_conflict.py) ---

STRONG = "STRONG"
MODERATE = "MODERATE"
WEAK = "WEAK"
VALID_SIGNALS = (STRONG, MODERATE, WEAK)
SIGNAL_TO_CAP = {STRONG: 0.75, MODERATE: 0.50, WEAK: 0.25}
DEMAND_BUCKETS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _truthful_signal(capability: float) -> str:
    if capability > 0.66:
        return STRONG
    if capability > 0.33:
        return MODERATE
    return WEAK


def _round_to_bucket(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return min(DEMAND_BUCKETS, key=lambda b: abs(b - x))


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()
    return json.loads(text)




## more vague about capability and war cost 
## strong, moderate or weak. --> in latent, 
# --- Prompts ---

GAME_RULES = """You are a state in a strategic bargaining game played on a grid of neighboring states. Each round, you bargain with all 8 Moore neighbors over a contested issue (worth 1 unit per dyad).

Your private information (others CANNOT directly see):
- Your true military capability (between 0.1 and 0.9)
- Your true war cost (between 0.1 and 0.5)

Each round proceeds in 3 phases:
1. SIGNAL: You broadcast ONE signal (STRONG / MODERATE / WEAK) visible to all 8 neighbors.
2. DECIDE: For each neighbor, you are either PROPOSER or RESPONDER (predetermined by position and step parity).
   - PROPOSER: pick a demand x in {0.1, 0.3, 0.5, 0.7, 0.9}.
   - RESPONDER: pick an acceptance threshold x* in [0, 1].
3. RESOLVE: For each dyad:
   - If demand <= threshold: peaceful settlement. Proposer gets demand, responder gets 1 - demand.
   - If demand > threshold: WAR. Each side pays its war_cost. Winner takes 1.0.
     P(you win) = your_cap / (your_cap + opp_cap).

Signal convention (shared by all agents):
- STRONG: claim of capability ~0.75
- MODERATE: claim ~0.50
- WEAK: claim ~0.25

Your goal: maximize total payoff summed across all 8 dyads each round."""

BLUFFING_HINT = "\n\nNote: your signal does not have to match your true capability. You may choose to signal strategically."

SIGNAL_PROMPT = """{game_rules}{bluffing_hint}

You are now in PHASE 1: SIGNAL. Choose what signal to broadcast this round.

Your private state:
- Your true capability: {capability:.2f}
- Your true war cost: {war_cost:.2f}
- Step: {step}
- Your previous signal: {own_previous_signal}
- Your last round payoff: {own_last_round_payoff:.2f}

Respond with ONLY a JSON object:
{{
  "signal": "STRONG" | "MODERATE" | "WEAK",
  "short_rationale": "<1-3 sentence explanation>"
}}"""

DECIDE_PROMPT = """{game_rules}{bluffing_hint}

You are now in PHASE 2: DECIDE. You have already broadcast your signal.
For each of your 8 neighbors, output your demand (if proposer) or
threshold (if responder).

Your private state:
- Your true capability: {capability:.2f}
- Your true war cost: {war_cost:.2f}
- Your signal this round: {self_signal}
- Step: {step}

Your neighbors:
{neighbor_table}

Respond with ONLY a JSON object:
{{
  "decisions": [
    {{"neighbor_id": <int>, "role": "proposer", "demand": <one of 0.1, 0.3, 0.5, 0.7, 0.9>, "rationale": "<short>"}},
    {{"neighbor_id": <int>, "role": "responder", "threshold": <float 0-1>, "rationale": "<short>"}}
  ],
  "short_rationale": "<1-3 sentence overall reasoning>"
}}"""


class LLMInterstateConflictAgent(BaseAgent):
    """LLM-driven Fearon bargaining agent."""

    def __init__(self, model, llm_provider: LLMProvider,
                 true_capability: float, true_war_cost: float):
        super().__init__(model, agent_type="llm_interstate_conflict")
        self.llm = llm_provider
        self.true_capability = true_capability
        self.true_war_cost = true_war_cost
        self.signal: Optional[str] = None
        self.demands: dict = {}
        self.thresholds: dict = {}
        self.payoff = 0.0
        self._last_signal_rationale: Optional[str] = None

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

    # --- Phase 1: signal broadcast ---

    def broadcast_signal(self):
        # Forced honest baseline: skip LLM call entirely.
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

    # --- Phase 2: per-neighbor demand/threshold ---

    def decide(self):
        self.demands = {}
        self.thresholds = {}

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

        table_lines = [
            "| neighbor_id | position | their_signal | last_signal | last_outcome | your_role |",
            "|---|---|---|---|---|---|",
        ]
        for ni in neighbor_info:
            table_lines.append(
                f"| {ni['id']} | {ni['pos']} | {ni['signal']} | "
                f"{ni['last_signal']} | {ni['last_outcome']} | {ni['role']} |"
            )
        neighbor_table = "\n".join(table_lines)

        prompt = DECIDE_PROMPT.format(
            game_rules=GAME_RULES,
            bluffing_hint=BLUFFING_HINT if self.model.bluffing_enabled else "",
            capability=self.true_capability,
            war_cost=self.true_war_cost,
            self_signal=self.signal,
            step=self.model.schedule_step,
            neighbor_table=neighbor_table,
        )
        raw = self.llm.query(prompt)

        parsed = None
        try:
            data = _parse_json(raw)
            for d in data.get("decisions", []):
                nid = int(d["neighbor_id"])
                ni = next((x for x in neighbor_info if x["id"] == nid), None)
                if ni is None:
                    continue
                # Use the model-determined role, not the LLM's claimed role.
                if ni["role"] == "proposer":
                    demand = float(d.get("demand", 0.5))
                    self.demands[nid] = _round_to_bucket(demand)
                else:
                    threshold = float(d.get("threshold", 0.5))
                    self.thresholds[nid] = max(0.0, min(1.0, threshold))
            parsed = data
        except Exception as e:
            if self.model.logger:
                self.model.logger.log_parse_failure(
                    step=self.model.schedule_step,
                    agent_id=self.unique_id,
                    raw_response=raw,
                    error=f"decide parse: {e}",
                )

        # Fill any missing decisions with the classic closed-form rule.
        for ni in neighbor_info:
            nid = ni["id"]
            if ni["role"] == "proposer" and nid not in self.demands:
                self.demands[nid] = self._classic_demand(ni["agent"])
            elif ni["role"] == "responder" and nid not in self.thresholds:
                self.thresholds[nid] = self._classic_threshold(ni["agent"])

        if self.model.logger:
            self.model.logger.log_llm_interaction(
                step=self.model.schedule_step,
                agent_id=self.unique_id,
                prompt=prompt,
                raw_response=raw,
                parsed={
                    "phase": "decide",
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
                "demands_to_neighbors": dict(self.demands),
                "thresholds_to_neighbors": dict(self.thresholds),
                "signal_rationale": self._last_signal_rationale,
            },
            action=self.signal,
            confidence=1.0 if parsed is not None else 0.0,
            short_rationale=(
                parsed.get("short_rationale", "")
                if parsed is not None
                else "Decide-phase parse failure; classic fallback used."
            ),
        )
        self.record_decision(decision)

    # --- Classic fallback formulas (duplicated from classic_interstate_conflict) ---

    def _classic_threshold(self, opponent) -> float:
        opp_signal = getattr(opponent, "signal", MODERATE)
        perceived_cap_opp = SIGNAL_TO_CAP.get(opp_signal, 0.5)
        perceived_p_self_wins = self.true_capability / (
            self.true_capability + perceived_cap_opp
        )
        return 1.0 - perceived_p_self_wins + self.true_war_cost

    def _classic_demand(self, opponent) -> float:
        opp_signal = getattr(opponent, "signal", MODERATE)
        perceived_cap_opp = SIGNAL_TO_CAP.get(opp_signal, 0.5)
        estimated_opp_threshold = (
            1.0
            - self.true_capability / (self.true_capability + perceived_cap_opp)
            + self.model.E_war_cost
        )
        return _round_to_bucket(estimated_opp_threshold)
