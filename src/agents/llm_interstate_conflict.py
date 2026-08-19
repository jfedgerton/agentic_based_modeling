"""LLM-based agent for the Interstate Conflict (Fearon Bargaining) game.

Supports three input modes — Calculator, Reasoner, Role-player — whose prompt
templates live in :mod:`src.prompts.ic_prompts`. Role-player has two
sub-variants controlled by ``model.signal_form``: ``"categorical"`` (default)
and ``"free_form_text"`` (LLM emits a free-form public statement instead of
a fixed label).

The LLM is queried twice per step:
  1. SIGNAL phase: chooses a signal label (categorical) or generates a
     free-form statement (only for Role-player + ``signal_form="free_form_text"``).
  2. DECIDE phase: outputs a per-neighbor demand (proposer) or threshold
     (responder), continuous in [0, 1].

On parse failure (or when ``bluffing_enabled=False``), the SIGNAL phase
falls back to a forced-truthful categorical signal and the DECIDE phase
falls back to the Fearon classic closed-form rule (which still buckets
demand into the discrete grid).
"""

import json
from typing import Literal, Optional

from src.agents.base import AgentDecision, BaseAgent
from src.llm.provider import LLMProvider
from src.prompts import get_prompt, BLUFFING_HINT

STRONG = "STRONG"
MODERATE = "MODERATE"
WEAK = "WEAK"
SIGNAL_TO_CAP = {STRONG: 0.75, MODERATE: 0.50, WEAK: 0.25}
DEMAND_BUCKETS = (0.1, 0.3, 0.5, 0.7, 0.9)

Mode = Literal["calculator", "reasoner", "roleplayer"]

# Signal surface labels the LLM is asked to output per mode.
_SIGNAL_SURFACE_LABELS: dict[str, tuple[str, ...]] = {
    "calculator": ("S1", "S2", "S3"),
    "reasoner": (STRONG, MODERATE, WEAK),
    "roleplayer": (STRONG, MODERATE, WEAK),
}

# Surface signal (LLM output) → canonical (STRONG / MODERATE / WEAK).
_SIGNAL_SURFACE_TO_CANONICAL: dict[str, dict[str, str]] = {
    "calculator": {"S1": STRONG, "S2": MODERATE, "S3": WEAK},
    "reasoner": {STRONG: STRONG, MODERATE: MODERATE, WEAK: WEAK},
    "roleplayer": {STRONG: STRONG, MODERATE: MODERATE, WEAK: WEAK},
}

# Canonical → surface, for echoing self/neighbor signals back in DECIDE prompts.
_SIGNAL_CANONICAL_TO_SURFACE: dict[str, dict[str, str]] = {
    mode: {can: surf for surf, can in mapping.items()}
    for mode, mapping in _SIGNAL_SURFACE_TO_CANONICAL.items()
}

# Role label rendering per mode (for neighbor_table).
_ROLE_LABEL: dict[str, dict[str, str]] = {
    "calculator": {"proposer": "role_X", "responder": "role_Y"},
    "reasoner": {"proposer": "proposer", "responder": "responder"},
    "roleplayer": {"proposer": "PROPOSER", "responder": "RESPONDER"},
}

# Outcome label rendering per mode.
_OUTCOME_LABEL: dict[str, dict[str, str]] = {
    "calculator": {"none": "O0", "settled": "O1", "won": "O2", "lost": "O3"},
    "reasoner": {"none": "none", "settled": "settled", "won": "won", "lost": "lost"},
    "roleplayer": {"none": "none", "settled": "settled", "won": "won", "lost": "lost"},
}

# "No prior signal recorded" rendering per mode.
_NONE_SIGNAL_RENDER: dict[str, str] = {
    "calculator": "—",
    "reasoner": "none",
    "roleplayer": "none",
}


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


class LLMInterstateConflictAgent(BaseAgent):
    """LLM-driven Fearon bargaining agent.

    Mode and signal_form are read from the model:
      - ``self.mode`` (set at __init__) controls prompt vocabulary and parsing.
      - ``self.model.signal_form`` controls SIGNAL output format ("categorical"
        or "free_form_text"; the latter is valid only for ``mode="roleplayer"``).
      - ``self.model.bluffing_enabled`` controls whether BLUFFING_HINT is injected
        and whether SIGNAL is allowed at all (False ⇒ forced truthful, no LLM call).
    """

    def __init__(self, model, llm_provider: LLMProvider,
                 true_capability: float, true_war_cost: float,
                 mode: Mode):
        super().__init__(model, agent_type="llm_interstate_conflict")
        if mode not in _SIGNAL_SURFACE_LABELS:
            raise ValueError(
                f"unknown mode: {mode!r}. Expected one of {list(_SIGNAL_SURFACE_LABELS)}"
            )
        self.mode = mode
        self.llm = llm_provider
        self.true_capability = true_capability
        self.true_war_cost = true_war_cost
        # Canonical signal (STRONG/MODERATE/WEAK) — always populated for downstream metrics
        self.signal: Optional[str] = None
        # Free-form text statement — only populated when signal_form="free_form_text"
        self.signal_text: Optional[str] = None
        self.demands: dict = {}
        self.thresholds: dict = {}
        self.payoff = 0.0
        self._last_signal_rationale: Optional[str] = None

    @property
    def signal_form(self) -> str:
        return self.model.signal_form

    # ---------------------------------------------------------------- Phase 1

    def broadcast_signal(self):
        # Forced honest baseline: skip LLM entirely.
        if not self.model.bluffing_enabled:
            self.signal = _truthful_signal(self.true_capability)
            self.signal_text = None
            self._last_signal_rationale = "Forced honest (bluffing disabled)."
            return

        self_hist = self.model._agent_history.get(
            self.unique_id,
            {"last_signal_self": None, "last_signal_text": None, "last_payoff": 0.0},
        )

        # Render own previous signal in the form the LLM expects:
        # - free-form: previous text statement
        # - categorical (Calc/Reasoner/Role-player a): mode-specific surface label
        if self.signal_form == "free_form_text":
            own_prev_signal = self_hist.get("last_signal_text") or "(no previous statement)"
        else:
            last_canonical = self_hist.get("last_signal_self")
            if last_canonical is None:
                own_prev_signal = "none"
            else:
                own_prev_signal = _SIGNAL_CANONICAL_TO_SURFACE[self.mode].get(
                    last_canonical, last_canonical
                )

        template = get_prompt("ic", self.mode, phase="signal",
                              signal_form=self.signal_form)
        format_kwargs = dict(
            capability=self.true_capability,
            war_cost=self.true_war_cost,
            step=self.model.schedule_step,
            own_previous_signal=own_prev_signal,
            own_last_round_payoff=float(self_hist.get("last_payoff", 0.0)),
        )
        # BLUFFING_HINT placeholder exists only in Reasoner / Role-player templates.
        if self.mode != "calculator":
            format_kwargs["BLUFFING_HINT"] = BLUFFING_HINT
        prompt = template.format(**format_kwargs)

        raw = self.llm.query(prompt)
        parsed = None
        try:
            data = _parse_json(raw)
            if self.signal_form == "free_form_text":
                text = str(data.get("signal_text", "")).strip()
                if not text:
                    raise ValueError("empty signal_text")
                self.signal_text = text
                # Set canonical signal to a neutral placeholder so classic
                # fallback formulas (which read opponent.signal) don't break.
                self.signal = MODERATE
            else:
                surface = str(data.get("signal", "")).upper()
                if surface not in _SIGNAL_SURFACE_LABELS[self.mode]:
                    raise ValueError(f"invalid signal value: {surface!r}")
                self.signal = _SIGNAL_SURFACE_TO_CANONICAL[self.mode][surface]
                self.signal_text = None
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
            self.signal_text = None
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
                    "signal_text": self.signal_text,
                    "rationale": self._last_signal_rationale,
                    "raw_parsed": parsed,
                },
            )

    # ---------------------------------------------------------------- Phase 2

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
                "signal_text": getattr(n, "signal_text", None),
                "role": role,
                "last_signal": hist.get("last_signal"),
                "last_outcome": hist.get("last_outcome") or "none",
            })

        # Render the neighbor block (categorical table vs free-form text block)
        if self.signal_form == "free_form_text":
            neighbor_block = self._render_neighbor_text_table(neighbor_info)
            template_var = "neighbor_text_table"
        else:
            neighbor_block = self._render_neighbor_table(neighbor_info)
            template_var = "neighbor_table"

        template = get_prompt("ic", self.mode, phase="decide",
                              signal_form=self.signal_form)
        format_kwargs = dict(
            capability=self.true_capability,
            war_cost=self.true_war_cost,
            self_signal=self._render_self_signal(),
            step=self.model.schedule_step,
        )
        format_kwargs[template_var] = neighbor_block
        if self.mode != "calculator":
            format_kwargs["BLUFFING_HINT"] = (
                BLUFFING_HINT if self.model.bluffing_enabled else ""
            )

        prompt = template.format(**format_kwargs)
        raw = self.llm.query(prompt)

        parsed = None
        try:
            data = _parse_json(raw)
            self._parse_decisions(data, neighbor_info)
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
                parsed.get("short_rationale", "") if parsed is not None
                else "Decide-phase parse failure; classic fallback used."
            ),
        )
        self.record_decision(decision)

    # --------------------------------------------------------- Render helpers

    def _render_self_signal(self) -> str:
        if self.signal_form == "free_form_text":
            return self.signal_text or ""
        if self.signal is None:
            return _NONE_SIGNAL_RENDER[self.mode]
        return _SIGNAL_CANONICAL_TO_SURFACE[self.mode].get(self.signal, str(self.signal))

    def _render_signal_label(self, canonical_signal: Optional[str]) -> str:
        if canonical_signal is None:
            return _NONE_SIGNAL_RENDER[self.mode]
        return _SIGNAL_CANONICAL_TO_SURFACE[self.mode].get(canonical_signal, str(canonical_signal))

    def _render_outcome_label(self, outcome: str) -> str:
        return _OUTCOME_LABEL[self.mode].get(outcome, outcome)

    def _render_neighbor_table(self, neighbor_info: list) -> str:
        """Categorical neighbor table (Calculator / Reasoner / Role-player-categorical)."""
        if self.mode == "roleplayer":
            col_their, col_last = "their_declaration", "last_declaration"
        else:
            col_their, col_last = "their_signal", "last_signal"
        header = f"| id | {col_their} | {col_last} | last_outcome | your_role |"
        sep = "|---|---|---|---|---|"
        rows = [header, sep]
        for ni in neighbor_info:
            their_sig = self._render_signal_label(ni["signal"])
            last_sig = self._render_signal_label(ni["last_signal"])
            last_outcome = self._render_outcome_label(ni["last_outcome"])
            role = _ROLE_LABEL[self.mode][ni["role"]]
            rows.append(
                f"| {ni['id']} | {their_sig} | {last_sig} | {last_outcome} | {role} |"
            )
        return "\n".join(rows)

    def _render_neighbor_text_table(self, neighbor_info: list) -> str:
        """Multi-line free-form neighbor block (Role-player free-form sub-variant)."""
        blocks = []
        for ni in neighbor_info:
            role_label = _ROLE_LABEL["roleplayer"][ni["role"]]
            stmt = ni.get("signal_text") or "(no statement)"
            # last-round statement text is not tracked in _dyad_history yet;
            # show the categorical fallback or "(not tracked)" placeholder.
            last_stmt = "(not tracked)"
            outcome = self._render_outcome_label(ni["last_outcome"])
            blocks.append(
                f"Neighbor {ni['id']} (your role: {role_label}):\n"
                f'  Statement this round: "{stmt}"\n'
                f'  Last round statement: "{last_stmt}"\n'
                f"  Last outcome: {outcome}"
            )
        return "\n\n".join(blocks)

    # ---------------------------------------------------- DECIDE JSON parsing

    def _parse_decisions(self, data: dict, neighbor_info: list) -> None:
        """Populate self.demands/thresholds from the LLM's 'decisions' list.

        Calculator JSON shape: ``{"id": int, "role": "role_X"/"role_Y",
        "value": float, ...}``.
        Reasoner / Role-player shape: ``{"neighbor_id": int,
        "role": "proposer"/"responder", "demand"/"threshold": float, ...}``.

        The simulator-assigned role is authoritative; the LLM's ``role`` field
        is treated as metadata only.
        """
        decisions = data.get("decisions", [])
        for d in decisions:
            # neighbor id key differs across mode JSON shapes
            raw_id = d.get("neighbor_id", d.get("id"))
            try:
                nid = int(raw_id)
            except (TypeError, ValueError):
                continue
            ni = next((x for x in neighbor_info if x["id"] == nid), None)
            if ni is None:
                continue
            sim_role = ni["role"]  # source of truth
            if self.mode == "calculator":
                val = float(d.get("value", 0.5))
            else:
                if sim_role == "proposer":
                    val = float(d.get("demand", d.get("value", 0.5)))
                else:
                    val = float(d.get("threshold", d.get("value", 0.5)))
            val = max(0.0, min(1.0, val))
            if sim_role == "proposer":
                self.demands[nid] = val
            else:
                self.thresholds[nid] = val

    # ----------------------------------------------- Classic fallback formulas

    def _classic_threshold(self, opponent) -> float:
        opp_signal = getattr(opponent, "signal", MODERATE) or MODERATE
        perceived_cap_opp = SIGNAL_TO_CAP.get(opp_signal, 0.5)
        perceived_p_self_wins = self.true_capability / (
            self.true_capability + perceived_cap_opp
        )
        return 1.0 - perceived_p_self_wins + self.true_war_cost

    def _classic_demand(self, opponent) -> float:
        opp_signal = getattr(opponent, "signal", MODERATE) or MODERATE
        perceived_cap_opp = SIGNAL_TO_CAP.get(opp_signal, 0.5)
        estimated_opp_threshold = (
            1.0
            - self.true_capability / (self.true_capability + perceived_cap_opp)
            + self.model.E_war_cost
        )
        return _round_to_bucket(estimated_opp_threshold)
