"""LLM-driven juror for the Jury Deliberation model.

Uses three LLM calls per round per juror:
- initialize_belief: LLM reads the case + testimony and produces an initial
  P(guilty) belief (Round 0 only).
- speak: LLM generates a 1-3 sentence natural-language argument given its
  current belief.
- update_belief: LLM reads the other 11 jurors' arguments (with vote /
  confidence / social_status metadata) and produces a new belief.

Parse failures fall back to:
- initialize_belief: classic 0.5 + Gaussian noise.
- speak: a templated string argument.
- update_belief: classic status-weighted DeGroot rule.
"""

import json
from typing import Optional

from src.agents.base import AgentDecision, BaseAgent
from src.llm.provider import LLMProvider

# Constants (keep in sync with src/models/jury_deliberation.py)
GUILTY = "GUILTY"
NOT_GUILTY = "NOT_GUILTY"


def _vote_of(belief: float) -> str:
    return GUILTY if belief > 0.5 else NOT_GUILTY


def _confidence_of(belief: float) -> float:
    return abs(belief - 0.5) * 2.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _parse_json(text: str) -> dict:
    """Extract JSON from an LLM response, handling markdown code fences."""
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


# --- Prompts ---

JURY_CONTEXT = """You are juror #{juror_id} in a 12-juror jury deliberating a criminal case.

Case: {case_description}

Testimony: {testimony}

Your social_status: {social_status:.2f} (other jurors weight your belief by this value when forming a status-weighted average; it is also visible to them)."""


INIT_PROMPT = JURY_CONTEXT + """

This is Round 0 — you have NOT yet heard any other juror. Based ONLY on
the testimony above, what is your initial belief that the defendant is
GUILTY (a probability between 0.0 and 1.0)?

Respond with ONLY a JSON object:
{{
  "belief": <float 0.0-1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}"""


SPEAK_PROMPT = JURY_CONTEXT + """

Round {round_num}.
Your current belief: P(guilty) = {belief:.2f} ({vote}, confidence {confidence:.2f}).

Generate a brief argument (1-3 sentences) to share with the other 11 jurors
this round. You may explain reasoning, point to evidence, raise doubt, or
attempt to persuade.

Respond with ONLY a JSON object:
{{
  "argument": "<your 1-3 sentence argument>",
  "short_rationale": "<your strategy in one sentence>"
}}"""


UPDATE_PROMPT = JURY_CONTEXT + """

Round {round_num}.
Your current belief: P(guilty) = {belief:.2f} ({vote}, confidence {confidence:.2f}).

You just heard the following arguments from the other 11 jurors:
{neighbor_table}

Update your belief based on what you heard. You may weigh argument content,
others' votes, their confidence, and their social_status as you see fit.

Respond with ONLY a JSON object:
{{
  "new_belief": <float 0.0-1.0>,
  "short_rationale": "<1-3 sentence explanation of how you updated>"
}}"""


def _format_neighbor_table(observations: list) -> str:
    """Build a markdown table of the 11 neighbors' arguments for the prompt."""
    lines = [
        "| juror_id | argument | vote | confidence | social_status |",
        "|---|---|---|---|---|",
    ]
    for o in observations:
        arg = o.get("argument") or {}
        text = arg.get("text", "(no text)") if isinstance(arg, dict) else str(arg)
        # Squash any newlines so the table stays single-row per neighbor.
        text = text.replace("\n", " ").strip()
        lines.append(
            f"| {o['id']} | {text} | {o['vote']} | "
            f"{o['confidence']:.2f} | {o['social_status']:.2f} |"
        )
    return "\n".join(lines)


class LLMJurorAgent(BaseAgent):
    """LLM-driven juror that produces natural-language arguments and updates."""

    def __init__(self, model, llm_provider: LLMProvider, social_status: float):
        super().__init__(model, agent_type="llm_juror")
        self.llm = llm_provider
        self.social_status = social_status
        self.belief: float = 0.5  # placeholder; set in initialize_belief()
        self.current_argument: Optional[dict] = None
        self._observations: list = []
        self._last_speak_rationale: Optional[str] = None

    # --- Required by BaseAgent ---

    def get_local_observation(self) -> dict:
        return {
            "my_belief": self.belief,
            "my_social_status": self.social_status,
            "my_vote": _vote_of(self.belief),
            "neighbors": list(self._observations),
        }

    # --- Round 0: form initial belief from testimony ---

    def initialize_belief(self):
        prompt = INIT_PROMPT.format(
            juror_id=self.unique_id,
            case_description=self.model.case_description,
            testimony=self.model.testimony,
            social_status=self.social_status,
        )
        raw = self.llm.query(prompt)
        parsed = None
        try:
            data = _parse_json(raw)
            self.belief = _clip01(float(data["belief"]))
            parsed = data
        except Exception as e:
            if self.model.logger:
                self.model.logger.log_parse_failure(
                    step=0,
                    agent_id=self.unique_id,
                    raw_response=raw,
                    error=f"init parse: {e}",
                )
            noise = self.random.gauss(0.0, 0.15)
            self.belief = _clip01(0.5 + noise)

        if self.model.logger:
            self.model.logger.log_llm_interaction(
                step=0,
                agent_id=self.unique_id,
                prompt=prompt,
                raw_response=raw,
                parsed={"phase": "init", "belief": self.belief, "raw_parsed": parsed},
            )

    # --- Phase 1: speak ---

    def speak(self):
        prompt = SPEAK_PROMPT.format(
            juror_id=self.unique_id,
            case_description=self.model.case_description,
            testimony=self.model.testimony,
            social_status=self.social_status,
            round_num=self.model.schedule_step,
            belief=self.belief,
            vote=_vote_of(self.belief),
            confidence=_confidence_of(self.belief),
        )
        raw = self.llm.query(prompt)
        parsed = None
        argument_text = None
        try:
            data = _parse_json(raw)
            argument_text = str(data["argument"]).strip()
            self._last_speak_rationale = data.get("short_rationale", "")
            parsed = data
        except Exception as e:
            if self.model.logger:
                self.model.logger.log_parse_failure(
                    step=self.model.schedule_step,
                    agent_id=self.unique_id,
                    raw_response=raw,
                    error=f"speak parse: {e}",
                )
            argument_text = (
                f"(parse failure; templated fallback) I currently believe "
                f"P(guilty) = {self.belief:.2f}."
            )
            self._last_speak_rationale = f"Parse failure: {e}"

        self.current_argument = {
            "text": argument_text,
            "vote": _vote_of(self.belief),
            "confidence": _confidence_of(self.belief),
            "social_status": self.social_status,
        }

        if self.model.logger:
            self.model.logger.log_llm_interaction(
                step=self.model.schedule_step,
                agent_id=self.unique_id,
                prompt=prompt,
                raw_response=raw,
                parsed={
                    "phase": "speak",
                    "argument": argument_text,
                    "rationale": self._last_speak_rationale,
                    "raw_parsed": parsed,
                },
            )

    # --- Phase 2: listen ---

    def listen(self):
        """Snapshot every other juror's argument/belief/status before any update."""
        self._observations = []
        for a in self.model.agents:
            if a is self:
                continue
            if not hasattr(a, "belief") or not hasattr(a, "social_status"):
                continue
            self._observations.append({
                "id": a.unique_id,
                "belief": float(a.belief),
                "social_status": float(a.social_status),
                "vote": _vote_of(a.belief),
                "confidence": _confidence_of(a.belief),
                "argument": getattr(a, "current_argument", None),
            })

    # --- Phase 3: update via LLM ---

    def update_belief(self):
        old_belief = self.belief

        prompt = UPDATE_PROMPT.format(
            juror_id=self.unique_id,
            case_description=self.model.case_description,
            testimony=self.model.testimony,
            social_status=self.social_status,
            round_num=self.model.schedule_step,
            belief=old_belief,
            vote=_vote_of(old_belief),
            confidence=_confidence_of(old_belief),
            neighbor_table=_format_neighbor_table(self._observations),
        )
        raw = self.llm.query(prompt)
        parsed = None
        try:
            data = _parse_json(raw)
            self.belief = _clip01(float(data["new_belief"]))
            rationale = data.get("short_rationale", "")
            parsed = data
        except Exception as e:
            if self.model.logger:
                self.model.logger.log_parse_failure(
                    step=self.model.schedule_step,
                    agent_id=self.unique_id,
                    raw_response=raw,
                    error=f"update parse: {e}",
                )
            # Fallback: status-weighted DeGroot on the observations.
            self.belief = self._classic_degroot_fallback(old_belief)
            rationale = (
                f"Parse failure; fell back to status-weighted DeGroot. "
                f"Error: {e}"
            )

        if self.model.logger:
            self.model.logger.log_llm_interaction(
                step=self.model.schedule_step,
                agent_id=self.unique_id,
                prompt=prompt,
                raw_response=raw,
                parsed={
                    "phase": "update",
                    "old_belief": old_belief,
                    "new_belief": self.belief,
                    "rationale": rationale,
                    "raw_parsed": parsed,
                },
            )

        decision = AgentDecision(
            observed_state_summary=(
                f"belief={old_belief:.2f}, status={self.social_status:.2f}, "
                f"vote={_vote_of(old_belief)}; heard {len(self._observations)} arguments"
            ),
            beliefs={
                "old_belief": float(old_belief),
                "new_belief": float(self.belief),
                "speak_rationale": self._last_speak_rationale,
                "n_neighbors": len(self._observations),
            },
            action=_vote_of(self.belief),
            confidence=_confidence_of(self.belief),
            short_rationale=rationale,
        )
        self.record_decision(decision)

    # --- Helpers ---

    def _classic_degroot_fallback(self, old_belief: float) -> float:
        if not self._observations:
            return old_belief
        alpha = self.model.mixing_alpha
        total_status = sum(o["social_status"] for o in self._observations)
        if total_status <= 0:
            weighted_avg = 0.5
        else:
            weighted_avg = sum(
                o["belief"] * o["social_status"] for o in self._observations
            ) / total_status
        return _clip01(alpha * old_belief + (1.0 - alpha) * weighted_avg)
