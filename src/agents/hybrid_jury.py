"""Hybrid juror for the Jury Deliberation model.

The LLM generates an argument (Phase 1) and scores each neighbor's argument
quality (Phase 3). A closed-form rule then combines those LLM-derived
quality scores with neighbors' beliefs to produce the new belief:

    new_belief = alpha * my_belief + (1 - alpha) * Σ(q_i * belief_i) / Σ q_i

This separates "argument interpretation" (LLM, via quality scoring) from
"decision aggregation" (rule, via weighted average), isolating the effect
of LLM content judgment from full LLM strategy choice.

The default quality-scoring prompt instructs the LLM to score based on
argument CONTENT and ignore social_status (status is still shown so the
prompt can be modified later to include it).
"""

import json
from typing import Optional

from src.agents.base import AgentDecision, BaseAgent
from src.llm.provider import LLMProvider

# Reuse shared prompts/helpers from llm_jury (single source of truth).
from src.agents.llm_jury import (
    GUILTY, NOT_GUILTY,
    JURY_CONTEXT, INIT_PROMPT, SPEAK_PROMPT,
    _vote_of, _confidence_of, _clip01, _parse_json,
    _format_neighbor_table,
)


HYBRID_UPDATE_PROMPT = JURY_CONTEXT + """

Round {round_num}.
Your current belief: P(guilty) = {belief:.2f} ({vote}, confidence {confidence:.2f}).

You just heard the following arguments from the other 11 jurors:
{neighbor_table}

Your job is NOT to choose a new belief. Instead, score each juror's
argument quality on a scale from 0.0 (very weak / irrelevant / confused)
to 1.0 (very strong / well-reasoned / persuasive). A downstream rule will
combine your scores with the jurors' beliefs to produce your new belief.

Score based on argument CONTENT — the reasoning, evidence, and
persuasiveness. Do NOT score by the speaker's social_status or vote.

Respond with ONLY a JSON object:
{{
  "quality_scores": [
    {{"juror_id": <int>, "quality_score": <float 0.0-1.0>, "rationale": "<short>"}}
  ],
  "short_rationale": "<1-3 sentence overall reasoning>"
}}"""


class HybridJurorAgent(BaseAgent):
    """LLM scores argument quality; closed-form rule aggregates."""

    def __init__(self, model, llm_provider: LLMProvider, social_status: float):
        super().__init__(model, agent_type="hybrid_juror")
        self.llm = llm_provider
        self.social_status = social_status
        self.belief: float = 0.5
        self.current_argument: Optional[dict] = None
        self._observations: list = []
        self._last_speak_rationale: Optional[str] = None
        self._last_quality_scores: dict = {}  # neighbor_id -> quality_score

    # --- Required by BaseAgent ---

    def get_local_observation(self) -> dict:
        return {
            "my_belief": self.belief,
            "my_social_status": self.social_status,
            "my_vote": _vote_of(self.belief),
            "neighbors": list(self._observations),
        }

    # --- Round 0: form initial belief via LLM (same as LLM juror) ---

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

    # --- Phase 1: speak (LLM generates argument, same as LLM juror) ---

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

    # --- Phase 2: listen (snapshot, identical to LLM juror) ---

    def listen(self):
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

    # --- Phase 3: LLM scores argument quality; rule aggregates ---

    def update_belief(self):
        old_belief = self.belief
        self._last_quality_scores = {}

        prompt = HYBRID_UPDATE_PROMPT.format(
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
            for entry in data.get("quality_scores", []):
                nid = int(entry["juror_id"])
                q = float(entry.get("quality_score", 0.5))
                self._last_quality_scores[nid] = max(0.0, min(1.0, q))
            parsed = data
        except Exception as e:
            if self.model.logger:
                self.model.logger.log_parse_failure(
                    step=self.model.schedule_step,
                    agent_id=self.unique_id,
                    raw_response=raw,
                    error=f"update (quality scoring) parse: {e}",
                )

        # Fill any missing quality scores with 0.5 (neutral) as fallback.
        for o in self._observations:
            self._last_quality_scores.setdefault(o["id"], 0.5)

        # Apply the closed-form rule:
        #   new_belief = alpha * my_belief + (1-alpha) * Σ(q_i * belief_i) / Σ q_i
        alpha = self.model.mixing_alpha
        total_q = sum(self._last_quality_scores.get(o["id"], 0.5)
                      for o in self._observations)
        if total_q <= 0 or not self._observations:
            weighted_avg = old_belief
        else:
            weighted_avg = sum(
                self._last_quality_scores.get(o["id"], 0.5) * o["belief"]
                for o in self._observations
            ) / total_q

        new_belief = alpha * old_belief + (1.0 - alpha) * weighted_avg
        self.belief = _clip01(new_belief)

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
                    "quality_scores": dict(self._last_quality_scores),
                    "weighted_avg_neighbor_belief": float(weighted_avg),
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
                "quality_scores": dict(self._last_quality_scores),
                "weighted_avg": float(weighted_avg),
                "alpha": float(alpha),
                "speak_rationale": self._last_speak_rationale,
            },
            action=_vote_of(self.belief),
            confidence=_confidence_of(self.belief),
            short_rationale=(
                "Hybrid: LLM-rated argument quality; rule combines quality-weighted "
                "neighbor beliefs with own belief."
            ),
        )
        self.record_decision(decision)
