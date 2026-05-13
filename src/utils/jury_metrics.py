"""Post-processing metrics for the Jury Deliberation model.

Reads a run's log directory (written by ExperimentLogger) and computes
Tier 2 / Tier 3 metrics that are too heavy to track live in the model.

Tier 2 (classic-vs-LLM comparison):
- deliberation_gain      final_accuracy - round_0_accuracy
- belief_trajectory      per-juror belief sequence across rounds
- opinion_polarization   variance of beliefs across the jury, per round
- conformity_index       mean shift of each juror toward the round-0 majority
- vote_flip_count        number of jurors who changed vote between round 0 and final
- time_to_convergence    number of rounds to reach unanimous (None if hung)

Tier 3 (LLM-only):
- rationale_distribution keyword-frequency summary of LLM short_rationale text
- status_bias_index      verdict / belief drift between a normal run and a
                         status-permuted run (requires two log dirs)
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Optional


GUILTY = "GUILTY"
NOT_GUILTY = "NOT_GUILTY"
HUNG = "HUNG"


# --- Loaders ---

def load_records(run_dir) -> list:
    path = Path(run_dir) / "records.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_llm_interactions(run_dir) -> list:
    path = Path(run_dir) / "llm_interactions.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Helpers ---

def _round_0_record(records):
    for r in records:
        if r.get("phase") == "round_0_baseline":
            return r
    return None


def _termination_record(records):
    for r in records:
        if r.get("phase") == "termination":
            return r
    return None


def _per_round_arg_records(records):
    """Yield (round_num, arguments_this_round dict) for rounds that have it."""
    for r in records:
        if "arguments_this_round" in r and r.get("phase") != "round_0_baseline":
            yield r.get("round"), r["arguments_this_round"]


# --- Tier 2: deliberation_gain ---

def deliberation_gain(records) -> Optional[float]:
    """final_accuracy - round_0_accuracy.

    Returns None if hung or missing data.
    """
    r0 = _round_0_record(records)
    rT = _termination_record(records)
    if r0 is None or rT is None:
        return None
    r0_acc = r0.get("round_0_accuracy")
    final_acc = rT.get("accuracy")
    if r0_acc is None or final_acc is None:
        return None
    return float(final_acc) - float(r0_acc)


# --- Tier 2: belief_trajectory ---

def belief_trajectory(records) -> dict:
    """Per-juror sequence of beliefs at the start of each deliberation round.

    Returns {juror_id (int): [(round_num, belief), ...]}.
    """
    trajectory: dict = {}
    for round_num, args in _per_round_arg_records(records):
        for jid_str, info in args.items():
            try:
                jid = int(jid_str)
            except (TypeError, ValueError):
                continue
            belief = info.get("belief_before_update")
            if belief is None:
                continue
            trajectory.setdefault(jid, []).append((round_num, float(belief)))
    return trajectory


# --- Tier 2: opinion_polarization ---

def opinion_polarization(records) -> dict:
    """Variance of jury beliefs at the start of each round.

    Returns {round_num: variance}.
    """
    var_by_round: dict = {}
    for round_num, args in _per_round_arg_records(records):
        beliefs = [
            float(info["belief_before_update"])
            for info in args.values()
            if info.get("belief_before_update") is not None
        ]
        if len(beliefs) < 2:
            continue
        mean = sum(beliefs) / len(beliefs)
        var = sum((b - mean) ** 2 for b in beliefs) / len(beliefs)
        var_by_round[round_num] = var
    return var_by_round


# --- Tier 2: conformity_index ---

def conformity_index(records) -> Optional[float]:
    """Mean per-juror shift of belief toward the round-0 majority direction.

    Positive value means jurors on average moved toward the round-0 majority.
    Returns None if data insufficient.
    """
    r0 = _round_0_record(records)
    if r0 is None:
        return None
    round_0_majority = r0.get("round_0_majority_verdict")
    if round_0_majority not in (GUILTY, NOT_GUILTY):
        return None
    # Majority direction: GUILTY means "increase belief"; NOT_GUILTY means "decrease".
    direction = 1.0 if round_0_majority == GUILTY else -1.0

    traj = belief_trajectory(records)
    if not traj:
        return None
    shifts = []
    for jid, series in traj.items():
        if len(series) < 2:
            continue
        first = series[0][1]
        last = series[-1][1]
        shifts.append((last - first) * direction)
    if not shifts:
        return None
    return sum(shifts) / len(shifts)


# --- Tier 2: vote_flip_count ---

def vote_flip_count(records) -> Optional[int]:
    """Number of jurors whose vote (GUILTY/NOT_GUILTY) at round 0 differs from the
    most recent recorded round's vote.

    Note: round 0's vote and the round-N pre-update vote are what's available.
    """
    traj = belief_trajectory(records)
    if not traj:
        return None
    flips = 0
    for jid, series in traj.items():
        if len(series) < 2:
            continue
        first_vote = GUILTY if series[0][1] > 0.5 else NOT_GUILTY
        last_vote = GUILTY if series[-1][1] > 0.5 else NOT_GUILTY
        if first_vote != last_vote:
            flips += 1
    return flips


# --- Tier 2: time_to_convergence ---

def time_to_convergence(records) -> Optional[int]:
    """Rounds taken to reach unanimous verdict; None if hung."""
    rT = _termination_record(records)
    if rT is None:
        return None
    if rT.get("hung"):
        return None
    return rT.get("round")


# --- Tier 3: rationale_distribution ---

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "to", "of", "and", "or", "in", "for", "i", "my",
    "they", "their", "this", "that", "be", "will", "would", "if", "it",
    "with", "on", "as", "so", "can", "may", "are", "was", "have", "has",
    "you", "me", "we", "our", "your", "but", "not", "no", "by", "at", "from",
    "than", "more", "less", "very", "do", "did", "does", "am",
})


def rationale_distribution(llm_interactions, top_n: int = 10) -> dict:
    """Aggregate LLM short_rationale strings into a keyword-frequency summary.

    v1: simple tokenization with a small stopword list. Embedding-based
    clustering is left to v2.
    """
    rationales: list = []
    for entry in llm_interactions:
        parsed = entry.get("parsed", {}) or {}
        for key in ("short_rationale", "rationale"):
            v = parsed.get(key)
            if isinstance(v, str):
                rationales.append(v.strip())
        raw = parsed.get("raw_parsed")
        if isinstance(raw, dict):
            v = raw.get("short_rationale")
            if isinstance(v, str):
                rationales.append(v.strip())

    if not rationales:
        return {"total": 0, "top_keywords": [], "by_phase": {}}

    counter: Counter = Counter()
    for r in rationales:
        for token in r.lower().split():
            token = token.strip(".,!?:;\"'()[]")
            if token and len(token) > 2 and token not in _STOPWORDS:
                counter[token] += 1

    # Also bucket by phase so we can see init vs speak vs update keyword profiles.
    by_phase: dict = {}
    for entry in llm_interactions:
        parsed = entry.get("parsed", {}) or {}
        phase = parsed.get("phase", "unknown")
        text = parsed.get("short_rationale") or parsed.get("rationale")
        if isinstance(text, str):
            by_phase.setdefault(phase, []).append(text.strip())
    phase_summary = {
        phase: {"count": len(texts), "sample": texts[:3]}
        for phase, texts in by_phase.items()
    }

    return {
        "total": len(rationales),
        "top_keywords": counter.most_common(top_n),
        "by_phase": phase_summary,
    }


# --- Tier 3: status_bias_index (requires two runs) ---

def status_bias_index(records_normal, records_permuted) -> dict:
    """Compare a normal run against a status-permuted run.

    Returns a dict with:
    - verdict_changed: bool, did the final verdict differ?
    - normal_verdict, permuted_verdict
    - final_belief_drift: |Δ mean_belief| between the two runs at termination
    """
    rT_n = _termination_record(records_normal)
    rT_p = _termination_record(records_permuted)
    if rT_n is None or rT_p is None:
        return {"verdict_changed": None, "normal_verdict": None,
                "permuted_verdict": None, "final_belief_drift": None}

    n_v = rT_n.get("verdict")
    p_v = rT_p.get("verdict")
    verdict_changed = (n_v != p_v) if (n_v and p_v) else None

    # Reconstruct round-by-round mean belief from arguments_this_round.
    def _final_mean_belief(records):
        last_round_args = None
        last_round_num = -1
        for r in records:
            if "arguments_this_round" not in r:
                continue
            rn = r.get("round", -1)
            if rn > last_round_num:
                last_round_num = rn
                last_round_args = r["arguments_this_round"]
        if not last_round_args:
            return None
        beliefs = [
            float(info["belief_before_update"])
            for info in last_round_args.values()
            if info.get("belief_before_update") is not None
        ]
        if not beliefs:
            return None
        return sum(beliefs) / len(beliefs)

    mean_n = _final_mean_belief(records_normal)
    mean_p = _final_mean_belief(records_permuted)
    drift = abs(mean_n - mean_p) if (mean_n is not None and mean_p is not None) else None

    return {
        "verdict_changed": verdict_changed,
        "normal_verdict": n_v,
        "permuted_verdict": p_v,
        "final_belief_drift": drift,
    }


# --- Convenience: compute everything for one run ---

def compute_all(run_dir) -> dict:
    """Compute every Tier 2/3 metric for a single run directory.

    For `status_bias_index` you must call it separately with two run dirs.
    """
    records = load_records(run_dir)
    llm_interactions = load_llm_interactions(run_dir)
    rT = _termination_record(records)

    return {
        # Tier 2
        "deliberation_gain": deliberation_gain(records),
        "belief_trajectory": belief_trajectory(records),
        "opinion_polarization_per_round": opinion_polarization(records),
        "conformity_index": conformity_index(records),
        "vote_flip_count": vote_flip_count(records),
        "time_to_convergence": time_to_convergence(records),
        # Tier 3
        "rationale_distribution": rationale_distribution(llm_interactions),
        # Meta
        "final_verdict": rT.get("verdict") if rT else None,
        "final_accuracy": rT.get("accuracy") if rT else None,
        "hung": rT.get("hung") if rT else None,
        "n_llm_calls": len(llm_interactions),
    }
