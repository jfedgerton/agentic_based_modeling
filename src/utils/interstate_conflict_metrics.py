"""Post-processing metrics for the Fearon Bargaining Grid.

Reads a run's log directory (the one written by ExperimentLogger) and
computes Tier 2 (classic-vs-LLM comparison) and Tier 3 (LLM-only) metrics
that are too heavy to compute live in the model's DataCollector.

Tier 2:
- mean_settlement_demand     mean demand on peaceful (non-war) dyads
- demand_calibration_gap     actual demand - classic-formula benchmark
- signal_informativeness     mutual information I(signal; capability_bin)
- spatial_clustering_war     fraction of war-dyads adjacent to other wars

Tier 3:
- rationale_distribution     LLM rationale frequency / top keywords (v1)
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Optional


# Constants kept in sync with src/models/interstate_conflict.py.
STRONG = "STRONG"
MODERATE = "MODERATE"
WEAK = "WEAK"
SIGNAL_TO_CAP = {STRONG: 0.75, MODERATE: 0.50, WEAK: 0.25}
DEMAND_BUCKETS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _round_to_bucket(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return min(DEMAND_BUCKETS, key=lambda b: abs(b - x))


# --- Loaders ---

def load_records(run_dir) -> list:
    """Load every line of records.jsonl."""
    path = Path(run_dir) / "records.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_llm_interactions(run_dir) -> list:
    """Load every line of llm_interactions.jsonl."""
    path = Path(run_dir) / "llm_interactions.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_dyad_outcomes(records: list) -> list:
    """Flatten the per-step dyad_outcomes lists into a single list."""
    out = []
    for r in records:
        if "dyad_outcomes" in r and isinstance(r["dyad_outcomes"], list):
            out.extend(r["dyad_outcomes"])
    return out


# --- Tier 2 metrics ---

def mean_settlement_demand(dyads: list) -> Optional[float]:
    """Mean demand across all settled (non-war) dyads."""
    settled = [d for d in dyads if not d.get("war")]
    if not settled:
        return None
    return sum(float(d["demand"]) for d in settled) / len(settled)


def demand_calibration_gap(dyads: list, E_war_cost: float) -> Optional[float]:
    """Average (actual_demand - benchmark_demand) across all dyads.

    The benchmark is the closed-form Fearon rule applied to the same
    observable inputs (proposer's true capability, responder's signal).
    Positive gap = more aggressive than benchmark, negative = more cautious.
    """
    gaps = []
    for d in dyads:
        cap_self = d.get("proposer_true_capability")
        opp_signal = d.get("responder_signal")
        actual = d.get("demand")
        if cap_self is None or opp_signal is None or actual is None:
            continue
        perceived_cap_opp = SIGNAL_TO_CAP.get(opp_signal, 0.5)
        benchmark_raw = (
            1.0 - cap_self / (cap_self + perceived_cap_opp) + E_war_cost
        )
        benchmark = _round_to_bucket(benchmark_raw)
        gaps.append(float(actual) - benchmark)
    if not gaps:
        return None
    return sum(gaps) / len(gaps)


def signal_informativeness(
    dyads: list, bin_threshold: float = 0.5
) -> Optional[float]:
    """Mutual information I(signal; capability_bin) in bits.

    Each dyad contributes 2 (capability, signal) samples (proposer +
    responder). True capability is binned by `bin_threshold`.
    """
    samples = []
    for d in dyads:
        for who in ("proposer", "responder"):
            cap = d.get(f"{who}_true_capability")
            sig = d.get(f"{who}_signal")
            if cap is None or sig is None:
                continue
            cap_bin = "STRONG_TYPE" if float(cap) > bin_threshold else "WEAK_TYPE"
            samples.append((cap_bin, sig))
    if not samples:
        return None

    n = len(samples)
    joint = Counter(samples)
    cap_marg = Counter(s[0] for s in samples)
    sig_marg = Counter(s[1] for s in samples)

    mi = 0.0
    for (c, s), n_cs in joint.items():
        p_cs = n_cs / n
        p_c = cap_marg[c] / n
        p_s = sig_marg[s] / n
        if p_cs > 0 and p_c > 0 and p_s > 0:
            mi += p_cs * math.log2(p_cs / (p_c * p_s))
    return mi


def spatial_clustering_war(
    dyads: list,
    grid_width: Optional[int] = None,
    grid_height: Optional[int] = None,
) -> Optional[float]:
    """Fraction of war-dyads whose proposer cell has at least one Moore-
    neighboring cell that is also a war-dyad in the same step.

    If grid_width / grid_height are provided, neighbors are computed on a
    torus (matching the model's grid). Otherwise edge cells lose some
    neighbors (minor effect).
    """
    wars_by_step: dict = {}
    for d in dyads:
        if not d.get("war"):
            continue
        step = d.get("step")
        pos = tuple(d.get("proposer_pos", ()))
        if not pos:
            continue
        wars_by_step.setdefault(step, []).append(pos)

    if not wars_by_step:
        return None

    total = 0
    clustered = 0
    for step, positions in wars_by_step.items():
        position_set = set(positions)
        for (px, py) in positions:
            total += 1
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if grid_width is not None and grid_height is not None:
                        nx = (px + dx) % grid_width
                        ny = (py + dy) % grid_height
                    else:
                        nx, ny = px + dx, py + dy
                    if (nx, ny) in position_set:
                        found = True
                        break
                if found:
                    break
            if found:
                clustered += 1

    return clustered / total if total > 0 else None


# --- Tier 3 metrics ---

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "to", "of", "and", "or", "in", "for", "i", "my",
    "they", "their", "this", "that", "be", "will", "would", "if", "it",
    "with", "on", "as", "so", "can", "may", "are", "was", "have", "has",
    "you", "me", "we", "our", "your", "but", "not", "no", "by", "at", "from",
    "than", "more", "less", "very", "do", "did", "does",
})


def rationale_distribution(
    llm_interactions: list, top_n: int = 10
) -> dict:
    """Aggregate LLM short_rationale strings into a keyword-frequency summary.

    v1: simple lowercase tokenization with a small stoplist. Embedding-
    based clustering is left for v2.
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
        return {"total": 0, "top_keywords": []}

    counter: Counter = Counter()
    for r in rationales:
        for token in r.lower().split():
            token = token.strip(".,!?:;\"'()[]")
            if token and len(token) > 2 and token not in _STOPWORDS:
                counter[token] += 1

    return {
        "total": len(rationales),
        "top_keywords": counter.most_common(top_n),
    }


# --- Convenience: compute everything for a run ---

def compute_all(
    run_dir,
    E_war_cost: float,
    grid_width: Optional[int] = None,
    grid_height: Optional[int] = None,
) -> dict:
    """Compute every Tier 2/3 metric for one run directory.

    Caller passes `E_war_cost` (the model's prior mean for war_cost used
    by the proposer rule) and optional grid dimensions for torus-aware
    spatial clustering.
    """
    records = load_records(run_dir)
    llm_interactions = load_llm_interactions(run_dir)
    dyads = extract_dyad_outcomes(records)

    return {
        # Tier 2
        "mean_settlement_demand": mean_settlement_demand(dyads),
        "demand_calibration_gap": demand_calibration_gap(dyads, E_war_cost),
        "signal_informativeness": signal_informativeness(dyads),
        "spatial_clustering_war": spatial_clustering_war(
            dyads, grid_width, grid_height
        ),
        # Tier 3
        "rationale_distribution": rationale_distribution(llm_interactions),
        # Meta
        "n_dyads": len(dyads),
        "n_wars": sum(1 for d in dyads if d.get("war")),
        "n_llm_calls": len(llm_interactions),
    }
