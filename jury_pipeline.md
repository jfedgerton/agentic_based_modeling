## DeGroot, M. H. (1974). Reaching a consensus. Journal of the American Statistical association, 69(345), 118-121.

# Jury Deliberation Pipeline (v1.1)

## Core research question

**Classic ABM**: jurors influence each other through **structural features** — social status, conformity pressure, majority effect. The *content* of arguments doesn't matter; only "who has status" and "how many voted which way" matters.

**L-ABM**: LLM jurors are genuinely persuaded by **argument content quality** and should in principle be **robust to social status**.

**Core hypothesis**: If we relabel the same set of arguments with permuted social-status tags (content unchanged):
- Classic verdicts should change (because classic weights by status)
- LLM verdicts should remain relatively stable (because LLM weighs content)
## or vise versa. This hypothesis can be revised and varied further 

## Structural differences from PD / CV / Interstate Conflict

| Dimension | PD / CV / Interstate Conflict | Jury |
|---|---|---|
| Space | 2D grid | No grid; 12 flat agents |
| Interaction | Synchronous, 8 Moore neighbors | Full mesh — everyone sees everyone |
| Agent output | Discrete action | **Natural-language argument + implicit vote** |
| Meaning of `step` | One game loop (fixed step count) | **One deliberation cycle (speak → listen → update); repeat until unanimous / hung** |
| Termination | Fixed step count | **Unanimous (12/12) or hung** |
| Ground truth | None | **Yes** (defendant is truly guilty or not guilty) |

---

## Pipeline

### 1. Case setup (once per simulation)

```
ground_truth ∈ {GUILTY, NOT_GUILTY}      # random or specified; balanced across simulations
case_description                          # short case text — TBD
testimony                                 # 1 testimony passage with built-in ambiguity — TBD
```

Testimony must be ambiguous enough to allow reasonable disagreement; otherwise deliberation is meaningless.

### 2. Initialize jury

12 jurors, each with:
```
belief         ∈ [0, 1]       # P(guilty), private
                              # Classic: 0.5 + Gaussian noise (σ=0.15)
                              # LLM:     LLM reads testimony and outputs a float directly
                              # Hybrid:  same as LLM
characteristics:
  social_status ∈ [0, 1]      # How much weight others place on this juror's belief; U(0,1) init
                              # (the only per-juror heterogeneity)
```

Derived from `belief`:
```
vote        = GUILTY if belief > 0.5 else NOT_GUILTY
confidence  = |belief - 0.5| × 2
```

**Important**: classic agents **do not read the testimony text**. They only see Gaussian noise around 0.5 — this simulates "noisy individual interpretation" without giving them textual access. Testimony only matters for LLM and Hybrid agents.

### 3. **Round 0** — pre-deliberation baseline (embedded Condorcet)

Every juror independently forms their belief and casts an opening vote. This is the "before deliberation" snapshot — the jury's collective accuracy without any discussion.

```
Record:
  round_0_individual_accuracy
  round_0_majority_verdict
  round_0_unanimous?
```

### 4. Deliberation rounds (round 1, 2, ..., max 10)

Each round has three synchronous phases (mirroring Interstate Conflict):
```
Phase 1 (speak):    All jurors simultaneously generate an argument
Phase 2 (listen):   Each juror sees the other 11 arguments
                    (only the latest round; no full history)
Phase 3 (update):   All jurors simultaneously update their belief
```

After each round, check termination.

### 5. Argument exchange (the core architectural difference)

| Architecture | Argument form | Update basis |
|---|---|---|
| **Classic** | No text; exposes only `(vote, confidence, social_status)` | DeGroot-style: status-weighted average of others' beliefs |
| **LLM** | Natural-language argument (1–3 sentences) | LLM reads all 11 arguments + statuses, produces new belief |
| **Hybrid** | LLM generates argument | LLM scores each argument's `quality_score`; rule weights by quality |

### 6. Update rule

**Classic** (status-weighted DeGroot, with a global fixed α):
```
new_belief = α × my_belief +
             (1 - α) × weighted_avg(neighbors' beliefs)

weight_i = social_status_i / Σ social_status_j
α        = 0.3 (global model parameter, identical for all jurors)
```

→ Classic agents **ignore argument content entirely** and update purely by status and majority direction.

**LLM**:
```
prompt = (case + testimony + my_state + 11 neighbors' {argument, vote, confidence, status})
LLM → new belief (float)
```

→ LLM sees both argument text and status, weighing them at its own discretion.

**Hybrid**:
```
LLM assigns each argument a quality_score ∈ [0, 1]
new_belief = α × my_belief + (1 - α) × Σ(q_i × belief_i) / Σ q_i
α          = 0.3 (same global value)
```

→ Hybrid replaces classic's status weights with LLM-derived **content quality** weights.

### 7. Termination

- **All 12 votes agree** → verdict reached
- **`max_rounds = 10` reached** → hung jury

### 8. Outcomes

| Tier | Metric |
|---|---|
| **Tier 1** | final_verdict, **accuracy** (verdict vs ground_truth), hung_rate, time_to_convergence (rounds), false_positive_rate, false_negative_rate |
| **Tier 2** | belief_trajectory, **deliberation_gain** (final_accuracy − round_0_accuracy), opinion_polarization, conformity_index, who_flipped_when |
| **Tier 3 (LLM-only)** | rationale themes, **argument_quality_vs_influence**, **status_bias_index** |

`deliberation_gain` is especially important: it answers *did deliberation improve or hurt accuracy?* — a question a single-shot vote cannot ask.

---

## Key ablation: status-permutation test

```
1. Run the simulation; record each juror's belief and argument
2. Randomly permute the social_status labels (content and beliefs unchanged — only the numeric status values are shuffled)
3. Re-run deliberation with permuted statuses
4. Compare verdicts:
   - Classic verdicts should change (status drives its weights)
   - LLM verdicts should be relatively stable
```

→ This directly **quantifies the LLM's robustness to status bias**.

---

## Main parameters

```
n_jurors             # default 12
max_rounds           # default 10
ground_truth         # fixed per simulation; balanced across simulations
mixing_alpha         # global self-anchoring coefficient; default 0.3
classic_noise_sigma  # initial Gaussian noise on classic belief; default 0.15
agent_type           # classic / llm / hybrid
case (description + testimony)   # stored inline in default.yaml
status_permutation   # bool — ablation switch
```

---

## Conditions matrix (experimental design)

| Condition | agent_type | status_permutation | Purpose |
|---|---|---|---|
| 1 | classic | False | Classic baseline |
| 2 | classic | True | Measures classic's status-driven shift |
| 3 | LLM | False | LLM baseline |
| 4 | LLM | True | Tests LLM's robustness to status (core hypothesis) |
| 5 | hybrid | False | Hybrid baseline |
| 6 | hybrid | True | Hybrid robustness check |

**Predictions**:
- Condition 1 vs 2: classic verdicts should differ noticeably
- Condition 3 vs 4: LLM verdicts should differ only minimally (core hypothesis)
- Condition 5 vs 6: hybrid should lie between classic and LLM

---

**v1.0 → v1.1 changes**:
- Removed per-juror `stubbornness ∈ [0,1]`
- Added global `mixing_alpha = 0.3` (shared by all jurors)
- `characteristics` now contains only `social_status` — cleaner core mechanism contrast
