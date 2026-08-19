# Output files


What this update added, and how to read every file a run produces.

---

## 1. What changed

### What is new

Each agent's **position** and the **neighbourhood it observed** were never recorded. The only trace of a neighbourhood in the old logs was the handful of counts embedded in each LLM prompt; position was absent entirely.

Every run now also produces one or two Parquet tables:

| File | Games | Contents |
|---|---|---|
| `agent_panel.parquet` | CV, PD | One row per agent per step: decision-time position, post-move position, state before and after, observed neighbourhood composition |
| `arrests.parquet` | CV | One row per arrest: who arrested whom, both cells, sentence length |
| `dyads.parquet` | IC | One row per dyad per step: both positions, both signals, demand/threshold, war outcome |

### Guarantee: existing results are untouched

The change is **read-only with respect to the simulation** — it draws no random numbers, does not alter activation order, and does not change a single byte of any prompt. That is not a claim but a verified property.

For each game, a completed run was re-executed with its original seed **and panel collection switched on**, then the regenerated `time_series.csv` was compared to the original **byte for byte**:

| Game | Cache hits | Cache misses | Byte-identical |
|---|---:|---:|:--:|
| CV | 111,991 | 0 | ✅ |
| IC | 80,000 | 0 | ✅ |
| PD | 40,000 | 0 | ✅ |

The CV run was checked for content as well: the neighbourhood counts, private parameters and final action in the panel were compared against the `llm_interactions.jsonl` written months earlier. All 111,991 decisions agree.


### New scripts

| Script | Purpose |
|---|---|
| `scripts/audit_runs.py` | Inventory every run directory; mark which are complete and replayable |
| `scripts/replay_one.py` | Replay a finished run and verify it reproduces byte for byte; `--panel` also produces the position data |
| `scripts/extract_ic_panel.py` | Extract the dyad table straight from old IC logs (**IC needs no replay**) |
| `scripts/verify_panel.py` | Cross-check a replayed panel against the original `llm_interactions.jsonl` |
| `scripts/verify_new_run.py` | Verify a fresh run, where there is no original output to diff against |

### What "replay" means

A replay is **re-running the simulation with the same seed and config**. Because the prompts come out byte-identical, every LLM call is served from the on-disk cache with the answer it received originally — so **no API calls, no cost, and results that match exactly**. The only difference is that this time positions are recorded.

Use a replay to backfill positions for runs that already exist. For conditions with no cache, just run them normally; the panel is produced on the first execution.

---

## 2. What a run directory contains

Taking `logs/default_experiment/cv_calculator_seed42_e91c193a/` as the example. The directory name is:

| File | Level | When present |
|---|---|---|
| `config.json` | metadata | always |
| `run.log` | metadata | always |
| `summary.json` | metadata | 
| `time_series.csv` | model | always |
| `records.jsonl` | model | always |
| `agent_panel.parquet` | **agent** | CV / PD, when the panel is on |
| `arrests.parquet` | **event** | CV, and only if an arrest occurred |
| `dyads.parquet` | **dyad** | IC, when the panel is on |
| `llm_interactions.jsonl` | agent | always created, but **0 bytes for the classic arm** |
| `parse_failures.jsonl` | agent | **only if a parse failed** |

---

## 3. Reading each file

### `summary.json` 

Its presence is **the reliable sign that a run finished** — it is written last.

```json
{
  "runtime_seconds": 41.6,
  "total_records": 100,              // = number of steps
  "total_llm_interactions": 111991,  // 0 for the classic arm
  "total_parse_failures": 29,
  "llm_stats": { "call_count": 0 }   // 0 = served entirely from cache
}
```

`llm_stats.call_count == 0` means the run made no API calls at all.

### `config.json` — full configuration snapshot

### `time_series.csv` — model-level time series

One row per step, columns are aggregate metrics.

Columns by game:

- **CV**: `rebellion_rate`, `quiet_rate`, `jailed_rate`, `arrests_this_step`
- **PD**: `cooperation_rate`, `mean_payoff`, `spatial_clustering`
- **IC**: `war_frequency`, `mean_payoff`, `welfare_loss`, `bluff_rate`, `bluff_success_rate`

All three CV rates use the **total citizen count** as denominator (jailed included), so `rebellion_rate + quiet_rate + jailed_rate = 1`.

### `records.jsonl` — step-level records

One JSON object per line, largely overlapping `time_series.csv` but with timestamps.

**IC is the important exception**: each line nests a `dyad_outcomes` array holding the full result of every dyad resolved that step (1600 entries per step on a 20×20 grid). This is why IC needs no replay — positions and both sides' actions were already there. `scripts/extract_ic_panel.py` simply flattens it into `dyads.parquet`.

### `run.log` — human-readable log

Opens with the full config, then one `Step N: {...}` line per step, parse failures as warnings, and closes with runtime and counts.

**Main use**: finding how far an **unfinished** run got — the only source of progress when `summary.json` is absent.

### `llm_interactions.jsonl` — every LLM call

```json
{
  "step": 1,
  "agent_id": "649",
  "prompt": "You are an agent in a repeated game...",
  "raw_response": "{\n  \"action\": \"QUIET\", ...}",
  "parsed": { "action": "QUIET", "confidence": 0.95, "short_rationale": "..." },
  "cached": false
}
```

**This file is 0 bytes for the classic arm** — rule-based agents never call an LLM.

`parsed.short_rationale` is the model's stated reasoning, useful for interpretability work. Treat it as **self-report**, not as the actual basis of the decision.

### `parse_failures.jsonl` — failed parses

Created only if a failure occurred. Fields: `step`, `agent_id`, `raw_response`, `error`.

On failure the agent falls back to the classical rule (`grievance − risk_aversion × arrest_prob > threshold`), so those decisions are not LLM-driven. They appear in `agent_panel.parquet` with `parse_failed = TRUE` and should usually be excluded or analysed separately.

---

## 4. The new tables

### `agent_panel.parquet` (CV)

One row per citizen per step.

| Column | Meaning |
|---|---|
| `step` | Step number (**1-based**) |
| `activation_idx` | Position in this step's activation order (0-based) |
| `agent_id` | Agent identifier |
| `agent_type` | `classic_citizen` or `llm_citizen` |
| `x`, `y` | Position **when the decision was made** (before moving) |
| `state_before` | State entering the step |
| `action` | Choice this step: `QUIET` / `ACTIVE` |
| `moved_to_x`, `moved_to_y` | Position **after moving** |
| `hardship` | Hardship (0–1), fixed per agent |
| `risk_aversion` | Risk aversion (0–1), fixed per agent |
| `legitimacy` | Regime legitimacy, shared by all agents |
| `grievance` | `hardship × (1 − legitimacy)` |
| `cops_nearby` | Cops within vision |
| `actives_nearby` | ACTIVE citizens within vision |
| `quiets_nearby` | QUIET citizens within vision |
| `arrest_prob` | The agent's own estimate of being arrested |
| `jail_term` | Remaining sentence **at the moment the row was written** |
| `confidence` | Self-reported LLM confidence; always 1.0 for classic |
| `parse_failed` | Whether the response failed to parse and the classical rule was used |

**"Neighbour actions" are the `cops_nearby` / `actives_nearby` / `quiets_nearby` columns** — the neighbourhood composition the agent genuinely saw at the moment it decided.

### `agent_panel.parquet` (PD)

| Column | Meaning |
|---|---|
| `step`, `agent_id`, `agent_type` | As above |
| `x`, `y` | Position. **PD agents never move**, so this is constant per agent |
| `action_before` → `action` | The action being replaced → the new one (`COOPERATE` / `DEFECT`) |
| `payoff` | This step's payoff, summed over all neighbours |
| `num_neighbors` / `coop_count` / `defect_count` / `coop_rate` | Neighbourhood composition |
| `best_neighbor_action` | Action of the highest-payoff neighbour |

PD positions are deterministic: `agent_id = x × height + y + 1`.

### `arrests.parquet` (CV)

| Column | Meaning |
|---|---|
| `step`, `activation_idx` | Step, and the arresting cop's activation index |
| `cop_id`, `cop_x`, `cop_y` | Cop and its position before moving |
| `target_id`, `target_x`, `target_y` | Arrested citizen and its position |
| `jail_term` | Sentence, drawn uniformly from 1 to `max_jail_term` |

### `dyads.parquet` (IC)

One row per dyad per step (1600 rows per step on a 20×20 grid).

| Column | Meaning |
|---|---|
| `step` | Step number |
| `proposer_id` / `responder_id` | Proposing and responding state |
| `proposer_x/y`, `responder_x/y` | Both positions (IC agents do not move either) |
| `proposer_signal` / `responder_signal` | Signals broadcast (`STRONG`/`MODERATE`/`WEAK`) |
| `demand` | The proposer's demand |
| `threshold` | The responder's acceptance threshold |
| `war` | War occurs when `demand > threshold` |
| `winner_id` | Winner of the war; `NA` when settled peacefully |
| `proposer_payoff` / `responder_payoff` | Payoffs |
| `*_true_capability` / `*_true_war_cost` | The true private parameters of both sides |

**The proposer role alternates each step**: the lexicographically smaller position proposes on even steps, and the roles swap on odd steps. So `proposer_id` flips back and forth for the same pair of neighbours.

---

## 5. Seven traps when analysing

**1. The grid is a torus — edges wrap**

A move from (1, 37) to (4, 4) looks like it crosses the whole grid; in fact y moved only +7 (37 → 38 → 39 → 0 → … → 4). **Distances must use torus distance**:

```r
torus_d <- function(a, b, size = 40) pmin(abs(a - b), size - abs(a - b))
# torus_d(37, 4) = 7, not 33
```

**2. `time_series.csv` is 0-indexed; `step` columns are 1-based**

Align before joining.

**3. Jailed citizens produce no panel rows**

`agent_panel.parquet` does **not** have (agents × steps) rows. A citizen arrested at step *t* with a sentence of *k* disappears from the panel for steps *t+1* through *t+k*. Its row for step *t* is still there — written before the arrest, so its `jail_term` reads 0; the real sentence lives only in `arrests.parquet`.

The shortfall equals the jailed-steps that fall **inside the run's horizon**; arrests near the end are truncated, so `sum(arrests$jail_term)` will not reconcile unless the last arrest is far enough from the end.

Because rows per step vary, **use the total citizen count as the denominator** (`n_distinct(agent_id)`), never the row count for that step.

**4. There are two positions — do not mix them up**

`x`/`y` is where the agent stood **when deciding**; `moved_to_x`/`moved_to_y` is where it ended up. Use the former to ask "what conditions produced this decision", the latter to study movement.

They chain: an agent's `x` at step *t* equals its `moved_to_x` at step *t−1*, provided it was not jailed at *t−1*.

**5. Activation is sequential over a shuffled order, not simultaneous**

The lower an agent's `activation_idx`, the more of the previous state it sees. The agent at index 0 observes a completely un-updated world.

**This means neighbourhood counts cannot be rebuilt from the position table afterwards** — the intermediate state at that agent's turn is not recoverable. The three count columns in the panel are the only ground truth.

**6. The panel holds more ACTIVE than `rebellion_rate` — this is not a bug**

A citizen chooses ACTIVE on its own turn (the panel records `action = ACTIVE`), then a cop activating **later in the same step** arrests it. By the time `rebellion_rate` is measured at the end of the step, it is JAILED and no longer counts as ACTIVE.

The correct identity is:

$$\text{rebellion\_rate}(t) \times n = \underbrace{\#\{\text{panel rows with action} = \text{ACTIVE}\}}_{\text{step } t} - \underbrace{\#\{\text{targets that had already acted}\}}_{\text{step } t}$$

The correction covers only arrests whose target had already acted that step. A citizen arrested before its own turn produces no panel row for that step, so it was never counted.

On a real classic run: 94 of 101 arrests needed the adjustment, after which all 100 steps agree exactly.

**7. `action` holds canonical labels**

The three arms ask the LLM for different vocabularies (calculator uses `A`/`B`, reasoner `QUIET`/`ACTIVE`, role-player `STAY_HOME`/`JOIN_PROTEST`), but `action` always stores `QUIET`/`ACTIVE` so arms are directly comparable. The raw LLM output is in `llm_interactions.jsonl`.

---
