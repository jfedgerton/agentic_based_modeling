## reference: Fearon, J. D. (1995). Rationalist explanations for war. International Organization, 49(3), 379-414.

# Interstate Conflict Pipeline (v3.3)

## Core research question

**Classic ABM**: agents bluff mechanically (a global probability `bluffing_rate`); their signal-to-capability inference is a fixed lookup; they have no memory across rounds.

**L-ABM**: LLM agents may bluff **strategically** (only when advantageous) and may **detect others' bluffs using one-round history** (reputational reasoning).

→ In the (bluff_rate, war_rate) plane, LLM points should fall on the "smart side" of the classic curve.

## Core hypotheses

| # | Hypothesis | How to test |
|---|---|---|
| **H1 (face validity)** | bluffing_enabled=False → war_frequency ≈ 0 | Condition 1 (Fearon's complete-info baseline) |
| **H2 (bluffing causes war)** | Classic war_frequency rises monotonically with bluffing_rate | Condition 2 sweep |
| **H3 (LLM emergent bluffing)** | LLM in bluffing_enabled=True spontaneously bluffs (measured rate > 0) | Condition 3 measurement |
| **H4 (LLM strategic bluffing)** | LLM's (bluff_rate, war_rate) falls on the "smart side" of the classic curve (lower war_rate at comparable bluff_rate) | Compare Conditions 2 vs 3 |
| **H5 (signal informativeness)** | LLM signals carry more mutual information about true_capability than classic signals at comparable bluff rates | `signal_informativeness` metric |

## 1. Initialize grid

Build a 2D grid (default **20×20**).
Each cell is a state with two true attributes (randomly initialized):
```
capability ~ Beta(2, 2) scaled to [0.1, 0.9]
war_cost   ~ Beta(2, 5) scaled to [0.1, 0.5]
```

## 2. Define neighbors

Each state has Moore neighbors (up to 8), with a torus boundary consistent with the PD grid.

## 3. Generate local disputes

In each round (one model step) every state simultaneously bargains with all of its neighbors.

**PD-style synchronous three-phase update**:
```
Phase 1 signal:    All states broadcast their signal in parallel
Phase 2 decide:    All states compute demand / threshold in parallel
Phase 3 resolve:   Pairwise payoffs computed in parallel
```

Each dyad's issue value = 1.

## 4. Private information

Each state knows its own `capability` and `war_cost`; it can only infer a neighbor's capability through the neighbor's signal and **one round of history**.

**Observation structure** (per-neighbor list):
```
neighbors = [
  {
    id, pos,
    signal_this_round,
    last_signal,          # the neighbor's signal in the previous round
    last_outcome,         # outcome of last round's dyad: "settled" / "war_won" / "war_lost" / "none"
  },
  ...  (up to 8)
]

own_state = {
  ...,
  last_signal_self,       # my own previous signal
  last_round_payoff,      # my own total payoff last round
}
```

## 5. Global public signaling / bluffing

Each round, every state broadcasts a single **global public signal** visible to all Moore neighbors.

Signal values: `STRONG / MODERATE / WEAK`

Fixed signal-to-capability lookup (shared by all agents):
```
signal_to_cap = {STRONG: 0.75, MODERATE: 0.50, WEAK: 0.25}
```

### Global bluffing switch (`bluffing_enabled`)

**Off (`False`)** — baseline:
- All agents (including LLM) are forced by code to emit a truthful signal.
```
truthful_signal(cap) = STRONG    if cap > 0.66
                       MODERATE  if cap > 0.33
                       WEAK      otherwise
```

**On (`True`)**:
- **Classic**: governed by `bluffing_rate` — weak states (cap<0.5) signal STRONG with probability `bluffing_rate`.
- **LLM**: free to choose signal, with access to its own previous signal + previous round payoff (bluff frequency is measured post-hoc).

## 6. Bargaining: TIOLI with step-parity proposer alternation

For each neighboring dyad `(A, B)`:
```
primary cell = whichever has the smaller (x, y) lex order
even step:  primary is proposer, the other is responder
odd step:   roles swap
```

Proposer makes a **discrete demand**: `x ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`
Responder pre-commits an `acceptance threshold x*`: `accept iff x ≤ x*`

### Decision rules

**Classic** (Fearon §pp.387, §pp.394 — **stays memoryless**, no history use):
```
Responder:  x* = 1 - cap_B/(cap_B + perceived_cap_A) + war_cost_B
Proposer:   x  = round_to_bucket(1 - cap_A/(cap_A + perceived_cap_B) + E[war_cost])

where perceived_cap = signal_to_cap[signal]     # fixed lookup; no reputational reasoning
```

**LLM** (DECIDE phase sees history):
- Prompt provides:
  - own cap, war_cost, this-round signal
  - for each of the 8 neighbors: this-round signal + last_signal + last_outcome
- LLM directly outputs demand/threshold (perception and decision are folded together)
- Can perform reputational reasoning: e.g. "this neighbor bluffed and lost the last war — its STRONG signal this round is suspicious."

**Hybrid** (INTERPRET phase sees history):
- LLM estimates per-neighbor `perceived_capability` using history.
- Rule then plugs LLM's `perceived_cap` into the Fearon formula above.

### Resolution

```
x ≤ x*:  accepted → proposer gets x, responder gets 1 - x
x > x*:  rejected → enter war lottery (Step 7)
```

## 7. War if bargaining fails

Victory probability: `P(A wins) = cap_A / (cap_A + cap_B)`
War payoffs: winner gets `1 - war_cost`; loser gets `-war_cost`.

## 8. Aggregate outcomes

Per round each state's payoff = sum over its 8 dyads.

End-of-round bookkeeping:
- `number of wars`, `number of settlements`
- `successful bluffs`, `failed bluffs`
- **`_dyad_history[(self, neighbor)] = {last_signal, last_outcome}`** (per dyad, bidirectional)
- **`_agent_history[agent] = {last_signal_self, last_payoff}`** (per agent)

## 9. Update after each round

**One-round memory** (added in v3.3, applies to LLM and Hybrid only; Classic **stays memoryless** to remain faithful to Fearon 1995):
- The next round's LLM/Hybrid prompts automatically receive last round's history.
- Classic ignores history and continues using the fixed SIGNAL_TO_CAP lookup.

A simulation runs for **50 steps** by default.

---

## Three architectures: side-by-side

| | Classic | Hybrid | LLM |
|---|---|---|---|
| Bluff (signal choice) | rule (`bluffing_rate`) | LLM free choice | LLM free choice |
| Bluff prompt uses history | — | ✅ (own previous signal/payoff) | ✅ |
| Perception (signal → cap) | fixed lookup | **LLM-estimated perceived_cap (uses history)** | implicit (LLM black box) |
| Decision (cap → demand/threshold) | Fearon formula | Fearon formula | LLM free |
| LLM calls / step | 0 | 2 (signal + interpret) | 2 (signal + decide) |

---

## Main parameters to vary

```
bluffing_enabled    (bool, global switch)
bluffing_rate       (Classic only; for LLM it is a measured output)
war_cost            (varied by adjusting the Beta distribution bounds)
neighbor_structure  (Moore vs. von Neumann, etc. — extension in v2)
```

## Main outcomes

### Tier 1 (Fearon core + welfare)
```
war_frequency, mean_payoff, welfare_loss, bluff_rate, bluff_success_rate
```

### Tier 2 (distinguishing classic vs. LLM)
```
mean_settlement_demand, demand_calibration_gap,
signal_informativeness, spatial_clustering_war
```

### Tier 3 (LLM-only)
```
rationale_distribution (text clustering / pattern analysis of LLM short_rationale)
```

---

## Experimental design

| Condition | bluffing_enabled | Agent type | bluffing_rate |
|---|---|---|---|
| 1 (baseline) | False | classic or LLM | n/a (forced honest) |
| 2 (classic + bluff) | True | classic | sweep {0.1, 0.3, 0.5, 0.7} |
| 3 (LLM + bluff) | True | LLM | measured output |

**Default simulation params**: 20×20 grid, 50 steps, ≥5 seeds per condition.

---

**v3.2 → v3.3 changes**:
- Added **Core hypotheses (H1–H5)** section
- **One-round history**: LLM/Hybrid prompts now expose `own_previous_signal` + `own_last_round_payoff` (SIGNAL phase) and per-neighbor `last_signal` + `last_outcome` (DECIDE/INTERPRET phase)
- Classic remains memoryless (Option α, faithful to Fearon 1995)
- Step 9 wording updated
- New three-architecture comparison table
