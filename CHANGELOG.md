# Changelog

All notable changes to this project.

## [Unreleased] — 2026-05-12

This session added two new games (Interstate Conflict, Jury Deliberation),
four new LLM providers, and updated the existing scaffolding to integrate
them. The Interstate Conflict game was later extended with 1-round history (v3.3).

### Added — Interstate Conflict (new game)

A spatial Fearon (1995) bargaining ABM where states on a torus grid signal,
bargain, and may go to war when bargaining fails. Tests whether LLM agents
bluff and detect bluffs more strategically than rule-based agents.

**New files (6):**
- `src/models/interstate_conflict.py` — `InterstateConflictModel`: 20×20 torus, synchronous
  3-phase step (signal → decide → resolve)
- `src/agents/classic_interstate_conflict.py` — `ClassicInterstateConflictAgent`: closed-form Fearon
  rules (status-weighted DeGroot demand / threshold)
- `src/agents/llm_interstate_conflict.py` — `LLMInterstateConflictAgent`: 2 LLM calls per step
  (signal + decide), with parse-failure fallback to classic
- `src/agents/hybrid_interstate_conflict.py` — `HybridInterstateConflictAgent`: LLM estimates
  `perceived_capability`, classic Fearon formula then aggregates
- `src/utils/interstate_conflict_metrics.py` — Tier 2/3 post-processing: welfare_loss,
  signal_informativeness (mutual info), bluff_rate, spatial_clustering_war,
  demand_calibration_gap, rationale_distribution
- `tests/test_interstate_conflict.py` — smoke + invariant tests (~8 tests)

**Pipeline docs:**
-  `interstate_conflict_pipeline.md` — added core
  hypotheses (H1–H5) and one-round history (see "Changed" below)


### Added — Jury Deliberation (new game) -- DeGroot (1974).

A 12-juror deliberation model with full-mesh communication, multi-round
synchronous (speak → listen → update) dynamics, and unanimous-or-hung
termination. Tests whether LLM jurors weigh argument content over social
status.

**New files (6):**
- `src/models/jury_deliberation.py` — `JuryDeliberationModel`: 12 jurors,
  full mesh, terminates on unanimous (12/12) or hung at `max_rounds=10`
- `src/agents/classic_jury.py` — `ClassicJurorAgent`: status-weighted
  DeGroot belief update; does NOT read testimony (noise-only init)
- `src/agents/llm_jury.py` — `LLMJurorAgent`: 3 LLM calls per round
  (init / speak / update), natural-language arguments
- `src/agents/hybrid_jury.py` — `HybridJurorAgent`: LLM rates argument
  quality, rule does quality-weighted DeGroot
- `src/utils/jury_metrics.py` — Tier 2/3 post-processing:
  deliberation_gain, belief_trajectory, opinion_polarization,
  conformity_index, vote_flip_count, time_to_convergence,
  rationale_distribution, status_bias_index (pair-wise)
- `tests/test_jury_deliberation.py` — smoke + invariant tests (~8 tests)

**Pipeline docs:**
-`jury_pipeline.md` — replaced
  `stubbornness` with a single global `mixing_alpha = 0.3`

---

### Added — LLM Providers

Extended `src/llm/provider.py` with three new commercial providers
(plus an existing factory registration):

| Provider | Default model | Env var | Backend |
|---|---|---|---|
| `gemini` | `gemini-3.1-flash-lite` | `GEMINI_API_KEY` | `google-generativeai` SDK |
| `deepseek` | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` | OpenAI SDK + base_url |
| `doubao` | `Doubao-Seed-2.0-lite` | `DOUBAO_API_KEY` | OpenAI SDK + base_url |

(Existing `openai` / `anthropic` / `mock` providers retained.)

---

### Changed

**OpenAI default model**: `gpt-4o-mini` → `gpt-5.4-mini` across:
- `src/llm/provider.py` (line 71)
- `src/experiments/runner.py` (line 46 fallback)
- `config/default.yaml` (line 54)
- `run_experiment.py` (help string)


### Outstanding / TBD

- **Interstate Conflict signaling format (open design question)**: The current
  implementation uses a three-category discrete signal
  (`STRONG / MODERATE / WEAK`). An alternative is to let the LLM emit
  a free-form textual claim about its capability. 
  --> But this would break the
  apples-to-apples comparison with classic agents (which have no
  natural-language output channel). 
  
- **Jury case content**: `case_description` and `testimony` in
  `default.yaml` are still placeholder strings. 
  


