# ABM-LLM Benchmark Framework

A research benchmarking framework for comparing classical agent-based models (ABMs) against ABMs where agents use large language models (LLMs) to make decisions.

## Research Question

How does LLM-empowered agency diverge from classic rule-based agency across canonical social simulations, and which dimensions of LLM capability — uncertainty perception, strategic reasoning, or decision-making — drive the divergence?

## Benchmark Models

1. **Prisoner's Dilemma Grid** — Strategic interaction on a spatial grid (Nowak & May 1992)
2. **Epstein Civil Violence** — Political behavior under grievance and repression (Epstein 2002)
3. **Interstate Conflict** — Spatial bargaining with private capability, costly war, and bluffing signals (Fearon 1995)

All three run on a **torus grid**: the edges wrap, so a cell at the top row neighbours the bottom row. Distances must be computed accordingly.

## Benchmark Arms

Each game runs under four arms — one classic baseline plus three LLM input modes that incrementally layer in semantic content. This **decomposition design** lets us attribute any Classic-vs-LLM divergence to specific layers (decision-making, reasoning, or perception):

| Arm | Description | What it isolates |
|---|---|---|
| **Classic** | Fixed game-specific rule (e.g. Nowak–May for PD, Fearon closed-form for IC) | Reference baseline |
| **Calculator** | LLM receives numeric payoffs and aggregated counts with abstract action labels (`A`/`B`, `S1`/`S2`/`S3`); no game framing | Adds LLM **decision-making** flexibility |
| **Reasoner** | LLM gets formal game mechanics + semantic labels (`COOPERATE`/`DEFECT`, `STRONG`/`MODERATE`/`WEAK`), without naming the game | Adds LLM **reasoning** capability |
| **Role-player** | LLM gets a discursive scenario framing (free trade, pro-democracy protest, territorial dispute) | Adds LLM **perception** and world priors |

Whatever surface labels an arm uses, the simulator stores **canonical** actions (`QUIET`/`ACTIVE`, `COOPERATE`/`DEFECT`), so arms are directly comparable. The raw LLM output is preserved in the interaction log.

Two game-specific IVs vary within the LLM arms:

- **CV**: prompt **language** (`en` / `zh`) — cross-LLM language ablation
- **IC**: **signal form** (`categorical` / `free_form_text`) — only the Role-player arm supports free-form text statements

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Classic PD baseline
python run_experiment.py --model pd_grid --agent classic --steps 100

# Mock LLM (no API call) — fast smoke test
python run_experiment.py --model pd_grid --agent reasoner --provider mock --steps 50

# CV with Mandarin prompts
python run_experiment.py --model civil_violence --agent reasoner --language zh --steps 50

# IC roleplayer with free-form text signals
python run_experiment.py --model interstate_conflict --agent roleplayer \
    --signal-form free_form_text --bluffing-enabled --steps 50

# Real LLM (requires API key)
export DEEPSEEK_API_KEY="your-key"
python run_experiment.py --model civil_violence --agent reasoner \
    --provider deepseek --llm-model deepseek-v4-flash --steps 100 --seeds 52 53 54

# Run tests
pytest tests/ -v
```

## Outputs

Each run writes a self-contained directory under `logs/{experiment}/{run_id}/`, where `run_id` is `{game}_{arm}_seed{seed}_{hex}`.

| File | Level | Contents |
|---|---|---|
| `summary.json` | run | Runtime, record counts, LLM usage. **Written last — its presence means the run finished** |
| `config.json` | run | The fully merged config this run actually used |
| `time_series.csv` | model | One row per step of aggregate metrics |
| `records.jsonl` | model | Step records; for IC also nests every dyad outcome |
| `agent_panel.parquet` | **agent** | One row per agent per step: position, observed neighbourhood, action |
| `arrests.parquet` | **event** | CV only: who arrested whom, where, and for how long |
| `dyads.parquet` | **dyad** | IC only: both positions, both signals, demand/threshold, war outcome |
| `llm_interactions.jsonl` | agent | Every prompt and response (empty for the classic arm) |
| `parse_failures.jsonl` | agent | Malformed LLM responses, if any |
| `run.log` | run | Human-readable progress log |

**See [OUTPUTS.md](OUTPUTS.md) for the full column dictionary and the pitfalls to avoid when analysing** — torus wrapping, the 0-vs-1 step indexing offset, jailed agents producing no rows, and why neighbourhood counts cannot be reconstructed from positions after the fact.

The agent panel is on by default. `--no-agent-panel` (or `logging.log_agent_panel: false`) restores the pre-panel output exactly.

## Reproducibility

Every LLM response is cached on disk, keyed by `(prompt, model, temperature)`. Because the models draw all randomness from a seeded RNG, re-running with the same seed regenerates an identical prompt sequence and is therefore served entirely from cache — **reproducing the original trajectory exactly, offline and at no cost**.

This makes three things possible:

- **Backfilling.** Runs finished before a logging feature existed can be replayed to produce the new tables, with results guaranteed identical to the originals.
- **Resuming.** Re-running an interrupted condition replays the completed steps from cache and continues live from where it stopped.
- **Verification.** A finished run can be re-executed and compared byte for byte against itself.


## Project Structure

```
├── config/
│   └── default.yaml     # Experiment configuration
├── src/
│   ├── agents/          # Agent implementations (classic + LLM per game)
│   ├── models/          # Mesa model definitions
│   ├── prompts/         # Centralized prompt templates (PD / CV / IC × 3 modes)
│   ├── llm/             # LLM provider abstraction and prompt caching
│   ├── experiments/     # Experiment runner and evaluation
│   └── utils/           # Logging, config, agent panel writer, replay harness
├── scripts/             # Audit, replay and verification tooling
├── tests/               # Test suite
├── outputs/             # Results JSON and replay artefacts
├── logs/                # Per-run output directories
├── _archive/            # Archived files from older designs (jury game, hybrid arm)
└── run_experiment.py    # Main entry point
```

## API Keys

For real LLM experiments, set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export DOUBAO_API_KEY="..."
```

Use `--provider mock` for testing without API keys.

## Evaluation Metrics

- **PD**: cooperation_rate, mean_payoff, spatial_clustering
- **CV**: rebellion_rate, quiet_rate, jailed_rate, arrests_this_step
- **IC**: war_frequency, mean_payoff, welfare_loss, bluff_rate, bluff_success_rate
- **All**: runtime, seed sensitivity, parse failure rate

The agent-level tables support finer-grained analysis than these aggregates — spatial clustering of rebellion, the neighbourhood conditions preceding a decision, arrest geography, and dyad-level bargaining outcomes.

## Testing

```bash
pytest tests/ -q
```

The suite covers the three models, the LLM provider and cache layers, response parsing, the panel writer, and the panel's consistency with the model-level time series.
