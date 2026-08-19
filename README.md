# ABM-LLM Benchmark Framework

A research-grade benchmarking framework for comparing classical rule-based agent-based models (ABMs) against ABMs where agents use large language models (LLMs) to make decisions.

## Research Question

How does LLM-empowered agency diverge from classic rule-based agency across canonical social simulations, and which dimensions of LLM capability — perception, reasoning, or decision-making — drive the divergence?

## Benchmark Models

1. **Prisoner's Dilemma Grid** — Strategic interaction on a spatial grid (Nowak & May 1992)
2. **Epstein Civil Violence** — Political behavior under grievance and repression (Epstein 2002)
3. **Interstate Conflict** — Spatial bargaining with private capability, costly war, and bluffing signals (Fearon 1995)

## Benchmark Arms

Each game runs under four arms — one classic baseline plus three LLM input modes that incrementally layer in semantic content. This **decomposition design** lets us attribute any Classic-vs-LLM divergence to specific layers (decision-making, reasoning, or perception):

| Arm | Description | What it isolates |
|---|---|---|
| **Classic** | Fixed game-specific rule (e.g. Nowak–May for PD, Fearon closed-form for IC) | Reference baseline |
| **Calculator** | LLM receives numeric payoffs and aggregated counts with abstract action labels (`A`/`B`, `S1`/`S2`/`S3`); no game framing | Adds LLM **decision-making** flexibility |
| **Reasoner** | LLM gets formal game mechanics + semantic labels (`COOPERATE`/`DEFECT`, `STRONG`/`MODERATE`/`WEAK`), without naming the game | Adds LLM **reasoning** capability |
| **Role-player** | LLM gets a discursive scenario framing (free trade, pro-democracy protest, territorial dispute) | Adds LLM **perception** and world priors |

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
export OPENAI_API_KEY="your-key"
python run_experiment.py --model pd_grid --agent reasoner \
    --provider openai --llm-model gpt-4o-mini --steps 100

# Run tests
pytest tests/ -v
```

## Project Structure

```
├── config/              # Experiment configurations
│   └── default.yaml
├── src/
│   ├── agents/          # Agent implementations (classic + LLM per game)
│   ├── models/          # Mesa model definitions
│   ├── prompts/         # Centralized prompt templates (PD / CV / IC × 3 modes)
│   ├── llm/             # LLM provider abstraction and caching
│   ├── experiments/     # Experiment runner and evaluation
│   └── utils/           # Logging and configuration
├── tests/               # Test suite
├── outputs/             # Experiment results
├── logs/                # Run logs
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

## Logging

Each run records: configuration, random seeds, prompts, responses, agent actions, rationales, parse failures, and runtime metrics. Logs are stored in `logs/` as structured JSON/JSONL files.

## Evaluation Metrics

- **PD**: cooperation_rate, spatial_clustering
- **CV**: rebellion_rate, arrest_count, jailed_rate
- **IC**: war_frequency, mean_payoff, welfare_loss, bluff_rate, bluff_success_rate, signal_informativeness
- **All**: runtime, seed sensitivity, parse failure rate
