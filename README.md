# ABM-LLM Benchmark Framework

A research-grade benchmarking framework for comparing classical rule-based agent-based models (ABMs) against ABMs where agents use large language models (LLMs) to make decisions.

## Research Question

Do LLM-based agents produce systematically different emergent dynamics compared to classical rule-based agents in canonical social simulations?

## Benchmark Models

1. **Prisoner's Dilemma Grid** — Strategic interaction on a spatial grid (Nowak & May 1992)
2. **Epstein Civil Violence** — Political behavior under grievance and repression (Epstein 2002)

## Agent Architectures

- **Classic**: Rule-based decision rules replicating canonical ABM behavior
- **LLM**: LLM API chooses among fixed actions using only local information
- **Hybrid**: LLM interprets environment, final decision constrained by rule structure

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run classic PD Grid baseline
python run_experiment.py --model pd_grid --agent classic --steps 100

# Run with mock LLM agents
python run_experiment.py --model pd_grid --agent llm --provider mock --steps 50

# Run Civil Violence model
python run_experiment.py --model civil_violence --agent classic --steps 100

# Run with real LLM (requires API key)
export OPENAI_API_KEY="your-key"
python run_experiment.py --model pd_grid --agent llm --provider openai --llm-model gpt-4o-mini

# Run tests
pytest tests/ -v
```

## Project Structure

```
├── config/              # Experiment configurations
│   └── default.yaml
├── src/
│   ├── agents/          # Agent implementations (classic, LLM, hybrid)
│   ├── models/          # Mesa model definitions
│   ├── llm/             # LLM provider abstraction and caching
│   ├── experiments/     # Experiment runner and evaluation
│   └── utils/           # Logging and configuration
├── tests/               # Test suite
├── outputs/             # Experiment results
├── logs/                # Run logs
├── notebooks/           # Analysis notebooks
└── run_experiment.py    # Main entry point
```

## API Keys

For real LLM experiments, set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Use `--provider mock` for testing without API keys.

## Logging

Each run records: configuration, random seeds, prompts, responses, agent actions, rationales, parse failures, and runtime metrics. Logs are stored in `logs/` as structured JSON/JSONL files.

## Evaluation Metrics

- **Emergent behavior**: cooperation rate, spatial clustering, rebellion rate, arrest dynamics
- **Benchmark**: runtime, seed sensitivity, temperature sensitivity, malformed response rate
- **Interpretability**: rationale theme frequency, theme-action relationships
