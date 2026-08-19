"""Main entry point for running benchmark experiments.

Usage:
    python run_experiment.py --model pd_grid --agent classic --seeds 42 43 44
    python run_experiment.py --model civil_violence --agent reasoner --provider mock
    python run_experiment.py --model civil_violence --agent roleplayer --language zh
    python run_experiment.py --model interstate_conflict --agent roleplayer \
        --signal-form free_form_text --bluffing-enabled
"""

import argparse
import sys

from src.experiments.runner import ExperimentRunner
from src.experiments.evaluation import BenchmarkEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="ABM-LLM Benchmark Experiment Runner")
    parser.add_argument("--model", type=str, default="pd_grid",
                        choices=["pd_grid", "civil_violence", "interstate_conflict"],
                        help="Which model to run")
    parser.add_argument("--agent", type=str, default="classic",
                        choices=["classic", "calculator", "reasoner", "roleplayer"],
                        help="Benchmark arm (one of 4)")
    parser.add_argument("--provider", type=str, default="mock",
                        choices=["openai", "anthropic", "gemini", "deepseek", "doubao", "mock"],
                        help="LLM provider (only used when --agent != classic)")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM model name (e.g. gpt-5.4-mini)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM temperature")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Random seeds for replications")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of simulation steps")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for results JSON")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable LLM prompt caching")

    # Game-specific IV overrides (only applied when relevant model is chosen)
    parser.add_argument("--language", type=str, default=None,
                        choices=["en", "zh"],
                        help="Prompt language (CV only)")
    parser.add_argument("--signal-form", type=str, default=None,
                        choices=["categorical", "free_form_text"],
                        help="Signal form (IC only; free_form_text valid only with roleplayer arm)")
    bluffing_group = parser.add_mutually_exclusive_group()
    bluffing_group.add_argument("--bluffing-enabled", dest="bluffing",
                                 action="store_true", default=None,
                                 help="Enable bluffing (IC only)")
    bluffing_group.add_argument("--no-bluffing", dest="bluffing",
                                 action="store_false",
                                 help="Disable bluffing — forced truthful signal (IC only)")

    args = parser.parse_args()

    # Build config overrides from CLI args
    overrides = {
        "experiment": {
            "num_steps": args.steps,
        },
        "agent": {
            "type": args.agent,
        },
        "llm": {
            "provider": args.provider,
            "temperature": args.temperature,
            "cache_enabled": not args.no_cache,
        },
    }
    if args.llm_model:
        overrides["llm"]["model"] = args.llm_model

    # Game-specific overrides (only when explicitly passed)
    cv_overrides = {}
    if args.language is not None:
        cv_overrides["language"] = args.language
    if cv_overrides:
        overrides["civil_violence"] = cv_overrides

    ic_overrides = {}
    if args.signal_form is not None:
        ic_overrides["signal_form"] = args.signal_form
    if args.bluffing is not None:
        ic_overrides["bluffing_enabled"] = args.bluffing
    if ic_overrides:
        overrides["interstate_conflict"] = ic_overrides

    runner = ExperimentRunner(config_path=args.config,
                              config_overrides=overrides)

    print(f"Running {args.model} with {args.agent} arm "
          f"({len(args.seeds)} seeds, {args.steps} steps)")

    results = runner.run_benchmark(
        model_name=args.model,
        seeds=args.seeds,
        num_steps=args.steps,
    )

    output_path = runner.save_results(output_path=args.output)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    evaluator = BenchmarkEvaluator(results)
    print(evaluator.generate_summary_table())


if __name__ == "__main__":
    main()
