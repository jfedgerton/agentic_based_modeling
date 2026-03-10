"""Main entry point for running benchmark experiments.

Usage:
    python run_experiment.py --model pd_grid --agent classic --seeds 42 43 44
    python run_experiment.py --model civil_violence --agent llm --provider mock
    python run_experiment.py --model pd_grid --agent hybrid --steps 50
"""

import argparse
import json
import sys

from src.experiments.runner import ExperimentRunner
from src.experiments.evaluation import BenchmarkEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="ABM-LLM Benchmark Experiment Runner")
    parser.add_argument("--model", type=str, default="pd_grid",
                        choices=["pd_grid", "civil_violence"],
                        help="Which model to run")
    parser.add_argument("--agent", type=str, default="classic",
                        choices=["classic", "llm", "hybrid"],
                        help="Agent architecture")
    parser.add_argument("--provider", type=str, default="mock",
                        choices=["openai", "anthropic", "mock"],
                        help="LLM provider (for llm/hybrid agents)")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM model name (e.g. gpt-4o-mini)")
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

    runner = ExperimentRunner(config_path=args.config,
                              config_overrides=overrides)

    print(f"Running {args.model} with {args.agent} agents "
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
