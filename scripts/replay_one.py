"""Replay a single finished run and verify it reproduces exactly.

Rebuilds the model from the run's own ``config.json`` and the seed embedded
in its directory name, serves every LLM call from the on-disk prompt cache,
and byte-compares the regenerated ``time_series.csv`` against the original.

A passing comparison is the gate for every later stage: once panel logging
is added, any new difference is attributable to that change alone.

Usage:
    python scripts/replay_one.py logs/default_experiment/cv_reasoner_seed52_06d2c412
"""

import argparse
import filecmp
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.agent_panel import (  # noqa: E402
    close_panel_writers,
    make_panel_writers,
)
from src.utils.replay import (  # noqa: E402
    CacheMissError,
    build_model,
    make_replay_provider,
    original_step_count,
    parse_run_id,
)




def report_first_difference(original: Path, replayed: Path) -> None:
    """Print the first differing line so divergence can be located by step."""
    with open(original) as f1, open(replayed) as f2:
        for i, (a, b) in enumerate(zip(f1, f2)):
            if a != b:
                label = "表头" if i == 0 else f"第 {i} 步"
                print(f"\n首个差异出现在 {label} (文件第 {i + 1} 行):")
                print(f"  原始: {a.rstrip()}")
                print(f"  重放: {b.rstrip()}")
                return
    print("\n逐行内容一致，差异只在文件长度上（行数不同）。")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one run and verify it")
    parser.add_argument("run_dir", help="Path to the original run directory")
    parser.add_argument("--out-dir", default="outputs/replay_check",
                        help="Where to write the regenerated time_series.csv")
    parser.add_argument("--panel", action="store_true",
                        help="Collect the agent panel during the replay. This "
                             "is the real gate: it proves panel logging does "
                             "not perturb the trajectory.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    original_ts = run_dir / "time_series.csv"
    if not original_ts.exists():
        print(f"ERROR: 原始 time_series.csv 不存在: {original_ts}", file=sys.stderr)
        return 1

    parts = parse_run_id(run_dir.name)
    if not parts:
        print(f"ERROR: 无法解析 run_id: {run_dir.name}", file=sys.stderr)
        return 1

    try:
        with open(run_dir / "config.json") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: 无法读取 config.json: {e}", file=sys.stderr)
        return 1

    game, arm, seed = parts["game"], parts["arm"], int(parts["seed"])
    num_steps = original_step_count(run_dir)
    llm_cfg = config["llm"]

    print(f"run_id     : {run_dir.name}")
    print(f"game / arm : {game} / {arm}")
    print(f"seed       : {seed}   (从 run_id 解析，不用 config.random_seed)")
    print(f"steps      : {num_steps}")
    print(f"model      : {llm_cfg.get('model')}  temperature={llm_cfg.get('temperature')}")
    print(f"cache_dir  : {llm_cfg.get('cache_dir')}\n")

    provider, cache = (None, None)
    if arm != "classic":
        provider, cache = make_replay_provider(llm_cfg)

    out_dir = Path(args.out_dir)
    writers = {}
    if args.panel:
        writers = make_panel_writers(game, out_dir / "panels",
                                     prefix=f"{run_dir.name}__")
        print(f"panel     : 开启 ({len(writers)} 张表)\n"
              if writers else "panel     : 该 game 暂不支持，已忽略\n")

    model = build_model(game, arm, seed, config, provider, **writers)

    print("重放中...")
    step = 0
    try:
        for step in range(1, num_steps + 1):
            model.step()
    except CacheMissError as e:
        print(f"\n❌ 第 {step} 步缓存未命中，重放中止。\n{e}", file=sys.stderr)
        if cache:
            print(f"\n中止前缓存命中 {cache.stats['hits']} 次", file=sys.stderr)
        return 2

    panel_summaries = close_panel_writers(writers)

    out_dir.mkdir(parents=True, exist_ok=True)
    replayed_ts = out_dir / f"{run_dir.name}__time_series.csv"
    model.datacollector.get_model_vars_dataframe().to_csv(replayed_ts)

    identical = filecmp.cmp(original_ts, replayed_ts, shallow=False)

    print("\n" + "=" * 60)
    if cache:
        stats = cache.stats
        print(f"缓存命中  : {stats['hits']}")
        print(f"缓存未命中: {stats['misses']}   (必须为 0)")
    for name, s in panel_summaries.items():
        print(f"{name:<13}: {s['rows']} 行 -> {s['path']}")
    print(f"time_series.csv 逐字节一致: {'✅ 是' if identical else '❌ 否'}")
    print("=" * 60)

    if not identical:
        report_first_difference(original_ts, replayed_ts)
        print(f"\n重放结果保留在: {replayed_ts}")
        return 3

    print("\n基线成立：当前代码可以忠实复现该 run。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
