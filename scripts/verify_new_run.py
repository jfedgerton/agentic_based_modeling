"""Verify a freshly produced run that has no prior output to compare against.

A replayed run can be checked against the original it reproduces. A new run
cannot — there is nothing to diff. This script substitutes internal
consistency plus one strong external check:

  * the panel aggregates back to the model-level time series
  * decision-time positions chain correctly onto the previous step's move
  * the run replays from its own cache, byte for byte, and regenerates a
    panel identical to the one on disk

The last check is free — it reads the cache and never calls the API — and
proves three things at once: every response was cached (so the run stays
replayable), panel logging did not perturb the trajectory, and the stored
panel is exactly what the model produced.

Usage:
    python scripts/verify_new_run.py logs/default_experiment/<run_id>
"""

import argparse
import filecmp
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

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

def _arrests_after_acting(panel: pd.DataFrame, arrests: pd.DataFrame,
                          steps) -> pd.Series:
    """Per step, arrests whose target had already acted that step.

    Such a citizen chose ACTIVE and recorded it, then a cop activated later
    in the same step and jailed it. By the time the model measures
    rebellion_rate at the end of the step it counts as JAILED, so the panel
    holds one more ACTIVE than the rate does.

    Targets that had not yet acted are already absent from the panel for
    that step, so they need no adjustment.
    """
    empty = pd.Series(0, index=steps)
    if arrests is None or arrests.empty:
        return empty
    acted = arrests.merge(panel[["step", "agent_id"]],
                          left_on=["step", "target_id"],
                          right_on=["step", "agent_id"], how="inner")
    if acted.empty:
        return empty
    return acted.groupby("step").size().reindex(steps, fill_value=0)


class Check:
    """Accumulates named pass/fail results."""

    def __init__(self):
        self.rows = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))

    def report(self) -> int:
        width = max(len(n) for n, _, _ in self.rows)
        failed = 0
        for name, ok, detail in self.rows:
            print(f"{'✅' if ok else '❌'} {name:<{width}}  {detail}")
            failed += not ok
        print()
        if failed:
            print(f"❌ {failed} 项未通过")
            return 1
        print("✅ 全部通过：该 run 自洽，且可从自身缓存完整复现")
        return 0


def check_structure(run_dir: Path, game: str, check: Check) -> Path:
    """Confirm the run finished and produced a panel; return the panel path."""
    check.add("run 已正常结束 (summary.json 存在)",
              (run_dir / "summary.json").exists())
    check.add("time_series.csv 存在", (run_dir / "time_series.csv").exists())

    stem = "dyads" if game == "ic" else "agent_panel"
    panel_path = run_dir / f"{stem}.parquet"
    check.add(f"{stem}.parquet 存在", panel_path.exists())
    return panel_path


def check_panel_aggregates(panel: pd.DataFrame, arrests: pd.DataFrame,
                           run_dir: Path, game: str, check: Check) -> None:
    """The panel must reproduce the model-level rate recorded beside it."""
    if game != "cv":
        return

    ts = pd.read_csv(run_dir / "time_series.csv", index_col=0)
    steps = range(1, len(ts) + 1)
    # Every citizen appears in the panel at some step, even if jailed later.
    n_citizens = panel["agent_id"].nunique()

    active = (panel[panel["action"] == "ACTIVE"]
              .groupby("step").size().reindex(steps, fill_value=0))
    adjustment = _arrests_after_acting(panel, arrests, steps)

    derived = ((active - adjustment) / n_citizens).to_numpy()
    worst = float(abs(derived - ts["rebellion_rate"].to_numpy()).max())
    check.add("panel 聚合出的 rebellion_rate == time_series.csv",
              worst < 1e-12,
              f"最大偏差 {worst:.2e}"
              + (f", 扣除当步被捕 {int(adjustment.sum())} 人次"
                 if adjustment.sum() else ""))


def check_position_continuity(panel: pd.DataFrame, check: Check) -> None:
    """Decision-time position must equal where the agent last moved to."""
    if "moved_to_x" not in panel.columns:
        return
    p = panel.sort_values(["agent_id", "step"])
    prev_x = p.groupby("agent_id")["moved_to_x"].shift(1)
    prev_y = p.groupby("agent_id")["moved_to_y"].shift(1)
    # Jailed agents skip steps, so only consecutive steps are comparable.
    comparable = (p.groupby("agent_id")["step"].diff() == 1) & prev_x.notna()
    bad = int(((p["x"] != prev_x) | (p["y"] != prev_y))[comparable].sum())
    check.add("位置连续性: 本步决策位置 == 上一步移动后位置",
              bad == 0, f"{bad} 处断裂 / 比对 {int(comparable.sum())} 条")


def self_replay(run_dir: Path, game: str, arm: str, seed: int, config: dict,
                panel: pd.DataFrame, panel_path: Path, check: Check) -> None:
    """Re-run from the run's own cache and compare everything it produced."""
    num_steps = original_step_count(run_dir)
    provider = cache = None
    if arm != "classic":
        provider, cache = make_replay_provider(config["llm"])

    tmp = Path(tempfile.mkdtemp(prefix="verify_run_"))
    try:
        writers = make_panel_writers(game, tmp)
        model = build_model(game, arm, seed, config, provider, **writers)

        try:
            for _ in range(num_steps):
                model.step()
        except CacheMissError:
            check.add("自我重放: 缓存完整", False,
                      "出现缓存未命中 — 该 run 已无法复现")
            return

        summaries = close_panel_writers(writers)
        if cache:
            check.add("自我重放: 缓存 0 未命中", cache.stats["misses"] == 0,
                      f"命中 {cache.stats['hits']}, 未命中 {cache.stats['misses']}")

        replayed_ts = tmp / "time_series.csv"
        model.datacollector.get_model_vars_dataframe().to_csv(replayed_ts)
        check.add("自我重放: time_series.csv 逐字节一致",
                  filecmp.cmp(run_dir / "time_series.csv", replayed_ts,
                              shallow=False))

        stem = panel_path.stem
        arg = next((a for a, s in summaries.items()
                    if s["path"] and Path(s["path"]).stem == stem), None)
        if arg is None:
            check.add("自我重放: 重新生成了 panel", False, "未产出对应表")
            return

        regenerated = pd.read_parquet(summaries[arg]["path"])
        same = regenerated.equals(panel)
        check.add("自我重放: 重新生成的 panel 与磁盘上的完全相同", same,
                  f"{len(regenerated):,} 行 vs {len(panel):,} 行")

        _check_against_agent_reporters(model, panel, check)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _check_against_agent_reporters(model, panel: pd.DataFrame,
                                   check: Check) -> None:
    """Compare the panel to mesa's own agent-level frame, a separate path."""
    if not getattr(model.datacollector, "agent_reporters", None):
        return
    if "moved_to_x" not in panel.columns:
        return
    av = model.datacollector.get_agent_vars_dataframe().reset_index()
    av = av.rename(columns={"Step": "step", "AgentID": "agent_id"})
    if not {"x", "y"} <= set(av.columns):
        return

    merged = panel.merge(av[["step", "agent_id", "x", "y"]],
                         on=["step", "agent_id"], how="inner",
                         suffixes=("", "_dc"))
    bad = int(((merged["moved_to_x"] != merged["x_dc"])
               | (merged["moved_to_y"] != merged["y_dc"])).sum())
    check.add("panel 移动后位置 == DataCollector 独立记录",
              bad == 0, f"{bad} 处不符 / 比对 {len(merged):,} 条")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a freshly produced run")
    parser.add_argument("run_dir")
    parser.add_argument("--skip-replay", action="store_true",
                        help="Only run the cheap consistency checks")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    parts = parse_run_id(run_dir.name)
    if not parts:
        print(f"ERROR: 无法解析 run_id: {run_dir.name}", file=sys.stderr)
        return 1

    game, arm, seed = parts["game"], parts["arm"], int(parts["seed"])
    try:
        with open(run_dir / "config.json") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: 无法读取 config.json: {e}", file=sys.stderr)
        return 1

    print(f"run_id     : {run_dir.name}")
    print(f"game / arm : {game} / {arm}   seed: {seed}\n")

    check = Check()
    panel_path = check_structure(run_dir, game, check)
    if not panel_path.exists():
        print("panel 不存在，无法继续。该 run 可能是在 panel 功能加入之前跑的。\n")
        return check.report()

    panel = pd.read_parquet(panel_path)
    print(f"panel 行数 : {len(panel):,}\n")

    arrests_path = run_dir / "arrests.parquet"
    arrests = pd.read_parquet(arrests_path) if arrests_path.exists() else None

    check_panel_aggregates(panel, arrests, run_dir, game, check)
    check_position_continuity(panel, check)
    if not args.skip_replay:
        self_replay(run_dir, game, arm, seed, config, panel, panel_path, check)

    return check.report()


if __name__ == "__main__":
    sys.exit(main())
