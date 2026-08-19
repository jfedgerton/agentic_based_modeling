"""Extract the Interstate Conflict dyad panel from an existing run.

IC runs already recorded everything the panel needs: ``records.jsonl``
carries, for every step, one entry per resolved dyad with both positions,
both signals, the demand and threshold, and the war outcome. So IC needs no
replay — only a format conversion.

The output uses IC_DYAD_SCHEMA, the same schema a live run writes, so tables
extracted from historical runs and tables produced by new runs are
interchangeable.

Usage:
    python scripts/extract_ic_panel.py logs/default_experiment/ic_calculator_seed31_1b7eed2f
    python scripts/extract_ic_panel.py --all
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.agent_panel import (  # noqa: E402
    IC_DYAD_SCHEMA,
    AgentPanelWriter,
    flatten_dyad_outcome,
)
from src.utils.replay import parse_run_id  # noqa: E402


def extract(run_dir: Path, out_dir: Path) -> dict:
    """Convert one run's nested dyad outcomes into a columnar table."""
    records = run_dir / "records.jsonl"
    if not records.exists():
        return {"run_id": run_dir.name, "ok": False, "reason": "缺 records.jsonl"}

    writer = AgentPanelWriter(out_dir / f"{run_dir.name}__dyads",
                              schema=IC_DYAD_SCHEMA)
    steps = 0
    malformed = 0
    with open(records) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            outcomes = rec.get("dyad_outcomes")
            if not outcomes:
                continue
            steps += 1
            for outcome in outcomes:
                writer.append(flatten_dyad_outcome(outcome))

    summary = writer.close()
    return {
        "run_id": run_dir.name,
        "ok": summary["rows"] > 0,
        "steps": steps,
        "rows": summary["rows"],
        "malformed_lines": malformed,
        "path": summary["path"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract IC dyad panels")
    parser.add_argument("run_dir", nargs="?", help="A single run directory")
    parser.add_argument("--all", action="store_true",
                        help="Process every IC run under --log-dir")
    parser.add_argument("--log-dir", default="logs/default_experiment")
    parser.add_argument("--out-dir", default="outputs/panels")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.all:
        log_dir = Path(args.log_dir)
        runs = sorted(d for d in log_dir.iterdir()
                      if d.is_dir() and parse_run_id(d.name).get("game") == "ic")
    elif args.run_dir:
        runs = [Path(args.run_dir)]
    else:
        parser.error("给出一个 run 目录，或使用 --all")

    print(f"待处理 IC run: {len(runs)}\n", flush=True)

    ok = failed = total_rows = 0
    for i, run in enumerate(runs, 1):
        result = extract(run, out_dir)
        if result["ok"]:
            ok += 1
            total_rows += result["rows"]
            print(f"[{i}/{len(runs)}] ✅ {result['run_id']}  "
                  f"{result['steps']} 步 / {result['rows']:,} 条 dyad", flush=True)
        else:
            failed += 1
            reason = result.get("reason", "无 dyad 记录")
            print(f"[{i}/{len(runs)}] ⏭  {result['run_id']}  跳过: {reason}",
                  flush=True)

    print(f"\n完成: {ok} 个成功, {failed} 个跳过, 合计 {total_rows:,} 条 dyad 记录")
    print(f"输出目录: {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
