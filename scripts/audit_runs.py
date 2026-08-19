"""Stage 0 — inventory every experiment run under a log directory.

Determines which runs are complete enough to be replayed. Writes a CSV
manifest and prints a summary grouped by game and arm.

Only small files (config.json, summary.json, time_series.csv) are read;
the multi-hundred-MB jsonl files are stat-ed, never parsed. Even so, a
scan over network-backed storage can take many minutes, so each row is
flushed to disk as it is produced and a re-run resumes where the last one
stopped.

Usage:
    python scripts/audit_runs.py
    python scripts/audit_runs.py --restart          # ignore existing rows
    python scripts/audit_runs.py --report-only      # just summarise the CSV
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# run_id format: {game}_{arm}_seed{seed}_{uuid8}
RUN_ID_RE = re.compile(
    r"^(?P<game>pd|cv|ic)_"
    r"(?P<arm>classic|calculator|reasoner|roleplayer)_"
    r"seed(?P<seed>\d+)_"
    r"(?P<uid>[0-9a-f]{8})$"
)

# Files a finalized run is expected to have. ExperimentLogger.finalize()
# writes summary.json last, so its absence means the run never completed.
CORE_FILES = ("config.json", "summary.json", "records.jsonl", "time_series.csv")
ALL_FILES = CORE_FILES + ("llm_interactions.jsonl", "parse_failures.jsonl")

FIELDNAMES = (
    ["run_id", "game", "arm", "seed", "malformed_run_id"]
    + [f"{p}_{n.split('.')[0]}" for n in ALL_FILES for p in ("has", "size")]
    + ["runtime_seconds", "total_llm_interactions", "total_parse_failures",
       "api_call_count", "num_steps_config", "llm_model", "temperature",
       "cache_dir", "language", "steps_recorded", "replayable"]
)

LLM_ARMS = ("calculator", "reasoner", "roleplayer")


def parse_run_id(run_id: str) -> dict:
    """Split a run directory name into its components, or return {} if malformed."""
    m = RUN_ID_RE.match(run_id)
    return m.groupdict() if m else {}


def _read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _count_lines(path: Path) -> int:
    """Line count for a small text file. Returns -1 if unreadable."""
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except OSError:
        return -1


def audit_run(run_dir: Path) -> dict:
    """Inspect one run directory and return a manifest row."""
    parts = parse_run_id(run_dir.name)

    # One scandir pass: name -> size, so each file is stat-ed exactly once.
    try:
        sizes = {e.name: e.stat().st_size for e in os.scandir(run_dir) if e.is_file()}
    except OSError:
        sizes = {}

    row = {
        "run_id": run_dir.name,
        "game": parts.get("game", ""),
        "arm": parts.get("arm", ""),
        "seed": parts.get("seed", ""),
        "malformed_run_id": not parts,
    }
    for name in ALL_FILES:
        key = name.split(".")[0]
        row[f"has_{key}"] = name in sizes
        row[f"size_{key}"] = sizes.get(name, 0)

    summary = _read_json(run_dir / "summary.json")
    llm_stats = summary.get("llm_stats") or {}
    row["runtime_seconds"] = summary.get("runtime_seconds", "")
    row["total_llm_interactions"] = summary.get("total_llm_interactions", "")
    row["total_parse_failures"] = summary.get("total_parse_failures", "")
    # call_count == 0 means the run was served entirely from cache.
    row["api_call_count"] = llm_stats.get("call_count", "")

    config = _read_json(run_dir / "config.json")
    row["num_steps_config"] = config.get("experiment", {}).get("num_steps", "")
    row["llm_model"] = config.get("llm", {}).get("model", "")
    row["temperature"] = config.get("llm", {}).get("temperature", "")
    row["cache_dir"] = config.get("llm", {}).get("cache_dir", "")
    row["language"] = config.get("civil_violence", {}).get("language", "")

    ts = run_dir / "time_series.csv"
    # Subtract the header row to get the number of simulated steps.
    row["steps_recorded"] = (_count_lines(ts) - 1) if ts.exists() else -1

    # An LLM arm with no interaction log cannot be verified against the cache.
    row["replayable"] = bool(
        parts
        and all(row[f"has_{n.split('.')[0]}"] for n in CORE_FILES)
        and row["steps_recorded"] > 0
        and (row["arm"] not in LLM_ARMS or row["has_llm_interactions"])
    )
    return row


def _load_done(out: Path) -> set:
    """run_ids already present in a partial manifest, so a re-run can resume."""
    if not out.exists():
        return set()
    try:
        with open(out, newline="") as f:
            return {r["run_id"] for r in csv.DictReader(f) if r.get("run_id")}
    except (OSError, csv.Error):
        return set()


def _as_bool(v) -> bool:
    """CSV round-trips booleans as the strings 'True'/'False'."""
    return v is True or v == "True"


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def scan(log_dir: Path, out: Path, restart: bool) -> None:
    """Walk every run directory, appending one manifest row at a time."""
    run_dirs = sorted(d for d in log_dir.iterdir() if d.is_dir())

    done = set() if restart else _load_done(out)
    todo = [d for d in run_dirs if d.name not in done]

    print(f"目录总数 {len(run_dirs)}  已扫描 {len(done)}  待扫描 {len(todo)}",
          flush=True)
    if not todo:
        print("无需扫描，直接汇总。", flush=True)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = restart or not out.exists()
    mode = "w" if write_header else "a"

    with open(out, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()
        for i, d in enumerate(todo, 1):
            writer.writerow(audit_run(d))
            # Flush every row: the scan is slow enough that an interrupted
            # run should never lose more than the directory in flight.
            f.flush()
            os.fsync(f.fileno())
            if i % 25 == 0 or i == len(todo):
                print(f"  ...{i}/{len(todo)}", flush=True)


def report(out: Path) -> None:
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("清单为空。")
        return

    for r in rows:
        r["replayable"] = _as_bool(r["replayable"])
        r["has_summary"] = _as_bool(r["has_summary"])
        r["has_llm_interactions"] = _as_bool(r["has_llm_interactions"])
        r["malformed_run_id"] = _as_bool(r["malformed_run_id"])

    print(f"\n清单: {out}   共 {len(rows)} 个 run\n")
    print("=" * 70)
    print(f"{'game':<5} {'arm':<12} {'总数':>6} {'可重放':>8} "
          f"{'缺summary':>11} {'缺LLM日志':>11}")
    print("-" * 70)

    for game, arm in sorted({(r["game"], r["arm"]) for r in rows}):
        sub = [r for r in rows if r["game"] == game and r["arm"] == arm]
        print(f"{game or '?':<5} {arm or '?':<12} {len(sub):>6} "
              f"{sum(r['replayable'] for r in sub):>8} "
              f"{sum(not r['has_summary'] for r in sub):>11} "
              f"{sum(not r['has_llm_interactions'] for r in sub):>11}")

    print("-" * 70)
    print(f"{'合计':<17} {len(rows):>6} {sum(r['replayable'] for r in rows):>8} "
          f"{sum(not r['has_summary'] for r in rows):>11} "
          f"{sum(not r['has_llm_interactions'] for r in rows):>11}")
    print("=" * 70)

    malformed = [r for r in rows if r["malformed_run_id"]]
    if malformed:
        print(f"\n⚠ run_id 格式异常 ({len(malformed)}): "
              f"{', '.join(r['run_id'] for r in malformed[:5])}")

    # Runs that hit the API are not pure cache replays; flag them explicitly.
    api = [r for r in rows if (_as_int(r["api_call_count"]) or 0) > 0]
    if api:
        print(f"\n⚠ 有真实 API 调用(非纯缓存)的 run: {len(api)} 个 — "
              f"{', '.join(r['run_id'] for r in api[:5])}")

    steps = sorted({_as_int(r["steps_recorded"]) for r in rows if r["replayable"]})
    print(f"\n可重放 run 的步数取值: {steps}")

    caches = sorted({r["cache_dir"] for r in rows if r["cache_dir"]})
    print("配置中出现的 cache_dir (都不能删):")
    for c in caches:
        n = sum(1 for r in rows if r["cache_dir"] == c and r["replayable"])
        print(f"  {c}   ({n} 个可重放 run 依赖它)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit experiment run directories")
    parser.add_argument("--log-dir", default="logs/default_experiment")
    parser.add_argument("--output", default="outputs/run_audit.csv")
    parser.add_argument("--restart", action="store_true",
                        help="Rescan everything instead of resuming")
    parser.add_argument("--report-only", action="store_true",
                        help="Summarise an existing manifest without scanning")
    args = parser.parse_args()

    out = Path(args.output)
    if not args.report_only:
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            print(f"ERROR: log dir not found: {log_dir}", file=sys.stderr)
            return 1
        scan(log_dir, out, args.restart)

    if not out.exists():
        print(f"ERROR: manifest not found: {out}", file=sys.stderr)
        return 1
    report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
