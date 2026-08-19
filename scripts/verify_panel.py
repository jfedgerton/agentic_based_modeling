"""Cross-check a replayed CV panel against the original run's own logs.

A byte-identical ``time_series.csv`` proves the replay followed the same
trajectory, but says nothing about whether the panel's contents are correct.
This script checks them against two independent sources recorded by the
original run:

  * ``llm_interactions.jsonl`` — the neighbourhood counts and private state
    embedded in each prompt, and the action the LLM actually returned. These
    were written months ago by code that knew nothing about the panel.
  * the replay's own ``agent_reporters`` frame — end-of-step positions
    collected by mesa's DataCollector, a separate code path from the panel.

The second check exploits a property of the model: an agent's position only
changes when it moves, so its decision-time position at step t must equal
its post-move position at step t-1.

Usage:
    python scripts/verify_panel.py \
        logs/default_experiment/cv_reasoner_seed52_06d2c412 \
        outputs/replay_check/panels/cv_reasoner_seed52_06d2c412__panel.parquet
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROMPT_FIELDS = {
    "cops_nearby": re.compile(r"enforcement agents nearby:\s*(\d+)"),
    "actives_nearby": re.compile(r"ACTIVE agents nearby:\s*(\d+)"),
    "quiets_nearby": re.compile(r"QUIET agents nearby:\s*(\d+)"),
    "hardship": re.compile(r"hardship:\s*([\d.]+)"),
    "risk_aversion": re.compile(r"risk_aversion:\s*([\d.]+)"),
}


def load_prompt_facts(path: Path, limit: int) -> pd.DataFrame:
    """Stream the interaction log, pulling the facts encoded in each prompt."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            prompt = rec["prompt"]
            row = {
                "step": int(rec["step"]),
                "agent_id": int(rec["agent_id"]),
                "logged_action": (rec.get("parsed") or {}).get("action"),
            }
            for name, pat in PROMPT_FIELDS.items():
                m = pat.search(prompt)
                row[name] = float(m.group(1)) if m else None
            rows.append(row)
    return pd.DataFrame(rows)


def check_against_prompts(panel: pd.DataFrame, facts: pd.DataFrame) -> list:
    """Compare panel columns with the same quantities taken from prompts."""
    merged = facts.merge(panel, on=["step", "agent_id"],
                         how="left", suffixes=("_log", "_panel"))
    results = []

    missing = int(merged["action"].isna().sum())
    results.append(("原始日志中的每条决策都能在 panel 中找到", missing == 0,
                    f"{missing} 条缺失 / 共 {len(merged)}"))

    for name in ("cops_nearby", "actives_nearby", "quiets_nearby"):
        a, b = merged[f"{name}_log"], merged[f"{name}_panel"]
        both = a.notna() & b.notna()
        bad = int((a[both] != b[both]).sum())
        results.append((f"{name} 与 prompt 一致", bad == 0,
                        f"{bad} 处不符 / 比对 {int(both.sum())} 条"))

    for name in ("hardship", "risk_aversion"):
        a, b = merged[f"{name}_log"], merged[f"{name}_panel"]
        both = a.notna() & b.notna()
        # Prompts render these rounded to 2 decimals.
        bad = int(((a[both] - b[both]).abs() > 5e-3).sum())
        results.append((f"{name} 与 prompt 一致(2位小数内)", bad == 0,
                        f"{bad} 处不符 / 比对 {int(both.sum())} 条"))

    known = merged["logged_action"].notna()
    bad = int((merged.loc[known, "logged_action"]
               != merged.loc[known, "action"]).sum())
    results.append(("action 与原始 LLM 解析结果一致", bad == 0,
                    f"{bad} 处不符 / 比对 {int(known.sum())} 条"))
    return results


def check_position_continuity(panel: pd.DataFrame) -> list:
    """An agent's decision-time position must be where it last moved to."""
    p = panel.sort_values(["agent_id", "step"])
    prev_x = p.groupby("agent_id")["moved_to_x"].shift(1)
    prev_y = p.groupby("agent_id")["moved_to_y"].shift(1)
    # Jailed agents skip steps entirely, so only consecutive steps qualify.
    consecutive = p.groupby("agent_id")["step"].diff() == 1
    comparable = consecutive & prev_x.notna()
    bad = int(((p["x"] != prev_x) | (p["y"] != prev_y))[comparable].sum())
    return [("位置连续性: 本步决策位置 == 上一步移动后位置", bad == 0,
             f"{bad} 处断裂 / 比对 {int(comparable.sum())} 条")]


def check_against_agent_reporters(panel: pd.DataFrame,
                                  agent_vars: Path) -> list:
    if not agent_vars.exists():
        return []
    av = pd.read_parquet(agent_vars).reset_index()
    av = av.rename(columns={"Step": "step", "AgentID": "agent_id"})
    merged = panel.merge(av, on=["step", "agent_id"], how="inner",
                         suffixes=("", "_dc"))
    bad = int(((merged["moved_to_x"] != merged["x_dc"])
               | (merged["moved_to_y"] != merged["y_dc"])).sum())
    return [("panel 移动后位置 == DataCollector 记录", bad == 0,
             f"{bad} 处不符 / 比对 {len(merged)} 条")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a replayed CV panel")
    parser.add_argument("run_dir", help="Original run directory")
    parser.add_argument("panel", help="Replayed panel .parquet")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only check the first N interactions (0 = all)")
    parser.add_argument("--agent-vars", default="",
                        help="Optional DataCollector agent-vars parquet")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    panel = pd.read_parquet(args.panel)
    print(f"panel 行数: {len(panel):,}")

    interactions = run_dir / "llm_interactions.jsonl"
    if not interactions.exists():
        print(f"ERROR: 找不到 {interactions}", file=sys.stderr)
        return 1

    print(f"读取原始交互日志 ({interactions.stat().st_size / 1e6:.0f} MB)...",
          flush=True)
    facts = load_prompt_facts(interactions, args.limit)
    print(f"解析出 {len(facts):,} 条原始决策\n")

    checks = check_against_prompts(panel, facts)
    checks += check_position_continuity(panel)
    if args.agent_vars:
        checks += check_against_agent_reporters(panel, Path(args.agent_vars))

    width = max(len(name) for name, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"{'✅' if ok else '❌'} {name:<{width}}  {detail}")
        failed += not ok

    print()
    if failed:
        print(f"❌ {failed} 项校验未通过")
        return 1
    print("✅ 全部校验通过：panel 内容与原始 run 的记录一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
