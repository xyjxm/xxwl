#!/usr/bin/env python3
"""Run one fast_safe trace with temporary policy overrides."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from .analyze_trace import print_summary, summarize
except ImportError:
    from analyze_trace import print_summary, summarize


STALE_ENV_KEYS = [
    "TRAFFIC_ACTOR_MODE",
    "TRAFFIC_COW_SCRIPTED",
    "TRAFFIC_COW_DELAY",
    "SCENARIO3_ARCLENGTH_TARGET",
    "RL_LINE_DAMPING_ENABLED",
    "RL_ACTOR_RESIDUAL_ENABLED",
    "RL_PEDESTRIAN_RESIDUAL_ENABLED",
    "RL_COW_RESIDUAL_ENABLED",
    "RL_STRICT_LANE_CONE_SPEED_CAP",
    "UPPER_VERTICAL_ENABLED",
    "RL_STRICT_LANE_FINAL_CROSSTRACK_SIGN",
]


def parse_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"override key is empty in {item!r}")
        overrides[key] = value.strip()
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="experiment")
    parser.add_argument("--set", dest="sets", action="append", default=[])
    parser.add_argument("--model", default="zero")
    parser.add_argument("--base_policy", default="fast_safe")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--runs_dir", default="runs")
    args = parser.parse_args()

    overrides = parse_overrides(args.sets)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.name)
    out = Path(args.runs_dir) / f"trace_{safe_name}_{timestamp}.csv"

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["RL_BASE_POLICY_OVERRIDES"] = json.dumps(overrides)
    for key in STALE_ENV_KEYS:
        env.pop(key, None)

    cmd = [
        sys.executable,
        "rl\\trace_residual_policy.py",
        "--scenario",
        "3",
        "--model",
        args.model,
        "--out",
        str(out),
        "--base_policy",
        args.base_policy,
        "--max_steps",
        str(args.max_steps),
        "--auto_reenter_plane",
    ]
    print(f"running: {' '.join(cmd)}")
    print(f"overrides: {overrides}")
    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    summary = summarize(out)
    print_summary(summary, show_penalties=True)

    summary_txt = out.with_suffix(".summary.txt")
    with summary_txt.open("w", encoding="utf-8") as handle:
        original_stdout = sys.stdout
        try:
            sys.stdout = handle
            print_summary(summary, show_penalties=True)
        finally:
            sys.stdout = original_stdout

    json_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"penalty_rows"}
    }
    json_summary["overrides"] = overrides
    out.with_suffix(".summary.json").write_text(
        json.dumps(json_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"TRACE_OUT={out}")
    print(f"SUMMARY_TXT={summary_txt}")


if __name__ == "__main__":
    main()
