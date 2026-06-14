#!/usr/bin/env python3
"""Write the residual-RL baseline report."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .baseline_registry import (
        ORIGINAL_BASELINE,
        STRONGEST_BASE_POLICY_ENV,
        USER_CONFIRMED_STRONGEST,
        baseline_to_dict,
        write_json,
    )
except ImportError:
    from baseline_registry import (
        ORIGINAL_BASELINE,
        STRONGEST_BASE_POLICY_ENV,
        USER_CONFIRMED_STRONGEST,
        baseline_to_dict,
        write_json,
    )


def build_report(scenario: int):
    strongest = USER_CONFIRMED_STRONGEST
    return {
        "scenario": scenario,
        "selection_rule": (
            "User-confirmed strongest baseline for this residual-RL experiment; "
            "RL hard-pass compares deterministic mean_final_time and mean_penalty_time against it."
        ),
        "original_baseline": baseline_to_dict(ORIGINAL_BASELINE),
        "strongest_available_baseline": baseline_to_dict(strongest),
        "strongest_base_policy_env": STRONGEST_BASE_POLICY_ENV,
        "notes": [
            "Zero residual must reproduce this base policy before training is trusted.",
            "Reward shaping uses 0.04m only inside rl/reward.py and does not alter PenaltySystem.py.",
            "Historical faster snapshots are not the target for this run because the user explicitly set the strongest baseline to 25.36+3.00=28.36.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=3)
    parser.add_argument("--out", default="runs/residual_rl/baseline_report.json")
    args = parser.parse_args()

    report = build_report(args.scenario)
    write_json(Path(args.out), report)
    print(f"Wrote baseline report to {args.out}")
    print(
        "strongest_available_baseline: "
        f"raw={report['strongest_available_baseline']['raw_time_seconds']:.2f}s, "
        f"penalty={report['strongest_available_baseline']['penalty_time_seconds']:.2f}s, "
        f"final={report['strongest_available_baseline']['final_time_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()
