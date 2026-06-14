"""Baseline records used by residual RL evaluation.

The strongest baseline is intentionally the user-confirmed current baseline for
this experiment. Older archived 20s policy-search snapshots are left out of the
hard-pass target because the user explicitly defined the baseline here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class BaselineRecord:
    name: str
    source: str
    scenario: int
    raw_time_seconds: float
    penalty_time_seconds: float
    final_time_seconds: float
    mean_raw_time_seconds: float
    mean_penalty_time_seconds: float
    mean_final_time_seconds: float
    compatible_with_current_residual_env: bool = True


USER_CONFIRMED_STRONGEST = BaselineRecord(
    name="user_confirmed_strongest_baseline_20260614",
    source="user message: raw=25.36s penalty=3.00s final=28.36s",
    scenario=3,
    raw_time_seconds=25.36,
    penalty_time_seconds=3.0,
    final_time_seconds=28.36,
    mean_raw_time_seconds=25.36,
    mean_penalty_time_seconds=3.0,
    mean_final_time_seconds=28.36,
    compatible_with_current_residual_env=True,
)

STRONGEST_BASE_POLICY_ENV = {
    "SCENARIO3_CRUISE_SPEED_LIMIT": "0.68",
    "SCENARIO3_FINAL_SPEED_LIMIT": "0.68",
    "LOWER_PEOPLE_HOLD_SECONDS": "0.00",
    "COW_HOLD_SECONDS": "0.00",
    "COW_CLEARANCE_HOLD_SECONDS": "5.00",
    "CONE_SETUP_X": "2.020",
    "CONE_PRE_X": "1.950",
    "CONE_LOWER_X": "1.985",
    "CONE_RECOVER_TARGET_X": "6.000",
    "CONE_TOP_BURST_FRAMES": "1",
    "CONE_RETURN_FORWARD": "2.050",
}

ORIGINAL_BASELINE = BaselineRecord(
    name="original_baseline_from_plan",
    source="plan_rl.md",
    scenario=3,
    raw_time_seconds=25.36,
    penalty_time_seconds=3.0,
    final_time_seconds=28.36,
    mean_raw_time_seconds=25.36,
    mean_penalty_time_seconds=3.0,
    mean_final_time_seconds=28.36,
    compatible_with_current_residual_env=True,
)


def baseline_to_dict(record: BaselineRecord) -> Dict[str, Any]:
    return asdict(record)


def summarize_episodes(episodes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = list(episodes)

    def values(key: str) -> List[float]:
        return [
            float(row[key])
            for row in rows
            if row.get("finished", True) and row.get(key) is not None
        ]

    raw = values("raw_time")
    penalty = values("penalty_time")
    final = values("final_time")
    strongest = USER_CONFIRMED_STRONGEST
    original = ORIGINAL_BASELINE

    def avg(items: List[float]) -> float | None:
        return mean(items) if items else None

    def std(items: List[float]) -> float | None:
        return pstdev(items) if len(items) > 1 else 0.0 if items else None

    invalid_count = sum(1 for row in rows if not row.get("finished", False))
    collision_count = sum(
        int(row.get("collision_count", 0) or 0)
        for row in rows
    )
    red_count = sum(
        int(row.get("red_light_violation_count", 0) or 0)
        for row in rows
    )
    reward_violations = [
        float(row.get("reward_threshold_0p04_violation_ratio", 0.0) or 0.0)
        for row in rows
    ]

    mean_final = avg(final)
    mean_penalty = avg(penalty)
    hard_pass = (
        mean_final is not None
        and mean_penalty is not None
        and mean_final < strongest.mean_final_time_seconds
        and mean_penalty <= strongest.mean_penalty_time_seconds
        and invalid_count == 0
    )
    soft_pass = (
        mean_final is not None
        and mean_final < original.mean_final_time_seconds
        and invalid_count == 0
    )

    return {
        "original_baseline": baseline_to_dict(original),
        "strongest_available_baseline": baseline_to_dict(strongest),
        "episodes": rows,
        "mean_raw_time": avg(raw),
        "std_raw_time": std(raw),
        "mean_penalty_time": avg(penalty),
        "std_penalty_time": std(penalty),
        "mean_final_time": mean_final,
        "std_final_time": std(final),
        "best_final_time": min(final) if final else None,
        "beat_original_baseline_rate": (
            sum(1 for value in final if value < original.mean_final_time_seconds) / len(final)
            if final else 0.0
        ),
        "beat_strongest_baseline_rate": (
            sum(1 for value in final if value < strongest.mean_final_time_seconds) / len(final)
            if final else 0.0
        ),
        "finish_success_rate": (
            sum(1 for row in rows if row.get("finished", False)) / len(rows)
            if rows else 0.0
        ),
        "invalid_episode_count": invalid_count,
        "collision_count": collision_count,
        "red_light_violation_count": red_count,
        "reward_threshold_0p04_violation_ratio": (
            mean(reward_violations) if reward_violations else 0.0
        ),
        "hard_pass": hard_pass,
        "soft_pass": soft_pass,
    }


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
