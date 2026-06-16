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
    "SCENARIO3_CRUISE_SPEED_LIMIT": "0.80",
    "SCENARIO3_FINAL_SPEED_LIMIT": "0.82",
    "STOP_SIGN_HOLD_SECONDS": "0.00",
    "STOP_SIGN_MAX_HOLD_SECONDS": "0.15",
    "LOWER_PEOPLE_HOLD_SECONDS": "1.80",
    "LOWER_PEOPLE_POST_HOLD_DISTANCE": "0.46",
    "LOWER_PEOPLE_POST_HOLD_SECONDS": "3.80",
    "LOWER_PEOPLE_CLEAR_SETTLE_SECONDS": "0.35",
    "LOWER_PEOPLE_MIN_CLEAR_SECONDS": "2.80",
    "LOWER_PEOPLE_CLEAR_Y": "-1.550",
    "LOWER_PEOPLE_FORCED_CLEAR_SECONDS": "3.20",
    "LOWER_PEOPLE_ENDPOINT_Y": "-1.600",
    "LOWER_PEOPLE_SPEED": "0.50",
    "TRAFFIC_PERSON2_ENDPOINT_Y": "-1.600",
    "LOWER_PEOPLE_GUARD_MIN_X": "0.26",
    "LOWER_PEOPLE_GUARD_MAX_X": "1.10",
    "LOWER_PEOPLE_GUARD_MIN_Y": "-1.36",
    "LOWER_PEOPLE_GUARD_MAX_Y": "-0.68",
    "LOWER_PEOPLE_PRE_WAIT_SPEED": "0.35",
    "LOWER_PEOPLE_PRE_WAIT_MIN_X": "-0.25",
    "LOWER_OVERRIDES_ENABLED": "1",
    "LOWER_ENTRY_SPEED": "0.85",
    "LOWER_ENTRY_TARGET_Y": "-1.090",
    "LOWER_AFTER_WAIT_TARGET_Y": "-1.045",
    "LOWER_AFTER_WAIT_SPEED": "0.52",
    "LOWER_MID_TARGET_Y": "-0.900",
    "LOWER_MID_SPEED": "0.86",
    "LOWER_EXIT_TARGET_Y": "-0.760",
    "LOWER_EXIT_SPEED": "0.90",
    "LOWER_EXIT_MAX_X": "2.05",
    "UPPER_PEOPLE_HOLD_SECONDS": "2.00",
    "UPPER_PEOPLE_STRICT_CLEAR_ENABLED": "1",
    "UPPER_PEOPLE_POST_HOLD_DISTANCE": "0.42",
    "UPPER_PEOPLE_POST_HOLD_SECONDS": "3.40",
    "UPPER_PEOPLE_CLEAR_SETTLE_SECONDS": "0.25",
    "UPPER_PEOPLE_MIN_CLEAR_SECONDS": "2.80",
    "UPPER_PEOPLE_FORCED_CLEAR_SECONDS": "4.20",
    "UPPER_PEOPLE_CLEAR_X": "-1.550",
    "UPPER_PEOPLE_ENDPOINT_X": "-1.451",
    "UPPER_PEOPLE_ENDPOINT_Y": "3.1722",
    "UPPER_PEOPLE_SPEED": "0.25",
    "UPPER_PEOPLE_PRE_REQUEST_ENABLED": "1",
    "UPPER_PEOPLE_PRE_REQUEST_MIN_X": "-0.60",
    "UPPER_PEOPLE_PRE_REQUEST_MAX_X": "0.12",
    "UPPER_PEOPLE_PRE_REQUEST_MIN_Y": "4.20",
    "UPPER_PEOPLE_PRE_REQUEST_MAX_Y": "4.58",
    "COW_HOLD_SECONDS": "0.00",
    "COW_CLEARANCE_HOLD_SECONDS": "3.00",
    "COW_CLEARANCE_POST_HOLD_SECONDS": "0.00",
    "COW_CLEARANCE_POST_CLEAR_Y": "4.03",
    "COW_CLEARANCE_POST_HOLD_DISTANCE": "0.54",
    "COW_ENTRY_TARGET_Y": "4.500",
    "COW_CLEARANCE_TARGET_Y": "4.500",
    "COW_AREA_SPEED_CAP": "0.72",
    "COW_EXIT_SPEED_CAP": "1.14",
    "COW_LATE_EXIT_SPEED_CAP": "0.92",
    "UPPER_APPROACH_SPEED_CAP": "0.62",
    "COW_RESTART_TICKS": "0",
    "COW_RESTART_SPEED": "0.28",
    "COW_RESTART_FORWARD_X": "0.08",
    "COW_RESTART_MIN_X": "0.32",
    "COW_RESTART_TARGET_Y": "4.455",
    "COW_SAFETY_CLEAR_DISTANCE": "0.00",
    "CONE_SETUP_X": "1.995",
    "CONE_PRE_X": "1.990",
    "CONE_LOWER_X": "2.000",
    "CONE_RECOVER_TARGET_X": "6.000",
    "CONE_TOP_TURN_START_Y": "1.315",
    "CONE_TOP_BURST_FRAMES": "1",
    "CONE_RETURN_FORWARD": "2.050",
}

FAST_SAFE_BASE_POLICY_ENV = {
    **STRONGEST_BASE_POLICY_ENV,
    "SCENARIO3_ARCLENGTH_TARGET": "1",
    "SCENARIO3_CRUISE_SPEED_LIMIT": "1.40",
    "SCENARIO3_FINAL_SPEED_LIMIT": "1.45",
    "LANE_ASSIST_ENABLED": "1",
    "LANE_ASSIST_START": "0.035",
    "LANE_ASSIST_GAIN": "1.35",
    "LANE_ASSIST_MAX": "0.060",
    "LANE_ASSIST_HEADING_GAIN": "0.060",
    "LANE_ASSIST_HEADING_MAX": "0.035",
    "LANE_ASSIST_SPEED_START": "0.050",
    "LANE_ASSIST_SPEED_CAP": "0.68",
    "LANE_ASSIST_CONE_MAX": "0.018",
    "LANE_ASSIST_CONE_SPEED_CAP": "0.52",
    "COW_CLEARANCE_HOLD_SECONDS": "1.20",
    "COW_CLEARANCE_POST_HOLD_SECONDS": "1.00",
    "COW_CLEARANCE_POST_CLEAR_Y": "3.90",
    "COW_CLEARANCE_POST_HOLD_DISTANCE": "0.85",
    "COW_CLEARANCE_MAX_WAIT_SECONDS": "99.00",
    "COW_CLEARANCE_ENDPOINT_Y": "3.780",
    "TRAFFIC_COW_ENDPOINT_Y": "3.780",
    "COW_CLEARANCE_ENDPOINT_SPEED": "0.35",
    "COW_PRE_MOVE_STOP_TICKS": "5",
    "COW_CLEARANCE_MAX_X": "0.68",
    "COW_CLEARANCE_MIN_X": "0.48",
    "COW_CLEARANCE_MIN_Y": "4.25",
    "COW_CLEARANCE_MAX_Y": "4.55",
    "COW_ENTRY_TARGET_Y": "4.400",
    "COW_CLEARANCE_TARGET_Y": "4.400",
    "COW_ENTRY_X_OFFSET": "0.22",
    "COW_CLEARANCE_X_OFFSET": "0.12",
    "COW_SAFE_CROSS_Y": "4.400",
    "COW_SAFE_MIN_X": "0.070",
    "COW_APPROACH_TARGET_X": "0.635",
    "COW_APPROACH_TARGET_Y": "4.340",
    "COW_APPROACH_SPEED_CAP": "0.30",
    "COW_AREA_SPEED_CAP": "0.54",
    "COW_EXIT_TARGET_Y": "4.395",
    "COW_LATE_EXIT_TARGET_Y": "4.395",
    "COW_EXIT_SPEED_CAP": "0.48",
    "COW_LATE_EXIT_SPEED_CAP": "0.50",
    "COW_EXIT_X_OFFSET": "0.46",
    "COW_LATE_EXIT_X_OFFSET": "0.52",
    "COW_EXIT_MAX_Y_DROP": "0.035",
    "FINAL_VERTICAL_PRE_START_Y": "2.80",
    "FINAL_VERTICAL_PRE_END_Y": "3.45",
    "FINAL_VERTICAL_PRE_TARGET_X": "-1.930",
    "FINAL_VERTICAL_PRE_FORWARD_Y": "0.36",
    "FINAL_VERTICAL_PRE_SPEED_CAP": "0.52",
    "FINAL_VERTICAL_SPEED_CAP": "0.52",
    "RL_ACTOR_RESIDUAL_ENABLED": "0",
    "RL_STRICT_LANE_START": "0.012",
    "RL_STRICT_LANE_GAIN": "1.85",
    "RL_STRICT_LANE_MAX": "0.105",
    "RL_STRICT_LANE_HEADING_GAIN": "0.100",
    "RL_STRICT_LANE_HEADING_MAX": "0.045",
    "RL_STRICT_LANE_SPEED_CAP": "0.66",
    "RL_STRICT_LANE_SPEED_START": "0.045",
    "RL_STRICT_LANE_CONE_MAX": "0.018",
    "RL_STRICT_LANE_CONE_SPEED_CAP": "0.52",
}

FAST_LEARN_BASE_POLICY_ENV = {
    **FAST_SAFE_BASE_POLICY_ENV,
    "RL_ACTOR_RESIDUAL_ENABLED": "1",
    "RL_PEDESTRIAN_RESIDUAL_ENABLED": "1",
    "RL_COW_RESIDUAL_ENABLED": "1",
    "RL_ACTOR_SPEED_DELTA_MIN": "-0.04",
    "RL_ACTOR_SPEED_DELTA_MAX": "0.02",
    "RL_ACTOR_STEERING_BIAS_LIMIT": "0.018",
    "RL_COW_SPEED_DELTA_MIN": "-0.04",
    "RL_COW_SPEED_DELTA_MAX": "0.02",
    "RL_COW_STEERING_BIAS_LIMIT": "0.016",
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
