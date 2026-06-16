#!/usr/bin/env python3
"""Trace a residual policy tick by tick for controller tuning."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from .qcar_residual_env import QCarResidualEnv
except ImportError:
    from qcar_residual_env import QCarResidualEnv


def load_policy(model_path: str):
    if model_path == "zero":
        return None
    from stable_baselines3 import SAC

    return SAC.load(model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=3)
    parser.add_argument("--model", default="zero")
    parser.add_argument("--out", default="runs/residual_rl/trace.csv")
    parser.add_argument("--base_policy", default="strongest")
    parser.add_argument("--max_steps", type=int, default=4500)
    parser.add_argument("--auto_reenter_plane", action="store_true")
    args = parser.parse_args()

    model = load_policy(args.model)
    env = QCarResidualEnv(
        scenario=args.scenario,
        base_policy=args.base_policy,
        zero_residual=args.model == "zero",
        max_steps=args.max_steps,
        auto_reenter_plane=args.auto_reenter_plane,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    try:
        obs, _ = env.reset(seed=0)
        done = False
        truncated = False
        while not done and not truncated:
            if model is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            row = {
                "step": info.get("step_count"),
                "x": (info.get("position") or [None, None])[0],
                "y": (info.get("position") or [None, None])[1],
                "yaw": info.get("yaw"),
                "speed": info.get("speed"),
                "segment_name": info.get("segment_name"),
                "lane_deviation": info.get("lane_deviation"),
                "penalty_increment": info.get("penalty_increment"),
                "total_penalty": (info.get("penalty_breakdown") or {}).get("total_penalty_time"),
                "raw_time": info.get("raw_time"),
                "penalty_time": info.get("penalty_time"),
                "final_time": info.get("final_time"),
                "finished": info.get("finished"),
            }
            final_action = info.get("final_action") or {}
            residual_action = info.get("residual_action") or {}
            row["final_forward"] = final_action.get("forward")
            row["final_turn"] = final_action.get("turn")
            row["residual_speed_delta"] = residual_action.get("speed_delta")
            row["residual_steering_bias"] = residual_action.get("steering_bias")
            row["strict_lane_correction"] = info.get("strict_lane_correction")
            row["strict_lane_heading_error"] = info.get("strict_lane_heading_error")
            row["strict_lane_signed_deviation"] = info.get("strict_lane_signed_deviation")
            row["strict_lane_speed_cap"] = info.get("strict_lane_speed_cap")
            rows.append(row)
    finally:
        env.close()

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["step"])
        writer.writeheader()
        writer.writerows(rows)

    offenders = [row for row in rows if float(row.get("lane_deviation") or 0.0) > 0.04]
    print(f"wrote {len(rows)} rows to {out}")
    print(f"deviation>0.04 count={len(offenders)}")
    if rows:
        print(
            "final: "
            f"raw={rows[-1].get('raw_time')} penalty={rows[-1].get('penalty_time')} "
            f"final={rows[-1].get('final_time')}"
        )


if __name__ == "__main__":
    main()
