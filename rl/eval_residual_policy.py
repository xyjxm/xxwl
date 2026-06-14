#!/usr/bin/env python3
"""Deterministic residual policy evaluation."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np

try:
    from .baseline_registry import USER_CONFIRMED_STRONGEST, summarize_episodes, write_json
    from .qcar_residual_env import QCarResidualEnv
except ImportError:
    from baseline_registry import USER_CONFIRMED_STRONGEST, summarize_episodes, write_json
    from qcar_residual_env import QCarResidualEnv


def load_policy(model_path: str):
    if model_path == "zero":
        return None
    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError("stable_baselines3 is required to evaluate SAC models.") from exc
    return SAC.load(model_path)


def evaluate(args):
    model = load_policy(args.model)
    episodes = []

    for episode_index in range(args.episodes):
        env = QCarResidualEnv(
            scenario=args.scenario,
            base_policy=args.base_policy,
            reward_minor_threshold=args.reward_minor_threshold,
            zero_residual=args.model == "zero",
            max_steps=args.max_steps,
            auto_reenter_plane=args.auto_reenter_plane,
        )
        try:
            obs, _ = env.reset(seed=args.seed + episode_index)
            done = False
            truncated = False
            last_info = {}
            total_reward = 0.0
            reward_violations = 0
            reward_samples = 0

            while not done and not truncated:
                if model is None:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += float(reward)
                last_info = info
                reward_samples += 1
                if info.get("lane_deviation", 0.0) > args.reward_minor_threshold:
                    reward_violations += 1

            penalty_breakdown = last_info.get("penalty_breakdown", {})
            collision_count = (
                int(penalty_breakdown.get("cone_collision_count", 0) or 0)
                + int(penalty_breakdown.get("people_collision_count", 0) or 0)
                + int(penalty_breakdown.get("cow_collision_count", 0) or 0)
            )
            episodes.append(
                {
                    "episode": episode_index + 1,
                    "finished": bool(done),
                    "truncated": bool(truncated),
                    "raw_time": last_info.get("raw_time"),
                    "penalty_time": last_info.get("penalty_time"),
                    "final_time": last_info.get("final_time"),
                    "total_reward": total_reward,
                    "collision_count": collision_count,
                    "red_light_violation_count": int(penalty_breakdown.get("red_light_violation_count", 0) or 0),
                    "reward_threshold_0p04_violation_ratio": (
                        reward_violations / reward_samples if reward_samples else 0.0
                    ),
                    "last_segment_name": last_info.get("segment_name"),
                    "elapsed_time": last_info.get("elapsed_time"),
                    "step_count": last_info.get("step_count"),
                    "position": last_info.get("position"),
                    "yaw": last_info.get("yaw"),
                    "speed": last_info.get("speed"),
                    "invalid_reason": last_info.get("invalid_reason"),
                    "stall_steps": last_info.get("stall_steps"),
                }
            )
        except Exception as exc:
            episodes.append(
                {
                    "episode": episode_index + 1,
                    "finished": False,
                    "truncated": False,
                    "raw_time": None,
                    "penalty_time": None,
                    "final_time": None,
                    "collision_count": 0,
                    "red_light_violation_count": 0,
                    "reward_threshold_0p04_violation_ratio": 0.0,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
        finally:
            env.close()

    report = summarize_episodes(episodes)
    report.update(
        {
            "scenario": args.scenario,
            "model": args.model,
            "base_policy": args.base_policy,
            "deterministic": not args.stochastic,
            "reward_minor_threshold": args.reward_minor_threshold,
        }
    )
    if args.model == "zero":
        mean_final = report.get("mean_final_time")
        mean_penalty = report.get("mean_penalty_time")
        final_diff = (
            abs(float(mean_final) - USER_CONFIRMED_STRONGEST.mean_final_time_seconds)
            if mean_final is not None else None
        )
        penalty_diff = (
            abs(float(mean_penalty) - USER_CONFIRMED_STRONGEST.mean_penalty_time_seconds)
            if mean_penalty is not None else None
        )
        report["zero_residual_check"] = {
            "target_final_time": USER_CONFIRMED_STRONGEST.mean_final_time_seconds,
            "target_penalty_time": USER_CONFIRMED_STRONGEST.mean_penalty_time_seconds,
            "mean_final_time_abs_diff": final_diff,
            "mean_penalty_time_abs_diff": penalty_diff,
            "tolerance_seconds": args.zero_tolerance_seconds,
            "matches_strongest_baseline": (
                final_diff is not None
                and penalty_diff is not None
                and final_diff <= args.zero_tolerance_seconds
                and penalty_diff <= 0.05
                and report.get("invalid_episode_count", 0) == 0
            ),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=3)
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", default="runs/residual_rl/eval.json")
    parser.add_argument("--base_policy", default="strongest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward_minor_threshold", type=float, default=0.04)
    parser.add_argument("--zero_tolerance_seconds", type=float, default=0.75)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--auto_reenter_plane", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    report = evaluate(args)
    write_json(Path(args.out), report)
    print(f"Wrote evaluation report to {args.out}")
    print(
        "hard_pass="
        f"{report['hard_pass']} mean_final_time={report['mean_final_time']} "
        f"mean_penalty_time={report['mean_penalty_time']}"
    )


if __name__ == "__main__":
    main()
