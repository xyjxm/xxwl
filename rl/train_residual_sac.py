#!/usr/bin/env python3
"""Train a single-env SAC residual policy."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

try:
    from .baseline_registry import USER_CONFIRMED_STRONGEST
    from .qcar_residual_env import QCarResidualEnv
except ImportError:
    from baseline_registry import USER_CONFIRMED_STRONGEST
    from qcar_residual_env import QCarResidualEnv


class TrainingLogCallback:
    """Small callback duck-typed for stable-baselines3 BaseCallback fallback."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=3)
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--save_dir", default="runs/residual_rl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_policy", default="strongest")
    parser.add_argument("--reward_minor_threshold", type=float, default=0.04)
    parser.add_argument("--init_model", default=None)
    parser.add_argument("--candidate_penalty_threshold", type=float, default=18.0)
    parser.add_argument("--no_auto_reenter_plane", action="store_true")
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
    except Exception as exc:
        raise SystemExit(
            "stable_baselines3 is required for SAC training. "
            "Install it in the Quanser Python environment before running training."
        ) from exc

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    training_log = save_dir / "training_log.csv"

    config = {
        "algorithm": "SAC",
        "n_envs": 1,
        "scenario": args.scenario,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "base_policy": args.base_policy,
        "reward_minor_threshold": args.reward_minor_threshold,
        "init_model": args.init_model,
        "candidate_penalty_threshold": args.candidate_penalty_threshold,
        "auto_reenter_plane": not args.no_auto_reenter_plane,
        "strongest_baseline": USER_CONFIRMED_STRONGEST.__dict__,
    }
    (save_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    class CsvCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best_rank = None
            self.header_written = False

        def _on_step(self) -> bool:
            infos = self.locals.get("infos") or []
            if not infos:
                return True
            info = infos[0]
            components = info.get("reward_components", {})
            row = {
                "num_timesteps": self.num_timesteps,
                "segment_id": info.get("segment_id"),
                "segment_name": info.get("segment_name"),
                "residual_enabled": info.get("residual_enabled"),
                "reward": info.get("reward"),
                "lane_deviation": info.get("lane_deviation"),
                "penalty_increment": info.get("penalty_increment"),
                "raw_time": info.get("raw_time"),
                "penalty_time": info.get("penalty_time"),
                "final_time": info.get("final_time"),
                "invalid_reason": info.get("invalid_reason"),
                "stall_steps": info.get("stall_steps"),
            }
            for key, value in components.items():
                row[f"reward_{key}"] = value

            write_header = not training_log.exists()
            with training_log.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

            final_time = info.get("final_time")
            penalty_time = info.get("penalty_time")
            if final_time is not None and penalty_time is not None:
                rank = (float(penalty_time), float(final_time))
                if self.best_rank is None or rank < self.best_rank:
                    self.best_rank = rank
                    self.model.save(save_dir / "best_by_penalty")
                if float(penalty_time) <= args.candidate_penalty_threshold:
                    safe_penalty = str(float(penalty_time)).replace(".", "p")
                    safe_final = str(round(float(final_time), 3)).replace(".", "p")
                    self.model.save(
                        save_dir / f"candidate_t{self.num_timesteps}_p{safe_penalty}_f{safe_final}"
                    )
            if (
                final_time is not None
                and penalty_time is not None
                and float(penalty_time) <= USER_CONFIRMED_STRONGEST.mean_penalty_time_seconds
                and (
                    self.best_rank is None
                    or float(final_time) < self.best_rank[1]
                )
            ):
                self.model.save(save_dir / "best_model")
            if self.num_timesteps > 0 and self.num_timesteps % 1000 == 0:
                self.model.save(save_dir / "last_model")
            return True

    env = QCarResidualEnv(
        scenario=args.scenario,
        base_policy=args.base_policy,
        reward_minor_threshold=args.reward_minor_threshold,
        zero_residual=False,
        auto_reenter_plane=not args.no_auto_reenter_plane,
    )
    if args.init_model:
        model = SAC.load(args.init_model, env=env)
        model.set_random_seed(args.seed)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            train_freq=1,
            gradient_steps=1,
            learning_starts=256,
            buffer_size=50000,
        )
    callback = CsvCallback()
    try:
        model.learn(total_timesteps=args.total_steps, callback=callback)
        model.save(save_dir / "last_model")
        if not (save_dir / "best_model.zip").exists():
            shutil.copyfile(save_dir / "last_model.zip", save_dir / "best_model.zip")
        if not (save_dir / "best_by_penalty.zip").exists():
            shutil.copyfile(save_dir / "last_model.zip", save_dir / "best_by_penalty.zip")
        if hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(save_dir / "replay_buffer.pkl")
    except Exception:
        model.save(save_dir / "last_model")
        if hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(save_dir / "replay_buffer.pkl")
        if not (save_dir / "best_model.zip").exists() and (save_dir / "last_model.zip").exists():
            shutil.copyfile(save_dir / "last_model.zip", save_dir / "best_model.zip")
        if not (save_dir / "best_by_penalty.zip").exists() and (save_dir / "last_model.zip").exists():
            shutil.copyfile(save_dir / "last_model.zip", save_dir / "best_by_penalty.zip")
        raise
    finally:
        env.close()

    print(f"Saved SAC residual artifacts to {save_dir}")


if __name__ == "__main__":
    main()
