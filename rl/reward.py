"""Reward shaping for residual RL.

The 0.04m threshold is only for reward shaping. It does not alter
PenaltySystem's official thresholds.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


REWARD_MINOR_DEVIATION_THRESHOLD = 0.04


class ResidualReward:
    def __init__(self, baseline_final_time: float = 28.36, minor_threshold: float = REWARD_MINOR_DEVIATION_THRESHOLD):
        self.baseline_final_time = float(baseline_final_time)
        self.minor_threshold = float(minor_threshold)

    def compute(
        self,
        previous_info: Dict,
        current_info: Dict,
        action_record: Dict,
        terminated: bool,
    ) -> Tuple[float, Dict[str, float]]:
        prev_progress = float(previous_info.get("route_progress", 0.0) or 0.0)
        progress = float(current_info.get("route_progress", 0.0) or 0.0)
        progress_delta = max(0.0, progress - prev_progress)
        deviation = float(current_info.get("lane_deviation", 0.0) or 0.0)
        penalty_increment = float(current_info.get("penalty_increment", 0.0) or 0.0)
        collision_count = float(current_info.get("collision_increment", 0.0) or 0.0)
        red_count = float(current_info.get("red_light_increment", 0.0) or 0.0)

        residual = action_record.get("residual_action", {}) if action_record else {}
        prev_residual = previous_info.get("residual_action", {})
        speed_delta = float(residual.get("speed_delta", 0.0) or 0.0)
        steering_bias = float(residual.get("steering_bias", 0.0) or 0.0)
        prev_speed_delta = float(prev_residual.get("speed_delta", 0.0) or 0.0)
        prev_steering_bias = float(prev_residual.get("steering_bias", 0.0) or 0.0)

        lane_margin_excess = max(0.0, deviation - self.minor_threshold * 0.80)
        lane_excess = max(0.0, deviation - self.minor_threshold)
        lane_deviation_penalty = -(lane_margin_excess * 80.0 + lane_excess * 260.0)
        if deviation > 0.08:
            lane_deviation_penalty -= (deviation - 0.08) * 120.0

        components = {
            "progress_reward": progress_delta * 40.0,
            "time_penalty": -0.02,
            "lane_deviation_penalty": lane_deviation_penalty,
            "official_penalty_increment": -8.0 * penalty_increment,
            "collision_penalty": -80.0 * collision_count,
            "red_light_penalty": -80.0 * red_count,
            "action_smoothness_penalty": -0.15 * (
                abs(speed_delta - prev_speed_delta) + abs(steering_bias - prev_steering_bias)
            ),
            "residual_magnitude_penalty": -0.20 * (
                abs(speed_delta) + abs(steering_bias)
            ),
            "finish_bonus": 10.0 if terminated else 0.0,
            "beat_baseline_bonus": 0.0,
        }

        final_time = current_info.get("final_time")
        if terminated and final_time is not None and float(final_time) < self.baseline_final_time:
            components["beat_baseline_bonus"] = 20.0

        reward = float(np.sum(list(components.values())))
        return reward, components
