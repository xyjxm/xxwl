"""Low-dimensional observation encoder for residual RL."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym


OBS_DIM = 29


class StateEncoder:
    def __init__(self):
        self.observation_dim = OBS_DIM

    def get_observation_space(self):
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

    def encode(
        self,
        core,
        segment,
        previous_base_action=None,
        previous_residual_action=None,
        previous_final_action=None,
    ) -> Tuple[np.ndarray, Dict]:
        missing: List[str] = []
        pos = getattr(core, "current_position", None)
        if pos is None:
            missing.append("current_position")
            x = y = 0.0
        else:
            x = float(pos[0])
            y = float(pos[1])

        yaw = float(getattr(core, "yaw", 0.0) or 0.0)
        speed = float(getattr(core, "speed", 0.0) or 0.0)

        deviation = 0.0
        is_right = 0.0
        penalty_system = getattr(core, "penalty_system", None)
        if penalty_system is None:
            missing.append("penalty_system")
        else:
            try:
                deviation, is_right_bool = penalty_system.calculate_lane_deviation([x, y])
                is_right = 1.0 if is_right_bool else 0.0
            except Exception:
                missing.append("lane_deviation")

        route_progress = 0.0
        next_dx = 0.0
        next_dy = 0.0
        path_points = getattr(core, "path_points", None)
        if path_points is None or len(path_points) == 0:
            missing.append("path_points")
        else:
            points = np.asarray(path_points, dtype=np.float32)
            distances = np.linalg.norm(points - np.array([x, y], dtype=np.float32), axis=1)
            nearest = int(np.argmin(distances))
            route_progress = nearest / max(1, len(points) - 1)
            target_index = min(nearest + 1, len(points) - 1)
            next_dx = float(points[target_index][0] - x)
            next_dy = float(points[target_index][1] - y)

        detections = self._detection_summary(core, missing)
        prev_base = self._pair(previous_base_action)
        prev_residual = self._pair(previous_residual_action, residual=True)
        prev_final = self._pair(previous_final_action)
        signed_deviation = float(deviation) if is_right else -float(deviation)

        obs = np.array(
            [
                x,
                y,
                float(np.sin(yaw)),
                float(np.cos(yaw)),
                speed,
                float(deviation),
                signed_deviation,
                route_progress,
                next_dx,
                next_dy,
                float(getattr(segment, "segment_id", 0)),
                detections["cone"],
                detections["people"],
                detections["cow"],
                detections["red"],
                detections["green"],
                detections["stop_line"],
                prev_base[0],
                prev_base[1],
                prev_residual[0],
                prev_residual[1],
                prev_final[0],
                prev_final[1],
                is_right,
                float(getattr(segment, "residual_enabled", False)),
                float(getattr(core, "max_speed", 0.0) or 0.0),
                float(getattr(core, "min_speed", 0.0) or 0.0),
                float(getattr(core, "ld", 0.0) or 0.0),
                float(getattr(core, "avoidance", False)),
            ],
            dtype=np.float32,
        )
        info = {
            "missing_fields": missing,
            "lane_deviation": float(deviation),
            "route_progress": float(route_progress),
        }
        return obs, info

    def _pair(self, action, residual: bool = False) -> Tuple[float, float]:
        if not action:
            return 0.0, 0.0
        if residual:
            return (
                float(action.get("speed_delta", 0.0)),
                float(action.get("steering_bias", 0.0)),
            )
        return (
            float(action.get("forward", 0.0)),
            float(action.get("turn", 0.0)),
        )

    def _detection_summary(self, core, missing: List[str]) -> Dict[str, float]:
        summary = {
            "cone": 0.0,
            "people": 0.0,
            "cow": 0.0,
            "red": 0.0,
            "green": 0.0,
            "stop_line": 0.0,
        }
        det = getattr(core, "last_detection_result", None)
        obj = getattr(core, "object_classes", None)
        if not det or len(det) == 0 or obj is None:
            missing.append("last_detection_result")
            return summary

        labels = det[0] if len(det) > 0 else []
        sizes = det[1] if len(det) > 1 else []
        mapping = {
            "cone": getattr(obj, "CONE", None),
            "people": getattr(obj, "PEOPLE", None),
            "cow": getattr(obj, "COW", None),
            "red": getattr(obj, "RED", None),
            "green": getattr(obj, "GREEN", None),
            "stop_line": getattr(obj, "STOP_LINE", None),
        }
        for name, label in mapping.items():
            if label in labels:
                index = labels.index(label)
                summary[name] = float(sizes[index]) if index < len(sizes) else 1.0
        return summary
