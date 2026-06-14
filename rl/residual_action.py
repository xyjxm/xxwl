"""Small residual action transformation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - local Quanser env currently has gym
    import gym


GLOBAL_SPEED_DELTA = (-0.20, 0.20)
GLOBAL_STEERING_BIAS = (-0.15, 0.15)


@dataclass(frozen=True)
class AppliedAction:
    base_action: Dict[str, float]
    residual_action: Dict[str, float]
    final_action: Dict[str, float]
    residual_enabled: bool


def get_action_space():
    return gym.spaces.Box(
        low=np.array([GLOBAL_SPEED_DELTA[0], GLOBAL_STEERING_BIAS[0]], dtype=np.float32),
        high=np.array([GLOBAL_SPEED_DELTA[1], GLOBAL_STEERING_BIAS[1]], dtype=np.float32),
        dtype=np.float32,
    )


def _clip(value: float, bounds: Tuple[float, float]) -> float:
    return float(np.clip(float(value), float(bounds[0]), float(bounds[1])))


class ResidualActionApplier:
    """Apply physical residuals to a base forward/turn command."""

    def __init__(self, zero_mode: bool = False, max_forward: float = 2.30):
        self.zero_mode = zero_mode
        self.max_forward = float(max_forward)

    def zero_action(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def apply(self, base_action: Dict[str, float], raw_action: Iterable[float], segment) -> AppliedAction:
        base_forward = float(base_action["forward"])
        base_turn = float(base_action["turn"])

        raw = np.asarray(raw_action if raw_action is not None else [0.0, 0.0], dtype=np.float32)
        if raw.shape != (2,):
            raw = np.zeros(2, dtype=np.float32)

        if self.zero_mode or not getattr(segment, "residual_enabled", False):
            speed_delta = 0.0
            steering_bias = 0.0
            residual_enabled = False
        else:
            speed_range = segment.residual_range.get("speed_delta", GLOBAL_SPEED_DELTA)
            steering_range = segment.residual_range.get("steering_bias", GLOBAL_STEERING_BIAS)
            speed_delta = _clip(raw[0], speed_range)
            steering_bias = _clip(raw[1], steering_range)
            residual_enabled = True

        final_forward = _clip(base_forward + speed_delta, (0.0, self.max_forward))
        final_turn = base_turn + steering_bias

        return AppliedAction(
            base_action={"forward": base_forward, "turn": base_turn},
            residual_action={"speed_delta": speed_delta, "steering_bias": steering_bias},
            final_action={"forward": final_forward, "turn": final_turn},
            residual_enabled=residual_enabled,
        )

    def apply_to_command(
        self,
        forward: float,
        turn: float,
        raw_action: Iterable[float],
        segment,
    ) -> Tuple[float, float, Dict[str, Any]]:
        applied = self.apply({"forward": forward, "turn": turn}, raw_action, segment)
        return (
            applied.final_action["forward"],
            applied.final_action["turn"],
            {
                "base_action": applied.base_action,
                "residual_action": applied.residual_action,
                "final_action": applied.final_action,
                "residual_enabled": applied.residual_enabled,
            },
        )

