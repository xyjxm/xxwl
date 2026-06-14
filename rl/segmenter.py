"""Track segmentation for residual action gating."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple


ResidualRange = Dict[str, Tuple[float, float]]


@dataclass(frozen=True)
class Segment:
    segment_id: int
    name: str
    residual_enabled: bool
    residual_range: ResidualRange


ZERO_RANGE: ResidualRange = {
    "speed_delta": (0.0, 0.0),
    "steering_bias": (0.0, 0.0),
}


class TrackSegmenter:
    """Position-based segmentation for scenario 3.

    The ranges are deliberately conservative: normal lane following is either
    off or almost off, while cone/actor/final sections allow small corrections.
    """

    def __init__(self):
        actor_residual_enabled = os.environ.get("RL_ACTOR_RESIDUAL_ENABLED", "0") == "1"
        normal_steering_limit = float(os.environ.get("RL_NORMAL_STEERING_BIAS_LIMIT", "0.035"))
        self.normal = Segment(
            0,
            "normal_lane_following",
            True,
            {
                "speed_delta": (-0.04, 0.03),
                "steering_bias": (-normal_steering_limit, normal_steering_limit),
            },
        )
        self.cone = Segment(
            1,
            "cone_area",
            True,
            {"speed_delta": (-0.08, 0.04), "steering_bias": (-0.055, 0.055)},
        )
        self.actor = Segment(
            2,
            "pedestrian_cow_area",
            actor_residual_enabled,
            (
                {"speed_delta": (-0.02, 0.01), "steering_bias": (-0.006, 0.006)}
                if actor_residual_enabled
                else ZERO_RANGE
            ),
        )
        self.final = Segment(
            3,
            "final_straight",
            True,
            {"speed_delta": (0.0, 0.06), "steering_bias": (-0.01, 0.01)},
        )
        self.finish = Segment(
            4,
            "finish_approach",
            True,
            {"speed_delta": (0.0, 0.04), "steering_bias": (-0.01, 0.01)},
        )

    def segment_for_position(self, position) -> Segment:
        if position is None:
            return self.normal
        x = float(position[0])
        y = float(position[1])

        if 1.55 <= x <= 2.55 and 0.20 <= y <= 1.85:
            return self.cone

        lower_people = 0.25 <= x <= 1.45 and -1.35 <= y <= -0.35
        upper_people = -2.25 <= x <= -1.05 and 2.90 <= y <= 4.25
        cow = -0.55 <= x <= 0.80 and 3.55 <= y <= 4.70
        if lower_people or upper_people or cow:
            return self.actor

        if -2.30 <= x <= -1.25 and 0.15 <= y <= 2.70:
            return self.final

        if -2.25 <= x <= -1.45 and -0.20 <= y <= 0.45:
            return self.finish

        return self.normal
