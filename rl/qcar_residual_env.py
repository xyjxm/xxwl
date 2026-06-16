"""Gym-style QCar residual RL environment."""

from __future__ import annotations

import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym

from AutonomousDriveCore import AutonomousDriveCore
from StudentDecision import StudentDecision

try:
    from .baseline_registry import FAST_LEARN_BASE_POLICY_ENV, FAST_SAFE_BASE_POLICY_ENV, STRONGEST_BASE_POLICY_ENV, USER_CONFIRMED_STRONGEST
    from .qlab_reentry import reenter_plane
    from .residual_action import ResidualActionApplier, get_action_space
    from .reward import ResidualReward
    from .segmenter import TrackSegmenter
    from .state_encoder import StateEncoder
except ImportError:  # script-style import fallback
    from baseline_registry import FAST_LEARN_BASE_POLICY_ENV, FAST_SAFE_BASE_POLICY_ENV, STRONGEST_BASE_POLICY_ENV, USER_CONFIRMED_STRONGEST
    from qlab_reentry import reenter_plane
    from residual_action import ResidualActionApplier, get_action_space
    from reward import ResidualReward
    from segmenter import TrackSegmenter
    from state_encoder import StateEncoder


class QCarResidualEnv(gym.Env):
    """Residual environment that keeps StudentDecision as the base policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: int = 3,
        base_policy: str = "strongest",
        reward_minor_threshold: float = 0.04,
        max_steps: int = 3000,
        zero_residual: bool = False,
        fixed_scene: bool = True,
        auto_reenter_plane: bool = False,
    ):
        super().__init__()
        self.scenario = int(scenario)
        self.base_policy = base_policy
        self.max_steps = int(max_steps)
        self.zero_residual = bool(zero_residual)
        self.fixed_scene = bool(fixed_scene)
        self.auto_reenter_plane = bool(
            auto_reenter_plane or os.environ.get("QLAB_AUTO_REENTER_PLANE", "0") == "1"
        )
        self._previous_env_values: Dict[str, Optional[str]] = {}
        self._apply_base_policy_env()

        self.segmenter = TrackSegmenter()
        self.encoder = StateEncoder()
        self.residual = ResidualActionApplier(zero_mode=self.zero_residual)
        self.reward_model = ResidualReward(
            baseline_final_time=USER_CONFIRMED_STRONGEST.mean_final_time_seconds,
            minor_threshold=reward_minor_threshold,
        )

        self.action_space = get_action_space()
        self.observation_space = self.encoder.get_observation_space()

        self.core: Optional[AutonomousDriveCore] = None
        self.student: Optional[StudentDecision] = None
        self.pending_action = self.residual.zero_action()
        self.step_count = 0
        self.lap_start = 0.0
        self.previous_info: Dict[str, Any] = {}
        self.previous_penalty = 0.0
        self.previous_penalty_breakdown: Dict[str, Any] = {}
        self._last_progress_position: Optional[np.ndarray] = None
        self._stall_steps = 0
        self._stall_distance_epsilon = float(os.environ.get("RL_STALL_DISTANCE_EPSILON", "0.0002"))
        self._max_stall_steps = int(os.environ.get("RL_MAX_STALL_STEPS", "360"))
        self.strict_lane_shield = os.environ.get("RL_STRICT_LANE_SHIELD", "1") == "1"
        self.strict_lane_start = float(os.environ.get("RL_STRICT_LANE_START", "0.015"))
        self.strict_lane_gain = float(os.environ.get("RL_STRICT_LANE_GAIN", "2.60"))
        self.strict_lane_max = float(os.environ.get("RL_STRICT_LANE_MAX", "0.110"))
        self.strict_lane_cone_max = float(os.environ.get("RL_STRICT_LANE_CONE_MAX", "0.020"))
        self.strict_lane_lower_left_max = float(
            os.environ.get("RL_STRICT_LANE_LOWER_LEFT_MAX", str(self.strict_lane_max))
        )
        self.strict_lane_lower_left_gain = float(
            os.environ.get("RL_STRICT_LANE_LOWER_LEFT_GAIN", str(self.strict_lane_gain))
        )
        self.strict_lane_heading_gain = float(os.environ.get("RL_STRICT_LANE_HEADING_GAIN", "0.00"))
        self.strict_lane_heading_max = float(os.environ.get("RL_STRICT_LANE_HEADING_MAX", "0.060"))
        self.strict_lane_speed_cap = float(os.environ.get("RL_STRICT_LANE_SPEED_CAP", "0.00"))
        self.strict_lane_cone_speed_cap = float(
            os.environ.get("RL_STRICT_LANE_CONE_SPEED_CAP", str(self.strict_lane_speed_cap))
        )
        self.strict_lane_speed_start = float(os.environ.get("RL_STRICT_LANE_SPEED_START", "0.055"))
        self.strict_lane_crosstrack_sign = float(os.environ.get("RL_STRICT_LANE_CROSSTRACK_SIGN", "1.0"))
        self.strict_lane_lower_entry_crosstrack_sign = float(
            os.environ.get("RL_STRICT_LANE_LOWER_ENTRY_CROSSTRACK_SIGN", str(self.strict_lane_crosstrack_sign))
        )
        self.strict_lane_lower_upcurve_crosstrack_sign = float(
            os.environ.get("RL_STRICT_LANE_LOWER_UPCURVE_CROSSTRACK_SIGN", str(self.strict_lane_crosstrack_sign))
        )
        self.strict_lane_final_crosstrack_sign = float(
            os.environ.get("RL_STRICT_LANE_FINAL_CROSSTRACK_SIGN", str(self.strict_lane_crosstrack_sign))
        )
        self.line_damping_enabled = os.environ.get("RL_LINE_DAMPING_ENABLED", "0") == "1"
        self.line_damping_start = float(os.environ.get("RL_LINE_DAMPING_START", "0.030"))
        self.line_damping_full = float(os.environ.get("RL_LINE_DAMPING_FULL", "0.050"))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.fixed_scene:
            os.environ.setdefault("TRAFFIC_LIGHT_MODE", "fixed_green")
            os.environ.setdefault("TRAFFIC_ACTOR_MODE", "triggered")

        reset_attempts = max(1, int(options.get("reset_attempts", 2)))
        retry_sleep = float(options.get("reset_retry_sleep", 1.5))
        last_error = None
        reentered_plane = False
        attempt = 0

        while True:
            attempt += 1
            try:
                self.close()
                self._apply_base_policy_env()
                self._push_env_var("QLAB_ALLOW_DIRECT_SETUP_ON_HEALTH_FAIL", "1")

                self.core = AutonomousDriveCore(scenario_num=self.scenario)
                self.core.single_tick_mode = True
                self.core.action_adapter = self._adapt_action
                self.core.initialize_system()
                self.student = StudentDecision(self.core)

                warmup_start = np.array(self.core.current_position, dtype=float)
                for _ in range(int(options.get("warmup_ticks", 5))):
                    self.pending_action = self.residual.zero_action()
                    self.student.pure_pursuit_control()
                    self.core.run_referee_checks()
                    time.sleep(0.01)
                warmup_distance = float(np.linalg.norm(np.array(self.core.current_position, dtype=float) - warmup_start))
                if warmup_distance < 1e-4:
                    raise RuntimeError(
                        "QLabs/QCar state did not advance during warmup; "
                        "the Plane scene likely needs to be exited and re-entered before evaluation."
                    )

                self.step_count = 0
                self.lap_start = time.time()
                self.previous_penalty = self._penalty_time()
                self.previous_penalty_breakdown = self._penalty_breakdown()
                self.previous_info = self._metrics_info()
                self._last_progress_position = np.array(self.core.current_position, dtype=float)
                self._stall_steps = 0
                obs, enc_info = self._observation()
                info = self._base_info(enc_info)
                if attempt > 1:
                    info["reset_retry_attempt"] = attempt
                return obs, info
            except Exception as exc:
                last_error = exc
                self.close()
                if attempt >= reset_attempts and self.auto_reenter_plane and not reentered_plane:
                    print(
                        "QCarResidualEnv reset failed after cleanup retries; "
                        "re-entering QLAB Plane without restarting QLAB."
                    )
                    reenter_plane()
                    reentered_plane = True
                    attempt = 0
                    time.sleep(retry_sleep)
                    continue
                if attempt >= reset_attempts:
                    raise
                print(f"QCarResidualEnv reset attempt {attempt}/{reset_attempts} failed: {exc}; retrying after cleanup.")
                time.sleep(retry_sleep)

        raise RuntimeError(f"QCarResidualEnv reset failed: {last_error}")

    def step(self, action):
        if self.core is None or self.student is None:
            raise RuntimeError("QCarResidualEnv.reset() must be called before step().")

        self.pending_action = np.asarray(action, dtype=np.float32)
        before_info = self._metrics_info()
        before_penalty = self._penalty_time()
        before_breakdown = self._penalty_breakdown()

        self.core.run_referee_checks()
        self._base_policy_tick()
        self.core.run_referee_checks()

        self.step_count += 1
        terminated = bool(self.core.check_endpoint())
        truncated = self.step_count >= self.max_steps
        invalid_reason = "max_steps" if truncated and not terminated else None

        after_penalty = self._penalty_time()
        after_breakdown = self._penalty_breakdown()
        obs, enc_info = self._observation()
        info = self._base_info(enc_info)
        info.update(self._penalty_delta_info(before_penalty, after_penalty, before_breakdown, after_breakdown))
        if not terminated and self._last_progress_position is not None:
            current_position = np.array(self.core.current_position, dtype=float)
            moved = float(np.linalg.norm(current_position - self._last_progress_position))
            final_action = info.get("final_action", {}) or {}
            commanded_forward = abs(float(final_action.get("forward", 0.0) or 0.0))
            if moved < self._stall_distance_epsilon and commanded_forward > 0.05:
                self._stall_steps += 1
            else:
                self._stall_steps = 0
                self._last_progress_position = current_position
            if self._stall_steps >= self._max_stall_steps:
                truncated = True
                invalid_reason = "qcar_no_progress"
        elapsed_time = time.time() - self.lap_start
        info["elapsed_time"] = elapsed_time
        info["position"] = [
            float(self.core.current_position[0]),
            float(self.core.current_position[1]),
        ]
        info["yaw"] = float(self.core.yaw)
        info["speed"] = float(self.core.speed)
        info["raw_time"] = elapsed_time if terminated else None
        info["penalty_time"] = after_penalty if terminated else None
        info["final_time"] = (
            info["raw_time"] + after_penalty
            if terminated and info["raw_time"] is not None
            else None
        )
        info["finished"] = terminated
        info["invalid_reason"] = invalid_reason
        info["stall_steps"] = self._stall_steps
        info["reward_threshold_0p04_violation_ratio"] = 1.0 if info["lane_deviation"] > 0.04 else 0.0

        reward_context = dict(info)
        reward, components = self.reward_model.compute(
            previous_info=before_info,
            current_info=reward_context,
            action_record=self.core.last_action_record,
            terminated=terminated,
        )
        info["reward_components"] = components
        info["reward"] = reward
        self.previous_info = info
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.core is not None:
            try:
                self.core.cleanup()
            except Exception:
                pass
        self.core = None
        self.student = None
        self._restore_base_policy_env()

    def _apply_base_policy_env(self):
        policy_envs = {
            "strongest": STRONGEST_BASE_POLICY_ENV,
            "fast_safe": FAST_SAFE_BASE_POLICY_ENV,
            "fast_learn": FAST_LEARN_BASE_POLICY_ENV,
        }
        policy_env = policy_envs.get(self.base_policy)
        if policy_env is None:
            return
        if self._previous_env_values:
            return
        for key, value in policy_env.items():
            self._previous_env_values[key] = os.environ.get(key)
            os.environ[key] = str(value)
        overrides_raw = os.environ.get("RL_BASE_POLICY_OVERRIDES")
        if overrides_raw:
            try:
                overrides = json.loads(overrides_raw)
            except json.JSONDecodeError as exc:
                raise ValueError("RL_BASE_POLICY_OVERRIDES must be a JSON object") from exc
            if not isinstance(overrides, dict):
                raise ValueError("RL_BASE_POLICY_OVERRIDES must be a JSON object")
            for key, value in overrides.items():
                if key not in self._previous_env_values:
                    self._previous_env_values[key] = os.environ.get(key)
                os.environ[str(key)] = str(value)

    def _restore_base_policy_env(self):
        if not self._previous_env_values:
            return
        for key, value in self._previous_env_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._previous_env_values = {}

    def _push_env_var(self, key: str, value: str):
        if key not in self._previous_env_values:
            self._previous_env_values[key] = os.environ.get(key)
        os.environ[key] = value

    def _adapt_action(self, forward: float, turn: float, context: str, core):
        segment = self.segmenter.segment_for_position(core.current_position)
        final_forward, final_turn, record = self.residual.apply_to_command(
            forward,
            turn,
            self.pending_action,
            segment,
        )
        line_damping = self._line_following_residual_damping(core, segment, record)
        if line_damping:
            final_turn += line_damping
            record.setdefault("final_action", {})["turn"] = float(final_turn)
            record.setdefault("final_action", {})["forward"] = float(final_forward)
        lane_correction, lane_speed_cap, lane_meta = self._strict_lane_correction(core, segment)
        if lane_correction:
            final_turn += lane_correction
            record.setdefault("final_action", {})["turn"] = float(final_turn)
            record.setdefault("final_action", {})["forward"] = float(final_forward)
        if lane_speed_cap > 0.0 and final_forward > lane_speed_cap:
            final_forward = lane_speed_cap
            record.setdefault("final_action", {})["turn"] = float(final_turn)
            record.setdefault("final_action", {})["forward"] = float(final_forward)
        record.update(
            {
                "segment_id": segment.segment_id,
                "segment_name": segment.name,
                "context": context,
                "strict_lane_correction": lane_correction,
                "strict_lane_heading_error": lane_meta.get("heading_error", 0.0),
                "strict_lane_signed_deviation": lane_meta.get("signed_deviation", 0.0),
                "strict_lane_speed_cap": lane_speed_cap,
                "line_damping_correction": line_damping,
            }
        )
        return final_forward, final_turn, record

    def _line_following_residual_damping(self, core, segment, record) -> float:
        """Cancel residual steering only when it fights lane-centering near the limit."""
        if (
            not self.line_damping_enabled
            or getattr(segment, "name", "") != "normal_lane_following"
            or core.penalty_system is None
            or core.current_position is None
        ):
            return 0.0

        deviation, is_right = core.penalty_system.calculate_lane_deviation(core.current_position)
        deviation = float(deviation)
        if deviation <= self.line_damping_start:
            return 0.0

        residual_bias = float(
            (record.get("residual_action") or {}).get("steering_bias", 0.0) or 0.0
        )
        if abs(residual_bias) < 1e-9:
            return 0.0

        signed_deviation = -deviation if is_right else deviation
        center_sign = float(np.sign(signed_deviation))
        if center_sign == 0.0 or residual_bias * center_sign >= 0.0:
            return 0.0

        span = max(1e-6, self.line_damping_full - self.line_damping_start)
        strength = min(1.0, max(0.0, (deviation - self.line_damping_start) / span))
        return float(-residual_bias * strength)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    def _lane_geometry(self, core):
        if core.penalty_system is None or core.current_position is None:
            return None
        deviation, is_right = core.penalty_system.calculate_lane_deviation(core.current_position)
        signed_deviation = -float(deviation) if is_right else float(deviation)

        heading_error = 0.0
        path_points = getattr(core, "path_points", None)
        if path_points is not None and len(path_points) >= 2:
            points = np.asarray(path_points, dtype=float)
            pos = np.asarray(core.current_position, dtype=float)
            nearest = int(np.argmin(np.linalg.norm(points - pos, axis=1)))
            p1_idx = nearest
            p2_idx = min(nearest + 1, len(points) - 1)
            if p1_idx == p2_idx and p1_idx > 0:
                p1_idx -= 1
            segment_vec = points[p2_idx] - points[p1_idx]
            if float(np.linalg.norm(segment_vec)) > 1e-6:
                path_heading = math.atan2(float(segment_vec[1]), float(segment_vec[0]))
                heading_error = self._wrap_angle(path_heading - float(core.yaw))

        return float(deviation), signed_deviation, heading_error

    def _strict_lane_correction(self, core, segment):
        """Small cross-track steering trim for the stricter 0.04m local experiment."""
        if not self.strict_lane_shield or core.penalty_system is None or core.current_position is None:
            return 0.0, 0.0, {}
        lane_geometry = self._lane_geometry(core)
        if lane_geometry is None:
            return 0.0, 0.0, {}
        deviation, signed_deviation, heading_error = lane_geometry

        x = float(core.current_position[0])
        y = float(core.current_position[1])
        if (
            getattr(segment, "name", "") == "pedestrian_cow_area"
            and -0.70 <= x <= 0.80
            and 3.55 <= y <= 4.72
            and not bool(getattr(self.student, "cow_clearance_yield_done", False))
        ):
            return 0.0, 0.0, {
                "signed_deviation": signed_deviation,
                "heading_error": heading_error,
            }

        start = self.strict_lane_start
        gain = self.strict_lane_gain
        max_correction = self.strict_lane_max
        crosstrack_sign = self.strict_lane_crosstrack_sign
        speed_cap = self.strict_lane_speed_cap
        if -1.05 <= x <= -0.88 and -1.04 <= y <= -0.95:
            gain = self.strict_lane_lower_left_gain
            max_correction = max(max_correction, self.strict_lane_lower_left_max)
            crosstrack_sign = self.strict_lane_lower_entry_crosstrack_sign
        if -0.08 <= x <= 0.24 and -1.06 <= y <= -0.99:
            crosstrack_sign = self.strict_lane_lower_entry_crosstrack_sign
        if 2.00 <= x <= 2.16 and -0.86 <= y <= -0.60:
            crosstrack_sign = self.strict_lane_lower_upcurve_crosstrack_sign
        if 0.95 <= y <= 1.34 and 1.86 <= x <= 2.10:
            max_correction = self.strict_lane_cone_max
            speed_cap = self.strict_lane_cone_speed_cap
        if -2.14 <= x <= -1.85 and 2.15 <= y <= 2.85:
            crosstrack_sign = self.strict_lane_final_crosstrack_sign

        excess = abs(signed_deviation) - start
        correction = 0.0
        if excess > 0.0:
            correction += float(
                crosstrack_sign
                * np.sign(signed_deviation)
                * min(max_correction, excess * gain)
            )

        if self.strict_lane_heading_gain > 0.0 and abs(heading_error) > 0.03:
            correction += float(
                np.clip(
                    -heading_error * self.strict_lane_heading_gain,
                    -self.strict_lane_heading_max,
                    self.strict_lane_heading_max,
                )
            )
        correction = float(np.clip(correction, -max_correction, max_correction))
        lane_speed_cap = speed_cap if abs(signed_deviation) > self.strict_lane_speed_start else 0.0
        return correction, lane_speed_cap, {
            "signed_deviation": signed_deviation,
            "heading_error": heading_error,
        }

    def _base_policy_tick(self):
        det_result = [[], [], []] if self.scenario == 3 else self.core.detection()
        self.core.last_detection_result = det_result

        if hasattr(self.student, "_service_single_tick_hold") and self.student._service_single_tick_hold():
            return

        if self.scenario != 3:
            self.student.pure_pursuit_control()
            return

        if self.student.should_position_yield():
            self.student.handle_people_cow_logic(det_result)
        elif self.student.should_focus_cone_only(det_result):
            self.student.handle_cone_avoidance_logic(det_result, False)
        elif det_result and len(det_result) > 0 and len(det_result[0]) > 0:
            detected_objects = det_result[0]
            object_classes = self.core.object_classes
            if (
                not self.student.should_prioritize_cone(det_result)
                and (
                    self.student.should_position_yield()
                    or object_classes.PEOPLE in detected_objects
                    or object_classes.COW in detected_objects
                )
            ):
                self.student.handle_people_cow_logic(det_result)
            elif (
                not self.student.should_prioritize_cone(det_result)
                and object_classes.STOP_SIGN in detected_objects
            ):
                self.student.handle_stop_sign_logic(det_result)
            else:
                self.student.handle_cone_avoidance_logic(det_result, False)
        else:
            if self.student.should_position_yield():
                self.student.handle_people_cow_logic(det_result)
            else:
                self.student.handle_cone_avoidance_logic(det_result, False)

    def _observation(self):
        segment = self.segmenter.segment_for_position(self.core.current_position)
        record = getattr(self.core, "last_action_record", {})
        return self.encoder.encode(
            self.core,
            segment,
            previous_base_action=record.get("base_action"),
            previous_residual_action=record.get("residual_action"),
            previous_final_action=record.get("final_action"),
        )

    def _base_info(self, encoder_info: Dict[str, Any]) -> Dict[str, Any]:
        record = getattr(self.core, "last_action_record", {})
        segment = self.segmenter.segment_for_position(self.core.current_position)
        info = {
            "segment_id": segment.segment_id,
            "segment_name": segment.name,
            "residual_enabled": bool(record.get("residual_enabled", False)),
            "base_action": record.get("base_action", {"forward": 0.0, "turn": 0.0}),
            "residual_action": record.get("residual_action", {"speed_delta": 0.0, "steering_bias": 0.0}),
            "final_action": record.get("final_action", {"forward": 0.0, "turn": 0.0}),
            "strict_lane_correction": float(record.get("strict_lane_correction", 0.0) or 0.0),
            "strict_lane_heading_error": float(record.get("strict_lane_heading_error", 0.0) or 0.0),
            "strict_lane_signed_deviation": float(record.get("strict_lane_signed_deviation", 0.0) or 0.0),
            "strict_lane_speed_cap": float(record.get("strict_lane_speed_cap", 0.0) or 0.0),
            "step_count": self.step_count,
        }
        info.update(encoder_info)
        return info

    def _metrics_info(self) -> Dict[str, Any]:
        _, enc_info = self._observation() if self.core is not None else (None, {})
        record = getattr(self.core, "last_action_record", {}) if self.core is not None else {}
        return {
            "route_progress": float(enc_info.get("route_progress", 0.0) or 0.0),
            "lane_deviation": float(enc_info.get("lane_deviation", 0.0) or 0.0),
            "residual_action": record.get("residual_action", {"speed_delta": 0.0, "steering_bias": 0.0}),
        }

    def _penalty_time(self) -> float:
        if self.core is None or self.core.penalty_system is None:
            return 0.0
        return float(self.core.penalty_system.total_penalty_time)

    def _penalty_breakdown(self) -> Dict[str, Any]:
        if self.core is None or self.core.penalty_system is None:
            return {}
        return dict(self.core.penalty_system.penalty_stats)

    def _penalty_delta_info(self, before_penalty, after_penalty, before_breakdown, after_breakdown):
        keys = [
            "cone_collision_count",
            "people_collision_count",
            "cow_collision_count",
            "red_light_violation_count",
        ]
        deltas = {
            key: int(after_breakdown.get(key, 0) or 0) - int(before_breakdown.get(key, 0) or 0)
            for key in keys
        }
        return {
            "penalty_increment": max(0.0, float(after_penalty) - float(before_penalty)),
            "collision_increment": max(
                0,
                deltas["cone_collision_count"]
                + deltas["people_collision_count"]
                + deltas["cow_collision_count"],
            ),
            "red_light_increment": max(0, deltas["red_light_violation_count"]),
            "collision_count": (
                int(after_breakdown.get("cone_collision_count", 0) or 0)
                + int(after_breakdown.get("people_collision_count", 0) or 0)
                + int(after_breakdown.get("cow_collision_count", 0) or 0)
            ),
            "red_light_violation_count": int(after_breakdown.get("red_light_violation_count", 0) or 0),
            "penalty_breakdown": after_breakdown,
        }
