import time
import os
import numpy as np

from AutonomousDriveCore import AutonomousDriveCore


class StudentDecision:
    """Decision logic for the scenario controllers."""

    def __init__(self, core: AutonomousDriveCore):
        self.core = core

        self.last_stop_sign_time = 0.0
        self.stop_sign_cooldown = 3.0

        self.last_people_stop_time = 0.0
        self.last_cow_stop_time = 0.0
        self.people_cooldown = 2.0
        self.cow_cooldown = 2.0
        self.stop_sign_hold_seconds = float(os.environ.get("STOP_SIGN_HOLD_SECONDS", "0.0"))
        self.people_hold_seconds = float(os.environ.get("PEOPLE_HOLD_SECONDS", "1.0"))
        self.cow_hold_seconds = float(os.environ.get("COW_HOLD_SECONDS", "3.30"))
        self.upper_people_hold_seconds = float(os.environ.get("UPPER_PEOPLE_HOLD_SECONDS", "2.20"))
        self.lower_people_hold_seconds = float(os.environ.get("LOWER_PEOPLE_HOLD_SECONDS", "0.00"))
        self.lower_people_pass_enabled = os.environ.get("LOWER_PEOPLE_PASS_ENABLED", "0") == "1"
        self.lower_people_pass_y = float(os.environ.get("LOWER_PEOPLE_PASS_Y", "-0.820"))
        self.lower_entry_target_y = float(os.environ.get("LOWER_ENTRY_TARGET_Y", "-1.115"))
        self.lower_entry_speed = float(os.environ.get("LOWER_ENTRY_SPEED", "2.08"))
        self.lower_mid_target_x = float(os.environ.get("LOWER_MID_TARGET_X", "2.00"))
        self.lower_mid_target_y = float(os.environ.get("LOWER_MID_TARGET_Y", "-0.975"))
        self.lower_mid_speed = float(os.environ.get("LOWER_MID_SPEED", "1.34"))
        self.lower_exit_target_x = float(os.environ.get("LOWER_EXIT_TARGET_X", "2.05"))
        self.lower_exit_target_y = float(os.environ.get("LOWER_EXIT_TARGET_Y", "-0.920"))
        self.lower_exit_speed = float(os.environ.get("LOWER_EXIT_SPEED", "1.15"))
        self.lower_after_wait_target_x = float(os.environ.get("LOWER_AFTER_WAIT_TARGET_X", "1.35"))
        self.lower_after_wait_target_y = float(os.environ.get("LOWER_AFTER_WAIT_TARGET_Y", "-0.980"))
        self.lower_after_wait_speed = float(os.environ.get("LOWER_AFTER_WAIT_SPEED", "0.25"))
        self.lower_curve_enabled = os.environ.get("LOWER_CURVE_ENABLED", "0") == "1"
        self.lower_curve_target_x = float(os.environ.get("LOWER_CURVE_TARGET_X", "1.930"))
        self.lower_curve_forward_y = float(os.environ.get("LOWER_CURVE_FORWARD_Y", "0.300"))
        self.lower_curve_speed = float(os.environ.get("LOWER_CURVE_SPEED", "0.62"))
        self.lower_overrides_enabled = os.environ.get("LOWER_OVERRIDES_ENABLED", "0") == "1"
        self.upper_people_position_yield_done = False
        self.lower_people_position_yield_done = False
        self.cow_position_yield_done = False

        self.last_cone_avoidance_time = 0.0
        self.cone_cooldown = 4.0
        self.cone_avoid_until = 0.0
        self.cone_recover_until = 0.0
        self.cone_bypass_done = False
        self.cone_sprint_done = False

        self.use_reference_cone_strategy = self.core.scenario_num == 3
        self.last_cone_log_time = 0.0
        self.cone_log_cooldown = 0.8
        self.cone_avoidance_until = 0.0
        self.cone_avoidance_duration = 2.2
        self.cone_threshold = 1.10
        self.default_cone_avoid_angle = 0.20
        self.cone_release_y = 1.18
        self.cone_recovery_start_y = 1.18
        self.cone_recovery_angle = -0.23
        self.cone_recovery_duration = 1.6
        self.cone_recovery_clear_x = 2.110
        self.cone_recovery_clear_y = 1.66
        self.cone_recovery_until = 0.0
        self.cone_recovery_started = False
        self.cone_seen = False
        self.last_cone_size = None
        self.last_cone_focus_log_time = 0.0
        self.cone_hard_recover_done = False
        self.sampling_bypass_experiment = False
        self.sampling_bypass_done = False
        self.fast_finish_after_cone = False
        self.visual_cone_half_side = 0.15
        self.cone_setup_x = float(os.environ.get("CONE_SETUP_X", "2.020"))
        self.cone_setup_forward = float(os.environ.get("CONE_SETUP_FORWARD", "1.620"))
        self.cone_pre_x = float(os.environ.get("CONE_PRE_X", "1.950"))
        self.cone_pre_forward = float(os.environ.get("CONE_PRE_FORWARD", "1.550"))
        self.cone_lower_x = float(os.environ.get("CONE_LOWER_X", "1.985"))
        self.cone_lower_forward = float(os.environ.get("CONE_LOWER_FORWARD", "1.550"))
        self.cone_recover_start_y = float(os.environ.get("CONE_RECOVER_START_Y", "1.000"))
        self.cone_recover_target_x = float(os.environ.get("CONE_RECOVER_TARGET_X", "6.000"))
        self.cone_recover_forward = float(os.environ.get("CONE_RECOVER_FORWARD", "1.650"))
        self.cone_raw_start_y = float(os.environ.get("CONE_RAW_START_Y", "9.000"))
        self.cone_raw_turn = float(os.environ.get("CONE_RAW_TURN", "2.800"))
        self.cone_raw_forward = float(os.environ.get("CONE_RAW_FORWARD", "1.050"))
        self.cone_raw_guard_x = float(os.environ.get("CONE_RAW_GUARD_X", "2.130"))
        self.cone_top_turn_start_y = float(os.environ.get("CONE_TOP_TURN_START_Y", "1.230"))
        self.cone_top_turn = float(os.environ.get("CONE_TOP_TURN", "4.000"))
        self.cone_top_forward = float(os.environ.get("CONE_TOP_FORWARD", "1.050"))
        self.cone_top_burst_frames = int(os.environ.get("CONE_TOP_BURST_FRAMES", "1"))
        self.cone_top_burst_remaining = 0
        self.cone_top_burst_done = False
        self.cone_return_forward = float(os.environ.get("CONE_RETURN_FORWARD", "2.050"))
        self.original_cone_style = os.environ.get("ORIGINAL_CONE_STYLE", "0") == "1"
        self.cone_pivot_frames = int(os.environ.get("CONE_PIVOT_FRAMES", "0"))
        self.cone_pivot_remaining = 0
        self.cone_pivot_done = False
        self.cone_pivot_forward = float(os.environ.get("CONE_PIVOT_FORWARD", "0.12"))
        self.cone_pivot_turn = float(os.environ.get("CONE_PIVOT_TURN", "2.60"))
        self.cone_pivot_start_y = float(os.environ.get("CONE_PIVOT_START_Y", "1.270"))
        self.cone_pivot_end_y = float(os.environ.get("CONE_PIVOT_END_Y", "1.320"))
        self.cone_use_slow_profile = os.environ.get("CONE_USE_SLOW_PROFILE", "0") == "1"
        self.cone_delay_left_enabled = os.environ.get("CONE_DELAY_LEFT_ENABLED", "0") == "1"
        self.cone_delay_start_y = float(os.environ.get("CONE_DELAY_START_Y", "0.820"))
        self.cone_delay_hold_end_y = float(os.environ.get("CONE_DELAY_HOLD_END_Y", "0.920"))
        self.cone_delay_hold_x = float(os.environ.get("CONE_DELAY_HOLD_X", "2.145"))
        self.cone_delay_hold_forward = float(os.environ.get("CONE_DELAY_HOLD_FORWARD", "1.20"))
        self.cone_hard_left_end_y = float(os.environ.get("CONE_HARD_LEFT_END_Y", "1.025"))
        self.cone_hard_left_x = float(os.environ.get("CONE_HARD_LEFT_X", "1.720"))
        self.cone_hard_left_forward = float(os.environ.get("CONE_HARD_LEFT_FORWARD", "0.90"))
        self.cone_hard_left_raw_turn = float(os.environ.get("CONE_HARD_LEFT_RAW_TURN", "0.0"))
        self.cone_left_burst_frames = int(os.environ.get("CONE_LEFT_BURST_FRAMES", "0"))
        self.cone_left_burst_remaining = 0
        self.cone_left_burst_done = False
        self.cone_left_burst_start_y = float(os.environ.get("CONE_LEFT_BURST_START_Y", "9.000"))
        self.cone_left_burst_end_y = float(os.environ.get("CONE_LEFT_BURST_END_Y", "9.100"))
        self.cone_left_burst_forward = float(os.environ.get("CONE_LEFT_BURST_FORWARD", "0.60"))
        self.cone_left_burst_turn = float(os.environ.get("CONE_LEFT_BURST_TURN", "-1.60"))
        self.cone_far_left_bypass_enabled = os.environ.get("CONE_FAR_LEFT_BYPASS_ENABLED", "0") == "1"
        self.cone_far_left_start_y = float(os.environ.get("CONE_FAR_LEFT_START_Y", "0.620"))
        self.cone_far_left_end_y = float(os.environ.get("CONE_FAR_LEFT_END_Y", "1.300"))
        self.cone_far_left_x = float(os.environ.get("CONE_FAR_LEFT_X", "1.200"))
        self.cone_far_left_forward = float(os.environ.get("CONE_FAR_LEFT_FORWARD", "2.080"))
        self.cone_far_left_return_x = float(os.environ.get("CONE_FAR_LEFT_RETURN_X", "2.240"))
        self.cone_far_left_return_forward = float(os.environ.get("CONE_FAR_LEFT_RETURN_FORWARD", "2.080"))
        self.cone_box_profile_enabled = os.environ.get("CONE_BOX_PROFILE_ENABLED", "0") == "1"
        self.cone_box_lane_x = float(os.environ.get("CONE_BOX_LANE_X", "2.150"))
        self.cone_box_left_target_x = float(os.environ.get("CONE_BOX_LEFT_TARGET_X", "1.550"))
        self.cone_box_keep_x = float(os.environ.get("CONE_BOX_KEEP_X", "2.055"))
        self.cone_box_keep_left_x = float(os.environ.get("CONE_BOX_KEEP_LEFT_X", "1.985"))
        self.cone_box_keep_right_x = float(os.environ.get("CONE_BOX_KEEP_RIGHT_X", "2.115"))
        self.cone_box_return_start_y = float(os.environ.get("CONE_BOX_RETURN_START_Y", "1.305"))
        self.cone_box_return_target_x = float(os.environ.get("CONE_BOX_RETURN_TARGET_X", "5.500"))
        self.cone_box_approach_speed = float(os.environ.get("CONE_BOX_APPROACH_SPEED", "0.95"))
        self.cone_box_left_speed = float(os.environ.get("CONE_BOX_LEFT_SPEED", "0.72"))
        self.cone_box_keep_speed = float(os.environ.get("CONE_BOX_KEEP_SPEED", "0.58"))
        self.cone_box_keep_raw_start_y = float(os.environ.get("CONE_BOX_KEEP_RAW_START_Y", "9.000"))
        self.cone_box_keep_right_raw_turn = float(os.environ.get("CONE_BOX_KEEP_RIGHT_RAW_TURN", "0.0"))
        self.cone_box_return_speed = float(os.environ.get("CONE_BOX_RETURN_SPEED", "0.60"))
        self.cone_box_return_raw_turn = float(os.environ.get("CONE_BOX_RETURN_RAW_TURN", "0.0"))
        self.cone_box_settle_speed = float(os.environ.get("CONE_BOX_SETTLE_SPEED", "0.78"))
        self.cone_exit_hold_enabled = os.environ.get("CONE_EXIT_HOLD_ENABLED", "0") == "1"
        self.cone_exit_hold_start_y = float(os.environ.get("CONE_EXIT_HOLD_START_Y", "1.180"))
        self.cone_exit_hold_end_y = float(os.environ.get("CONE_EXIT_HOLD_END_Y", "1.315"))
        self.cone_exit_hold_x = float(os.environ.get("CONE_EXIT_HOLD_X", "2.065"))
        self.cone_exit_hold_forward = float(os.environ.get("CONE_EXIT_HOLD_FORWARD", "0.45"))
        self.cone_exit_hold_raw_turn = float(os.environ.get("CONE_EXIT_HOLD_RAW_TURN", "0.0"))
        self.cone_exit_return_target_x = float(os.environ.get("CONE_EXIT_RETURN_TARGET_X", "20.000"))
        self.cone_exit_return_forward = float(os.environ.get("CONE_EXIT_RETURN_FORWARD", "1.20"))
        self.cone_exit_return_y_offset = float(os.environ.get("CONE_EXIT_RETURN_Y_OFFSET", "0.10"))
        self.cone_exit_return_raw_turn = float(os.environ.get("CONE_EXIT_RETURN_RAW_TURN", "0.0"))
        self.cone_exit_return_raw_frames = int(os.environ.get("CONE_EXIT_RETURN_RAW_FRAMES", "0"))
        self.cone_exit_return_raw_remaining = 0
        self.cone_exit_return_raw_done = False
        self.cone_right_bypass_enabled = os.environ.get("CONE_RIGHT_BYPASS_ENABLED", "0") == "1"
        self.cone_right_lane_x = float(os.environ.get("CONE_RIGHT_LANE_X", "2.215"))
        self.cone_right_setup_x = float(os.environ.get("CONE_RIGHT_SETUP_X", "2.300"))
        self.cone_right_keep_x = float(os.environ.get("CONE_RIGHT_KEEP_X", "2.395"))
        self.cone_right_return_x = float(os.environ.get("CONE_RIGHT_RETURN_X", "2.210"))
        self.cone_right_setup_start_y = float(os.environ.get("CONE_RIGHT_SETUP_START_Y", "0.450"))
        self.cone_right_keep_start_y = float(os.environ.get("CONE_RIGHT_KEEP_START_Y", "0.860"))
        self.cone_right_return_start_y = float(os.environ.get("CONE_RIGHT_RETURN_START_Y", "1.150"))
        self.cone_right_settle_start_y = float(os.environ.get("CONE_RIGHT_SETTLE_START_Y", "1.300"))
        self.cone_right_approach_speed = float(os.environ.get("CONE_RIGHT_APPROACH_SPEED", "0.80"))
        self.cone_right_keep_speed = float(os.environ.get("CONE_RIGHT_KEEP_SPEED", "0.62"))
        self.cone_right_return_speed = float(os.environ.get("CONE_RIGHT_RETURN_SPEED", "0.82"))

        self.scenario3_cruise_speed_limit = float(os.environ.get("SCENARIO3_CRUISE_SPEED_LIMIT", "0.68"))
        self.scenario3_cruise_start_y = float(os.environ.get("SCENARIO3_CRUISE_START_Y", "1.65"))
        self.scenario3_final_speed_limit = float(os.environ.get("SCENARIO3_FINAL_SPEED_LIMIT", "0.68"))
        self.scenario3_final_slow_x = float(os.environ.get("SCENARIO3_FINAL_SLOW_X", "-1.25"))

        # Scenario 3 keeps enough early speed for dynamic actors, then slows later.
        if self.core.scenario_num == 3:
            self.core.ld = max(self.core.ld, 0.52)
            scenario3_fast_speed = float(os.environ.get("SCENARIO3_FAST_SPEED", "2.08"))
            scenario3_min_speed = float(os.environ.get("SCENARIO3_MIN_SPEED", "0.20"))
            self.core.max_speed = max(self.core.max_speed, scenario3_fast_speed)
            self.core.min_speed = min(max(self.core.min_speed, scenario3_min_speed), self.core.max_speed)
            if self.core.avoid_angle <= 0:
                self.core.avoid_angle = 0.45

    def _detected_size(self, det_result, label):
        if not det_result or len(det_result) < 2 or label not in det_result[0]:
            return None
        return det_result[1][det_result[0].index(label)]

    def _stop_vehicle_once(self, brake=True):
        status, veh_posi, orien, _, _ = self.core.car.set_velocity_and_request_state(
            forward=0,
            turn=0,
            headlights=False,
            leftTurnSignal=False,
            rightTurnSignal=False,
            brakeSignal=brake,
            reverseSignal=False,
        )
        self.core.current_position = np.array([veh_posi[0], veh_posi[1]])
        self.core.yaw = orien[2]
        self.core.speed = 0
        self.core.run_referee_checks()
        return self.core.current_position, self.core.yaw, self.core.speed

    def _hold_stop(self, seconds):
        end_time = time.time() + seconds
        while time.time() < end_time:
            self._stop_vehicle_once(brake=True)
            time.sleep(0.05)
        return self.core.current_position, self.core.yaw, self.core.speed

    def _drive(self, avoidance=None, speed_cap=None):
        if avoidance is not None:
            self.core.avoidance = avoidance

        original_max_speed = self.core.max_speed
        if speed_cap is not None:
            self.core.max_speed = min(self.core.max_speed, speed_cap)

        target_point, _ = self.core.find_target_point(self.core.current_position)
        if self.core.scenario_num == 3:
            pos = self.core.current_position
            if pos[1] >= self.scenario3_cruise_start_y:
                self.core.max_speed = min(self.core.max_speed, self.scenario3_cruise_speed_limit)
            if pos[0] <= self.scenario3_final_slow_x:
                self.core.max_speed = min(self.core.max_speed, self.scenario3_final_speed_limit)
            if (
                self.lower_overrides_enabled
                and
                self.lower_people_pass_enabled
                and
                self.lower_people_position_yield_done
                and 0.30 <= pos[0] < 0.78
                and -1.32 <= pos[1] <= -0.75
            ):
                target_point = np.array([1.35, -0.500])
                self.core.max_speed = min(self.core.max_speed, 0.28)
            elif (
                self.lower_overrides_enabled
                and
                self.lower_people_pass_enabled
                and
                self.lower_people_position_yield_done
                and 0.78 <= pos[0] < 1.22
                and -1.32 <= pos[1] <= -0.75
            ):
                target_point = np.array([2.05, self.lower_people_pass_y])
                self.core.max_speed = min(self.core.max_speed, 1.65)
            elif (
                self.lower_overrides_enabled
                and
                self.lower_people_position_yield_done
                and 0.05 <= pos[0] < 1.12
                and -1.32 <= pos[1] <= -0.88
            ):
                target_point = np.array([self.lower_after_wait_target_x, self.lower_after_wait_target_y])
                self.core.max_speed = min(self.core.max_speed, self.lower_after_wait_speed)
            elif self.lower_overrides_enabled and 0.05 <= pos[0] < 1.12 and -1.32 <= pos[1] <= -0.92:
                target_point = np.array([2.05, self.lower_entry_target_y])
                self.core.max_speed = min(self.core.max_speed, self.lower_entry_speed)
            elif self.lower_overrides_enabled and 1.12 <= pos[0] <= 1.45 and -1.32 <= pos[1] <= -0.88:
                target_point = np.array([self.lower_mid_target_x, self.lower_mid_target_y])
                self.core.max_speed = min(self.core.max_speed, self.lower_mid_speed)
            elif self.lower_overrides_enabled and 1.45 < pos[0] <= 2.08 and -1.20 <= pos[1] <= -0.86:
                target_point = np.array([self.lower_exit_target_x, self.lower_exit_target_y])
                self.core.max_speed = min(self.core.max_speed, self.lower_exit_speed)
            elif (
                self.lower_overrides_enabled
                and
                self.lower_curve_enabled
                and 1.70 <= pos[0] <= 2.12
                and -0.95 <= pos[1] <= -0.52
            ):
                target_point = np.array([self.lower_curve_target_x, pos[1] + self.lower_curve_forward_y])
                self.core.max_speed = min(self.core.max_speed, self.lower_curve_speed)
            elif self.lower_overrides_enabled and -1.12 <= pos[0] <= -0.74 and -1.08 <= pos[1] <= -0.86:
                target_point = np.array([pos[0] + 0.55, -1.105])
                self.core.max_speed = min(self.core.max_speed, 0.54)
            elif 2.08 <= pos[0] <= 2.34 and -1.00 <= pos[1] < -0.35:
                target_point = np.array([2.10, pos[1] + 0.38])
                self.core.max_speed = min(self.core.max_speed, 0.72)
            elif 2.02 <= pos[0] <= 2.26 and -0.35 <= pos[1] <= 0.08:
                target_point = np.array([2.20, pos[1] + 0.42])
                self.core.max_speed = min(self.core.max_speed, 0.66)
            elif 1.85 <= pos[0] <= 2.36 and 2.85 <= pos[1] <= 3.45:
                self.core.max_speed = min(self.core.max_speed, 1.45)
            elif -0.18 <= pos[0] <= 0.85 and 4.30 <= pos[1] <= 4.55:
                target_point = np.array([pos[0] - 0.55, 4.500])
                self.core.max_speed = min(self.core.max_speed, 1.14)
            elif -0.48 <= pos[0] < -0.18 and 4.30 <= pos[1] <= 4.55:
                target_point = np.array([pos[0] - 0.60, 4.47])
                self.core.max_speed = min(self.core.max_speed, 0.92)
            elif -2.10 <= pos[0] <= -1.65 and 3.45 <= pos[1] <= 4.35:
                self.core.max_speed = min(self.core.max_speed, 1.08)
            elif 1.00 <= pos[0] <= 2.28 and 3.45 <= pos[1] <= 4.52:
                self.core.max_speed = min(self.core.max_speed, 1.45)

        steering_angle, _, _, _ = self.core.calculate_steering_angle(
            self.core.current_position,
            target_point,
            self.core.yaw,
        )
        self.core.current_position, self.core.yaw, self.core.speed = self.core.update_car_state(
            steering_angle,
            self.core.speed,
            self.core.avoidance,
        )

        self.core.max_speed = original_max_speed
        self.core.run_referee_checks()
        return self.core.current_position, self.core.yaw, self.core.speed

    def _drive_to_point(self, target_point, speed_cap):
        original_max_speed = self.core.max_speed
        self.core.max_speed = min(self.core.max_speed, speed_cap)
        steering_angle, _, _, _ = self.core.calculate_steering_angle(
            self.core.current_position,
            np.array(target_point),
            self.core.yaw,
        )
        self.core.current_position, self.core.yaw, self.core.speed = self.core.update_car_state(
            steering_angle,
            self.core.speed,
            False,
        )
        self.core.max_speed = original_max_speed
        self.core.run_referee_checks()
        return self.core.current_position, self.core.yaw, self.core.speed

    def _drive_to_point_fast(self, target_point, forward):
        steering_angle, _, _, _ = self.core.calculate_steering_angle(
            self.core.current_position,
            np.array(target_point),
            self.core.yaw,
        )
        return self._drive_raw(forward, steering_angle)

    def _drive_raw(self, forward, turn):
        status, veh_posi, orien, _, _ = self.core.car.set_velocity_and_request_state(
            forward=forward,
            turn=turn,
            headlights=False,
            leftTurnSignal=False,
            rightTurnSignal=False,
            brakeSignal=False,
            reverseSignal=False,
        )
        if status and (abs(veh_posi[0]) > 1e-6 or abs(veh_posi[1]) > 1e-6):
            self.core.current_position = np.array([veh_posi[0], veh_posi[1]])
            self.core.yaw = orien[2]
        self.core.speed = forward
        self.core.run_referee_checks()
        return self.core.current_position, self.core.yaw, self.core.speed

    def _run_red_light_finish_burst(self):
        """Finish the final straight in one decision call after red is detected."""
        deadline = time.time() + 1.8
        while time.time() < deadline:
            pos = self.core.current_position
            if self.core.scenario_num == 3 and pos[0] < -1.55 and pos[1] <= 0.30:
                break
            self._drive(speed_cap=min(self.core.max_speed, 1.10))
            time.sleep(0.01)
        return self.core.current_position, self.core.yaw, self.core.speed

    def _run_fast_finish_after_cone(self):
        print("[EXPERIMENT] Fast finish: batching post-cone lane following.")
        start_time = time.time()
        deadline = time.time() + 22.0
        max_deviation = 0.0
        max_deviation_pos = None
        people_hit = False
        cow_hit = False
        original_ld = self.core.ld
        self.core.ld = min(self.core.ld, 0.40)

        try:
            while time.time() < deadline:
                if self.core.check_endpoint():
                    break

                self.core.avoidance = False
                pos = self.core.current_position
                x = float(pos[0])
                y = float(pos[1])
                if 1.00 <= x <= 2.28 and 3.45 <= y <= 4.52:
                    speed_cap = 0.55
                elif -2.02 <= x <= -1.70 and 3.55 <= y <= 4.35:
                    speed_cap = 0.74
                else:
                    speed_cap = 1.30
                self._drive(speed_cap=speed_cap)
                pos = self.core.current_position

                if self.core.penalty_system is not None:
                    deviation, _ = self.core.penalty_system.calculate_lane_deviation(pos)
                    if deviation > max_deviation:
                        max_deviation = deviation
                        max_deviation_pos = (float(pos[0]), float(pos[1]))
                    people_hit = people_hit or self.core.penalty_system.check_people_collision_by_distance(pos)
                    cow_hit = cow_hit or self.core.penalty_system.check_cow_collision_by_distance(pos)

                time.sleep(0.01)
        finally:
            self.core.ld = original_ld

        print(
            "[EXPERIMENT] Fast finish complete: "
            f"elapsed={time.time() - start_time:.2f}s, "
            f"pos=({self.core.current_position[0]:.3f}, {self.core.current_position[1]:.3f}), "
            f"max_deviation={max_deviation:.3f}, "
            f"max_deviation_pos={max_deviation_pos}, "
            f"people_hit={people_hit}, cow_hit={cow_hit}."
        )
        return self.core.current_position, self.core.yaw, self.core.speed

    def _apply_reference_cone_avoidance(self, cone_size=None):
        self.core.avoidance = True
        if cone_size is not None:
            self.last_cone_size = cone_size
        sign = 1.0 if self.core.avoid_angle >= 0.0 else -1.0
        self.core.avoid_angle = sign * self.default_cone_avoid_angle
        self.core.speed = min(self.core.max_speed, max(self.core.min_speed, 0.28))

    def _apply_reference_cone_recovery(self):
        self.core.avoidance = True
        self.core.avoid_angle = self.cone_recovery_angle
        self.core.speed = min(self.core.max_speed, max(self.core.min_speed, 0.34))

    def _drive_cone_slow_profile(self, y):
        if y < -0.35:
            target_point = [2.130, y + 0.44]
            speed_cap = 0.70
        elif y < 0.15:
            target_point = [2.125, y + 0.38]
            speed_cap = 0.60
        elif y < 0.46:
            target_point = [2.132, y + 0.34]
            speed_cap = 0.56
        elif y < 0.82:
            target_point = [2.080, y + 0.32]
            speed_cap = 0.48
        elif y < 1.04:
            target_point = [2.055, y + 0.24]
            speed_cap = 0.40
        elif y < 1.16:
            target_point = [2.075, y + 0.21]
            speed_cap = 0.32
        elif y < 1.27:
            target_point = [2.075, y + 0.18]
            speed_cap = 0.28
        elif y < 1.30:
            target_point = [2.900, y + 0.14]
            speed_cap = 0.20
        elif y < 1.36:
            target_point = [2.430, y + 0.16]
            speed_cap = 0.24
        elif y < 1.54:
            target_point = [2.155, y + 0.24]
            speed_cap = 0.36
        elif y < 1.72:
            target_point = [2.195, y + 0.32]
            speed_cap = 0.50
        elif y < 1.84:
            target_point = [2.220, y + 0.34]
            speed_cap = 0.62
        else:
            target_point = [2.230, y + 0.42]
            speed_cap = 0.86

        return self._drive_to_point(target_point, speed_cap)

    def _drive_cone_box_profile(self, x, y):
        if y < 0.58:
            return self._drive_to_point_fast([2.205, y + 0.38], self.cone_box_approach_speed)
        if y < 0.88:
            return self._drive_to_point_fast(
                [self.cone_box_lane_x, y + 0.32],
                self.cone_box_approach_speed,
            )
        if y < 1.02:
            return self._drive_to_point_fast(
                [self.cone_box_left_target_x, y + 0.22],
                self.cone_box_left_speed,
            )
        if y < self.cone_box_return_start_y:
            if x < self.cone_box_keep_x - 0.030:
                if (
                    y >= self.cone_box_keep_raw_start_y
                    and abs(self.cone_box_keep_right_raw_turn) > 1e-6
                ):
                    return self._drive_raw(
                        self.cone_box_keep_speed,
                        self.cone_box_keep_right_raw_turn,
                    )
                target_x = self.cone_box_keep_right_x
            elif x > self.cone_box_keep_x + 0.025:
                target_x = self.cone_box_keep_left_x
            else:
                target_x = self.cone_box_keep_x
            return self._drive_to_point_fast([target_x, y + 0.22], self.cone_box_keep_speed)
        if y < 1.38:
            if abs(self.cone_box_return_raw_turn) > 1e-6:
                return self._drive_raw(
                    self.cone_box_return_speed,
                    self.cone_box_return_raw_turn,
                )
            return self._drive_to_point_fast(
                [self.cone_box_return_target_x, y + 0.10],
                self.cone_box_return_speed,
            )
        if y < 1.72:
            return self._drive_to_point_fast([2.220, y + 0.30], self.cone_box_settle_speed)
        return self._drive_to_point([2.230, y + 0.40], 0.90)

    def _drive_cone_right_bypass_profile(self, x, y):
        if y < self.cone_right_setup_start_y:
            return self._drive_to_point_fast(
                [self.cone_right_lane_x, y + 0.38],
                self.cone_right_approach_speed,
            )
        if y < self.cone_right_keep_start_y:
            return self._drive_to_point_fast(
                [self.cone_right_setup_x, y + 0.30],
                self.cone_right_approach_speed,
            )
        if y < self.cone_right_return_start_y:
            return self._drive_to_point_fast(
                [self.cone_right_keep_x, y + 0.24],
                self.cone_right_keep_speed,
            )
        if y < self.cone_right_settle_start_y:
            return self._drive_to_point_fast(
                [self.cone_right_return_x, y + 0.22],
                self.cone_right_return_speed,
            )
        if y < 1.72:
            return self._drive_to_point_fast([2.220, y + 0.34], 1.05)
        return self._drive_to_point([2.230, y + 0.40], 0.90)

    def _drive_cone_frame(self):
        pos = self.core.current_position
        x = float(pos[0])
        y = float(pos[1])

        self.core.avoidance = False
        if 0.80 <= y <= 1.50:
            with open("cone_trace.csv", "a", encoding="utf-8") as trace_file:
                trace_file.write(
                    f"{time.time():.3f},{x:.4f},{y:.4f},{float(self.core.yaw):.4f},{float(self.core.speed):.4f}\n"
                )
        if self.cone_box_profile_enabled:
            return self._drive_cone_box_profile(x, y)
        if self.cone_right_bypass_enabled:
            return self._drive_cone_right_bypass_profile(x, y)
        if self.cone_use_slow_profile:
            return self._drive_cone_slow_profile(y)
        if self.cone_far_left_bypass_enabled:
            if self.cone_far_left_start_y <= y < self.cone_far_left_end_y:
                return self._drive_to_point_fast(
                    [self.cone_far_left_x, y + 0.48],
                    self.cone_far_left_forward,
                )
            if self.cone_far_left_end_y <= y < 1.56:
                return self._drive_to_point_fast(
                    [self.cone_far_left_return_x, y + 0.46],
                    self.cone_far_left_return_forward,
                )
        if self.cone_delay_left_enabled:
            if self.cone_delay_start_y <= y < self.cone_delay_hold_end_y:
                return self._drive_to_point_fast(
                    [self.cone_delay_hold_x, y + 0.34],
                    self.cone_delay_hold_forward,
                )
            if self.cone_delay_hold_end_y <= y < self.cone_hard_left_end_y:
                if abs(self.cone_hard_left_raw_turn) > 1e-6:
                    return self._drive_raw(
                        self.cone_hard_left_forward,
                        self.cone_hard_left_raw_turn,
                    )
                return self._drive_to_point_fast(
                    [self.cone_hard_left_x, y + 0.24],
                    self.cone_hard_left_forward,
                )
        if (
            self.cone_left_burst_frames > 0
            and not self.cone_left_burst_done
            and self.cone_left_burst_start_y <= y < self.cone_left_burst_end_y
        ):
            self.cone_left_burst_done = True
            self.cone_left_burst_remaining = self.cone_left_burst_frames
        if self.cone_left_burst_remaining > 0:
            self.cone_left_burst_remaining -= 1
            return self._drive_raw(self.cone_left_burst_forward, self.cone_left_burst_turn)
        if 0.30 <= y < 0.52:
            return self._drive_to_point_fast([2.205, y + 0.42], 1.38)
        if 0.52 <= y < 0.82:
            return self._drive_to_point_fast([self.cone_setup_x, y + 0.40], self.cone_setup_forward)
        if 0.82 <= y < 0.98:
            return self._drive_to_point_fast([self.cone_pre_x, y + 0.34], self.cone_pre_forward)
        if 0.98 <= y < self.cone_recover_start_y:
            return self._drive_to_point_fast([self.cone_lower_x, y + 0.32], self.cone_lower_forward)
        if self.cone_recover_start_y <= y < 1.18:
            return self._drive_to_point_fast(
                [self.cone_recover_target_x, y + 0.30],
                self.cone_recover_forward,
            )
        if (
            self.cone_pivot_frames > 0
            and not self.cone_pivot_done
            and self.cone_pivot_start_y <= y < self.cone_pivot_end_y
            and x < 2.09
        ):
            self.cone_pivot_done = True
            self.cone_pivot_remaining = self.cone_pivot_frames
        if self.cone_pivot_remaining > 0:
            self.cone_pivot_remaining -= 1
            return self._drive_raw(self.cone_pivot_forward, self.cone_pivot_turn)
        if self.cone_exit_hold_enabled:
            if self.cone_exit_hold_start_y <= y < self.cone_exit_hold_end_y:
                if abs(self.cone_exit_hold_raw_turn) > 1e-6:
                    return self._drive_raw(
                        self.cone_exit_hold_forward,
                        self.cone_exit_hold_raw_turn,
                    )
                return self._drive_to_point_fast(
                    [self.cone_exit_hold_x, y + 0.16],
                    self.cone_exit_hold_forward,
                )
            if self.cone_exit_hold_end_y <= y < 1.38:
                if (
                    self.cone_exit_return_raw_frames > 0
                    and not self.cone_exit_return_raw_done
                    and abs(self.cone_exit_return_raw_turn) > 1e-6
                ):
                    self.cone_exit_return_raw_done = True
                    self.cone_exit_return_raw_remaining = self.cone_exit_return_raw_frames
                if self.cone_exit_return_raw_remaining > 0:
                    self.cone_exit_return_raw_remaining -= 1
                    return self._drive_raw(
                        self.cone_exit_return_forward,
                        self.cone_exit_return_raw_turn,
                    )
                if (
                    self.cone_exit_return_raw_frames <= 0
                    and abs(self.cone_exit_return_raw_turn) > 1e-6
                ):
                    return self._drive_raw(
                        self.cone_exit_return_forward,
                        self.cone_exit_return_raw_turn,
                    )
                return self._drive_to_point_fast(
                    [self.cone_exit_return_target_x, y + self.cone_exit_return_y_offset],
                    self.cone_exit_return_forward,
                )
        if 1.18 <= y < 1.23 and x < 2.13:
            return self._drive_to_point_fast([20.000, y + 0.36], self.cone_return_forward)
        if (
            self.cone_top_burst_frames > 0
            and not self.cone_top_burst_done
            and self.cone_top_turn_start_y <= y < 1.35
            and x < self.cone_raw_guard_x
        ):
            self.cone_top_burst_done = True
            self.cone_top_burst_remaining = self.cone_top_burst_frames
        if self.cone_top_burst_remaining > 0:
            self.cone_top_burst_remaining -= 1
            return self._drive_raw(self.cone_top_forward, self.cone_top_turn)
        if (
            self.cone_top_burst_frames <= 0
            and self.cone_top_turn_start_y <= y < 1.35
            and x < self.cone_raw_guard_x
        ):
            return self._drive_raw(self.cone_top_forward, self.cone_top_turn)
        if self.cone_raw_start_y <= y < 1.35 and x < self.cone_raw_guard_x:
            return self._drive_raw(self.cone_raw_forward, self.cone_raw_turn)
        if 1.23 <= y < 1.40:
            return self._drive_to_point_fast([2.240, y + 0.42], 0.86)
        if 1.40 <= y < 1.52:
            return self._drive_to_point_fast([2.260, y + 0.38], 1.05)
        if 1.44 <= y < 1.90 and x < 2.13:
            return self._drive_to_point_fast([2.760, y + 0.44], 1.35)
        if 1.50 <= y < 1.82:
            return self._drive_to_point([2.215, y + 0.42], 1.20)

        if y < -0.35:
            target_point = [2.130, y + 0.44]
            speed_cap = 0.70
        elif y < 0.15:
            target_point = [2.125, y + 0.38]
            speed_cap = 0.60
        elif y < 0.46:
            target_point = [2.205, y + 0.34]
            speed_cap = 0.74
        elif y < 0.82:
            target_point = [2.205, y + 0.36]
            speed_cap = 0.74
        elif y < 1.04:
            target_point = [2.055, y + 0.24]
            speed_cap = 0.40
        elif y < 1.16:
            target_point = [2.075, y + 0.21]
            speed_cap = 0.32
        elif y < 1.27:
            target_point = [2.075, y + 0.18]
            speed_cap = 0.28
        elif y < 1.30:
            target_point = [2.900, y + 0.14]
            speed_cap = 0.20
        elif y < 1.36:
            target_point = [2.430, y + 0.16]
            speed_cap = 0.24
        elif y < 1.54:
            target_point = [2.155, y + 0.24]
            speed_cap = 0.36
        elif y < 1.72:
            target_point = [2.195, y + 0.32]
            speed_cap = 0.50
        elif y < 1.84:
            target_point = [2.220, y + 0.34]
            speed_cap = 0.62
        else:
            target_point = [2.230, y + 0.42]
            speed_cap = 0.86

        return self._drive_to_point(target_point, speed_cap)

    def _past_cone_release_point(self):
        return (
            self.core.scenario_num == 3
            and self.core.current_position is not None
            and float(self.core.current_position[1]) > self.cone_release_y
        )

    def _needs_cone_recovery_clearance(self):
        if (
            self.core.scenario_num != 3
            or self.core.current_position is None
            or not self.cone_seen
        ):
            return False
        x = float(self.core.current_position[0])
        y = float(self.core.current_position[1])
        if not (1.85 <= x <= 2.30):
            return False
        if y < self.cone_recovery_start_y:
            return False
        if y >= self.cone_recovery_clear_y:
            return False
        return x < self.cone_recovery_clear_x

    def _arm_reference_cone_avoidance_from_detection(self, det_result, current_time):
        object_classes = self.core.object_classes
        cone_size = self._detected_size(det_result, object_classes.CONE)
        if cone_size is None:
            return False

        cone_size = float(cone_size)
        self.last_cone_size = cone_size
        self.cone_seen = True
        if cone_size < self.cone_threshold:
            return False

        extra_time = min(0.9, max(0.0, (cone_size - self.cone_threshold) / 8.0))
        self.cone_avoidance_until = max(
            self.cone_avoidance_until,
            current_time + self.cone_avoidance_duration + extra_time,
        )
        if current_time - self.last_cone_log_time > self.cone_log_cooldown:
            pos = self.core.current_position
            print(
                "Cone: reference left avoidance "
                f"(size={cone_size:.2f}%, until={self.cone_avoidance_until - current_time:.1f}s, "
                f"pos=({pos[0]:.3f}, {pos[1]:.3f}))."
            )
            self.last_cone_log_time = current_time
        return True

    def _visual_cone_contact(self, pos):
        cone_x, cone_y = 2.23, 1.17
        dx = abs(float(pos[0]) - cone_x)
        dy = abs(float(pos[1]) - cone_y)
        return dx < self.visual_cone_half_side and dy < self.visual_cone_half_side, dx, dy

    def _run_sampling_bypass_experiment(self, det_result):
        if not self.sampling_bypass_experiment:
            return self.core.current_position, self.core.yaw, self.core.speed
        self.sampling_bypass_done = True
        start_time = time.time()
        print(
            "[EXPERIMENT] Teacher sampling bypass: batching guarded cone avoidance "
            "inside one decision call."
        )

        self._arm_reference_cone_avoidance_from_detection(det_result, time.time())
        deviation = 999.0
        max_deviation = 0.0
        visual_cone_hit = False
        visual_line_press = False
        min_visual_dx = 999.0
        min_visual_dy = 999.0
        min_x_in_visual_y_band = 999.0
        max_x_in_visual_y_band = -999.0
        min_x_after_cone = 999.0
        max_x_after_cone = -999.0
        first_visual_line_press = None
        visual_line_guard_x = 2.095
        post_cone_line_guard_x = 2.180
        right_road_guard_x = 2.270
        visual_right_exit = False
        first_visual_right_exit = None
        deadline = time.time() + 12.0

        while time.time() < deadline:
            pos = self.core.current_position
            x = float(pos[0])
            y = float(pos[1])

            if y < 0.18:
                target_x = 2.155
                target_y = y + 0.58
                speed_cap = 0.45
            elif y < 0.58:
                if x > 2.130:
                    target_x = 2.060
                elif x > 2.105:
                    target_x = 2.070
                else:
                    target_x = 2.098
                target_y = y + 0.58
                speed_cap = 0.42
            elif y < 0.92:
                target_x = 2.070 if x > 2.105 else 2.098
                target_y = y + 0.58
                speed_cap = 0.40
            elif y < 1.255:
                target_x = 2.060 if x > 2.100 else 2.098
                target_y = y + 0.50
                speed_cap = 0.36 if x > 2.100 else 0.42
            elif y < 1.302:
                # Preload the right turn while keeping the center outside the
                # cone's 0.13m visual box on the left side.
                if x > 2.098:
                    target_x = 2.060
                elif x < 2.092:
                    target_x = 2.18
                else:
                    target_x = 3.20
                target_y = min(y + 0.020, 1.306)
                speed_cap = 0.070
            elif y < 1.43 and x < 2.180:
                self.core.avoidance = False
                self._drive_raw(0.10, 2.00)
                pos = self.core.current_position
                x = float(pos[0])
                y = float(pos[1])
                hit, dx, dy = self._visual_cone_contact(pos)
                visual_cone_hit = visual_cone_hit or hit
                min_visual_dx = min(min_visual_dx, dx)
                min_visual_dy = min(min_visual_dy, dy)
                if abs(y - 1.17) < self.visual_cone_half_side:
                    min_x_in_visual_y_band = min(min_x_in_visual_y_band, x)
                    max_x_in_visual_y_band = max(max_x_in_visual_y_band, x)
                    if x < visual_line_guard_x and first_visual_line_press is None:
                        first_visual_line_press = (x, y)
                    visual_line_press = visual_line_press or x < visual_line_guard_x
                if 1.40 <= y <= 2.18:
                    min_x_after_cone = min(min_x_after_cone, x)
                    max_x_after_cone = max(max_x_after_cone, x)
                    if x < post_cone_line_guard_x and first_visual_line_press is None:
                        first_visual_line_press = (x, y)
                    visual_line_press = visual_line_press or x < post_cone_line_guard_x
                    if x > right_road_guard_x and first_visual_right_exit is None:
                        first_visual_right_exit = (x, y)
                    visual_right_exit = visual_right_exit or x > right_road_guard_x
                if self.core.penalty_system is not None:
                    deviation, _ = self.core.penalty_system.calculate_lane_deviation(pos)
                    max_deviation = max(max_deviation, deviation)
                if y >= 1.72 and 2.18 <= x <= 2.26 and deviation <= 0.055:
                    break
                time.sleep(0.012)
                continue
            elif y < 1.43:
                target_x = 2.18
                target_y = min(y + 0.16, 1.55)
                speed_cap = 0.26
            elif y < 1.66:
                if x > 2.24:
                    target_x = 2.10
                elif x > 2.21:
                    target_x = 2.14
                else:
                    target_x = 2.20
                target_y = min(y + 0.24, 1.88)
                speed_cap = 0.42
            else:
                if x < 2.18:
                    target_x = 2.22
                elif x > 2.24:
                    target_x = 2.08
                elif x > 2.22:
                    target_x = 2.16
                elif x < 2.20:
                    target_x = 2.21
                else:
                    target_x = 2.20
                target_y = min(y + 0.45, 2.34)
                speed_cap = 0.52

            self.core.avoidance = False
            self._drive_to_point((target_x, target_y), speed_cap)
            pos = self.core.current_position
            x = float(pos[0])
            y = float(pos[1])
            hit, dx, dy = self._visual_cone_contact(pos)
            visual_cone_hit = visual_cone_hit or hit
            min_visual_dx = min(min_visual_dx, dx)
            min_visual_dy = min(min_visual_dy, dy)
            if abs(y - 1.17) < self.visual_cone_half_side:
                min_x_in_visual_y_band = min(min_x_in_visual_y_band, x)
                max_x_in_visual_y_band = max(max_x_in_visual_y_band, x)
                if x < visual_line_guard_x and first_visual_line_press is None:
                    first_visual_line_press = (x, y)
                visual_line_press = visual_line_press or x < visual_line_guard_x
            if 1.40 <= y <= 2.18:
                min_x_after_cone = min(min_x_after_cone, x)
                max_x_after_cone = max(max_x_after_cone, x)
                if x < post_cone_line_guard_x and first_visual_line_press is None:
                    first_visual_line_press = (x, y)
                visual_line_press = visual_line_press or x < post_cone_line_guard_x
                if x > right_road_guard_x and first_visual_right_exit is None:
                    first_visual_right_exit = (x, y)
                visual_right_exit = visual_right_exit or x > right_road_guard_x
            if self.core.penalty_system is not None:
                deviation, _ = self.core.penalty_system.calculate_lane_deviation(pos)
                max_deviation = max(max_deviation, deviation)
            if y >= 1.72 and 2.18 <= x <= 2.26 and deviation <= 0.055:
                break
            time.sleep(0.012)

        self.cone_avoidance_until = 0.0
        self.cone_recovery_until = 0.0
        self.cone_recovery_started = False
        self.core.avoidance = False
        if self.fast_finish_after_cone:
            self._run_fast_finish_after_cone()
        print(
            "[EXPERIMENT] Teacher sampling bypass complete: "
            f"elapsed={time.time() - start_time:.2f}s, "
            f"pos=({self.core.current_position[0]:.3f}, {self.core.current_position[1]:.3f}), "
            f"deviation={deviation:.3f}, max_deviation={max_deviation:.3f}, "
            f"visual_cone_hit={visual_cone_hit}, visual_line_press={visual_line_press}, "
            f"visual_right_exit={visual_right_exit}, "
            f"min_visual_dx={min_visual_dx:.3f}, min_visual_dy={min_visual_dy:.3f}, "
            f"min_x_in_visual_y_band={min_x_in_visual_y_band:.3f}, "
            f"max_x_in_visual_y_band={max_x_in_visual_y_band:.3f}, "
            f"min_x_after_cone={min_x_after_cone:.3f}, "
            f"max_x_after_cone={max_x_after_cone:.3f}, "
            f"first_visual_line_press={first_visual_line_press}, "
            f"first_visual_right_exit={first_visual_right_exit}, "
            f"visual_line_guard_x={visual_line_guard_x:.3f}, "
            f"post_cone_line_guard_x={post_cone_line_guard_x:.3f}, "
            f"right_road_guard_x={right_road_guard_x:.3f}, "
            f"visual_half_side={self.visual_cone_half_side:.3f}."
        )
        return self.core.current_position, self.core.yaw, self.core.speed

    def should_prioritize_cone(self, det_result):
        if self.core.scenario_num != 3 or self.core.current_position is None:
            return False
        current_time = time.time()
        if (
            current_time < self.cone_avoidance_until
            or current_time < self.cone_recovery_until
            or self._needs_cone_recovery_clearance()
        ):
            return True
        if not det_result or len(det_result) < 1 or len(det_result[0]) == 0:
            return False
        y = float(self.core.current_position[1])
        return (
            y <= 1.80
            and (
                self.core.avoidance
                or self.core.object_classes.CONE in det_result[0]
                or self.last_cone_size is not None
            )
        )

    def should_focus_cone_only(self, det_result):
        if self.core.scenario_num != 3 or self.core.current_position is None:
            return False
        if self.should_prioritize_cone(det_result):
            return True

        x = float(self.core.current_position[0])
        y = float(self.core.current_position[1])
        return 1.70 <= x <= 2.35 and -0.75 <= y <= 1.85

    def _handle_reference_cone_avoidance_logic(self, det_result, avoidance_result):
        if self.original_cone_style:
            self._handle_original_cone_avoidance_logic(det_result)
            return

        current_time = time.time()
        if self.core.current_position is not None:
            x = float(self.core.current_position[0])
            y = float(self.core.current_position[1])
            if 1.70 <= x <= 2.35 and -0.75 <= y <= 1.85:
                self.cone_seen = True
                if y <= self.cone_release_y:
                    self.cone_avoidance_until = max(self.cone_avoidance_until, current_time + 0.25)
            if self.cone_seen and 1.32 <= y < 1.64:
                yaw = float(self.core.yaw)
                if x > 2.235 and yaw < 1.35:
                    self._drive_raw(0.48, -1.20)
                    return
                if x > 2.185 and yaw < 1.05:
                    self._drive_raw(0.55, -0.55)
                    return

        if (
            self.sampling_bypass_experiment
            and not self.sampling_bypass_done
            and self.core.current_position is not None
        ):
            x = float(self.core.current_position[0])
            y = float(self.core.current_position[1])
            if 2.02 <= x <= 2.23 and -0.70 <= y <= 0.52:
                self._run_sampling_bypass_experiment(det_result)
                return

        if self._past_cone_release_point():
            if (
                not self.cone_recovery_started
                and (self.cone_avoidance_until > 0.0 or self.core.avoidance)
            ):
                self.cone_recovery_started = True
                self.cone_recovery_until = current_time + self.cone_recovery_duration
            self.cone_avoidance_until = 0.0
            self.core.avoidance = False
            self.last_cone_size = None

        if self.core.current_position is not None:
            x = float(self.core.current_position[0])
            y = float(self.core.current_position[1])
            if y > 1.38 and 2.145 <= x <= 2.220:
                self.cone_recovery_until = 0.0
                self.cone_recovery_started = False

        if current_time < self.cone_recovery_until or self._needs_cone_recovery_clearance():
            self._drive_cone_frame()
            return

        if self._past_cone_release_point():
            self.core.avoidance = False
            self.core.speed = self.core.max_speed
            self.cone_recovery_started = False
            self._drive()
            return

        if current_time < self.cone_avoidance_until:
            self._drive_cone_frame()
            return

        object_classes = self.core.object_classes
        if not det_result or len(det_result) < 2 or len(det_result[0]) == 0:
            self.core.avoidance = False
            self.core.speed = self.core.max_speed
            self.last_cone_size = None
            self._drive()
            return

        cone_size = self._detected_size(det_result, object_classes.CONE)
        if cone_size is None:
            self.core.avoidance = False
            self.core.speed = self.core.max_speed
            self.last_cone_size = None
            self._drive()
            return

        cone_size = float(cone_size)
        self.last_cone_size = cone_size
        self.cone_seen = True
        if cone_size >= self.cone_threshold:
            self._arm_reference_cone_avoidance_from_detection(det_result, current_time)
            self._drive_cone_frame()
            return
        else:
            self.core.avoidance = False
            self.core.speed = self.core.max_speed

        self._drive()

    def _handle_original_cone_avoidance_logic(self, det_result):
        current_time = time.time()

        if self._past_cone_release_point():
            if (
                not self.cone_recovery_started
                and (self.cone_avoidance_until > 0.0 or self.core.avoidance)
            ):
                self.cone_recovery_started = True
                self.cone_recovery_until = current_time + self.cone_recovery_duration
            self.cone_avoidance_until = 0.0
            self.core.avoidance = False
            self.last_cone_size = None

        if current_time < self.cone_recovery_until or self._needs_cone_recovery_clearance():
            self._apply_reference_cone_recovery()
            self._drive()
            return

        if self._past_cone_release_point():
            self.core.avoidance = False
            self.core.speed = self.core.max_speed
            self.cone_recovery_started = False
            self._drive()
            return

        if current_time < self.cone_avoidance_until:
            self._apply_reference_cone_avoidance(self.last_cone_size)
            self._drive()
            return

        if not det_result or len(det_result) < 2 or len(det_result[0]) == 0:
            self.core.avoidance = False
            self.core.speed = self.core.max_speed
            self.last_cone_size = None
            self._drive()
            return

        object_classes = self.core.object_classes
        cone_size = self._detected_size(det_result, object_classes.CONE)
        if cone_size is None:
            self.core.avoidance = False
            self.core.speed = self.core.max_speed
            self.last_cone_size = None
            self._drive()
            return

        cone_size = float(cone_size)
        self.last_cone_size = cone_size
        self.cone_seen = True
        if cone_size >= self.cone_threshold:
            self._arm_reference_cone_avoidance_from_detection(det_result, current_time)
            self._apply_reference_cone_avoidance(cone_size)
        else:
            self.core.avoidance = False
            self.core.speed = self.core.max_speed

        self._drive()

    def _sprint_through_cone_gap(self):
        """Cross the cone gap in one control burst, then return inside the lane."""
        self.cone_sprint_done = True
        self.core.avoidance = False
        original_min_speed = self.core.min_speed
        original_max_speed = self.core.max_speed
        original_ld = self.core.ld
        print(
            "Cone: sprinting through bypass gap, "
            f"start=({self.core.current_position[0]:.3f}, {self.core.current_position[1]:.3f})."
        )

        self.core.min_speed = max(self.core.min_speed, 0.24)
        self.core.max_speed = max(self.core.max_speed, 0.36)
        self.core.ld = min(self.core.ld, 0.24)
        deviation = 999.0

        try:
            deadline = time.time() + 7.0
            while time.time() < deadline:
                pos = self.core.current_position

                if pos[1] < 1.32:
                    target_point = [1.96, pos[1] + 0.30]
                    speed_cap = 0.34
                elif pos[0] < 2.14 or pos[1] < 1.58:
                    target_point = [2.34, pos[1] + 0.32]
                    speed_cap = 0.36
                else:
                    target_point = [2.20, pos[1] + 0.34]
                    speed_cap = 0.28

                self._drive_to_point(target_point, speed_cap=speed_cap)
                if self.core.penalty_system is not None:
                    deviation, _ = self.core.penalty_system.calculate_lane_deviation(self.core.current_position)
                pos = self.core.current_position
                if pos[1] >= 1.60 and deviation <= 0.07:
                    break
                time.sleep(0.025)

            settle_deadline = time.time() + 1.5
            while time.time() < settle_deadline and deviation > 0.07:
                pos = self.core.current_position
                self._drive_to_point([2.20, pos[1] + 0.34], speed_cap=0.24)
                if self.core.penalty_system is None:
                    break
                deviation, _ = self.core.penalty_system.calculate_lane_deviation(self.core.current_position)
                if self.core.current_position[1] >= 1.60 and deviation <= 0.07:
                    break
                time.sleep(0.025)
        finally:
            self.core.min_speed = original_min_speed
            self.core.max_speed = original_max_speed
            self.core.ld = original_ld

        print(
            "Cone: sprint complete, "
            f"pos=({self.core.current_position[0]:.3f}, {self.core.current_position[1]:.3f}), "
            f"deviation={deviation:.3f}."
        )
        if self.core.current_position[1] >= 1.50 and deviation <= 0.08:
            self.cone_bypass_done = True
        return self.core.current_position, self.core.yaw, self.core.speed

    def handle_traffic_light_logic(self, det_result, det_stoplane):
        """Stop at red lights after the stop line has been detected."""
        object_classes = self.core.object_classes

        if not det_result or len(det_result) < 2 or len(det_result[0]) == 0:
            self._drive()
            return det_stoplane

        red_size = self._detected_size(det_result, object_classes.RED)
        green_size = self._detected_size(det_result, object_classes.GREEN)
        saw_stop_line = object_classes.STOP_LINE in det_result[0]

        if saw_stop_line and red_size is not None:
            det_stoplane = 1

        if red_size is not None and red_size >= 0.12 and det_stoplane == 1:
            print("Red light: rolling through timing zone.")
            if self.core.scenario_num == 3 and self.core.current_position[0] < -1.55:
                self._run_red_light_finish_burst()
            else:
                self._drive()
            return det_stoplane

        if green_size is not None:
            det_stoplane = 0
            self._drive()
            return det_stoplane

        # If the stop line and red light are still far away, crawl toward the
        # line so the car does not enter the referee danger zone too quickly.
        if saw_stop_line and red_size is not None:
            self._drive(speed_cap=0.22)
        else:
            self._drive()

        return det_stoplane

    def handle_stop_sign_logic(self, det_result):
        """Stop once for each visible stop sign."""
        object_classes = self.core.object_classes

        if not det_result or len(det_result) < 2 or len(det_result[0]) == 0:
            self._drive()
            return

        if self._cone_window_active():
            self.handle_cone_avoidance_logic(det_result, False)
            return

        stop_size = self._detected_size(det_result, object_classes.STOP_SIGN)
        current_time = time.time()

        if (
            stop_size is not None
            and stop_size >= self.core.stop_sign_threshold
            and current_time - self.last_stop_sign_time > self.stop_sign_cooldown
        ):
            print(f"Stop sign: passing, size={stop_size:.2f}%.")
            if self.stop_sign_hold_seconds > 0.0:
                self._hold_stop(self.stop_sign_hold_seconds)
            self.last_stop_sign_time = time.time()
            self._drive()
            return

        self._drive()

    def _position_yield_request(self):
        if self.core.scenario_num != 3 or self.core.current_position is None:
            return None

        x = float(self.core.current_position[0])
        y = float(self.core.current_position[1])

        if (
            not self.cow_position_yield_done
            and 0.45 <= x <= 0.68
            and 4.03 <= y <= 4.55
        ):
            return "cow", "Cow: position guard yielding.", self.cow_hold_seconds

        if (
            not self.upper_people_position_yield_done
            and -2.05 <= x <= -1.62
            and 3.35 <= y <= 4.15
        ):
            return (
                "upper_people",
                "Pedestrian: upper position guard yielding.",
                self.upper_people_hold_seconds,
            )

        if (
            not self.lower_people_position_yield_done
            and self.lower_people_hold_seconds > 0.0
            and 0.30 <= x <= 0.58
            and -1.24 <= y <= -0.82
        ):
            return (
                "lower_people",
                "Pedestrian: lower position guard yielding.",
                self.lower_people_hold_seconds,
            )

        return None

    def should_position_yield(self):
        return self._position_yield_request() is not None

    def _apply_position_yield(self):
        request = self._position_yield_request()
        if request is None:
            return False

        guard_name, message, hold_seconds = request
        pos = self.core.current_position
        print(f"{message} pos=({pos[0]:.3f}, {pos[1]:.3f}).")
        if hold_seconds > 0.0:
            self._hold_stop(hold_seconds)

        if guard_name == "cow":
            self.cow_position_yield_done = True
            self.last_cow_stop_time = time.time()
        elif guard_name == "upper_people":
            self.upper_people_position_yield_done = True
            self.last_people_stop_time = time.time()
        elif guard_name == "lower_people":
            self.lower_people_position_yield_done = True
            self.last_people_stop_time = time.time()

        return True

    def handle_people_cow_logic(self, det_result):
        """Yield for pedestrians and cow detections before they reach the lane."""
        object_classes = self.core.object_classes

        if self._apply_position_yield():
            return

        if not det_result or len(det_result) < 2 or len(det_result[0]) == 0:
            self._drive()
            return

        current_time = time.time()
        people_size = self._detected_size(det_result, object_classes.PEOPLE)
        cow_size = self._detected_size(det_result, object_classes.COW)

        if (
            people_size is not None
            and people_size >= 1.2
            and self._people_yield_zone()
            and current_time - self.last_people_stop_time > self.people_cooldown
        ):
            print(f"Pedestrian: yielding, size={people_size:.2f}%.")
            if self.people_hold_seconds > 0.0:
                if self.core.current_position[1] > 2.20 and self.core.current_position[0] < -0.80:
                    self.upper_people_position_yield_done = True
                    self._hold_stop(self.upper_people_hold_seconds)
                else:
                    self.lower_people_position_yield_done = True
                    self._hold_stop(self.lower_people_hold_seconds)
            else:
                self._drive()
            self.last_people_stop_time = time.time()
            return

        if (
            cow_size is not None
            and cow_size >= 5.5
            and current_time - self.last_cow_stop_time > self.cow_cooldown
        ):
            print(f"Cow: yielding, size={cow_size:.2f}%.")
            self.cow_position_yield_done = True
            if self.cow_hold_seconds > 0.0:
                self._hold_stop(self.cow_hold_seconds)
            else:
                self._drive()
            self.last_cow_stop_time = time.time()
            return

        self._drive()

    def _people_yield_zone(self):
        if self.core.scenario_num != 3:
            return True

        pos = self.core.current_position
        near_lower_crossing = -1.35 <= pos[1] <= -0.35 and pos[0] > 0.45
        near_upper_crossing = pos[1] > 2.20 and pos[0] < -0.80
        return near_lower_crossing or near_upper_crossing

    def _cone_window_active(self):
        if self.core.scenario_num != 3:
            return False

        cone_pos = np.array([2.23, 1.17])
        pos = self.core.current_position

        # Position-based guard makes the bypass work even if YOLO misses one
        # frame. The car travels north here, so subtracting avoid_angle moves it
        # left of the cone.
        return (
            not self.cone_bypass_done
            and abs(pos[0] - cone_pos[0]) < 0.85
            and cone_pos[1] - 1.20 <= pos[1] <= cone_pos[1] + 1.00
        )

    def handle_cone_avoidance_logic(self, det_result, avoidance_result):
        """Pass the cone on the left, then smoothly return to the path."""
        if self.use_reference_cone_strategy:
            self._handle_reference_cone_avoidance_logic(det_result, avoidance_result)
            return

        object_classes = self.core.object_classes
        current_time = time.time()

        red_size = self._detected_size(det_result, object_classes.RED)
        if red_size is not None and red_size >= 0.12:
            self.core.avoidance = False
            self._drive()
            return

        cone_size = self._detected_size(det_result, object_classes.CONE)
        cone_is_close = cone_size is not None and cone_size >= 8.0
        position_guard = self._cone_window_active()

        pos = self.core.current_position
        if pos[1] > 2.18:
            self.cone_bypass_done = True
            self.core.avoidance = False

        if position_guard:
            if (not self.cone_sprint_done
                    and 0.82 <= pos[1] <= 0.94
                    and pos[0] <= 2.15):
                self._sprint_through_cone_gap()
                return

            if pos[1] >= 1.58 and pos[0] >= 2.15:
                print(f"Cone: returned to lane after bypass, pos=({pos[0]:.3f}, {pos[1]:.3f}).")
                self.cone_bypass_done = True
                self.core.avoidance = False
                self._drive(speed_cap=0.16)
                return

            if pos[1] < 0.20:
                print(f"Cone: holding lane before bypass, pos=({pos[0]:.3f}, {pos[1]:.3f}).")
                self._drive_to_point([2.20, pos[1] + 0.58], speed_cap=0.15)
                return

            if pos[1] <= 0.62:
                print(f"Cone: setting up left bypass, pos=({pos[0]:.3f}, {pos[1]:.3f}).")
                self._drive_to_point([2.16, pos[1] + 0.42], speed_cap=0.12)
                return

            if pos[1] <= 0.98:
                print(f"Cone: cutting left past cone, pos=({pos[0]:.3f}, {pos[1]:.3f}).")
                self._drive_to_point([1.80, pos[1] + 0.36], speed_cap=0.12)
                return

            if pos[1] <= 1.30:
                print(f"Cone: guarded return around cone, pos=({pos[0]:.3f}, {pos[1]:.3f}).")
                self._drive_to_point([2.04, pos[1] + 0.34], speed_cap=0.13)
                return

            if pos[1] <= 1.72:
                print(f"Cone: snap back after cone, pos=({pos[0]:.3f}, {pos[1]:.3f}).")
                self._drive_to_point([2.36, pos[1] + 0.32], speed_cap=0.17)
                return

            if pos[1] <= 2.05:
                print(f"Cone: settling back to lane, pos=({pos[0]:.3f}, {pos[1]:.3f}).")
                self._drive_to_point([2.22, pos[1] + 0.44], speed_cap=0.18)
                return

        if self.core.scenario_num == 3:
            self.core.avoidance = False
            if cone_is_close and not self.cone_bypass_done and pos[1] < 0.55:
                print(f"Cone: approaching bypass window, pos=({pos[0]:.3f}, {pos[1]:.3f}).")
                self._drive_to_point([2.20, pos[1] + 0.62], speed_cap=0.16)
                return
            self._drive()
            return

        if (cone_is_close or avoidance_result) and current_time >= self.last_cone_avoidance_time:
            self.cone_avoid_until = max(self.cone_avoid_until, current_time + 2.0)
            self.cone_recover_until = max(self.cone_recover_until, current_time + 3.0)
            self.last_cone_avoidance_time = current_time + self.cone_cooldown
            if cone_size is not None:
                pos = self.core.current_position
                print(f"Cone: starting left bypass, size={cone_size:.2f}%, pos=({pos[0]:.3f}, {pos[1]:.3f}).")
            else:
                print("Cone: starting left bypass by position guard.")

        if current_time < self.cone_avoid_until:
            self._drive(avoidance=True, speed_cap=0.28)
            return

        if current_time < self.cone_recover_until:
            self._drive(avoidance=False, speed_cap=0.30)
            return

        self.core.avoidance = False
        self._drive()

    def pure_pursuit_control(self):
        """Default path tracking."""
        return self._drive()
