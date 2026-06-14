#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动驾驶主程序 - 重构版本
使用分离的核心系统和学生决策逻辑
"""

import time
import signal
import cv2
import argparse
from AutonomousDriveCore import AutonomousDriveCore
from StudentDecision import StudentDecision

def main():
    """
    主程序函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=1, help="场景编号 (1-3)")
    parser.add_argument("--ld", type=float, default=0.50, help="预瞄距离")
    parser.add_argument("--max_speed", type=float, default=0.80, help="最大速度")
    parser.add_argument("--min_speed", type=float, default=0.20, help="最小速度")
    parser.add_argument("--stop_sign_threshold", type=float, default=0.80, help="停止标志阈值")
    parser.add_argument("--avoid_angle", type=float, default=0.0, help="避障角度")
    parser.add_argument("--show_camera", action="store_true", help="显示摄像头窗口")
    parser.add_argument("--stop_after_y", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_runtime", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # 初始化核心系统
    core = AutonomousDriveCore(
        scenario_num=args.scenario,
        ld=args.ld,
        max_speed=args.max_speed,
        min_speed=args.min_speed,
        stop_sign_threshold=args.stop_sign_threshold,
        avoid_angle=args.avoid_angle,
        show_camera=args.show_camera
    )
    
    # 初始化系统
    if args.scenario == 3 and not core.wait_for_qlab_actor_spawner(timeout=3.0):
        print("QLAB actor spawner 未响应：请在不重启 QLAB 的前提下退出当前场景并重新进入 Self-Driving -> Plane。")
        return
    core.initialize_system()
    
    # 初始化学生决策系统
    student_decision = StudentDecision(core)
    
    # 场景相关变量
    det_stoplane = 0  # 停止线检测标志
    
    try:
        # 初始化循环
        for _ in range(5):
            student_decision.pure_pursuit_control()
            time.sleep(0.01)
        
        # 开始计时
        lap_start = time.time()
        last_status_time = lap_start
        loop_count = 0
        
        # 主循环
        while True:
            loop_count += 1
            # Scenario 3 uses position guards for the fast lap. Avoid running
            # YOLO every control frame; it is only preloaded during startup.
            if args.scenario == 3:
                det_result = [[], [], []]
            else:
                det_result = core.detection()
            
            # 获取激光雷达数据（场景3）- 默认关闭以提高性能
            # success, angle_front, dist_front, avoidance_result = core.get_front_lidar()
            success, angle_front, dist_front, avoidance_result = True, [], [], False
            
            # 更新惩罚系统（碰撞检测等）
            core.run_referee_checks()
            
            # 学生决策系统处理检测结果 - 根据场景调用不同方法
            if args.scenario == 1:
                # 场景1：仅红绿灯
                if det_result and len(det_result) > 0 and len(det_result[0]) > 0:
                    det_stoplane = student_decision.handle_traffic_light_logic(det_result, det_stoplane)
                    # 确保在处理完红绿灯逻辑后进行正常巡线
                    if not (core.object_classes.RED in det_result[0] and det_stoplane == 1):
                        # 没有红灯停车时，正常巡线
                        student_decision.pure_pursuit_control()
                else:
                    # 没有检测结果时，正常巡线
                    student_decision.pure_pursuit_control()
                
            elif args.scenario == 2:
                # 场景2：红绿灯 + 停止标志
                if det_result and len(det_result) > 0 and len(det_result[0]) > 0:
                    det_stoplane = student_decision.handle_traffic_light_logic(det_result, det_stoplane)
                    
                    # 检查停止标志
                    if core.object_classes.STOP_SIGN in det_result[0]:
                        student_decision.handle_stop_sign_logic(det_result)
                    else:
                        # 正常巡线
                        student_decision.pure_pursuit_control()
                else:
                    # 没有检测结果时，正常巡线
                    student_decision.pure_pursuit_control()
                    
            elif args.scenario == 3:
                # 场景3：红绿灯 + 停止标志 + 锥桶避障 + 行人/牛检测
                if student_decision.should_position_yield():
                    student_decision.handle_people_cow_logic(det_result)
                elif student_decision.should_focus_cone_only(det_result):
                    student_decision.handle_cone_avoidance_logic(det_result, avoidance_result)
                elif det_result and len(det_result) > 0 and len(det_result[0]) > 0:
                    detected_objects = det_result[0]
                    has_traffic_signal = (
                        core.object_classes.RED in detected_objects
                        or core.object_classes.GREEN in detected_objects
                        or core.object_classes.STOP_LINE in detected_objects
                    )
                    if has_traffic_signal:
                        det_stoplane = student_decision.handle_traffic_light_logic(det_result, det_stoplane)
                    
                    # 检查是否正在红灯停车状态
                    is_red_light_stopping = (core.object_classes.RED in det_result[0] and 
                                            det_result[1][det_result[0].index(core.object_classes.RED)] >= 0.15 and
                                            det_stoplane == 1)
                    
                    if is_red_light_stopping:
                        # 红灯停车时，跳过其他逻辑，避免被覆盖
                        print("🚦 红灯停车中，跳过其他检测逻辑")
                        pass  # 已经在traffic_light_logic中处理了停车
                    # 检查停止标志
                    # 检查行人或牛
                    elif (
                        not student_decision.should_prioritize_cone(det_result)
                        and (
                            student_decision.should_position_yield()
                            or core.object_classes.PEOPLE in detected_objects
                            or core.object_classes.COW in detected_objects
                        )
                    ):
                        student_decision.handle_people_cow_logic(det_result)
                    elif (
                        not student_decision.should_prioritize_cone(det_result)
                        and core.object_classes.STOP_SIGN in detected_objects
                    ):
                        student_decision.handle_stop_sign_logic(det_result)
                    else:
                         # 锥桶避障逻辑：使用学生决策系统处理（纯图像判断）
                        student_decision.handle_cone_avoidance_logic(det_result, avoidance_result)
                else:
                    # 没有检测结果时，使用学生决策系统处理（纯图像判断）
                    if student_decision.should_position_yield():
                        student_decision.handle_people_cow_logic(det_result)
                    else:
                        student_decision.handle_cone_avoidance_logic(det_result, avoidance_result)
            
            # 检查是否到达终点
            core.run_referee_checks()

            now = time.time()
            if now - last_status_time >= 3.0:
                penalty_time = 0.0
                if core.penalty_system is not None:
                    penalty_time = core.penalty_system.total_penalty_time
                with open("run_status.txt", "a", encoding="utf-8") as status_file:
                    status_file.write(
                        f"t={now - lap_start:.2f}, loop={loop_count}, "
                        f"pos=({core.current_position[0]:.3f},{core.current_position[1]:.3f}), "
                        f"yaw={core.yaw:.3f}, speed={core.speed:.3f}, "
                        f"penalty={penalty_time:.1f}\n"
                    )
                last_status_time = now

            if core.check_endpoint():
                # 车辆停止
                core.car.set_velocity_and_request_state(
                    forward=0, turn=0, 
                    headlights=False, leftTurnSignal=False, rightTurnSignal=True, 
                    brakeSignal=False, reverseSignal=False
                )
                
                # 计算最终时间
                lap_time = time.time() - lap_start
                penalty_stats = core.penalty_system.get_penalty_stats()
                total_penalty = penalty_stats['total_penalty_time']
                final_time = lap_time + total_penalty
                
                # 显示最终统计信息
                core.penalty_system.show_final_summary(lap_time)
                print(f"LAP TIME : {final_time:.2f} sec")
                break

            if args.stop_after_y is not None and core.current_position[1] >= args.stop_after_y:
                core.car.set_velocity_and_request_state(
                    forward=0, turn=0,
                    headlights=False, leftTurnSignal=False, rightTurnSignal=True,
                    brakeSignal=False, reverseSignal=False
                )
                penalty_time = 0.0
                if core.penalty_system is not None:
                    penalty_time = core.penalty_system.total_penalty_time
                print(
                    "TUNE STOP: "
                    f"y={core.current_position[1]:.3f}, "
                    f"pos=({core.current_position[0]:.3f},{core.current_position[1]:.3f}), "
                    f"penalty={penalty_time:.1f}"
                )
                break
            
            # 添加信号处理
            if args.max_runtime is not None and now - lap_start >= args.max_runtime:
                core.car.set_velocity_and_request_state(
                    forward=0, turn=0,
                    headlights=False, leftTurnSignal=False, rightTurnSignal=True,
                    brakeSignal=False, reverseSignal=False
                )
                penalty_time = 0.0
                if core.penalty_system is not None:
                    penalty_time = core.penalty_system.total_penalty_time
                print(
                    "TUNE TIMEOUT: "
                    f"elapsed={now - lap_start:.2f}, "
                    f"pos=({core.current_position[0]:.3f},{core.current_position[1]:.3f}), "
                    f"penalty={penalty_time:.1f}"
                )
                break

            signal.signal(signal.SIGINT, core.terminate_handler)
            signal.signal(signal.SIGTERM, core.terminate_handler)
    
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 清理资源
        core.cleanup()

if __name__ == "__main__":
    main() 
