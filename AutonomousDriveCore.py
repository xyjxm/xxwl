import cv2, time, math
import numpy as np
import os
import sys
import runtime_paths

BASE_DIR = runtime_paths.configure()

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.basic_shape import QLabsBasicShape
from qvl.system import QLabsSystem
from qvl.walls import QLabsWalls
from qvl.flooring import QLabsFlooring
from qvl.stop_sign import QLabsStopSign
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_cone import QLabsTrafficCone
from qvl.traffic_light import QLabsTrafficLight
from qvl.generic_sensor import QLabsGenericSensor
from qvl.person import QLabsPerson
from qvl.animal import QLabsAnimal
from ultralytics import YOLO

from Setup_Competition import *
from PenaltySystem import PenaltySystem
import subprocess
import signal
import argparse
import threading
import runpy

# YOLO Model Pretrained Object List Binding to ID
# YOLO模型预训练对象列表与ID绑定
class YoloObject():
    """
    Class defining constants for YOLO object detection labels, their string representations and colors
    定义YOLO对象检测标签、字符串表示和颜色的常量类
    
    Attributes:
        CONE (int): Traffic cone label / 交通锥标签
        COW (int): Cow label / 奶牛标签
        CROSSWALK (int): Crosswalk label / 人行横道标签
        GREEN (int): Green traffic light label / 绿灯标签
        PEOPLE (int): People label / 行人标签
        RED (int): Red traffic light label / 红灯标签
        STOP_LINE (int): Stop line label / 停止线标签
        STOP_SIGN (int): Stop sign label / 停止标志标签
        LABELS (list): String representations of the labels
        COLORS (list): RGB color tuples for visualization
    """
    CONE = 0        # Traffic cone / 交通锥
    COW = 1         # Cow / 奶牛
    CROSSWALK = 2   # Crosswalk / 人行横道
    GREEN = 3       # Green traffic light / 绿灯
    PEOPLE = 4      # People / 行人
    RED = 5         # Red traffic light / 红灯
    STOP_LINE = 6   # Stop line / 停止线
    STOP_SIGN = 7   # Stop sign / 停止标志

    # String labels for each object
    LABELS = ['Cone', 'Cow', 'Crosswalk', 'GREEN', 'People', 'RED', 'Stop Line', 'Stop Sign']
    
    # RGB colors for visualization (B,G,R format)
    COLORS = [(50,50,255), (0,204,0), (194,153,255), (255,204,51), 
              (255,102,204), (0,153,255), (255,0,0), (255,255,0)]

class AutonomousDriveCore:
    """
    自动驾驶核心系统
    包含YOLO检测、Pure Pursuit控制、激光雷达处理等功能
    """
    
    def __init__(self, scenario_num=1, ld=0.50, max_speed=0.80, min_speed=0.20, 
                 stop_sign_threshold=0.80, avoid_angle=0.0, show_camera=False):
        """
        初始化自动驾驶核心系统
        
        Args:
            scenario_num: 场景编号 (1-3)
            ld: 预瞄距离
            max_speed: 最大速度
            min_speed: 最小速度
            stop_sign_threshold: 停止标志阈值
            avoid_angle: 避障角度
            show_camera: 是否显示摄像头窗口
        """
        # 基本参数
        self.scenario_num = scenario_num
        self.ld = ld
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.stop_sign_threshold = stop_sign_threshold
        self.avoid_angle = avoid_angle
        self.show_camera = show_camera
        self.fast_lane_following = (scenario_num == 3)
        
        # 系统组件
        self.qlabs = None
        self.car = None
        self.model = None
        self.process = None
        self.traffic_thread = None
        self.traffic_stop_requested = False
        self.traffic_qlabs = None
        self.traffic_stdout = None
        self.traffic_stderr = None
        self.penalty_system = None
        self.people_actors = {}
        self.cow_actors = {}
        self.last_dynamic_referee_update = 0.0
        self.dynamic_referee_update_interval = 0.20
        
        # 路径和控制参数
        self.path_points = None
        self.L = 0.27  # 轴距
        self.current_position = np.array([0.0, 0.0])
        self.yaw = 0.0
        self.speed = 0.8
        
        # 检测结果
        self.last_detection_result = None
        self.object_classes = YoloObject()
        
        # 状态变量
        self.avoidance = False
        self.avoidance_prev = []
        self.det_stoplane = 0
        self.det_trafficcone_time = 0
        self.referee_monitoring_active = False
        self.action_adapter = None
        self.single_tick_mode = False
        self.last_action_record = {
            "base_action": {"forward": 0.0, "turn": 0.0},
            "residual_action": {"speed_delta": 0.0, "steering_bias": 0.0},
            "final_action": {"forward": 0.0, "turn": 0.0},
            "segment_id": 0,
            "segment_name": "uninitialized",
            "residual_enabled": False,
        }
        
        # 场景控制开关
        self.ENABLE_CONE_AVOIDANCE = (scenario_num == 3)
        self.ENABLE_PEOPLE_COW = (scenario_num == 3)
        
        # 终点位置
        self.endpoint = np.array((-1.9501, 0.20551))
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self.terminate_handler)
        signal.signal(signal.SIGTERM, self.terminate_handler)
    
    def terminate_handler(self, signum, frame):
        """
        信号处理函数
        """
        print(f"收到信号 {signum}，正在清理资源...")
        self.cleanup()
        exit(0)
    
    def cleanup(self):
        """
        清理系统资源
        """
        try:
            print("🧹 正在清理资源...")
            
            # 关闭摄像头窗口
            cv2.destroyAllWindows()
            
            # 停止车辆
            if self.car:
                self.car.set_velocity_and_request_state(
                    forward=0, turn=0, 
                    headlights=False, leftTurnSignal=False, rightTurnSignal=False, 
                    brakeSignal=False, reverseSignal=False
                )
            
            # 终止外部进程
            self.stop_traffic_control()

            self.traffic_stop_requested = True
            if self.traffic_thread and self.traffic_thread.is_alive():
                self.traffic_thread.join(timeout=1.0)
            if self.traffic_qlabs:
                self.traffic_qlabs.close()
                self.traffic_qlabs = None

            if self.traffic_stdout:
                self.traffic_stdout.close()
                self.traffic_stdout = None
            if self.traffic_stderr:
                self.traffic_stderr.close()
                self.traffic_stderr = None
            
            # 清理QLabs资源
            if self.qlabs:
                self.qlabs.destroy_all_spawned_actors()
                self.qlabs.close()
            
            print("✅ 资源清理完成")
            
        except Exception as e:
            print(f"⚠️ 资源清理时出现错误: {e}")

    def wait_for_traffic_ready(self, timeout=20.0):
        """
        Wait until the traffic subprocess confirms scenario actors are loaded.
        """
        if self.scenario_num != 3:
            return

        ready_file = os.path.join(BASE_DIR, "traffic_ready.flag")
        status_file = os.path.join(BASE_DIR, "traffic_status.txt")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if os.path.exists(ready_file):
                with open(ready_file, "r", encoding="utf-8") as file:
                    ready_line = file.read().strip()
                print(f"Traffic actors confirmed before run: {ready_line}")
                return

            if self.process and self.process.poll() is not None:
                details = ""
                if os.path.exists(status_file):
                    with open(status_file, "r", encoding="utf-8") as file:
                        details = file.read()[-1200:]
                raise RuntimeError(f"Traffic process exited before actors were ready.\n{details}")

            time.sleep(0.2)

        details = ""
        if os.path.exists(status_file):
            with open(status_file, "r", encoding="utf-8") as file:
                details = file.read()[-1200:]
        raise TimeoutError(f"Timed out waiting for people/cow to load.\n{details}")

    def start_traffic_control(self):
        """
        Start traffic/pedestrian/cow control in its original separate process.
        """
        ready_file = os.path.join(BASE_DIR, "traffic_ready.flag")
        status_file = os.path.join(BASE_DIR, "traffic_status.txt")
        for path in (ready_file, status_file):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        self.traffic_stop_requested = False
        self.traffic_thread = None
        self.traffic_qlabs = None
        self.traffic_stdout = open(
            os.path.join(BASE_DIR, "traffic_process.log"),
            "w",
            encoding="utf-8",
            errors="replace",
        )
        self.traffic_stderr = open(
            os.path.join(BASE_DIR, "traffic_process.err.log"),
            "w",
            encoding="utf-8",
            errors="replace",
        )
        self.process = subprocess.Popen(
            [sys.executable, "Traffic_Lights_Competition.py", "--scenario", str(self.scenario_num)],
            cwd=BASE_DIR,
            stdout=self.traffic_stdout,
            stderr=self.traffic_stderr,
            text=True,
        )

    def stop_traffic_control(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)
        self.process = None

        if self.traffic_stdout:
            self.traffic_stdout.close()
            self.traffic_stdout = None
        if self.traffic_stderr:
            self.traffic_stderr.close()
            self.traffic_stderr = None

    def reset_qlab_plane_scene(self):
        guide_path = os.path.join(BASE_DIR, "qlab_reenter_plane.md")
        raise RuntimeError(
            "Traffic actors did not load. Do not restart QLAB or use the old "
            f"reset script; follow the manual Plane re-entry flow in {guide_path}."
        )

    def qlab_actor_spawner_healthy(self):
        qlabs = None
        try:
            qlabs = QuanserInteractiveLabs()
            if not qlabs.open("localhost"):
                return False
            return qlabs.destroy_all_spawned_actors() != -1
        except Exception:
            return False
        finally:
            if qlabs:
                qlabs.close()

    def wait_for_qlab_actor_spawner(self, timeout=12.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.qlab_actor_spawner_healthy():
                return True
            time.sleep(1.0)
        return False

    def setup_competition_scene(self):
        if not self.wait_for_qlab_actor_spawner():
            if os.environ.get("QLAB_ALLOW_DIRECT_SETUP_ON_HEALTH_FAIL", "0") != "1":
                raise RuntimeError("QLAB actor spawner is not responding; cannot rebuild Plane actors without a working QLAB scene.")
            print(
                "QLAB actor spawner health check failed; attempting direct setup. "
                "The caller must verify QCar movement before accepting this episode."
            )
        self.qlabs = None
        self.car = setup(scenario_num=self.scenario_num)
        self.qlabs = getattr(self.car, "_qlabs", None)
        time.sleep(3.5)

    def start_traffic_with_recovery(self, max_attempts=3):
        last_error = None
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self.stop_traffic_control()
                self.reset_qlab_plane_scene()

            self.setup_competition_scene()
            self.start_traffic_control()
            try:
                self.wait_for_traffic_ready(timeout=25.0)
                return
            except Exception as exc:
                last_error = exc
                print(f"Traffic actor load attempt {attempt}/{max_attempts} failed: {exc}")
                self.stop_traffic_control()

        raise RuntimeError(f"Traffic actors failed to load after {max_attempts} attempts: {last_error}")

    def initialize_system(self):
        """
        初始化系统组件
        """
        # setup() owns the QLabs connection used by the car. If the traffic
        # actors fail to spawn, re-enter Plane and rebuild the scene.
        self.start_traffic_with_recovery()

        # Load YOLO Model
        self.model = YOLO("qlabspretrained.pt")

        # Load Path Points
        with open('path.txt', 'r') as file:
            lines = file.readlines()
            self.path_points = [list(map(float, line.strip().split())) for line in lines]
        self.path_points = np.array(self.path_points)
        if self.scenario_num == 3 and os.environ.get("SCENARIO3_CONTROL_PATH_BIAS", "0") == "1":
            self.path_points = self._adjust_scenario3_path_points(self.path_points)

        # Initialize Penalty System
        self.penalty_system = PenaltySystem(path_points_file='path.txt', 
                                          scenario_num=self.scenario_num, 
                                          use_file_output=True)
        print("✅ 罚时系统已启动，使用文件输出模式")

        # Initialize car position
        x_offset = 0.13
        y_offset = 1.67
        initial_location = [-1.335 + x_offset, -2.5 + y_offset, 0.005]
        initial_yaw = -45 * np.pi / 180
        self.car.set_transform_and_request_state_degrees(
            location=initial_location,
            rotation=[0, 0, -45], 
            enableDynamics=True, 
            headlights=False, 
            leftTurnSignal=False, 
            rightTurnSignal=False, 
            brakeSignal=False, 
            reverseSignal=False, 
            waitForConfirmation=True
        )
        self.current_position = np.array(initial_location[:2])
        self.yaw = initial_yaw

        # Initialize referee system with god view
        self.initialize_referee_system()

        # Initialize detection
        _ = self.detection()
        time.sleep(1)  # Wait for YOLO preloading

        # Get initial state
        self.current_position, self.yaw, self.speed = self.update_car_state(0, self.speed, False)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.terminate_handler)
        signal.signal(signal.SIGTERM, self.terminate_handler)

    def initialize_referee_system(self):
        """
        初始化裁判系统
        """
        # 裁判系统使用PenaltySystem的功能，不需要额外的QLabs API
        self.add_penalty_message("🎯 裁判系统已初始化，开始监控违规行为")
        
        # 初始化上帝视角需要的变量
        self.last_detection_result = None
        self.attach_referee_actor_handles()
        self.referee_monitoring_active = True

    def _adjust_scenario3_path_points(self, path_points):
        """Bias only the control path near the cone; referee still uses path.txt."""
        adjusted = np.array(path_points, dtype=float, copy=True)
        cone_bias = float(os.environ.get("CONE_PATH_BIAS", "0.24"))
        cone_center_y = float(os.environ.get("CONE_PATH_BIAS_CENTER_Y", "1.12"))
        lower_width = float(os.environ.get("CONE_PATH_BIAS_LOWER_WIDTH", "0.40"))
        upper_width = float(os.environ.get("CONE_PATH_BIAS_UPPER_WIDTH", "0.16"))

        for point in adjusted:
            x, y = point[0], point[1]
            if 1.82 <= x <= 2.30 and -0.45 <= y <= 1.70:
                width = lower_width if y <= cone_center_y else upper_width
                cone_weight = math.exp(-((y - cone_center_y) / width) ** 2)
                point[0] = x - cone_bias * cone_weight
            if 4.12 <= y <= 4.46 and -1.00 <= x <= 1.95:
                if x > 1.55:
                    weight = (1.95 - x) / 0.40
                elif x < -0.80:
                    weight = (x + 1.00) / 0.20
                else:
                    weight = 1.0
                point[1] = min(4.46, y + 0.02 * max(0.0, min(1.0, weight)))
        return adjusted

    def attach_referee_actor_handles(self):
        self.people_actors = {}
        self.cow_actors = {}
        self.last_dynamic_referee_update = 0.0
        if self.scenario_num != 3 or self.qlabs is None:
            return

        for actor_number in (0, 1):
            actor = QLabsPerson(self.qlabs)
            actor.actorNumber = actor_number
            self.people_actors[actor_number] = actor

        cow = QLabsAnimal(self.qlabs)
        cow.actorNumber = 0
        self.cow_actors[0] = cow

    def should_update_dynamic_referee_positions(self):
        if self.scenario_num != 3 or self.current_position is None:
            return False

        x = float(self.current_position[0])
        y = float(self.current_position[1])
        near_lower_people = 0.05 <= x <= 1.65 and -1.45 <= y <= -0.35
        near_cow = -0.75 <= x <= 1.35 and 3.55 <= y <= 4.75
        near_upper_people = -2.35 <= x <= -1.00 and 2.75 <= y <= 4.55
        if not (near_lower_people or near_cow or near_upper_people):
            return False

        now = time.time()
        if now - self.last_dynamic_referee_update < self.dynamic_referee_update_interval:
            return False
        self.last_dynamic_referee_update = now
        return True

    def detection(self):
        """
        Detect objects using YOLO model and visualize the results
        使用文件夹中提供的 qlabspretrained.pt YOLO模型检测物体并可视化结果

        Returns:
            list: Detection results containing:
                 检测结果包含：
                 [0] - List of detected object IDs / 检测到的物体ID列表
                 [1] - List of object sizes (percentage of image) / 物体大小列表（占图像百分比）
                 [2] - List of bounding box coordinates / 边界框坐标列表
        """
        # Get image from car camera
        # 从车载摄像头获取图像
        try:
            success, raw_img = self.car.get_image(4)
            if not success or raw_img is None:
                print("⚠️ 摄像头获取图像失败，返回空检测结果")
                return [[], [], []]
            img = raw_img.copy()
        except Exception as e:
            print(f"⚠️ 摄像头访问错误: {e}")
            return [[], [], []]
        
        # Run YOLO prediction
        # 运行YOLO预测
        result = self.model.predict(source=img, verbose=False, save=False)
        
        try:
            obj_list = YoloObject.LABELS
            box_color_list = YoloObject.COLORS
            det_result_obj = []    # List to store detected object IDs / 存储检测到的物体ID的列表
            det_result_size = []   # List to store object sizes / 存储物体大小的列表
            det_result_coord = result[0].boxes.xyxy.tolist()  # Get bounding box coordinates / 获取边界框坐标
            
            # Process each detected object
            # 处理每个检测到的物体
            for i in range(len(result[0].boxes.cls.tolist())):
                # Get bounding box coordinates
                # 获取边界框坐标
                x1 = int(result[0].boxes.xyxy.tolist()[i][0])
                x2 = int(result[0].boxes.xyxy.tolist()[i][2])
                y1 = int(result[0].boxes.xyxy.tolist()[i][1])
                y2 = int(result[0].boxes.xyxy.tolist()[i][3])
                
                # Store detection results
                # 存储检测结果
                label_id = int(result[0].boxes.cls[i])
                det_result_obj.append(label_id)
                box_color = box_color_list[int(result[0].boxes.cls[i])]

                # Calculate object size as percentage of image
                # 计算物体大小占图像的百分比
                obj_size = round((x2-x1)*(y2-y1)/(img.shape[0]*img.shape[1])*100, 3)
                det_result_size.append(obj_size)
                
                # Only draw if camera display is enabled
                # 仅在启用摄像头显示时绘制
                if self.show_camera:
                    # Draw detection visualization
                    # 绘制检测可视化
                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Add label text
                    # 添加标签文本
                    txt_loc = (max(x1+2, 0), max(y1+2, 0))
                    txt = obj_list[label_id]
                    txt = f'{txt}({det_result_size[i]:.1f}%)'
                    img_h, img_w, _ = img.shape
                    
                    # Check if text location is valid
                    # 检查文本位置是否有效
                    if txt_loc[0] < img_w and txt_loc[1] < img_h:
                        # Draw label background and text
                        # 绘制标签背景和文本
                        margin = 3
                        size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        w = size[0][0] + margin * 2
                        h = size[0][1] + margin * 2
                        cv2.rectangle(img, (x1-1, y1-1-h), (x1+w, y1), box_color, -1)
                        cv2.putText(img, txt, (x1+margin, y1-margin-2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), lineType=cv2.LINE_AA)
            
            # Show detection results only if camera display is enabled
            # 仅在启用摄像头显示时显示检测结果
            if self.show_camera:
                cv2.imshow('YOLO Detection Results', img)
                cv2.waitKey(1)
            
            # Save detection results for referee system
            # 保存检测结果供裁判系统使用
            self.last_detection_result = [det_result_obj, det_result_size, det_result_coord]
            
            return self.last_detection_result
            
        except Exception as e:
            print(f"Detection error: {e}")
            return [[]]

    def find_nearest_point_index(self, current_position):
        """
        找到路径上最接近当前位置的点的索引
        """
        distances = np.linalg.norm(self.path_points - current_position, axis=1)
        return np.argmin(distances)

    def find_target_point(self, current_position):
        """
        基于前瞻距离在路径上找到目标点
        """
        if self.scenario_num == 3 and os.environ.get("SCENARIO3_ARCLENGTH_TARGET", "0") == "1":
            return self._find_target_point_by_arclength(current_position)

        nearest_point_index = self.find_nearest_point_index(current_position)

        # Look ahead on path to find target point
        for i in range(nearest_point_index, len(self.path_points)):
            distance = np.linalg.norm(self.path_points[i] - current_position)
            if distance > self.ld:
                return (self.path_points[i], distance)
        
        # If no point found, return second point
        return (self.path_points[2], distance)

    def _find_target_point_by_arclength(self, current_position):
        """Find a smoother scenario-3 lookahead target by projecting onto path segments."""
        points = self.path_points
        car_pos = np.array(current_position, dtype=float)
        if points is None or len(points) < 2:
            return (car_pos, 0.0)

        best_idx = 0
        best_t = 0.0
        best_dist = float("inf")
        segment_count = len(points)

        for idx in range(segment_count):
            p1 = points[idx]
            p2 = points[(idx + 1) % segment_count]
            segment = p2 - p1
            length_sq = float(np.dot(segment, segment))
            if length_sq < 1e-9:
                continue
            t = float(np.clip(np.dot(car_pos - p1, segment) / length_sq, 0.0, 1.0))
            projection = p1 + t * segment
            dist = float(np.linalg.norm(car_pos - projection))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
                best_t = t

        remaining = float(self.ld)
        idx = best_idx
        t = best_t
        for _ in range(segment_count + 1):
            p1 = points[idx]
            p2 = points[(idx + 1) % segment_count]
            segment = p2 - p1
            seg_len = float(np.linalg.norm(segment))
            if seg_len < 1e-9:
                idx = (idx + 1) % segment_count
                t = 0.0
                continue
            available = seg_len * (1.0 - t)
            if remaining <= available:
                target_t = t + remaining / seg_len
                target = p1 + target_t * segment
                return (target, float(np.linalg.norm(target - car_pos)))
            remaining -= available
            idx = (idx + 1) % segment_count
            t = 0.0

        fallback = points[(best_idx + 1) % segment_count]
        return (fallback, float(np.linalg.norm(fallback - car_pos)))

    def calculate_steering_angle(self, current_position, target_point, yaw):
        """
        使用Pure Pursuit算法计算转向角
        """
        # Calculate vector to target
        target_vector = np.array([target_point[0] - current_position[0], 
                                target_point[1] - current_position[1]])
        
        # Rotate vector to car's coordinate frame
        rotated_x = math.cos(yaw)*target_vector[0] + math.sin(yaw) * target_vector[1]
        rotated_y = math.sin(yaw)*target_vector[0] - math.cos(yaw) * target_vector[1]

        # Calculate target angle
        target_angle = np.arctan2(rotated_y, rotated_x)

        # Calculate steering angle using Pure Pursuit formula.
        # Do not let high straight-line speed shrink the steering command too
        # much; otherwise the car understeers through the tight curves.
        effective_speed = max(0.35, min(abs(self.speed), 0.75))
        delta = np.arctan2((2 * self.L * np.sin(target_angle)), self.ld * effective_speed * 1.55) 

        # Limit steering angle
        str_max = 0.52 if self.scenario_num == 3 else 0.4
        return max(min(delta, str_max), -str_max), target_angle, rotated_x, rotated_y

    def adjust_speed_based_on_steering_angle(self, steering_angle, current_speed):
        """
        基于转向角调整速度以获得更好的控制效果
        """
        abs_angle = np.abs(steering_angle)
        angle_threshold = 0.02

        if current_speed == 0 and abs_angle == 0:
            target_speed = 0
            return target_speed

        if self.fast_lane_following:
            if self.avoidance and self.ENABLE_CONE_AVOIDANCE:
                target_speed = min(self.max_speed, max(self.min_speed, min(current_speed, 0.40)))
            else:
                if abs_angle < 0.16:
                    target_speed = self.max_speed
                elif abs_angle < 0.24:
                    target_speed = min(self.max_speed, max(self.min_speed, 1.30))
                elif abs_angle < 0.32:
                    target_speed = min(self.max_speed, max(self.min_speed, 1.02))
                else:
                    target_speed = min(self.max_speed, max(self.min_speed, 0.78))

            max_change = 0.32
            if target_speed > current_speed:
                return min(target_speed, current_speed + max_change)
            return target_speed

        else:
            if abs_angle > angle_threshold:
                target_speed = self.min_speed
            else:
                target_speed = self.max_speed

        # Limit acceleration for smooth control
        max_change = 0.2
        if target_speed > current_speed:
            new_speed = min(target_speed, current_speed + max_change)
        else:
            new_speed = target_speed

        return new_speed

    def update_car_state(self, steering_angle, speed, avoidance):
        """
        使用新的控制输入更新车辆状态
        """
        # Adjust speed based on steering angle
        speed = self.adjust_speed_based_on_steering_angle(steering_angle, speed)
        
        # Apply avoidance steering adjustment if needed
        if avoidance and self.ENABLE_CONE_AVOIDANCE:
            steering_angle -= self.avoid_angle

        speed, steering_angle = self.apply_action_adapter(
            speed,
            steering_angle,
            context="pure_pursuit",
        )
        
        # Update car state and get new position
        status, veh_posi, orien, _, _ = self.car.set_velocity_and_request_state(
            forward=speed, turn=steering_angle, headlights=False,
            leftTurnSignal=False, rightTurnSignal=self.scenario_num == 1,
            brakeSignal=False, reverseSignal=False)

        if (not status) or (abs(veh_posi[0]) < 1e-6 and abs(veh_posi[1]) < 1e-6):
            return self.current_position, self.yaw, speed

        # Extract new position and orientation
        yaw = orien[2]
        current_position = np.array([veh_posi[0], veh_posi[1]])

        return current_position, yaw, speed

    def apply_action_adapter(self, forward, turn, context="drive"):
        base_action = {"forward": float(forward), "turn": float(turn)}
        if self.action_adapter is None:
            self.last_action_record = {
                "base_action": base_action.copy(),
                "residual_action": {"speed_delta": 0.0, "steering_bias": 0.0},
                "final_action": base_action.copy(),
                "segment_id": 0,
                "segment_name": context,
                "residual_enabled": False,
            }
            return forward, turn

        final_forward, final_turn, record = self.action_adapter(
            forward=float(forward),
            turn=float(turn),
            context=context,
            core=self,
        )
        if record is None:
            record = {}
        record.setdefault("base_action", base_action.copy())
        record.setdefault("residual_action", {"speed_delta": 0.0, "steering_bias": 0.0})
        record.setdefault("final_action", {"forward": float(final_forward), "turn": float(final_turn)})
        record.setdefault("segment_id", 0)
        record.setdefault("segment_name", context)
        record.setdefault("residual_enabled", False)
        self.last_action_record = record
        return float(final_forward), float(final_turn)

    def get_front_lidar(self):
        """
        获取前方激光雷达数据并检测障碍物（仅场景3）
        """
        if not self.ENABLE_CONE_AVOIDANCE:
            return True, [], [], False
        
        obstacle_ahead = False
        success, angle, distance = self.car.get_lidar(samplePoints=400)

        angle_deg = 180*angle/np.pi
        angle_deg = np.where(angle_deg > 180, angle_deg - 360, angle_deg)

        angle_front, dist_front = [], []

        for i in range(len(angle_deg)):
            if 0 <= angle_deg[i] <= 30:
                angle_front.append(angle_deg[i])
                dist_front.append(distance[i])

        # 对数据进行3:1降采样
        dist_slice = dist_front[::3]
        angle_slice = angle_front[::3]

        gradient = abs(np.gradient(dist_slice, angle_slice))
        gradient_dif = abs(np.diff(gradient))

        thres = 0.9
        for i in dist_front:
            if i < thres and min(dist_front) >= 0.1:
                if max(gradient) > 0.5*i and max(gradient_dif) > 0.1:
                    obstacle_ahead = True

        return success, angle_front, dist_front, obstacle_ahead

    def check_collision_and_penalties(self):
        """
        检查碰撞并处理惩罚（学生不需要修改）
        """
        current_time = time.time()
        
        # Check people collision by distance
        people_distance_collision = self.penalty_system.check_people_collision_by_distance(self.current_position)
        if people_distance_collision:
            self.penalty_system.add_people_collision_penalty()
            print(f"🚶💥 行人碰撞！(距离碰撞检测) 罚时{self.penalty_system.PEOPLE_COLLISION_PENALTY}秒")
            return True
        
        # Check cow collision by distance
        cow_distance_collision = self.penalty_system.check_cow_collision_by_distance(self.current_position)
        if cow_distance_collision:
            self.penalty_system.add_cow_collision_penalty()
            print(f"🐄💥 牛碰撞！(距离碰撞检测) 罚时{self.penalty_system.COW_COLLISION_PENALTY}秒")
            return True
        
        return False

    def get_traffic_light_state_god_view(self, traffic_light_id):
        """
        使用上帝视角获取交通灯状态
        """
        try:
            # 这里我们跟踪Traffic_Lights_Competition.py中的状态变化
            # 由于QLabs交通灯没有get_state方法，我们需要从外部进程获取状态
            # 暂时使用文件通信或直接检查传感器触发状态
            
            # 目前简化实现：基于时间和位置判断
            # 在实际应用中，这应该从Traffic_Lights_Competition.py获取
            current_time = time.time()
            
            # 获取车辆与交通灯的距离
            if traffic_light_id == 0:
                # 交通灯0位置
                x_offset = 0.13
                y_offset = 1.67
                traffic_light_pos = np.array([2.3 + x_offset, y_offset])
                distance = np.linalg.norm(self.current_position - traffic_light_pos)
                
                # 如果车辆接近交通灯，可能触发红灯
                if distance < 1.5:  # 1.5米触发距离
                    return QLabsTrafficLight.STATE_RED
                else:
                    return QLabsTrafficLight.STATE_GREEN
            elif traffic_light_id == 1:
                # 交通灯1位置
                x_offset = 0.13
                y_offset = 1.67
                traffic_light_pos = np.array([-2.3 + x_offset, -1 + y_offset])
                distance = np.linalg.norm(self.current_position - traffic_light_pos)
                
                if distance < 1.5:
                    return QLabsTrafficLight.STATE_RED
                else:
                    return QLabsTrafficLight.STATE_GREEN
            
            return QLabsTrafficLight.STATE_GREEN
            
        except Exception as e:
            print(f"⚠️ 获取交通灯状态失败: {e}")
            return QLabsTrafficLight.STATE_GREEN

    def get_people_positions_god_view(self):
        """
        使用上帝视角获取行人位置
        """
        people_positions = []
        if self.scenario_num == 3:
            try:
                for i, people_actor in self.people_actors.items():
                    success, location, rotation, scale = people_actor.get_world_transform()
                    if success:
                        x, y = location[:2]
                        if i == 0 and -2.25 <= x <= -1.40 and 3.05 <= y <= 3.30:
                            people_positions.append(location[:2])  # 只取x,y坐标
                        elif i == 1 and 0.95 <= x <= 1.25 and -1.35 <= y <= -0.50:
                            people_positions.append(location[:2])  # 只取x,y坐标
            except Exception as e:
                print(f"⚠️ 获取行人位置失败: {e}")
        
        return people_positions

    def get_cow_positions_god_view(self):
        """
        使用上帝视角获取牛位置
        """
        cow_positions = []
        if self.scenario_num == 3:
            try:
                for i, cow_actor in self.cow_actors.items():
                    success, location, rotation, scale = cow_actor.get_world_transform()
                    if success:
                        x, y = location[:2]
                        if -0.30 <= x <= -0.02 and 3.85 <= y <= 4.65:
                            cow_positions.append(location[:2])  # 只取x,y坐标
            except Exception as e:
                print(f"⚠️ 获取牛位置失败: {e}")
        
        return cow_positions

    def monitor_red_light_violation_god_view(self):
        """
        裁判系统：监控红灯违规行为（上帝视角）
        不依赖学生的摄像头检测结果
        """
        # 基于车辆位置和速度监控红灯违规
        # 这里使用时间和位置的组合来判断是否可能闯红灯
        
        # 获取当前位置
        current_position = self.current_position
        
        # 交通灯位置 (基于Setup_Competition.py中的设置)
        x_offset = 0.13
        y_offset = 1.67
        traffic_light_pos = np.array([2.3 + x_offset, y_offset])
        
        # 计算与交通灯的距离
        distance_to_light = np.linalg.norm(current_position - traffic_light_pos)
        
        # 如果车辆接近交通灯区域且速度过快，可能闯红灯
        if distance_to_light < 1.0 and hasattr(self, 'speed') and self.speed > 0.1:
            # 检查最近的YOLO检测结果中是否有红灯
            if hasattr(self, 'last_detection_result') and self.last_detection_result:
                det_result = self.last_detection_result
                if len(det_result) > 0 and YoloObject.RED in det_result[0]:
                    if YoloObject.STOP_LINE in det_result[0]:
                        return

                    # 检测到红灯且车辆在危险区域内高速行驶
                    red_light_idx = det_result[0].index(YoloObject.RED)
                    red_light_size = det_result[1][red_light_idx]
                    
                    # 红灯足够大（近距离）且车辆速度过快
                    if red_light_size >= 0.15:
                        # 添加闯红灯罚时
                        self.penalty_system.add_red_light_violation_penalty()
                        self.add_penalty_message(f"🚨 裁判系统检测到闯红灯！罚时{self.penalty_system.RED_LIGHT_VIOLATION_PENALTY}秒")
    
    def monitor_people_cow_collision_god_view(self):
        """
        裁判系统：监控行人和牛碰撞（上帝视角）
        使用QLAB动态位置和距离检测，不依赖学生摄像头
        """
        current_position = self.current_position
        if self.should_update_dynamic_referee_positions():
            people_positions = self.get_people_positions_god_view()
            cow_positions = self.get_cow_positions_god_view()

            if people_positions:
                self.penalty_system.update_people_position(people_positions)
            if cow_positions:
                self.penalty_system.update_cow_position(cow_positions)
        
        # 检查行人碰撞
        if self.penalty_system.check_people_collision_by_distance(current_position):
            self.penalty_system.add_people_collision_penalty()
            people = np.array(self.penalty_system.people_positions, dtype=float)
            distances = np.linalg.norm(people - np.array(current_position), axis=1)
            index = int(np.argmin(distances))
            nearest = people[index]
            self.add_penalty_message(
                f"🚶💥 裁判系统检测到行人碰撞！罚时{self.penalty_system.PEOPLE_COLLISION_PENALTY}秒 "
                f"| car=({current_position[0]:.3f},{current_position[1]:.3f}) "
                f"| people#{index}=({nearest[0]:.3f},{nearest[1]:.3f}) "
                f"| dist={distances[index]:.3f}"
            )
        
        # 检查牛碰撞
        if self.penalty_system.check_cow_collision_by_distance(current_position):
            self.penalty_system.add_cow_collision_penalty()
            cows = np.array(self.penalty_system.cow_positions, dtype=float)
            distances = np.linalg.norm(cows - np.array(current_position), axis=1)
            index = int(np.argmin(distances))
            nearest = cows[index]
            self.add_penalty_message(
                f"🐄💥 裁判系统检测到牛碰撞！罚时{self.penalty_system.COW_COLLISION_PENALTY}秒 "
                f"| car=({current_position[0]:.3f},{current_position[1]:.3f}) "
                f"| cow#{index}=({nearest[0]:.3f},{nearest[1]:.3f}) "
                f"| dist={distances[index]:.3f}"
            )

    def run_referee_checks(self):
        if self.penalty_system is None:
            return
        self.penalty_system.update_penalty(self.current_position)
        if self.referee_monitoring_active:
            self.monitor_people_cow_collision_god_view()
    
    def add_penalty_message(self, message):
        """添加罚时信息到文件"""
        try:
            with open('penalty_messages.txt', 'a', encoding='utf-8') as f:
                f.write(f"{message}|{self.penalty_system.total_penalty_time}\n")
        except:
            print(message)
    
    def monitor_red_light_violation(self, det_result):
        """
        监控红灯违规行为 - 裁判系统负责
        已废弃：现在使用上帝视角监控，不再依赖摄像头检测
        """
        print("⚠️ 警告：使用了已废弃的基于摄像头的红灯监控方法")
        print("   请使用 monitor_red_light_violation_god_view() 方法")
        # 调用新的上帝视角方法
        self.monitor_red_light_violation_god_view()

    def proceed_normally(self, avoidance):
        """
        正常前进，不进行特殊处理
        """
        target_point, distance = self.find_target_point(self.current_position)
        steering_angle, alpha, rotated_x, rotated_y = self.calculate_steering_angle(
            self.current_position, target_point, self.yaw)
        self.current_position, self.yaw, self.speed = self.update_car_state(
            steering_angle, self.speed, avoidance)
        
        return self.current_position, self.yaw, self.speed

    def check_endpoint(self):
        """
        检查是否到达终点
        """
        dist_to_end = np.linalg.norm(self.current_position - self.endpoint)
        return dist_to_end < 0.2

    def show_final_results(self, lap_start):
        """
        显示最终结果
        """
        self.car.set_velocity_and_request_state(
            forward=0, turn=0, headlights=False, leftTurnSignal=False, 
            rightTurnSignal=True, brakeSignal=False, reverseSignal=False)
        
        lap_time = time.time() - lap_start
        
        # Get final penalty statistics
        penalty_stats = self.penalty_system.get_penalty_stats()
        total_penalty = penalty_stats['total_penalty_time']
        final_time = lap_time + total_penalty
        
        # Display final summary
        self.penalty_system.show_final_summary(lap_time)
        print("="*60)
        print(f"LAP TIME : {final_time:.2f} sec") 
