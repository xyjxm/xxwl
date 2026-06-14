import time
import sys
import numpy as np

class PenaltySystem:
    """
    竞赛计时罚时系统
    参考强化学习代码中的惩罚机制，为学生竞赛系统添加：
    1. 出界罚时：轻微出界3秒，严重出界6秒
    2. 锥桶碰撞罚时
    3. 持续出界按每1.0秒计时
    """
    
    def __init__(self, path_points_file="path.txt", scenario_num=3, use_file_output=False):
        """
        初始化惩罚系统
        
        Args:
            path_points_file: 路径点文件
            scenario_num: 场景编号
            use_file_output: 是否使用文件输出
        """
        self.scenario_num = scenario_num
        self.use_file_output = use_file_output
        
        # === 车道偏离罚时参数 ===
        # 参考强化学习代码中的阈值设置
        self.MINOR_DEVIATION_THRESHOLD = 0.08      # 轻微出界阈值：0.08m
        self.LEFT_MAJOR_DEVIATION_THRESHOLD = 0.12  # 左侧中等出界阈值：0.12m
        self.LEFT_TERMINAL_THRESHOLD = 0.18         # 左侧严重出界阈值：0.18m
        self.RIGHT_MAJOR_DEVIATION_THRESHOLD = 0.12 # 右侧中等出界阈值：0.12m
        self.RIGHT_TERMINAL_THRESHOLD = 0.18        # 右侧严重出界阈值：0.18m
        
        # === 罚时设置 ===
        self.MINOR_DEVIATION_PENALTY = 3.0         # 轻微出界罚时：3秒
        self.MAJOR_DEVIATION_PENALTY = 6.0         # 严重出界罚时：6秒
        self.CONE_COLLISION_PENALTY = 10.0          # 锥桶碰撞罚时：10秒
        self.CONTINUOUS_DEVIATION_PENALTY = 3.0    # Continuous line/out-of-bounds penalty per interval
        self.CONTINUOUS_DEVIATION_INTERVAL = 1.5   # Continuous line/out-of-bounds interval: 1.5s
        self.PEOPLE_COLLISION_PENALTY = 10.0       # 行人碰撞罚时：10秒
        self.COW_COLLISION_PENALTY = 10.0          # 牛碰撞罚时：10秒
        self.RED_LIGHT_VIOLATION_PENALTY = 10.0     # 闯红灯罚时：10秒
        
        # === 锥桶碰撞检测参数 ===
        self.CONE_SQUARE_HALF_SIDE = 0.15          # 锥桶碰撞检测范围：±0.15m
        
        # 加载路径点
        self.REFEREE_SAMPLE_STEP = 0.02
        self.last_sampled_position = None
        self.path_points = self._load_path_points(path_points_file)
        
        # 设置锥桶位置（参考强化学习代码）
        self.cone_positions = []
        if scenario_num == 3:
            # 场景3的锥桶位置（与Setup_Competition.py保持一致）
            x_offset = 0.13
            y_offset = 1.67
            cone_x = 2.1 + x_offset
            cone_y = y_offset - 0.5
            self.cone_positions.append((cone_x, cone_y))
        
        # 设置行人和牛的初始位置（场景3）
        self.people_positions = []
        self.cow_positions = []
        if scenario_num == 3:
            # 行人位置（从Traffic_Lights_Competition.py获取）
            self.people_positions = [
                [-1.451, 3.172],  # 行人1起始位置
                [1.1, -0.56]      # 行人2起始位置  
            ]
            # 牛位置
            self.cow_positions = [
                [-0.159, 3.9]     # 牛起始位置
            ]
        
        # 碰撞检测距离阈值
        self.PEOPLE_COLLISION_DISTANCE = 0.3   # 行人碰撞距离阈值：0.3m
        self.COW_COLLISION_DISTANCE = 0.5      # 牛碰撞距离阈值：0.5m
        
        # 状态跟踪
        self.total_penalty_time = 0.0              # 总罚时
        self.last_deviation_time = 0.0             # 上次偏离时间
        self.continuous_deviation_start = 0.0      # 连续偏离开始时间
        self.last_continuous_penalty_time = 0.0    # 上次计算持续出界罚时的时间
        self.is_continuously_deviating = False     # 是否正在连续偏离
        self.was_out_of_bounds_last_check = False  # 上次检查是否出界（简化逻辑）
        self.last_deviation_section = None
        self.collision_cooldown = {}               # 碰撞冷却时间（防止重复检测）
        self.last_penalty_print_time = 0.0        # 上次打印罚时的时间
        
        # 统计信息
        self.penalty_stats = {
            'minor_deviation_count': 0,
            'major_deviation_count': 0,
            'cone_collision_count': 0,
            'continuous_deviation_time': 0.0,
            'people_collision_count': 0,
            'cow_collision_count': 0,
            'red_light_violation_count': 0
        }
        
        self._print_penalty_message(f"PenaltySystem initialized for scenario {scenario_num}")
        self._print_penalty_message(f"Cone positions: {self.cone_positions}")
        self._print_penalty_message(f"Cone collision half side: {self.CONE_SQUARE_HALF_SIDE}m")
        self._print_penalty_message(f"Penalty settings: Minor={self.MINOR_DEVIATION_PENALTY}s, Major={self.MAJOR_DEVIATION_PENALTY}s, Cone={self.CONE_COLLISION_PENALTY}s")
        self._print_penalty_message(f"Collision penalties: People={self.PEOPLE_COLLISION_PENALTY}s, Cow={self.COW_COLLISION_PENALTY}s, RedLight={self.RED_LIGHT_VIOLATION_PENALTY}s")
        self._print_penalty_message(f"Distance collision thresholds: People={self.PEOPLE_COLLISION_DISTANCE}m, Cow={self.COW_COLLISION_DISTANCE}m")
        self._print_penalty_message(f"People positions: {self.people_positions}")
        self._print_penalty_message(f"Cow positions: {self.cow_positions}")
    
    def _load_path_points(self, path_file):
        """加载路径点文件"""
        try:
            with open(path_file, 'r') as file:
                lines = file.readlines()
                path_points = np.array([list(map(float, line.strip().split())) for line in lines])
                self._print_penalty_message(f"Loaded {len(path_points)} path points")
                return path_points
        except Exception as e:
            self._print_penalty_message(f"Error loading path points: {e}")
            return np.array([])
    
    def calculate_lane_deviation(self, current_position):
        """
        计算车道偏离距离
        参考强化学习代码中的车道偏离计算逻辑
        
        Args:
            current_position: 当前车辆位置 [x, y]
            
        Returns:
            tuple: (偏离距离, 是否右侧偏离)
        """
        if len(self.path_points) < 2:
            return 0.0, False
            
        car_pos = np.array(current_position)
        
        # 找到最近的路径点索引
        distances_to_points = np.linalg.norm(self.path_points - car_pos, axis=1)
        nearest_point_idx = np.argmin(distances_to_points)
        
        # 确定线段的两个点 P1 和 P2
        if nearest_point_idx == 0:
            p1_idx = 0
            p2_idx = 1
        elif nearest_point_idx == len(self.path_points) - 1:
            p1_idx = len(self.path_points) - 2
            p2_idx = len(self.path_points) - 1
        else:
            # 使用与强化学习代码相同的逻辑
            p_prev = self.path_points[nearest_point_idx - 1]
            p_curr = self.path_points[nearest_point_idx]
            p_next = self.path_points[nearest_point_idx + 1] if nearest_point_idx + 1 < len(self.path_points) else p_curr
            
            vec_prev_curr = p_curr - p_prev
            vec_car_prev = car_pos - p_prev
            vec_curr_car = car_pos - p_curr
            
            if np.dot(vec_car_prev, vec_prev_curr) >= 0 and np.dot(vec_curr_car, vec_prev_curr) <= 0:
                p1_idx = nearest_point_idx - 1
                p2_idx = nearest_point_idx
            else:
                p1_idx = nearest_point_idx
                p2_idx = min(nearest_point_idx + 1, len(self.path_points) - 1)
                if p1_idx == p2_idx and p1_idx > 0:
                    p1_idx = p1_idx - 1
        
        p1 = self.path_points[p1_idx]
        p2 = self.path_points[p2_idx]
        
        # 计算点到线段的垂直距离
        if np.array_equal(p1, p2):
            distance_to_segment = np.linalg.norm(car_pos - p1)
            is_right_deviation = False
        else:
            # 计算垂直距离
            numerator = abs((p2[0] - p1[0]) * (p1[1] - car_pos[1]) - (p1[0] - car_pos[0]) * (p2[1] - p1[1]))
            denominator = np.sqrt(np.sum((p2 - p1)**2))
            
            if denominator < 1e-6:
                distance_to_segment = np.linalg.norm(car_pos - p1)
                is_right_deviation = False
            else:
                distance_to_segment = numerator / denominator
                
                # 计算车辆相对于路径的位置（左侧还是右侧）
                path_vector = p2 - p1
                car_vector = car_pos - p1
                cross_product_z = path_vector[0] * car_vector[1] - path_vector[1] * car_vector[0]
                
                # 赛道是逆时针方向，右侧是墙壁，所以叉积为负表示车辆在右侧
                is_right_deviation = cross_product_z < 0
        
        return distance_to_segment, is_right_deviation
    
    def check_cone_collision(self, current_position):
        """
        检查锥桶碰撞
        参考强化学习代码中的锥桶碰撞检测逻辑
        
        Args:
            current_position: 当前车辆位置 [x, y]
            
        Returns:
            bool: 是否发生碰撞
        """
        current_time = time.time()
        car_pos = np.array(current_position)
        
        for i, cone_pos in enumerate(self.cone_positions):
            # 检查碰撞冷却时间（防止重复检测）
            if i in self.collision_cooldown:
                if current_time - self.collision_cooldown[i] < 2.0:  # 2秒冷却时间
                    continue
            
            # 检查是否在锥桶的方形碰撞区域内
            if (abs(car_pos[0] - cone_pos[0]) < self.CONE_SQUARE_HALF_SIDE and
                abs(car_pos[1] - cone_pos[1]) < self.CONE_SQUARE_HALF_SIDE):
                
                # 记录碰撞冷却时间
                self.collision_cooldown[i] = current_time
                return True
        
        return False

    def add_people_collision_penalty(self):
        """
        添加行人碰撞罚时
        
        Returns:
            bool: 是否成功添加罚时
        """
        current_time = time.time()
        
        # 检查冷却时间（防止重复检测）
        if 'people_collision' in self.collision_cooldown:
            if current_time - self.collision_cooldown['people_collision'] < 3.0:  # 3秒冷却时间
                return False
        
        # 添加罚时
        self.total_penalty_time += self.PEOPLE_COLLISION_PENALTY
        self.penalty_stats['people_collision_count'] += 1
        self.collision_cooldown['people_collision'] = current_time
        
        message = f"🚶 行人碰撞罚时！+{self.PEOPLE_COLLISION_PENALTY:.1f}秒 (总计: {self.total_penalty_time:.1f}秒)"
        self._print_penalty_message(message)
        
        return True
    
    def add_cow_collision_penalty(self):
        """
        添加牛碰撞罚时
        
        Returns:
            bool: 是否成功添加罚时
        """
        current_time = time.time()
        
        # 检查冷却时间（防止重复检测）
        if 'cow_collision' in self.collision_cooldown:
            if current_time - self.collision_cooldown['cow_collision'] < 3.0:  # 3秒冷却时间
                return False
        
        # 添加罚时
        self.total_penalty_time += self.COW_COLLISION_PENALTY
        self.penalty_stats['cow_collision_count'] += 1
        self.collision_cooldown['cow_collision'] = current_time
        
        message = f"🐄 牛碰撞罚时！+{self.COW_COLLISION_PENALTY:.1f}秒 (总计: {self.total_penalty_time:.1f}秒)"
        self._print_penalty_message(message)
        
        return True
    
    def add_red_light_violation_penalty(self):
        """
        添加闯红灯罚时
        
        Returns:
            bool: 是否成功添加罚时
        """
        current_time = time.time()
        
        # 检查冷却时间（防止重复检测）
        if 'red_light_violation' in self.collision_cooldown:
            if current_time - self.collision_cooldown['red_light_violation'] < 5.0:  # 5秒冷却时间
                return False
        
        # 添加罚时
        self.total_penalty_time += self.RED_LIGHT_VIOLATION_PENALTY
        self.penalty_stats['red_light_violation_count'] += 1
        self.collision_cooldown['red_light_violation'] = current_time
        
        message = f"🚨 闯红灯罚时！+{self.RED_LIGHT_VIOLATION_PENALTY:.1f}秒 (总计: {self.total_penalty_time:.1f}秒)"
        self._print_penalty_message(message)
        
        return True
    
    def update_people_position(self, people_positions):
        """
        更新行人位置
        
        Args:
            people_positions: 行人位置列表 [[x1, y1], [x2, y2], ...]
        """
        if people_positions:
            self.people_positions = people_positions
    
    def update_cow_position(self, cow_positions):
        """
        更新牛位置
        
        Args:
            cow_positions: 牛位置列表 [[x1, y1], [x2, y2], ...]
        """
        if cow_positions:
            self.cow_positions = cow_positions
    
    def check_people_collision_by_distance(self, current_position):
        """
        基于距离检查行人碰撞
        
        Args:
            current_position: 当前车辆位置 [x, y]
            
        Returns:
            bool: 是否发生碰撞
        """
        if not self.people_positions:
            return False
            
        current_time = time.time()
        car_pos = np.array(current_position)
        
        for i, people_pos in enumerate(self.people_positions):
            # 检查碰撞冷却时间
            people_key = f'people_collision_{i}'
            if people_key in self.collision_cooldown:
                if current_time - self.collision_cooldown[people_key] < 3.0:
                    continue
            
            # 计算距离
            distance = np.linalg.norm(car_pos - np.array(people_pos))
            
            # 检查是否在碰撞范围内
            if distance <= self.PEOPLE_COLLISION_DISTANCE:
                self.collision_cooldown[people_key] = current_time
                return True
        
        return False
    
    def check_cow_collision_by_distance(self, current_position):
        """
        基于距离检查牛碰撞
        
        Args:
            current_position: 当前车辆位置 [x, y]
            
        Returns:
            bool: 是否发生碰撞
        """
        if not self.cow_positions:
            return False
            
        current_time = time.time()
        car_pos = np.array(current_position)
        
        for i, cow_pos in enumerate(self.cow_positions):
            # 检查碰撞冷却时间
            cow_key = f'cow_collision_{i}'
            if cow_key in self.collision_cooldown:
                if current_time - self.collision_cooldown[cow_key] < 3.0:
                    continue
            
            # 计算距离
            distance = np.linalg.norm(car_pos - np.array(cow_pos))
            
            # 检查是否在碰撞范围内
            if distance <= self.COW_COLLISION_DISTANCE:
                self.collision_cooldown[cow_key] = current_time
                return True
        
        return False

    def _deviation_section(self, current_position):
        if not self.cone_positions:
            return "default"

        car_pos = np.array(current_position)
        for i, cone_pos in enumerate(self.cone_positions):
            cone_x, cone_y = cone_pos
            near_cone_x = abs(car_pos[0] - cone_x) <= 0.95
            near_cone_y = cone_y - 1.35 <= car_pos[1] <= cone_y + 1.25
            if near_cone_x and near_cone_y:
                if car_pos[1] <= cone_y + self.CONE_SQUARE_HALF_SIDE:
                    return f"cone_{i}_before"
                return f"cone_{i}_after"

        return "default"

    def _is_cone_before_after_transition(self, previous_section, current_section):
        if previous_section is None or current_section is None:
            return False
        if previous_section == current_section:
            return False
        return (
            previous_section.startswith("cone_")
            and current_section.startswith("cone_")
        )

    def _interpolated_referee_positions(self, current_position):
        current = np.array(current_position, dtype=float)
        if self.last_sampled_position is None:
            return [current]

        previous = np.array(self.last_sampled_position, dtype=float)
        distance = np.linalg.norm(current - previous)
        if not np.isfinite(distance) or distance <= self.REFEREE_SAMPLE_STEP:
            return [current]

        sample_count = int(np.ceil(distance / self.REFEREE_SAMPLE_STEP))
        return [
            previous + (current - previous) * (i / sample_count)
            for i in range(1, sample_count + 1)
        ]
    
    def update_penalty(self, current_position):
        """
        更新罚时系统
        
        Args:
            current_position: 当前车辆位置 [x, y]
            
        Returns:
            dict: 罚时信息
        """
        current_time = time.time()
        penalty_info = {
            'new_penalty': 0.0,
            'total_penalty': self.total_penalty_time,
            'reason': '',
            'deviation_distance': 0.0,
            'is_right_deviation': False
        }
        sample_positions = self._interpolated_referee_positions(current_position)
        report_position = np.array(current_position, dtype=float)
        
        # 1. 检查车道偏离
        deviation_distance = 0.0
        is_right_deviation = False
        deviation_position = np.array(current_position, dtype=float)
        for sample_position in sample_positions:
            sample_deviation, sample_is_right = self.calculate_lane_deviation(sample_position)
            if sample_deviation > deviation_distance:
                deviation_distance = sample_deviation
                is_right_deviation = sample_is_right
                deviation_position = sample_position
        penalty_info['deviation_distance'] = deviation_distance
        penalty_info['is_right_deviation'] = is_right_deviation
        
        # 2. 检查锥桶碰撞
        cone_collision_detected = False
        for sample_position in sample_positions:
            if not self.check_cone_collision(sample_position):
                continue
            cone_collision_detected = True
            report_position = sample_position
            penalty_info['new_penalty'] += self.CONE_COLLISION_PENALTY
            penalty_info['reason'] += f"锥桶碰撞(+{self.CONE_COLLISION_PENALTY}s) "
            self.penalty_stats['cone_collision_count'] += 1
            break
        
        # 3. 处理车道偏离罚时
        current_is_out_of_bounds = (
            deviation_distance > self.MINOR_DEVIATION_THRESHOLD
        )
        deviation_section = self._deviation_section(deviation_position)
        
        if current_is_out_of_bounds:
            # 只在从正常状态进入出界状态时记录一次出界罚时
            if not self.was_out_of_bounds_last_check:
                # 第一次出界，记录出界罚时
                
                # 判断偏离严重程度
                if is_right_deviation:
                    # 右侧偏离（更严格）
                    if deviation_distance > self.RIGHT_TERMINAL_THRESHOLD:
                        # 严重右侧偏离
                        penalty_info['new_penalty'] += self.MAJOR_DEVIATION_PENALTY
                        penalty_info['reason'] += f"严重右侧出界(+{self.MAJOR_DEVIATION_PENALTY}s) "
                        self.penalty_stats['major_deviation_count'] += 1
                    elif deviation_distance > self.RIGHT_MAJOR_DEVIATION_THRESHOLD:
                        # 中等右侧偏离
                        penalty_info['new_penalty'] += self.MINOR_DEVIATION_PENALTY
                        penalty_info['reason'] += f"中等右侧出界(+{self.MINOR_DEVIATION_PENALTY}s) "
                        self.penalty_stats['minor_deviation_count'] += 1
                    else:
                        # 轻微右侧偏离
                        penalty_info['new_penalty'] += self.MINOR_DEVIATION_PENALTY
                        penalty_info['reason'] += f"轻微右侧出界(+{self.MINOR_DEVIATION_PENALTY}s) "
                        self.penalty_stats['minor_deviation_count'] += 1
                else:
                    # 左侧偏离
                    if deviation_distance > self.LEFT_TERMINAL_THRESHOLD:
                        # 严重左侧偏离
                        penalty_info['new_penalty'] += self.MAJOR_DEVIATION_PENALTY
                        penalty_info['reason'] += f"严重左侧出界(+{self.MAJOR_DEVIATION_PENALTY}s) "
                        self.penalty_stats['major_deviation_count'] += 1
                    elif deviation_distance > self.LEFT_MAJOR_DEVIATION_THRESHOLD:
                        # 中等左侧偏离
                        penalty_info['new_penalty'] += self.MINOR_DEVIATION_PENALTY
                        penalty_info['reason'] += f"中等左侧出界(+{self.MINOR_DEVIATION_PENALTY}s) "
                        self.penalty_stats['minor_deviation_count'] += 1
                    else:
                        # 轻微左侧偏离
                        penalty_info['new_penalty'] += self.MINOR_DEVIATION_PENALTY
                        penalty_info['reason'] += f"轻微左侧出界(+{self.MINOR_DEVIATION_PENALTY}s) "
                        self.penalty_stats['minor_deviation_count'] += 1
                
                # 开始持续出界计时
                self.is_continuously_deviating = True
                self.continuous_deviation_start = current_time
                self.last_continuous_penalty_time = current_time
            
            # 处理持续出界罚时（不论是否第一次出界）
            if self.is_continuously_deviating:
                # 继续持续出界，计算额外罚时
                continuous_time = current_time - self.continuous_deviation_start
                if continuous_time >= self.CONTINUOUS_DEVIATION_INTERVAL:
                    # 只对新增的时间进行罚时计算
                    time_since_last_penalty = current_time - self.last_continuous_penalty_time
                    if time_since_last_penalty >= self.CONTINUOUS_DEVIATION_INTERVAL:
                        # 计算新增的完整1.0秒段数
                        additional_intervals = int(time_since_last_penalty / self.CONTINUOUS_DEVIATION_INTERVAL)
                        if additional_intervals > 0:  # 确保有新增时间段
                            additional_duration = additional_intervals * self.CONTINUOUS_DEVIATION_INTERVAL
                            additional_penalty = additional_intervals * self.CONTINUOUS_DEVIATION_PENALTY
                            penalty_info['new_penalty'] += additional_penalty
                            penalty_info['reason'] += f"持续出界{additional_duration:.1f}s(+{additional_penalty}s) "
                            self.penalty_stats['continuous_deviation_time'] += additional_penalty
                            # 更新上次计算罚时的时间
                            self.last_continuous_penalty_time += additional_duration
        else:
            # 没有偏离，重置所有出界状态
            self.is_continuously_deviating = False
            self.continuous_deviation_start = 0.0
            self.last_continuous_penalty_time = 0.0
        
        if current_is_out_of_bounds and not cone_collision_detected:
            report_position = deviation_position
        
        # 更新上次检查状态
        self.was_out_of_bounds_last_check = current_is_out_of_bounds
        self.last_deviation_section = deviation_section
        
        # 更新总罚时
        self.total_penalty_time += penalty_info['new_penalty']
        penalty_info['total_penalty'] = self.total_penalty_time
        
        # 实时打印罚时信息
        if penalty_info['new_penalty'] > 0:
            # 有新罚时时立即显示详情
            self._print_penalty_message(
                f"🚨 罚时: {penalty_info['reason']}| 总罚时: {self.total_penalty_time:.1f}s "
                f"| pos=({report_position[0]:.3f}, {report_position[1]:.3f})")
            
            # 显示详细的偏离信息
            if deviation_distance > self.MINOR_DEVIATION_THRESHOLD:
                side = "右侧" if is_right_deviation else "左侧"
                self._print_penalty_message(f"   📍 {side}偏离距离: {deviation_distance:.3f}m (阈值: {self.MINOR_DEVIATION_THRESHOLD:.3f}m)")
            
            self.last_penalty_print_time = current_time
        elif (current_time - self.last_penalty_print_time) > 5.0:
            # 定期显示总罚时（避免刷屏）
            if self.total_penalty_time > 0:
                self._print_penalty_message(f"📊 当前总罚时: {self.total_penalty_time:.1f}s")
            self.last_penalty_print_time = current_time
        
        self.last_sampled_position = np.array(current_position, dtype=float)
        return penalty_info
    
    def get_penalty_stats(self):
        """获取罚时统计信息"""
        return {
            'total_penalty_time': self.total_penalty_time,
            'penalty_breakdown': self.penalty_stats.copy(),
            'cone_positions': self.cone_positions
        }
    
    def reset(self):
        """重置罚时系统"""
        self.total_penalty_time = 0.0
        self.last_deviation_time = 0.0
        self.continuous_deviation_start = 0.0
        self.last_continuous_penalty_time = 0.0
        self.is_continuously_deviating = False
        self.was_out_of_bounds_last_check = False
        self.last_deviation_section = None
        self.last_sampled_position = None
        self.collision_cooldown = {}
        self.last_penalty_print_time = 0.0
    

    
    def _safe_print(self, message=""):
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        text = str(message)
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    
    def _print_penalty_message(self, message):
        """打印罚时信息（支持回调函数、文件输出和终端）"""
        # 优先使用回调函数
        if hasattr(self, 'gui_callback') and self.gui_callback:
            try:
                self.gui_callback(message, self.total_penalty_time)
                return
            except:
                pass
        
        # 其次使用文件输出
        if self.use_file_output:
            try:
                with open('penalty_messages.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{message}|{self.total_penalty_time}\n")
                return
            except:
                pass
        
        # 最后回退到终端打印
        self._safe_print(message)
    

    
    def show_final_summary(self, lap_time):
        """显示最终统计信息"""
        # 优先使用回调函数
        if hasattr(self, 'gui_callback') and self.gui_callback:
            try:
                # 通过回调显示最终结果
                summary_message = f"🏁 LAP COMPLETE!\n⏱️  原始时间: {lap_time:.2f} 秒\n⚠️  总罚时: {self.total_penalty_time:.2f} 秒\n🏆 最终时间: {lap_time + self.total_penalty_time:.2f} 秒"
                self.gui_callback(summary_message, self.total_penalty_time, lap_time)
                return
            except:
                pass
        
        # 其次使用文件输出
        if self.use_file_output:
            try:
                summary_message = f"🏁 LAP COMPLETE! ⏱️  原始时间: {lap_time:.2f} 秒 ⚠️  总罚时: {self.total_penalty_time:.2f} 秒 🏆 最终时间: {lap_time + self.total_penalty_time:.2f} 秒"
                with open('penalty_messages.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{summary_message}|{self.total_penalty_time}|{lap_time}\n")
                
                # 添加详细统计信息
                stats_message = f"📊 罚时详情: 轻微出界: {self.penalty_stats['minor_deviation_count']} 次, 严重出界: {self.penalty_stats['major_deviation_count']} 次, 锥桶碰撞: {self.penalty_stats['cone_collision_count']} 次, 持续出界罚时: {self.penalty_stats['continuous_deviation_time']:.1f} 秒, 行人碰撞: {self.penalty_stats['people_collision_count']} 次, 牛碰撞: {self.penalty_stats['cow_collision_count']} 次, 闯红灯: {self.penalty_stats['red_light_violation_count']} 次"
                with open('penalty_messages.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{stats_message}|{self.total_penalty_time}|{lap_time}\n")
                
                # 同时在终端显示详细信息
                self._safe_print(summary_message)
                self._safe_print(stats_message)
                return
            except:
                pass
        

        
        # 最后回退到终端显示
        self._safe_print("\n============================================================")
        self._safe_print("🏁 LAP COMPLETE!")
        self._safe_print("============================================================")
        self._safe_print(f"⏱️  原始时间: {lap_time:.2f} 秒")
        self._safe_print(f"⚠️  总罚时: {self.total_penalty_time:.2f} 秒")
        self._safe_print(f"🏆 最终时间: {lap_time + self.total_penalty_time:.2f} 秒")
        self._safe_print("============================================================")
        self._safe_print("📊 罚时详情:")
        self._safe_print(f"   轻微出界: {self.penalty_stats['minor_deviation_count']} 次")
        self._safe_print(f"   严重出界: {self.penalty_stats['major_deviation_count']} 次")
        self._safe_print(f"   锥桶碰撞: {self.penalty_stats['cone_collision_count']} 次")
        self._safe_print(f"   持续出界罚时: {self.penalty_stats['continuous_deviation_time']:.1f} 秒")
        self._safe_print(f"   行人碰撞: {self.penalty_stats['people_collision_count']} 次")
        self._safe_print(f"   牛碰撞: {self.penalty_stats['cow_collision_count']} 次")
        self._safe_print(f"   闯红灯: {self.penalty_stats['red_light_violation_count']} 次")
        self._safe_print("============================================================")
    
    def set_gui_callback(self, gui_callback):
        """设置GUI回调函数"""
        self.gui_callback = gui_callback 
