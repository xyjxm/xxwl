import sys
import os
import time
import runtime_paths
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QLabel, QDoubleSpinBox, QGroupBox,
                            QGridLayout, QFormLayout, QDialog, QDialogButtonBox,
                            QHBoxLayout, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import subprocess
import signal
import gym
from gym import spaces
import numpy as np

BASE_DIR = runtime_paths.configure()

class ParameterDialog(QDialog):
    def __init__(self, scenario_num, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"赛题 {scenario_num} 配置")
        self.setModal(True)
        self.scenario_num = scenario_num
        
        # Get stored parameters if available
        stored_params = parent.stored_params.get(scenario_num) if parent else None
        
        layout = QVBoxLayout(self)
        
        # Create form for parameters
        form_layout = QFormLayout()
        
        # Create parameter inputs with stored values if available
        self.ld_input = QDoubleSpinBox()
        self.ld_input.setRange(0.1, 1.0)
        self.ld_input.setValue(stored_params['ld'] if stored_params else 0.50)
        self.ld_input.setSingleStep(0.05)
        
        self.max_speed_input = QDoubleSpinBox()
        self.max_speed_input.setRange(0.1, 3.0)
        self.max_speed_input.setValue(stored_params['max_speed'] if stored_params else 0.80)
        self.max_speed_input.setSingleStep(0.1)
        
        self.min_speed_input = QDoubleSpinBox()
        self.min_speed_input.setRange(0.1, 3.0)
        self.min_speed_input.setValue(stored_params['min_speed'] if stored_params else 0.20)
        self.min_speed_input.setSingleStep(0.1)
        
        # Add camera display option
        self.show_camera_checkbox = QCheckBox("显示摄像头窗口")
        self.show_camera_checkbox.setChecked(stored_params.get('show_camera', False) if stored_params else False)
        self.show_camera_checkbox.setToolTip("启用以显示YOLO检测窗口（会影响性能）")
        
        # Add inputs to form layout
        form_layout.addRow("预瞄距离(m):", self.ld_input)
        form_layout.addRow("最大速度(m/s):", self.max_speed_input)
        form_layout.addRow("最小速度(m/s):", self.min_speed_input)
        form_layout.addRow("", self.show_camera_checkbox)  # Empty label for checkbox

        if scenario_num == 2 or scenario_num == 3:
            self.stop_sign_input = QDoubleSpinBox()
            self.stop_sign_input.setRange(0.1, 20)
            self.stop_sign_input.setValue(stored_params['stop_sign'] if stored_params else 0.80)
            self.stop_sign_input.setSingleStep(0.05)
            form_layout.addRow("停止标志有效大小(%):", self.stop_sign_input)

            if scenario_num == 3:
                self.avoid_angle = QDoubleSpinBox()
                self.avoid_angle.setRange(0, 0.5)
                self.avoid_angle.setValue(stored_params['avoid_angle'] if stored_params else 0)
                self.avoid_angle.setSingleStep(0.01)
                form_layout.addRow("避障拐弯角度(rads):", self.avoid_angle)
        
        layout.addLayout(form_layout)
        
        # Add performance notice
        notice_label = QLabel("💡 性能提示：关闭摄像头窗口可显著提高运行性能")
        notice_label.setStyleSheet("color: #666; font-size: 10px; margin: 10px 0;")
        layout.addWidget(notice_label)
        
        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #2196F3;
                border-radius: 3px;
            }
            QCheckBox {
                padding: 5px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

    def accept(self):
        # Store the current values before accepting
        if self.parent():
            stored_values = {
                'ld': self.ld_input.value(),
                'max_speed': self.max_speed_input.value(),
                'min_speed': self.min_speed_input.value(),
                'show_camera': self.show_camera_checkbox.isChecked()
            }
            
            if hasattr(self, 'stop_sign_input'):
                stored_values['stop_sign'] = self.stop_sign_input.value()
                
            if hasattr(self, 'avoid_angle'):
                stored_values['avoid_angle'] = self.avoid_angle.value()
                
            self.parent().stored_params[self.scenario_num] = stored_values
            
        super().accept()

class ProcessThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(self.cmd, cwd=BASE_DIR, env=os.environ.copy())
            self.process.wait()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        if self.process:
            try:
                # Send SIGTERM signal first for graceful shutdown
                self.process.send_signal(signal.SIGTERM)
                
                # Wait for up to 5 seconds for process to terminate
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # If process doesn't terminate within timeout, force kill it
                    self.process.kill()
                    self.process.wait()
                    
            except Exception as e:
                print(f"Error stopping process: {e}")
            finally:
                self.process = None

class CompetitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("竞赛控制面板")
        self.setGeometry(100, 100, 500, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 120px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2196F3;
                border-radius: 5px;
                margin-top: 10px;
                padding: 15px;
            }
        """)
        self.cmd = []
        
        # Initialize variables
        self.process_thread = None
        self.current_scenario_params = None
        # Initialize parameter storage for each scenario
        self.stored_params = {
            1: {'ld': 0.50, 'max_speed': 0.80, 'min_speed': 0.20, 'show_camera': False},
            2: {'ld': 0.50, 'max_speed': 0.80, 'min_speed': 0.20, 'stop_sign': 0.80, 'show_camera': False},
            3: {'ld': 0.50, 'max_speed': 0.80, 'min_speed': 0.20, 'stop_sign': 0.80, 'avoid_angle': 0, 'show_camera': False}
        }
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create scenarios group
        scenarios_group = QGroupBox("赛题选择")
        scenarios_layout = QHBoxLayout()
        
        # Create scenario buttons
        for i in range(1, 4):
            button = QPushButton(f"赛题 {i}")
            button.clicked.connect(lambda checked, x=i: self.show_scenario_dialog(x))
            scenarios_layout.addWidget(button)
        
        scenarios_group.setLayout(scenarios_layout)
        main_layout.addWidget(scenarios_group)
        
        # Create control buttons group
        control_group = QGroupBox("程序控制")
        control_layout = QHBoxLayout()
        
        # Create control buttons
        self.start_button = QPushButton("▶ Start")
        self.stop_button = QPushButton("⬛ Stop")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Connect buttons to their functions
        self.start_button.clicked.connect(self.start_autonomous)
        self.stop_button.clicked.connect(self.stop_autonomous)
        
        # Create penalty display
        self.create_penalty_display(main_layout)
        
        # Start penalty monitoring
        self.start_penalty_monitoring()
        
    def show_scenario_dialog(self, scenario_num:int):
        dialog = ParameterDialog(scenario_num, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.current_scenario_params = {
                'scenario': scenario_num,
                'ld': dialog.ld_input.value(),
                'max_speed': dialog.max_speed_input.value(),
                'min_speed': dialog.min_speed_input.value(),
                'show_camera': dialog.show_camera_checkbox.isChecked(),
                'stop_sign_threshold': dialog.stop_sign_input.value() if scenario_num == 2 or scenario_num == 3 else None,
                'avoid_angle': dialog.avoid_angle.value() if scenario_num == 3 else None
            }
            self.start_button.setEnabled(True)
            # 使用重构后的自动驾驶程序
            self.cmd = [sys.executable, 'Autonomous_Drive_New.py',
                '--scenario', str(self.current_scenario_params['scenario']),
                '--ld', str(self.current_scenario_params['ld']),
                '--max_speed', str(self.current_scenario_params['max_speed']),
                '--min_speed', str(self.current_scenario_params['min_speed'])]
            
            # 添加摄像头显示选项
            if self.current_scenario_params['show_camera']:
                self.cmd.append('--show_camera')
            
            # 根据场景添加额外参数
            if scenario_num >= 2:
                self.cmd.extend(['--stop_sign_threshold', str(self.current_scenario_params['stop_sign_threshold'])])
            if scenario_num >= 3:
                self.cmd.extend(['--avoid_angle', str(self.current_scenario_params['avoid_angle'])])
            
    def start_autonomous(self):
        if self.process_thread is None and self.current_scenario_params is not None:
            
            try:
                # 创建罚时通信文件
                import os
                if os.path.exists('penalty_messages.txt'):
                    os.remove('penalty_messages.txt')
                
                self.add_penalty_message("🚗 自动驾驶程序启动中...")
                
                # Start the autonomous drive process
                self.process_thread = ProcessThread(self.cmd)
                self.process_thread.finished.connect(self.on_process_finished)
                self.process_thread.error.connect(self.on_process_error)
                self.process_thread.start()
                
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                
                # 启动罚时监控
                self.start_penalty_monitoring()
                
            except Exception as e:
                print(f"Error starting autonomous drive: {e}")

    def stop_autonomous(self):
        if self.process_thread is not None:
            self.process_thread.stop()
            self.process_thread = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            # 停止罚时监控
            if hasattr(self, 'penalty_timer'):
                self.penalty_timer.stop()
                
            self.add_penalty_message("🛑 自动驾驶程序已停止")

    def on_process_finished(self):
        self.process_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # 停止罚时监控
        if hasattr(self, 'penalty_timer'):
            self.penalty_timer.stop()
            
        self.add_penalty_message("✅ 自动驾驶程序已完成")

    def on_process_error(self, error_msg):
        print(f"Process error: {error_msg}")
        self.process_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
    def create_penalty_display(self, main_layout):
        """创建罚时显示区域"""
        # 罚时显示组
        penalty_group = QGroupBox("罚时信息")
        penalty_layout = QVBoxLayout()
        
        # 时间显示行
        time_layout = QHBoxLayout()
        
        # 原始时间
        self.original_time_label = QLabel("⏱️ 原始时间\n0.00 秒")
        self.original_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_time_label.setStyleSheet("background-color: #2E7D32; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        time_layout.addWidget(self.original_time_label)
        
        # 总罚时
        self.penalty_time_label = QLabel("⚠️ 总罚时\n0.00 秒")
        self.penalty_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.penalty_time_label.setStyleSheet("background-color: #F57F17; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        time_layout.addWidget(self.penalty_time_label)
        
        # 最终时间
        self.final_time_label = QLabel("🏆 最终时间\n0.00 秒")
        self.final_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.final_time_label.setStyleSheet("background-color: #1976D2; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        time_layout.addWidget(self.final_time_label)
        
        penalty_layout.addLayout(time_layout)
        
        # 实时罚时详情
        penalty_details_label = QLabel("实时罚时详情:")
        penalty_details_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        penalty_layout.addWidget(penalty_details_label)
        
        # 罚时详情文本框
        from PyQt5.QtWidgets import QTextEdit
        self.penalty_details = QTextEdit()
        self.penalty_details.setReadOnly(True)
        self.penalty_details.setMaximumHeight(150)
        self.penalty_details.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        penalty_layout.addWidget(self.penalty_details)
        
        penalty_group.setLayout(penalty_layout)
        main_layout.addWidget(penalty_group)
        
        # 初始化罚时显示
        self.original_time = 0.0
        self.penalty_time = 0.0
        self.final_time = 0.0
        
        # 添加初始信息
        self.add_penalty_message("🎯 系统启动，等待比赛开始...")
        
        # 创建罚时系统实例
        self.penalty_system = None
        
    def add_penalty_message(self, message):
        """添加罚时信息"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.penalty_details.append(formatted_message)
        
        # 自动滚动到底部
        cursor = self.penalty_details.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.penalty_details.setTextCursor(cursor)
        
    def update_penalty_time(self, penalty_time):
        """更新罚时显示"""
        self.penalty_time = penalty_time
        self.penalty_time_label.setText(f"⚠️ 总罚时\n{penalty_time:.2f} 秒")
        
        # 根据罚时变化颜色
        if penalty_time == 0:
            self.penalty_time_label.setStyleSheet("background-color: #2E7D32; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        elif penalty_time < 10:
            self.penalty_time_label.setStyleSheet("background-color: #F57F17; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        elif penalty_time < 30:
            self.penalty_time_label.setStyleSheet("background-color: #FF9800; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        else:
            self.penalty_time_label.setStyleSheet("background-color: #F44336; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        
        self.update_final_time()
        
    def update_original_time(self, original_time):
        """更新原始时间"""
        self.original_time = original_time
        self.original_time_label.setText(f"⏱️ 原始时间\n{original_time:.2f} 秒")
        self.update_final_time()
        
    def update_final_time(self):
        """更新最终时间"""
        self.final_time = self.original_time + self.penalty_time
        self.final_time_label.setText(f"🏆 最终时间\n{self.final_time:.2f} 秒")
        
    def penalty_gui_callback(self, message, penalty_time, original_time=None):
        """罚时系统回调函数"""
        self.add_penalty_message(message)
        self.update_penalty_time(penalty_time)
        if original_time is not None:
            self.update_original_time(original_time)
    
    def start_penalty_monitoring(self):
        """启动罚时监控"""
        from PyQt5.QtCore import QTimer
        self.penalty_timer = QTimer()
        self.penalty_timer.timeout.connect(self.read_penalty_messages)
        self.penalty_timer.start(500)  # 每500ms检查一次
        
    def read_penalty_messages(self):
        """读取罚时信息文件"""
        try:
            import os
            if os.path.exists('penalty_messages.txt'):
                with open('penalty_messages.txt', 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # 清空文件
                with open('penalty_messages.txt', 'w', encoding='utf-8') as f:
                    f.write('')
                
                # 处理每一行
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            message = parts[0].strip()
                            penalty_time = float(parts[1].strip())
                            original_time = float(parts[2].strip()) if len(parts) > 2 else None
                            
                            self.penalty_gui_callback(message, penalty_time, original_time)
                        else:
                            # 简单消息
                            self.add_penalty_message(line)
                            
        except Exception as e:
            # 静默处理错误，避免干扰程序运行
            pass
            


class QCarRLEnv(gym.Env):
    # ... 其余代码不变 ...
    def reset(self, ld=None, max_speed=None, min_speed=None, stop_sign_threshold=None, avoid_angle=None):
        # 支持外部传参
        self.ld = ld if ld is not None else 0.5
        self.max_speed = max_speed if max_speed is not None else 0.8
        self.min_speed = min_speed if min_speed is not None else 0.2
        self.stop_sign_threshold = stop_sign_threshold if stop_sign_threshold is not None else 0.8
        self.avoid_angle = avoid_angle if avoid_angle is not None else 0.0
        # ... 其余reset逻辑 ...

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CompetitionGUI()
    window.show()
    sys.exit(app.exec())
