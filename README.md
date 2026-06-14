# xxwl QLabs Self-Driving Project

这是 QLabs Self-Driving Car Studio 的课程实验代码导出版。仓库只保留可读源码、必要模型文件和关键验证记录，未上传本地运行环境和大型依赖目录。

## 仓库内容

- `Autonomous_Drive_New.py`：当前主入口。
- `StudentDecision.py`：当前默认控制策略，已采用 `speed075` 学习候选参数。
- `AutonomousDriveCore.py`：QLabs/QCar 初始化、传感器读取、车辆控制与检测核心。
- `PenaltySystem.py`：teacher 罚时逻辑。
- `Traffic_Lights_Competition.py`：红绿灯、行人、牛等场景对象控制。
- `qlabspretrained.pt`：运行 YOLO 检测所需的小模型文件，约 5.4 MB。
- `baselines/teacher_baseline_20260614_5pass_penalty3_raw25_30/`：满足 5 次连续通过要求的 baseline 快照与结果。
- `learning_runs/`：保留两次关键学习/验证记录。
- `run_archives/20260614_011907_fixed_green_5run_validation_reenter_fullwait/`：baseline 的 5 次连续验证日志。
- `agent.md`、`qlab_reenter_plane.md`：实验过程说明和 QLabs 退出重进 Plane 流程。

## 未上传内容

以下内容没有进入仓库：

- `quanser/`、`python/`、`qcar_test/`、`rtmodels/` 等大型本地依赖或运行环境。
- 大量历史 `scenario3*.log`、截图、窗口状态图、临时 `.pid`、`__pycache__/`、`.ultralytics/`。
- 非关键调参过程产物。

## 运行前提

本仓库不是完整离线环境，需要本机已经安装并能打开 Quanser Interactive Labs / QLabs，以及对应 Python 依赖。运行前请先在 QLabs 中进入：

`Self-Driving Car Studio -> Plane`

常用运行命令：

```powershell
python Autonomous_Drive_New.py --scenario 3
```

如果使用课程目录里的 Quanser Python 运行时，可以在本地恢复 `quanser/` 后执行：

```powershell
quanser\python.exe Autonomous_Drive_New.py --scenario 3
```

固定场景验证时使用的环境变量：

```powershell
$env:TRAFFIC_LIGHT_MODE = "fixed_green"
$env:TRAFFIC_ACTOR_MODE = "triggered"
python Autonomous_Drive_New.py --scenario 3
```

## 当前结果

Baseline 要求：

- 连续 5 次通过。
- 原始用时在 25 到 30 秒之间。
- 总罚时不超过 3 秒。
- 每次行人和牛均加载成功：`people=2 cow=1`。

Baseline 结果：

| Run | Raw time (s) | Penalty (s) |
| --- | ---: | ---: |
| 1 | 25.2341964244843 | 3 |
| 2 | 25.285905122757 | 3 |
| 3 | 25.35822057724 | 3 |
| 4 | 25.2149469852448 | 3 |
| 5 | 25.2907538414001 | 3 |

学习后默认策略验证：

- Raw time: `23.964003324508667s`
- Penalty: `3.0s`
- Final time: `26.964003324508667s`
- 相比 baseline 最优原始用时提升约 `1.2509s`
- Cone / pedestrian / cow / red-light collision: `0`

## 罚时逻辑摘要

- 普通压线/偏离使用车体中心点到 `path.txt` 中心线的距离。
- 轻微偏离阈值：`0.08m`。
- 左右侧较大偏离阈值：`0.12m`。
- 连续压线额外罚时触发间隔：`1.5s`。
- 锥桶碰撞虚拟框：`abs(car_x - cone_x) < 0.15` 且 `abs(car_y - cone_y) < 0.15`。

详细记录见 `baselines/`、`learning_runs/` 和 `run_archives/`。
