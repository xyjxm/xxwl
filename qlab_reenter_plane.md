# QLAB 退出并重新进入 Plane 流程

这个流程用于 QLAB 场景卡住、行人/牛没有刷新、或者需要重新加载 Plane 场景时使用。

核心原则：

- 禁止重启 QLAB。
- 不移动、不缩放 QLAB 窗口。
- 以屏幕上实际看到的按钮为准，人工退出并重新进入 Plane。
- 不使用旧的固定坐标 Python 脚本。
- 旧的 `reset_qlab_plane.py` 已删除，不要重新创建或调用。
- 每次跑车前必须确认行人和牛加载成功。
- 每次跑车后必须等一圈结束，或者先清理残留 Python 控制进程，再开始下一次。

## 1. 先清理残留控制进程

如果上一圈卡住，先只清理本项目启动的 Python 控制进程，不关闭 QLAB。

检查：

```powershell
Get-Process python,pythonw -ErrorAction SilentlyContinue |
    Select-Object Id,ProcessName,Path,StartTime
```

如果看到 `qimodazuoye_teacher\qimodazuoye\quanser\python.exe` 残留，可以停止这些 Python 进程。不要停止 `QLabs-Win64-Shipping`。

## 2. 确认 QLAB 当前窗口

不要移动或缩放 QLAB 窗口。只确认 QLAB 窗口仍在屏幕上，并且当前显示的是比赛场景或 QLAB 菜单。

Windows 缩放会导致普通截图裁掉 QLAB 右侧按钮，所以不要仅凭普通截图判断按钮被遮挡。如果必须截图诊断，截图、窗口坐标、点击坐标都应先调用 DPI-aware。

确认 QLAB 窗口矩形时应看到类似：

```text
Window : L=651 T=69 R=1871 B=793 W=1220 H=724
Client : Origin=661,108 W=1200 H=675
```

如果不用 DPI-aware，可能会误判为右上角按钮被挡住。

## 3. 从当前场景退出到主菜单

在 QLAB 当前场景右上角点击退出图标。

注意：

- 这是退出当前场景，不是退出 QLAB 程序。
- 点击后应进入 QLAB 主菜单。
- 主菜单中能看到 `Self-Driving Car Studio` 卡片。

## 4. 重新进入 Plane

按下面顺序点击：

1. 主菜单中的 `Self-Driving Car Studio` 卡片。
2. 自驾场景选择页右侧的红色箭头，向右翻到 `Plane`。
3. `Plane` 卡片。

进入成功后，QLAB 应显示空白网格地面。此时还没有赛道、车、行人、牛，这正常；这些由项目启动脚本重新生成。

## 5. 启动项目并确认动态物体加载

进入 Plane 后再运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\quanser\python.exe -u .\Autonomous_Drive_New.py --scenario 3
```

启动后必须看到类似：

```text
READY: required actors loaded; people=2 cow=1
```

只有看到 `people=2 cow=1` 后，才认为行人和牛已经加载成功，可以继续这一圈。

## 6. 跑圈结束后检查

必须等程序输出 `LAP COMPLETE` 或 `LAP TIME` 后再开始下一次。

如果程序长时间卡住，检查：

```powershell
Get-Content .\run_status.txt -Tail 80
Get-Content .\traffic_status.txt -Tail 80
Get-Process python,pythonw -ErrorAction SilentlyContinue |
    Select-Object Id,ProcessName,Path,StartTime
```

如果 `run_status.txt` 中车辆位置长时间不变，说明这圈已经卡住。此时清理残留 Python 进程，然后按本文流程重新进入 Plane。

## 7. 常见错误

- 不要用普通主屏截图判断 QLAB 按钮是否被遮挡；在 DPI 缩放下截图可能被裁剪。
- 不要移动或缩放 QLAB 窗口来修正截图问题。
- 不要重启 QLAB。
- 不要在上一圈还没结束时再次启动项目。
- 不要只看罚时日志；如果 QLAB 视觉上撞行人、撞牛、撞锥桶或明显压线，这圈仍然无效。
