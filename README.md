# QIMODAZUOYE Teacher - 0.08 罚时口径稳定版

这个包是当前调好的 `qimodazuoye_teacher` 可运行版本。目标口径：

- 罚时系统阈值使用 `PENALTY_MINOR_DEVIATION_THRESHOLD=0.08`
- 锥桶后的不可避免压线允许 3 秒罚时
- 行人、牛、上方行人后到最后直道入口不应再产生 0.08 口径罚时
- 最近验证结果：连续 3 个有效样本均为 `penalty=3.0s`

## 运行前准备

1. 打开 QLAB，但不要重启 QLAB。
2. 进入 `Self-Driving Car Studio -> Plane` 场景。
3. 不要移动或缩放 QLAB 窗口。
4. 确认本目录下存在 `quanser\python.exe` 和 `qlabspretrained.pt`。

如果场景卡住或行人/牛没有加载，运行脚本会尝试执行“退出 Plane 再重新进入 Plane”的流程，但不会重启 QLAB。

## 推荐运行方法

双击：

```bat
run_penalty008_fast_safe.bat
```

或者在 PowerShell 中运行：

```powershell
cd /d 解压后的\qimodazuoye
$env:PENALTY_MINOR_DEVIATION_THRESHOLD = "0.08"
$env:PYTHONPATH = "$PWD\python;$env:PYTHONPATH"
.\quanser\python.exe rl\trace_residual_policy.py --base_policy fast_safe --model zero --auto_reenter_plane --out runs\trace_penalty008_latest.csv
```

注意：PowerShell 里不要使用带空格的反引号换行；推荐直接使用上面的一整行命令。

## 结果查看

运行结束后，终端会输出类似：

```text
final: raw=28.xx penalty=3.0 final=31.xx
```

生成的轨迹文件在：

```text
runs\trace_penalty008_latest.csv
```

可以分析轨迹：

```powershell
.\quanser\python.exe rl\analyze_trace.py runs\trace_penalty008_latest.csv
```

## 已验证样本

当前版本的有效验证轨迹：

- `runs\trace_fast_safe_penalty008_cowexit_soft_20260615_180719.csv`
- `runs\trace_fast_safe_penalty008_cowexit_soft_stability2_20260615_180851.csv`
- `runs\trace_fast_safe_penalty008_cowexit_soft_stability3_20260615_181222.csv`

三次结果都只有锥桶区 3 秒罚时；行人、牛、最后直道入口没有 0.08 口径罚时。

## 其他入口

GUI 入口仍然保留：

```bat
start.bat
```

但本版本最终验证使用的是 `rl\trace_residual_policy.py --base_policy fast_safe --model zero`，建议比赛/复现实验优先使用推荐运行方法。
