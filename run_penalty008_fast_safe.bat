@echo off
setlocal
cd /d "%~dp0"

set "PENALTY_MINOR_DEVIATION_THRESHOLD=0.08"
set "PYTHONPATH=%CD%\python;%PYTHONPATH%"
set "QAL_DIR=%CD%"
set "RTMODELS_DIR=%CD%\rtmodels"
set "YOLO_CONFIG_DIR=%CD%\.ultralytics"

if not exist runs mkdir runs

echo Running fast_safe baseline with 0.08 penalty threshold...
echo QLAB must already be open and in or near the Plane scene.
echo.

".\quanser\python.exe" "rl\trace_residual_policy.py" --base_policy fast_safe --model zero --auto_reenter_plane --out "runs\trace_penalty008_latest.csv"

echo.
echo Done. Trace saved to runs\trace_penalty008_latest.csv
pause
