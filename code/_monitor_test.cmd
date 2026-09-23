@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _monitor_test.log del _monitor_test.log
python -u monitor_progress.py --once --seeds 2 >> _monitor_test.log 2>&1
echo FINISHED >> _monitor_test.log
