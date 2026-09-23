@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _metrics_check.log del _metrics_check.log
python -u -m py_compile 02_model\losses.py 02_model\make_tables.py >> _metrics_check.log 2>&1
if errorlevel 1 (echo COMPILE_FAIL >> _metrics_check.log) else (echo COMPILE_OK >> _metrics_check.log)
python -u _metrics_check.py >> _metrics_check.log 2>&1
echo. >> _metrics_check.log
echo === 当前训练进度 === >> _metrics_check.log
powershell -NoProfile -Command "Get-Content '_trainv3.log' | Select-String -Pattern 'ep[0-9]|best valid|early|VALID|TEST|ALL_TRAINING' | Select-Object -Last 8" >> _metrics_check.log 2>&1
echo FINISHED >> _metrics_check.log
