@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _frozen.log del _frozen.log
echo === 第 1 次采样 === >> _frozen.log
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { 'PID {0}  CPU(s)={1:N1}' -f $_.Id, $_.CPU }" >> _frozen.log 2>&1
powershell -NoProfile -Command "Start-Sleep -Seconds 12" >> _frozen.log 2>&1
echo === 12 秒后第 2 次采样 === >> _frozen.log
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { 'PID {0}  CPU(s)={1:N1}' -f $_.Id, $_.CPU }" >> _frozen.log 2>&1
echo === 训练日志最后写入时间 === >> _frozen.log
powershell -NoProfile -Command "Get-ChildItem '_trainv3.log' | ForEach-Object { $_.LastWriteTime }" >> _frozen.log 2>&1
echo FINISHED >> _frozen.log
