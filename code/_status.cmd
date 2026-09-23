@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _status.log del _status.log
echo === python 进程 === >> _status.log
tasklist /FI "IMAGENAME eq python.exe" >> _status.log 2>&1
echo. >> _status.log
echo === 训练日志文件信息 === >> _status.log
dir _trainfinal.log >> _status.log 2>&1
echo. >> _status.log
echo === 当前系统时间 === >> _status.log
echo %DATE% %TIME% >> _status.log
echo. >> _status.log
echo === 日志末尾 === >> _status.log
powershell -NoProfile -Command "Get-Content '_trainfinal.log' -Tail 10" >> _status.log 2>&1
echo. >> _status.log
echo === runs\q2 内容 === >> _status.log
dir /b "%~dp0..\runs\q2" >> _status.log 2>&1
echo FINISHED >> _status.log
