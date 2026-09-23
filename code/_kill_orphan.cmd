@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _kill.log del _kill.log
echo === 终止残留的 seed43 进程 (PID 25940) === >> _kill.log
taskkill /F /PID 25940 >> _kill.log 2>&1
echo. >> _kill.log
echo === 剩余 python 进程 === >> _kill.log
tasklist /FI "IMAGENAME eq python.exe" >> _kill.log 2>&1
echo. >> _kill.log
echo === 训练日志末尾 === >> _kill.log
powershell -NoProfile -Command "Get-Content '_trainfinal.log' -Tail 6" >> _kill.log 2>&1
echo FINISHED >> _kill.log
