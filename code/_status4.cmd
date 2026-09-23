@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _status4.log del _status4.log
echo === 当前时间 === >> _status4.log
echo %DATE% %TIME% >> _status4.log
echo === 日志文件时间戳 === >> _status4.log
dir _trainv2.log _trainfinal.log 2>nul | findstr /R "trainv2 trainfinal" >> _status4.log
echo === python CPU 占用 === >> _status4.log
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,WS | Format-Table -AutoSize" >> _status4.log 2>&1
echo === _trainv2.log 末尾 8 行 === >> _status4.log
powershell -NoProfile -Command "Get-Content '_trainv2.log' -Tail 8" >> _status4.log 2>&1
echo FINISHED >> _status4.log
