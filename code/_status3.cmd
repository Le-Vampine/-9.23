@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _status3.log del _status3.log
echo === python 进程 === >> _status3.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Select-Object ProcessId,CreationDate,CommandLine | Format-List" >> _status3.log 2>&1
echo. >> _status3.log
echo === _trainv2.log 末尾 === >> _status3.log
powershell -NoProfile -Command "Get-Content '_trainv2.log' -Tail 20" >> _status3.log 2>&1
echo. >> _status3.log
echo === _trainfinal.log 末尾（旧链，应已停止） === >> _status3.log
powershell -NoProfile -Command "Get-Content '_trainfinal.log' -Tail 4" >> _status3.log 2>&1
echo FINISHED >> _status3.log
