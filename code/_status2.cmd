@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _status2.log del _status2.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Select-Object ProcessId,CreationDate,WorkingSetSize,CommandLine | Format-List" >> _status2.log 2>&1
echo === CPU 占用 === >> _status2.log
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,WS | Format-Table -AutoSize" >> _status2.log 2>&1
echo FINISHED >> _status2.log
