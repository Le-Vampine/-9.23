@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _killfinal.log del _killfinal.log
echo === before: python processes === >> _killfinal.log
tasklist /FI "IMAGENAME eq python.exe" >> _killfinal.log 2>&1
echo. >> _killfinal.log
echo === killing suspended training PID 18088 === >> _killfinal.log
taskkill /F /PID 18088 >> _killfinal.log 2>&1
echo. >> _killfinal.log
echo === after: python processes === >> _killfinal.log
tasklist /FI "IMAGENAME eq python.exe" >> _killfinal.log 2>&1
echo. >> _killfinal.log
echo === checkpoints on disk === >> _killfinal.log
dir /b "%~dp0..\runs\q2" >> _killfinal.log 2>&1
echo --- q2v3 --- >> _killfinal.log
dir /b "%~dp0..\runs\q2v3" >> _killfinal.log 2>&1
echo --- v3check --- >> _killfinal.log
dir /b "%~dp0..\runs\v3check" >> _killfinal.log 2>&1
echo. >> _killfinal.log
echo === deliverables === >> _killfinal.log
dir /b "%~dp0..\submission" >> _killfinal.log 2>&1
echo FINISHED >> _killfinal.log
