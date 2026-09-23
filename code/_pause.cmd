@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _pause.log del _pause.log
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pause_training.ps1" >> _pause.log 2>&1
echo FINISHED >> _pause.log
