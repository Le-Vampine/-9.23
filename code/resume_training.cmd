@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _resume.log del _resume.log
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0resume_training.ps1" >> _resume.log 2>&1
echo FINISHED >> _resume.log
