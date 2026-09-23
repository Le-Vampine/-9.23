@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _diag.log del _diag.log
python -u diagnose_nan.py >> _diag.log 2>&1
echo FINISHED >> _diag.log
