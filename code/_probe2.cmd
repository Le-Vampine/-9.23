@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _probe2.log del _probe2.log
python -u probe_text.py >> _probe2.log 2>&1
echo FINISHED >> _probe2.log
