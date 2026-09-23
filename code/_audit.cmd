@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _audit.log del _audit.log
python -u check_deliverables.py >> _audit.log 2>&1
python -u 02_model\tune_hyperparams.py --list >> _audit.log 2>&1
echo FINISHED >> _audit.log
