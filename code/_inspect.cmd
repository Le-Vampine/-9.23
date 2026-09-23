@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _inspect.log del _inspect.log
python -u inspect_data.py --version aligned --save "%~dp0..\runs\data_report_real.json" >> _inspect.log 2>&1
echo FINISHED >> _inspect.log
