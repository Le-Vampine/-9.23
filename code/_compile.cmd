@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _compile.log del _compile.log
python -u -m py_compile 02_model\runtime.py 02_model\calibrate_missing.py 02_model\viz_results.py 02_model\error_analysis.py 02_model\tune_hyperparams.py 02_model\train.py 02_model\missing_sim.py 02_model\config.py 02_model\infer_att3.py 02_model\data_utils.py 02_model\losses.py 02_model\model.py 02_model\analyze_sweep.py check_deliverables.py inspect_data.py inspect_label.py >> _compile.log 2>&1
if errorlevel 1 (echo COMPILE_FAIL >> _compile.log) else (echo COMPILE_ALL_OK >> _compile.log)
python -u 02_model\tune_hyperparams.py --list >> _compile.log 2>&1
echo FINISHED >> _compile.log
