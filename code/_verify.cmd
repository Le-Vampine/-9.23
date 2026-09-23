@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _verify.log del _verify.log
python -u -m py_compile 02_model\config.py 02_model\data_utils.py 02_model\missing_sim.py 02_model\model.py 02_model\losses.py 02_model\train.py 02_model\infer_att3.py 02_model\analyze_sweep.py inspect_data.py make_dummy_data.py >> _verify.log 2>&1
if errorlevel 1 (echo COMPILE_FAIL >> _verify.log) else (echo COMPILE_OK >> _verify.log)
del /q _tidy.cmd _tidy.done 2>nul
echo DONE >> _verify.log
