@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _val.log del _val.log
echo ===== viz_results ===== >> _val.log
python -u 02_model\viz_results.py --ckpt_dir "%~dp0..\runs\smoke" --out_dir "%~dp0..\figs" --data_dir "%~dp0..\data_dummy" --tag smoke --device cpu >> _val.log 2>&1
echo ===== error_analysis ===== >> _val.log
python -u 02_model\error_analysis.py --ckpt_dir "%~dp0..\runs\smoke" --out_dir "%~dp0..\runs\smoke" --data_dir "%~dp0..\data_dummy" --device cpu >> _val.log 2>&1
echo ===== calibrate_missing ===== >> _val.log
python -u 02_model\calibrate_missing.py --pkl "%~dp0..\data_dummy\att3_like.pkl" --out_dir "%~dp0..\runs\smoke" >> _val.log 2>&1
echo ===== tune list ===== >> _val.log
python -u 02_model\tune_hyperparams.py --list >> _val.log 2>&1
echo FINISHED >> _val.log
