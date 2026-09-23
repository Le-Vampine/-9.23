@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _err2.log del _err2.log
echo ===== 错误归因（修正 theta 后重跑） ===== >> _err2.log
python -u 02_model\error_analysis.py --ckpt_dir "%~dp0..\runs\q2" --out_dir "%~dp0..\runs\q2" --device cpu >> _err2.log 2>&1
echo. >> _err2.log
echo ===== v3 训练进度 ===== >> _err2.log
echo %DATE% %TIME% >> _err2.log
powershell -NoProfile -Command "Get-Content '_trainv3.log' | Select-String -Pattern 'ep[0-9]|SEED|best|early|VALID|TEST|SAVE|ALL_TRAINING' | Select-Object -Last 12" >> _err2.log 2>&1
echo FINISHED >> _err2.log
