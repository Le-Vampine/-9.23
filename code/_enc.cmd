@echo off
set PYTHONUTF8=1
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=E:\hf_cache
cd /d "%~dp0"
if exist _enc.log del _enc.log
python -u encode_att_text.py --kind att3 --version aligned >> _enc.log 2>&1
python -u encode_att_text.py --kind att4 --version aligned >> _enc.log 2>&1
echo ===== calibrate att3 ===== >> _enc.log
python -u 02_model\calibrate_missing.py --pkl "%~dp0..\data_att\att3_aligned.pkl" --out_dir "%~dp0..\runs\q2" >> _enc.log 2>&1
echo FINISHED >> _enc.log
