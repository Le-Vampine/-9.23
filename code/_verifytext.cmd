@echo off
set PYTHONUTF8=1
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=E:\hf_cache
cd /d "%~dp0"
if exist _verifytext.log del _verifytext.log
python -u verify_text_encoder.py --n 8 >> _verifytext.log 2>&1
echo FINISHED >> _verifytext.log
