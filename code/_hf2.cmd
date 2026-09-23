@echo off
set PYTHONUTF8=1
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=E:\hf_cache
cd /d "%~dp0"
if exist _hf2.log del _hf2.log
python -u download_bert.py >> _hf2.log 2>&1
echo FINISHED >> _hf2.log
