@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _probe.log del _probe.log
python -u probe_att34.py >> _probe.log 2>&1
echo ===== HF 连通性测试 ===== >> _probe.log
python -u -c "import urllib.request as u; [print(s, '->', (lambda r: r.status)(u.urlopen(u.Request(s, method='HEAD'), timeout=12))) for s in ['https://huggingface.co','https://hf-mirror.com','https://pypi.tuna.tsinghua.edu.cn']]" >> _probe.log 2>&1
echo FINISHED >> _probe.log
