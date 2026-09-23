@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _att.log del _att.log
echo ===== 附件路径发现 ===== >> _att.log
python -u att_paths.py >> _att.log 2>&1
echo ===== 附件3 对齐版本 结构探查 ===== >> _att.log
python -u inspect_att34.py --kind att3 --version aligned >> _att.log 2>&1
echo FINISHED >> _att.log
