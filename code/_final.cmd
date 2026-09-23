@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _final.log del _final.log
echo === 1. python 进程（应已全部退出） === >> _final.log
tasklist /FI "IMAGENAME eq python.exe" >> _final.log 2>&1
echo. >> _final.log
echo === 2. 重新生成迁移包 === >> _final.log
python -u make_migration_package.py >> _final.log 2>&1
echo. >> _final.log
echo === 3. 交接包内容 === >> _final.log
dir /b "%~dp0..\迁移交接" >> _final.log 2>&1
echo. >> _final.log
echo === 4. 交付物自检 === >> _final.log
python -u check_deliverables.py >> _final.log 2>&1
echo FINISHED >> _final.log
