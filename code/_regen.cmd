@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _regen.log del _regen.log
echo === python processes (should be none) === >> _regen.log
tasklist /FI "IMAGENAME eq python.exe" >> _regen.log 2>&1
echo. >> _regen.log
echo === regenerate migration package === >> _regen.log
python -u make_migration_package.py >> _regen.log 2>&1
echo. >> _regen.log
echo === training artifacts === >> _regen.log
dir /b "%~dp0..\runs" >> _regen.log 2>&1
echo FINISHED >> _regen.log
