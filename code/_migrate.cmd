@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _migrate.log del _migrate.log
python -u -m py_compile make_migration_package.py >> _migrate.log 2>&1
if errorlevel 1 (echo COMPILE_FAIL >> _migrate.log) else (echo COMPILE_OK >> _migrate.log)
python -u make_migration_package.py >> _migrate.log 2>&1
echo === 生成的交接包内容 === >> _migrate.log
dir /b "%~dp0..\迁移交接" >> _migrate.log 2>&1
echo FINISHED >> _migrate.log
