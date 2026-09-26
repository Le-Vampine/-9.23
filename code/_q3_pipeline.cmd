@echo off
REM Problem 3 one-click pipeline: Shapley/IG/occlusion -> fuse -> metrics -> evidence
REM -> cards -> submission CSV -> valid analysis -> figures -> self-check.
REM Logs are written under runs\q3\logs\ (UTF-8, one file per step).
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
REM 优先使用项目 venv；若不存在则回退到 PATH 上的 python
set PYEXE=D:\venv-mosei\Scripts\python.exe
if not exist "%PYEXE%" set PYEXE=python
if exist _q3_pipeline.log del _q3_pipeline.log
"%PYEXE%" -u 03_xai\run_pipeline.py >> _q3_pipeline.log 2>&1
echo FINISHED >> _q3_pipeline.log
