@echo off
REM Problem 3 interpretability head training (frozen backbone, parallel evidence head).
REM Args: %1 = epochs (default 12), %2 = alpha_f (default 1.0)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set PYEXE=D:\venv-mosei\Scripts\python.exe
if not exist "%PYEXE%" set PYEXE=python
set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=12
set ALPHA=%2
if "%ALPHA%"=="" set ALPHA=1.0
if exist _q3_head.log del _q3_head.log
"%PYEXE%" -u 03_xai\train_evidence_head.py --epochs %EPOCHS% --alpha_f %ALPHA% >> _q3_head.log 2>&1
echo FINISHED >> _q3_head.log
