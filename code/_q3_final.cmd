@echo off
REM Problem 3 finalization: problem-2 attachment-3 report refresh (D1=A), interpretability
REM head training, paper section fill-in, handover package, self-check.
REM Log: code\_q3_final.log  (UTF-8 content written by python; read with Get-Content -Encoding UTF8)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
REM 优先使用项目 venv；若不存在则回退到 PATH 上的 python
set PYEXE=D:\venv-mosei\Scripts\python.exe
if not exist "%PYEXE%" set PYEXE=python
if exist _q3_final.log del _q3_final.log
echo === 1. att3 report and figures (D1=A refresh) === >> _q3_final.log
"%PYEXE%" -u 02_model\att3_predict_report.py >> _q3_final.log 2>&1
echo === 2. evidence head training (frozen backbone) === >> _q3_final.log
"%PYEXE%" -u 03_xai\train_evidence_head.py --epochs 12 >> _q3_final.log 2>&1
echo === 3. fill paper section === >> _q3_final.log
"%PYEXE%" -u 03_xai\fill_paper_section.py --apply >> _q3_final.log 2>&1
echo === 4. handover package === >> _q3_final.log
"%PYEXE%" -u 03_xai\make_handover.py >> _q3_final.log 2>&1
echo === 5. self check === >> _q3_final.log
"%PYEXE%" -u 03_xai\verify_q3.py >> _q3_final.log 2>&1
echo ALL_DONE >> _q3_final.log
