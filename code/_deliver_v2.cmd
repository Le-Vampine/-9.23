@echo off
REM Produce the first complete deliverable set using the finished v2 model (runs\q2).
set PYTHONUTF8=1
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=E:\hf_cache
cd /d "%~dp0"
if exist _deliver.log del _deliver.log
echo ===== 1. 附件3 正式推理 ===== >> _deliver.log
python -u 02_model\infer_att3.py --pkl "%~dp0..\data_att\att3_aligned.pkl" --ckpt_dir "%~dp0..\runs\q2" --out_csv "%~dp0..\submission\pred_att3.csv" --device cpu >> _deliver.log 2>&1
echo ===== 2. 附件4 正式推理 ===== >> _deliver.log
python -u 02_model\infer_att3.py --pkl "%~dp0..\data_att\att4_aligned.pkl" --ckpt_dir "%~dp0..\runs\q2" --out_csv "%~dp0..\submission\pred_att4.csv" --device cpu >> _deliver.log 2>&1
echo ===== 3. 论文图 ===== >> _deliver.log
python -u 02_model\viz_results.py --ckpt_dir "%~dp0..\runs\q2" --out_dir "%~dp0..\figs" --tag q2 --device cpu >> _deliver.log 2>&1
echo ===== 4. 错误归因 ===== >> _deliver.log
python -u 02_model\error_analysis.py --ckpt_dir "%~dp0..\runs\q2" --out_dir "%~dp0..\runs\q2" --device cpu >> _deliver.log 2>&1
echo ===== 5. 论文表格 ===== >> _deliver.log
python -u 02_model\make_tables.py >> _deliver.log 2>&1
echo FINISHED >> _deliver.log
