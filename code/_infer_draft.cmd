@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _infer_draft.log del _infer_draft.log
echo ===== 附件3（真实）推理链路验证 ===== >> _infer_draft.log
python -u 02_model\infer_att3.py --pkl "%~dp0..\data_att\att3_aligned.pkl" --ckpt_dir "%~dp0..\runs\timing" --out_csv "%~dp0..\submission\pred_att3_draft.csv" --device cpu >> _infer_draft.log 2>&1
echo ===== 附件4（真实）推理链路验证 ===== >> _infer_draft.log
python -u 02_model\infer_att3.py --pkl "%~dp0..\data_att\att4_aligned.pkl" --ckpt_dir "%~dp0..\runs\timing" --out_csv "%~dp0..\submission\pred_att4_draft.csv" --device cpu >> _infer_draft.log 2>&1
echo FINISHED >> _infer_draft.log
