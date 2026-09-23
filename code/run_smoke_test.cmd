@echo off
REM One-click smoke test: synth data -> tiny train (teacher+student+KD) -> att3 inference
REM Uses relative paths only, so no non-ASCII literal appears in this file.
set PYTHONUTF8=1
cd /d "%~dp0"

echo [1/3] generate synthetic data
python -u make_dummy_data.py --out_dir "%~dp0..\data_dummy" --version aligned --n_train 256 --n_valid 128 --n_test 128 > runs_smoke.log 2>&1

echo [2/3] tiny train with KD (override data_dir to synthetic folder)
python -u 02_model\train.py --data_dir "%~dp0..\data_dummy" --limit 192 --epochs_teacher 10 --epochs_student 12 --seed 42 --hidden 128 --batch_size 64 --out_dir "%~dp0..\runs\smoke" --device cpu >> runs_smoke.log 2>&1

echo [3/3] att3 inference
python -u 02_model\infer_att3.py --pkl "%~dp0..\data_dummy\att3_like.pkl" --ckpt_dir "%~dp0..\runs\smoke" --out_csv "%~dp0..\submission\pred_att3_smoke.csv" --batch_size 64 --device cpu >> runs_smoke.log 2>&1

echo SMOKE TEST FINISHED >> runs_smoke.log
