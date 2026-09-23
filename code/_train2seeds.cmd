@echo off
REM Full-scale training: 2 seeds, teacher 20 epochs + student 30 epochs (CPU, ~4 h total)
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _train2seeds.log del _train2seeds.log
echo ===== seed 42 ===== >> _train2seeds.log
python -u 02_model\train.py --version aligned --epochs_teacher 20 --epochs_student 30 --seed 42 --hidden 128 --batch_size 64 --out_dir "%~dp0..\runs\q2" >> _train2seeds.log 2>&1
echo ===== seed 43 ===== >> _train2seeds.log
python -u 02_model\train.py --version aligned --epochs_teacher 20 --epochs_student 30 --seed 43 --hidden 128 --batch_size 64 --out_dir "%~dp0..\runs\q2" >> _train2seeds.log 2>&1
echo ALL_TRAINING_FINISHED >> _train2seeds.log
