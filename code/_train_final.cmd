@echo off
REM Full-scale training (rerun): 2 seeds, teacher 20 + student 30 epochs
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _trainfinal.log del _trainfinal.log
echo ===== seed 42 ===== >> _trainfinal.log
python -u 02_model\train.py --version aligned --epochs_teacher 20 --epochs_student 30 --seed 42 --hidden 128 --batch_size 64 --out_dir "%~dp0..\runs\q2" >> _trainfinal.log 2>&1
echo ===== seed 43 ===== >> _trainfinal.log
python -u 02_model\train.py --version aligned --epochs_teacher 20 --epochs_student 30 --seed 43 --hidden 128 --batch_size 64 --out_dir "%~dp0..\runs\q2" >> _trainfinal.log 2>&1
echo ALL_TRAINING_FINISHED >> _trainfinal.log
