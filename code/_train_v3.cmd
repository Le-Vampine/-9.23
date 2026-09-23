@echo off
REM Stop the v2 chain, then run the regularized (v3) training into runs\q2v3.
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _trainv3.log del _trainv3.log
echo ==== stop v2 chain ==== >> _trainv3.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='cmd.exe'\" | Where-Object { $_.CommandLine -like '*_train_v2*' } | ForEach-Object { Write-Output ('kill cmd ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >> _trainv3.log 2>&1
taskkill /F /IM python.exe >> _trainv3.log 2>&1
echo ==== v3 hyperparams: dropout .35 wd 5e-4 lr 5e-5/5e-4 noise .15 ls .1 ==== >> _trainv3.log

echo ===== QUICK CHECK v3: 4 teacher epochs (v2 ref: 0.7745/0.6534/0.5926/0.6186) ===== >> _trainv3.log
python -u 02_model\train.py --version aligned --epochs_teacher 4 --epochs_student 1 --seed 42 --hidden 128 --batch_size 64 --dropout 0.35 --weight_decay 5e-4 --lr_backbone 5e-5 --lr_head 5e-4 --noise_sigma 0.15 --label_smoothing 0.1 --out_dir "%~dp0..\runs\v3check" >> _trainv3.log 2>&1

echo ===== seed 42 (v3) ===== >> _trainv3.log
python -u 02_model\train.py --version aligned --epochs_teacher 20 --epochs_student 30 --seed 42 --hidden 128 --batch_size 64 --dropout 0.35 --weight_decay 5e-4 --lr_backbone 5e-5 --lr_head 5e-4 --noise_sigma 0.15 --label_smoothing 0.1 --out_dir "%~dp0..\runs\q2v3" >> _trainv3.log 2>&1
echo ===== seed 43 (v3) ===== >> _trainv3.log
python -u 02_model\train.py --version aligned --epochs_teacher 20 --epochs_student 30 --seed 43 --hidden 128 --batch_size 64 --dropout 0.35 --weight_decay 5e-4 --lr_backbone 5e-5 --lr_head 5e-4 --noise_sigma 0.15 --label_smoothing 0.1 --out_dir "%~dp0..\runs\q2v3" >> _trainv3.log 2>&1
echo ALL_TRAINING_FINISHED >> _trainv3.log
