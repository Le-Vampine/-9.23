@echo off
REM Stop the old training chain (cmd + python) then restart with the v2 regression loss.
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _trainv2.log del _trainv2.log
echo ==== stop old chain ==== >> _trainv2.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='cmd.exe'\" | Where-Object { $_.CommandLine -like '*_train_final*' } | ForEach-Object { Write-Output ('kill cmd ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >> _trainv2.log 2>&1
taskkill /F /IM python.exe >> _trainv2.log 2>&1
echo ==== start v2 training ==== >> _trainv2.log
echo ===== QUICK CHECK: 3 teacher epochs (compare with old loss 0.748/0.677/0.593) ===== >> _trainv2.log
python -u 02_model\train.py --version aligned --epochs_teacher 3 --epochs_student 1 --seed 42 --hidden 128 --batch_size 64 --out_dir "%~dp0..\runs\v2check" >> _trainv2.log 2>&1
echo ===== seed 42 ===== >> _trainv2.log
python -u 02_model\train.py --version aligned --epochs_teacher 20 --epochs_student 30 --seed 42 --hidden 128 --batch_size 64 --out_dir "%~dp0..\runs\q2" >> _trainv2.log 2>&1
echo ===== seed 43 ===== >> _trainv2.log
python -u 02_model\train.py --version aligned --epochs_teacher 20 --epochs_student 30 --seed 43 --hidden 128 --batch_size 64 --out_dir "%~dp0..\runs\q2" >> _trainv2.log 2>&1
echo ALL_TRAINING_FINISHED >> _trainv2.log
