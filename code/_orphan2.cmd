@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _orphan2.log del _orphan2.log
echo === before: python processes === >> _orphan2.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Select-Object ProcessId,CreationDate,CommandLine | Format-List" >> _orphan2.log 2>&1
echo === kill parent cmd chain (_train_v3) === >> _orphan2.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='cmd.exe'\" | Where-Object { $_.CommandLine -like '*_train_v3*' } | ForEach-Object { Write-Output ('kill cmd ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >> _orphan2.log 2>&1
echo === kill all python === >> _orphan2.log
taskkill /F /IM python.exe >> _orphan2.log 2>&1
echo === after: python processes (should be none) === >> _orphan2.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Select-Object ProcessId,CreationDate,CommandLine | Format-List" >> _orphan2.log 2>&1
echo. >> _orphan2.log
echo === after: cmd processes running _train === >> _orphan2.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='cmd.exe'\" | Where-Object { $_.CommandLine -like '*_train*' } | Select-Object ProcessId,CommandLine | Format-List" >> _orphan2.log 2>&1
echo FINISHED >> _orphan2.log
