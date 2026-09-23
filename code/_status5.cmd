@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if exist _status5.log del _status5.log
echo === 当前时间 === >> _status5.log
echo %DATE% %TIME% >> _status5.log
echo === 日志时间戳 === >> _status5.log
dir /O-D _train*.log 2>nul | findstr /R "train" >> _status5.log
echo === python 进程 === >> _status5.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Select-Object ProcessId,CreationDate,CommandLine | Format-List" >> _status5.log 2>&1
echo === _trainv3.log 全文（去掉数据核查部分） === >> _status5.log
powershell -NoProfile -Command "Get-Content '_trainv3.log' | Select-String -Pattern 'ep[0-9]|QUICK|SEED|best|early|TH\]|VALID|TEST|SAVE|ALL_TRAINING|Error|Traceback' | Select-Object -Last 60" >> _status5.log 2>&1
echo FINISHED >> _status5.log
