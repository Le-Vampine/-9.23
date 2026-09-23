@echo off
REM Real-time training progress monitor (terminal refresh + writes runs\progress.md)
REM usage: monitor.cmd         refresh every 15 s
REM        monitor.cmd 5       refresh every 5 s
set PYTHONUTF8=1
cd /d "%~dp0"
if "%1"=="" (set IV=15) else (set IV=%1)
python -u monitor_progress.py --interval %IV%
