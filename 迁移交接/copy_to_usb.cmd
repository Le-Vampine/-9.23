@echo off
REM Wrapper: run the Python copy script (avoids cmd.exe encoding issues with CJK paths)
set PYTHONUTF8=1
cd /d "%~dp0"
python -u copy_to_usb.py %*
pause