@echo off
rem Shortcut: skip the picker and start the CUDA server type.
rem Accepts the same flags as Start-TTS.bat (--model, --list, --dry-run, -new, ...).
setlocal
cd /d "%~dp0"

set "PYTHON=.venv-gpu\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo .venv-gpu is missing; run Setup-TTS-GPU.bat first.
    pause
    exit /b 1
)

"%PYTHON%" start_sylvia_server.py --device gpu %*
if errorlevel 1 (
    echo.
    echo Startup failed. Check the message above and logs\sylvia-server-*.log
    pause
    exit /b 1
)
exit /b 0
