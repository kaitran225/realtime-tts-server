@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Creating Python 3.12 virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
)

if not exist ".venv\Scripts\realtimetts-qwen-server.exe" (
    echo Installing RealtimeTTS CPU server...
    "%PYTHON%" -m pip install "realtimetts[qwen-cpu-server]"
    if errorlevel 1 goto failed
)

if not exist ".venv\Scripts\realtimetts-qwen-server.exe" (
    echo RealtimeTTS server executable was not installed.
    goto failed
)

echo.
echo Setup complete. Start the server with Start-TTS.bat.
exit /b 0

:failed
echo.
echo Setup failed. Check the message above.
pause
exit /b 1
