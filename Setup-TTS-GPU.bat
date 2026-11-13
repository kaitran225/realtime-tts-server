@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv-gpu\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Creating Python 3.12 GPU virtual environment in .venv-gpu ...
    py -3.12 -m venv .venv-gpu
    if errorlevel 1 goto failed
)

"%PYTHON%" -m pip show realtimetts-qwen-native >nul 2>nul
if errorlevel 1 (
    echo Installing RealtimeTTS CUDA server runtime ^(~2 GB of wheels^) ...
    "%PYTHON%" -m pip install -U pip
    if errorlevel 1 goto failed
    "%PYTHON%" -m pip install "realtimetts[qwen-server]"
    if errorlevel 1 goto failed
)

"%PYTHON%" -m pip show realtimetts-qwen-native >nul 2>nul
if errorlevel 1 (
    echo RealtimeTTS CUDA runtime was not installed.
    goto failed
)

where.exe nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo WARNING: nvidia-smi not found; GPU mode will not start.
) else (
    echo NVIDIA GPU:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
)

echo.
echo Setup complete. Start the server with Start-TTS.bat and pick "gpu".
exit /b 0

:failed
echo.
echo GPU setup failed. Check the message above.
pause
exit /b 1
