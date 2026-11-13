@echo off
setlocal
cd /d "%~dp0"

set "MODEL_DIR=D:\Shared\GGUF\Serveurperso\Qwen3-TTS-GGUF"
set "PYTHON=.venv\Scripts\python.exe"
set "REALTIMETTS_API_KEY="

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

echo.
echo First-time setup: run Setup-TTS.bat once; missing setup is also automatic.
echo Startup pipeline: reuse Sylvia's reference or generate it with 1.7B VoiceDesign,
echo then serve it with 0.6B Base Q4_K_M by default.
echo CPU worker pools: 6 generation + 6 codec threads; no core pinning.
echo Request limit: 1 active synthesis, no queued requests.
echo Pass --Q8 to serve with the 0.6B Base Q8 model instead.
echo VoiceDesign defaults to Q4_K_M; pass --bf16 to select BF16 assets.
echo Existing reference is reused; pass -new to archive it and generate another.
echo API authentication is disabled; restrict access with Windows Firewall.
echo Studio: http://127.0.0.1:2558/studio
echo LAN:    http://^<this-PC's-LAN-IP^>:2558/studio
where.exe tailscale >nul 2>nul
if not errorlevel 1 (
    echo Tailscale IP:
    tailscale ip -4
)
echo Keep this window open while the server is running.
echo.

"%PYTHON%" start_sylvia_server.py %*
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Startup failed. Check the message above.
pause
exit /b 1
