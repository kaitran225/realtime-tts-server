@echo off
setlocal
cd /d "%~dp0"

rem Single entry point: asks for the server type (cpu / gpu), then the model to serve.
rem Server types: cpu -> .venv (realtimetts-qwen-native-cpu), gpu -> .venv-gpu (CUDA).
rem Run Setup-TTS-GPU.bat once to create the GPU environment.

set "REALTIMETTS_API_KEY="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else if exist ".venv-gpu\Scripts\python.exe" (
    set "PYTHON=.venv-gpu\Scripts\python.exe"
) else (
    set "PYTHON=.venv\Scripts\python.exe"
)

if not exist "%PYTHON%" (
    echo Creating Python 3.12 CPU virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
    set "PYTHON=.venv\Scripts\python.exe"
    echo Installing RealtimeTTS CPU server...
    "%PYTHON%" -m pip install "realtimetts[qwen-cpu-server]"
    if errorlevel 1 goto failed
)

echo.
echo Launcher: %PYTHON%
echo Models:   .config\models.json  (F:\LLM\Serveurperso\Qwen3-TTS-GGUF)
echo Logs:     logs\sylvia-server-YYYYMMDD.log
echo.
echo You will be asked for the server type, then the model to serve.
echo Flags: --list catalog, --model ^<id^>, --device cpu^|gpu, --dry-run, --yes, -new
echo CPU worker pools: 6 generation + 6 codec threads. Request limit: 1 active, none queued.
echo GPU mode needs Setup-TTS-GPU.bat once; it is offered automatically when installed.
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
echo Startup failed. Check the message above and logs\sylvia-server-*.log
pause
exit /b 1
