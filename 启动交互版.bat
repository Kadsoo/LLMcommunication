@echo off
rem One-click launcher for the interactive Gradio demo (demo/app.py).
rem Usage: double-click this file. Close the window to stop the server.
rem NOTE: keep this file pure ASCII so it parses under any console codepage.
cd /d "%~dp0"
set "PORT=7860"
rem Reserved for Phase 2 (live model test in the page). No effect for now.
rem set "LLM_MODEL=Qwen/Qwen3-0.6B"

echo [1/4] Checking Python ...
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo [FAIL] Python 3.10+ not found. Reinstall Python with "Add to PATH" checked.
  pause
  exit /b 1
)
echo     Using %PY%

echo [2/4] Checking Gradio (auto-install if missing; torch untouched) ...
"%PY%" -c "import gradio" >nul 2>&1
if errorlevel 1 (
  echo     gradio not found, running: pip install gradio
  "%PY%" -m pip install gradio
  if errorlevel 1 (
    echo [FAIL] pip install gradio failed. Check network and retry.
    pause
    exit /b 1
  )
)

echo [3/4] Checking port %PORT% ...
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo [FAIL] Port %PORT% is busy. Close the old demo/app.py first.
  pause
  exit /b 1
)

echo [4/4] Starting demo at http://127.0.0.1:%PORT% ...
echo     Browser opens in 5s. First start takes 10-20s; refresh if not loaded yet.
echo     Close this window to stop the server.
start /b cmd /c "ping -n 6 127.0.0.1 >nul & start """" http://127.0.0.1:%PORT%"
"%PY%" demo\app.py
pause
