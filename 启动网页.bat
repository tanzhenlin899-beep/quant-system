@echo off
rem ============================================================
rem  Quant Web Launcher  -  double click to start the web UI
rem  URL: http://127.0.0.1:8000
rem ============================================================
title Quant Web 8000
cd /d "%~dp0"

echo ============================================================
echo   Multi-Factor Quant System  -  Web UI
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found in this folder.
  echo         Run first:  py -3.11 -m venv .venv
  echo.
  pause
  exit /b 1
)

if not exist "web\app.py" (
  echo [ERROR] web\app.py not found. Keep this file inside D:\quant
  echo.
  pause
  exit /b 1
)

echo Starting server, browser will open in 5 seconds...
echo If it does not open automatically, visit: http://127.0.0.1:8000
echo.
echo Close this window to stop the server.
echo ------------------------------------------------------------

start "" /min cmd /c "timeout /t 5 >nul && start http://127.0.0.1:8000"

".venv\Scripts\python.exe" -m uvicorn web.app:app --host 127.0.0.1 --port 8000

echo.
echo ------------------------------------------------------------
echo Server stopped. Press any key to close.
pause >nul
