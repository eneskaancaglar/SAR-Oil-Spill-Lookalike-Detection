@echo off
chcp 65001 >nul
setlocal
title SAR Oil Spill Scanner
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment was not found.
    pause
    exit /b 1
)

if not exist "app_final.py" (
    echo ERROR: app_final.py was not found.
    pause
    exit /b 1
)

echo Starting SAR Oil Spill Scanner...
echo Browser address: http://localhost:8504
echo Keep this window open while using the application.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://localhost:8504'"
".venv\Scripts\python.exe" -m streamlit run "app_final.py" --server.port 8504 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false

echo Application stopped.
pause
endlocal
