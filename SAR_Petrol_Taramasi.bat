@echo off
title SAR Oil Spill Detection
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment not found.
    echo Expected: .venv\Scripts\python.exe
    pause
    exit /b 1
)

if not exist "app_final.py" (
    echo app_final.py not found.
    pause
    exit /b 1
)

echo Starting SAR Oil Spill Detection...
echo.
echo Browser address: http://localhost:8504
echo Do not close this window while using the application.
echo.

start "" "http://localhost:8504"

".venv\Scripts\python.exe" -m streamlit run "app_final.py" --server.port 8504 --server.address 127.0.0.1 --browser.gatherUsageStats false

echo.
echo Application stopped.
pause
