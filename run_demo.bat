@echo off
title SAR Oil Spill Detector v0.3
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo HATA: .venv Python bulunamadi.
    echo Proje klasorundeki sanal ortam eksik.
    echo.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo.
    echo HATA: app.py bulunamadi.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo SAR Petrol Sizintisi Tespit Sistemi v0.3
echo ============================================================
echo.
echo Uygulama baslatiliyor...
echo Tarayici otomatik acilacaktir.
echo Kapatmak icin bu pencerede Ctrl+C kullanin.
echo.

".venv\Scripts\python.exe" -m streamlit run "app.py"

pause
