@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   DisasterAware Dashboard - Launcher
echo ============================================
echo.

REM ---- find Python ----
set "PY="
python --version >nul 2>&1 && set "PY=python"
if not defined PY ( py --version >nul 2>&1 && set "PY=py" )
if not defined PY (
    echo [HATA] Python bulunamadi. python.org'dan kurun, "Add Python to PATH" secin.
    pause & exit /b 1
)
echo [OK] Python: %PY%

REM ---- install required packages only if missing ----
for %%P in (flask numpy) do (
    %PY% -c "import %%P" >nul 2>&1
    if errorlevel 1 (
        echo [KUR] %%P yukleniyor...
        %PY% -m pip install %%P
        if errorlevel 1 ( echo [HATA] %%P yuklenemedi. & pause & exit /b 1 )
    ) else ( echo [OK] %%P zaten kurulu. )
)
echo.
echo Tarayici acilacak: http://127.0.0.1:5000
echo Kapatmak icin bu pencerede Ctrl+C.
echo.
%PY% app.py
pause
