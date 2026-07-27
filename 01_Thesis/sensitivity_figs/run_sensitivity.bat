@echo off
REM DisasterAware - sensitivity sweep + figures
setlocal
cd /d "%~dp0"

where py >nul 2>nul && (set "PY=py") || (set "PY=python")

if not exist ".venv\Scripts\activate.bat" (
    echo [setup] Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [error] Could not create the virtual environment. Is Python installed?
        pause & exit /b 1
    )
)
call ".venv\Scripts\activate.bat"
set PYTHONDONTWRITEBYTECODE=1
echo [setup] Installing dependencies...
python -m pip install --upgrade pip >nul 2>nul
pip install -r requirements.txt
if errorlevel 1 ( echo [error] Dependency install failed. & pause & exit /b 1 )

echo [run] Running the sensitivity sweep ^(a few minutes; resumable^)...
python runner.py
if errorlevel 1 ( echo [error] Sweep failed. & pause & exit /b 1 )

echo [plot] Generating figures...
python plot.py
if errorlevel 1 ( echo [error] Plotting failed. & pause & exit /b 1 )

echo [done] sens.csv + fig_tornado.png, fig_capacity.png, fig_robust.png
pause
