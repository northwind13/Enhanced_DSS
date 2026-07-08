@echo off
REM DisasterAware simulator - dashboard launcher
setlocal
cd /d "%~dp0"

where py >nul 2>nul && (set "PY=py") || (set "PY=python")

if not exist ".venv\Scripts\activate.bat" (
    echo [setup] Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [error] Could not create virtual environment. Is Python installed?
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
REM stale .pyc files have caused "old code running" bugs after updates:
REM never write bytecode and clear any leftover caches on every start
set PYTHONDONTWRITEBYTECODE=1
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
echo [setup] Installing dependencies...
python -m pip install --upgrade pip >nul 2>nul
pip install -r requirements.txt
if errorlevel 1 (
    echo [error] Dependency installation failed.
    pause
    exit /b 1
)

echo [run] Starting dashboard. Close this window or press Ctrl+C to stop.
streamlit run app\streamlit_app.py
pause
