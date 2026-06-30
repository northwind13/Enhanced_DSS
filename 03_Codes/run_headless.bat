@echo off
REM DisasterAware simulator - headless example run (no dashboard)
setlocal
cd /d "%~dp0"

where py >nul 2>nul && (set "PY=py") || (set "PY=python")
if not exist ".venv\Scripts\activate.bat" (
    %PY% -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip >nul 2>nul
pip install -r requirements.txt >nul
python examples\run_headless.py
pause
