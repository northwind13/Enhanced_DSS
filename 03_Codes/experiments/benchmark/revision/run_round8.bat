@echo off
setlocal
REM Round 8: K35, K37, K38, K39 (and K36 if the hindcast cache exists), all CPU,
REM in parallel, then commit and push. W_CPU sets the total workers (default 12).
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python experiments\benchmark\revision\run_round8.py
pause
endlocal
