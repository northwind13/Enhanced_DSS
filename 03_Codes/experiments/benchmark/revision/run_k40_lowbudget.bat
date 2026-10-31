@echo off
setlocal
REM K40: evolutionary and cross-entropy searches at 6 and 12 forecasts, CPU only, in parallel, then commit and push.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python experiments\benchmark\revision\run_k40_lowbudget.py
pause
endlocal
