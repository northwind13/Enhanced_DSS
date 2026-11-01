@echo off
setlocal
REM Which model the run scripts will use, and whether each CLI is logged in.
cd /d "%~dp0"
call "%~dp0llm_env.bat"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m dss.llm status
endlocal
pause
