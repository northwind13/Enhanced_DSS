@echo off
REM Regenerate the Section 5.4.3 open-decision-space figures from the
REM committed logs (logs/DSS_*/cycles.jsonl + logs/dss_generated_state.json).
REM Output goes to 01_Thesis\figures\fig_openspace_*.png
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python experiments\plot_openspace.py
echo.
echo Done. Figures written to ..\01_Thesis\figures\
pause
