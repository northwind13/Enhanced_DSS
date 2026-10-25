@echo off
REM K2b: tactics derived from seeds 201-215, frozen, evaluated on the unseen seeds 216-230 with the closed layer. CPU only, ~10 min. Resumable.
setlocal
cd /d "%~dp0..\..\.."
call ".venv\Scripts\activate.bat"
set "R=experiments\benchmark\revision"
python %R%\k2_predefined.py --mode canon --derive-seeds 201-215 --seeds 216-230 --workers 16 --out %R%\k2b_generalization
python %R%\revision_tables.py
pause
endlocal
