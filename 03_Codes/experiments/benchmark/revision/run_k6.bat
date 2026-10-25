@echo off
REM K6: coating effect weaker in the live world, planning on the nominal coating.
REM Part 1 closed+replay: CPU only, 900 runs, about 1.5 h with 16 workers.
REM Part 2 live: VODA run anew on the perturbed world, 450 runs with model calls, about 4 h with 3 workers.
REM Resumable: run again, finished runs are skipped.
setlocal
cd /d "%~dp0..\..\.."
call ".venv\Scripts\activate.bat"
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "W=16"
if not "%~1"=="" set "W=%~1"
set "R=experiments\benchmark\revision"
set "LOG=%R%\revision_K6.log"
echo ==== K6 started %date% %time%, workers %W% >> "%LOG%"
echo === K6 part 1. closed layer and replay under coat_half and coat_off ===
python %R%\k6_coating.py --arms closed,replay --workers %W%
echo K6 part 1 finished %errorlevel% %time% >> "%LOG%"
echo === K6 part 2. VODA live under coat_half and coat_off, model calls ===
python %R%\k6_coating.py --arms live --workers 3
echo K6 part 2 finished %errorlevel% %time% >> "%LOG%"
python %R%\revision_tables.py
echo ==== K6 done %date% %time% >> "%LOG%"
pause
endlocal
