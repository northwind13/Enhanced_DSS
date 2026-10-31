@echo off
setlocal
REM K31. Coating effectiveness sweep, four levels below nominal, closed and
REM replay arms, two modes (world only; planner and world). CPU only, no
REM model call; 2400 runs, about the time of two K6 part-1 runs (3 h with
REM 16 workers). Resumable.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "DSS_NO_CONCEPT=1"
set "R=experiments\benchmark\revision"
if "%W_CPU%"=="" set "W_CPU=16"
echo ==== K31 started %date% %time% >> "%R%\revision_K31.log"
python "%R%\k31_coating_sweep.py" --mode world --workers %W_CPU%
echo ==== K31 world done %errorlevel% %date% %time% >> "%R%\revision_K31.log"
python "%R%\k31_coating_sweep.py" --mode both --workers %W_CPU%
echo ==== K31 done %errorlevel% %date% %time% >> "%R%\revision_K31.log"
endlocal
