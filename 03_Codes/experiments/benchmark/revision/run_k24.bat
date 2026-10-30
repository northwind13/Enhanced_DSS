@echo off
setlocal
REM K24. Marginal contribution of each effect: replay of the admitted objects
REM with every object using one effect withheld, seven effects x 150 worlds.
REM CPU only, ~1 h with 12 workers. Resumable.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "DSS_NO_CONCEPT=1"
set "R=experiments\benchmark\revision"
if "%W_CPU%"=="" set "W_CPU=12"
echo ==== K24 started %date% %time% >> "%R%\revision_K24.log"
python "%R%\k24_effect_removal.py" --workers %W_CPU%
echo ==== K24 done %errorlevel% %date% %time% >> "%R%\revision_K24.log"
pause
endlocal
