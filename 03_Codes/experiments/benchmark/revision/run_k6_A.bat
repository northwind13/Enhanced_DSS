@echo off
REM K6 window A: closed layer and replay under coat_half, coat_off, coat_price_x2. CPU only, 900 runs, about 1.5 h with 16 workers. Resumable.
setlocal
cd /d "%~dp0..\..\.."
call ".venv\Scripts\activate.bat"
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "R=experiments\benchmark\revision"
echo ==== K6 A started %date% %time% >> "%R%\revision_K6.log"
python %R%\k6_coating.py --arms closed,replay --workers 16
echo K6 A finished %errorlevel% %time% >> "%R%\revision_K6.log"
python %R%\revision_tables.py
pause
endlocal
