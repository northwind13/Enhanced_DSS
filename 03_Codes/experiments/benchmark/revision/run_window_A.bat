@echo off
REM Window A: K1, K2, K3 one after another. CPU only, no model calls. Resumable: run again, finished runs are skipped.
setlocal
cd /d "%~dp0..\..\.."
REM the project environment (rasterio, numpy, ...) lives in 03_Codes\.venv
call ".venv\Scripts\activate.bat"
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_REQUIRE_GENAI="
set "W=16"
if not "%~1"=="" set "W=%~1"
set "B=experiments\benchmark"
set "R=%B%\revision"
set "LOG=%R%\revision_A.log"
echo ==== window A started %date% %time%, workers %W% >> "%LOG%"
echo === K1. closed layer, whole written doctrine, arm F40Ev, 150 worlds ===
python experiments\open_space\campaign.py --exp E2P --arms F40Ev --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers %W% --out %B%\final\t13_full_doctrine
echo K1 finished %errorlevel% %time% >> "%LOG%"
echo === K2. discovered tactics given to the closed layer at minute 0 ===
python %R%\k2_predefined.py --mode canon --workers %W%
echo K2 canon finished %errorlevel% %time% >> "%LOG%"
python %R%\k2_predefined.py --mode own --workers %W%
echo K2 own finished %errorlevel% %time% >> "%LOG%"
echo === K3. planning on the nominal model, evaluated under model error ===
python %R%\k3_model_error.py --workers %W%
echo K3 finished %errorlevel% %time% >> "%LOG%"
echo === summary, K4 may still be running in window B ===
python %R%\revision_tables.py
echo ==== window A finished %date% %time% >> "%LOG%"
echo Window A done. If window B has finished too, run: python %R%\revision_tables.py
pause
endlocal
