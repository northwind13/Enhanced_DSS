@echo off
setlocal
REM K15. One random candidate per call admitted after G1 to G4 only, the
REM forecast gates G5 to G8 switched off (arm RandBlind-ungated). CPU only,
REM no model call, about 15 minutes with 16 workers. Pairs with
REM final/t4_random_single (the same proposer under the full chain) and
REM with K12u (the model proposer without the forecast gates).
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "DSS_NO_CONCEPT="
set "R=experiments\benchmark\revision"
echo ==== K15 started %date% %time% >> "%R%\revision_K15.log"
python experiments\open_space\campaign.py --exp E10 --arms RandBlind-ungated --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers 16 --out %R%\k15_random_ungated
echo ==== K15 done %errorlevel% %date% %time% >> "%R%\revision_K15.log"
pause
endlocal
