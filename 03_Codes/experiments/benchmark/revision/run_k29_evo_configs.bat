@echo off
setlocal
REM K29. Evolutionary search under two further population/generation settings
REM at the same 36-forecast budget (reported: 12 x 3). CPU only, no model
REM call; each setting about the time of K16 (15 min with 16 workers).
REM Resumable. Pairs with final/t1_reference and revision/k16_evo.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "DSS_NO_CONCEPT="
set "R=experiments\benchmark\revision"
if "%W_CPU%"=="" set "W_CPU=16"
echo ==== K29 started %date% %time% >> "%R%\revision_K29.log"
set "DSS_EVO_POP=6"
set "DSS_EVO_GENS=5"
python experiments\open_space\campaign.py --exp E15 --arms Evo --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers %W_CPU% --out %R%\k29_evo_6x5
echo ==== K29 6x5 done %errorlevel% %date% %time% >> "%R%\revision_K29.log"
set "DSS_EVO_POP=18"
set "DSS_EVO_GENS=1"
python experiments\open_space\campaign.py --exp E15 --arms Evo --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers %W_CPU% --out %R%\k29_evo_18x1
echo ==== K29 done %errorlevel% %date% %time% >> "%R%\revision_K29.log"
set "DSS_EVO_POP="
set "DSS_EVO_GENS="
endlocal
