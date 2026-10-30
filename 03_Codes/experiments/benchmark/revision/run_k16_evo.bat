@echo off
setlocal
REM K16. Evolutionary search over the composed grammar: population 12, three
REM generations, tournament selection, mutation and clause crossover, the
REM 36-forecast budget and the gate chain of GODA. CPU only, no model call,
REM about 15 minutes with 16 workers. Pairs with final/t5_random_screened and
REM final/t6_voda.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "DSS_NO_CONCEPT="
set "R=experiments\benchmark\revision"
echo ==== K16 started %date% %time% >> "%R%\revision_K16.log"
python experiments\open_space\campaign.py --exp E15 --arms Evo --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers 16 --out %R%\k16_evo
echo ==== K16 done %errorlevel% %date% %time% >> "%R%\revision_K16.log"
pause
endlocal
