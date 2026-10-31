@echo off
setlocal
REM K33. Evolutionary search restricted to the six families (rules and macros
REM only, DSS_NO_COMPOSED=1 through the -fam arm): the receding-horizon
REM optimization baseline over the closed vocabulary at the same 36-forecast
REM budget and gates. CPU only, no model call, about 15 minutes with 16
REM workers. Resumable (rows already in k33_evo_fam/runs.csv are skipped).
REM Pairs with final/t1_reference, revision/k13_closed_vocab and k33_evo_fam.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "DSS_NO_CONCEPT="
set "R=experiments\benchmark\revision"
if "%W_CPU%"=="" set "W_CPU=16"
echo ==== K33 started %date% %time% >> "%R%\revision_K33.log"
python experiments\open_space\campaign.py --exp E15 --arms Evo-fam --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers %W_CPU% --out %R%\k33_evo_fam
echo ==== K33 done %errorlevel% %date% %time% >> "%R%\revision_K33.log"
pause
endlocal
