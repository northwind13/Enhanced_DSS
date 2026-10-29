@echo off
setlocal
REM K13 and K14, CPU only, no model call. Resumable: finished worlds are skipped.
REM   K13  Random-fam   closed-vocabulary search: rules and macros over the six
REM                     families, the 36-forecast budget and the gate chain of
REM                     VODA; the forecast-matched closed baseline
REM   K14  RandomVar    six random seeds plus the variation operator of the
REM                     model search, same budget and gates; pairs with VODA
REM About one hour in all with 16 workers.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "DSS_NO_CONCEPT="
set "DSS_LLM_SCREEN_N=6"
set "R=experiments\benchmark\revision"
set "SC=S1,S2,S3,S4,S5"
echo ==== K13 started %date% %time% >> "%R%\revision_K13_K14.log"
python experiments\open_space\campaign.py --exp E13 --arms Random-fam --scenarios %SC% --seeds 201-230 --workers 16 --out %R%\k13_closed_vocab
echo ==== K13 done %errorlevel% %date% %time% >> "%R%\revision_K13_K14.log"
echo ==== K14 started %date% %time% >> "%R%\revision_K13_K14.log"
python experiments\open_space\campaign.py --exp E14 --arms RandomVar --scenarios %SC% --seeds 201-230 --workers 16 --out %R%\k14_random_var
echo ==== K14 done %errorlevel% %date% %time% >> "%R%\revision_K13_K14.log"
pause
endlocal
