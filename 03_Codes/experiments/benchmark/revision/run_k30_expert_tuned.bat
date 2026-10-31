@echo off
setlocal
REM K30. The six hand-written tactics of Table IV tuned on the forecast:
REM band and amount chosen among six variants by the one-future forecast the
REM first time a tactic fires, six forecasts per tactic, then installed as
REM doctrine without the gates (arm ExpertTune, exp E17). CPU only, no model
REM call, about the time of K11. Resumable. Pairs with final/t1_reference,
REM revision/k11_expert_doctrine and revision/k12_doctrine_seeded.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "DSS_NO_CONCEPT=1"
set "R=experiments\benchmark\revision"
if "%W_CPU%"=="" set "W_CPU=16"
echo ==== K30 started %date% %time% >> "%R%\revision_K30.log"
python experiments\open_space\campaign.py --exp E17 --arms ExpertTune --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers %W_CPU% --out %R%\k30_expert_tuned
echo ==== K30 done %errorlevel% %date% %time% >> "%R%\revision_K30.log"
endlocal
