@echo off
setlocal
REM K21. Forecast-budget sweep. The article has the random proposer and GODA
REM at 6 and 36 first-future forecasts per call; this adds 12 and 24 for both,
REM same 150 worlds, same gates. Random arms CPU (~20 min each), GODA arms LLM
REM (~1 h each). Resumable. Pairs with final\t1_reference (F5Ev).
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_MODEL=claude-opus-5"
set "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "DSS_LLM_SCREEN_FILL=1"
set "DSS_NO_CONCEPT=1"
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "R=experiments\benchmark\revision"
if "%W_CPU%"=="" set "W_CPU=16"
if "%W_LLM%"=="" set "W_LLM=6"
set "SC=S1,S2,S3,S4,S5"
set "SEEDS=201-230"
echo ==== K21 started %date% %time% >> "%R%\revision_K21.log"
set "DSS_REQUIRE_GENAI="
python experiments\open_space\campaign.py --exp E8 --arms Random --budget 12 --scenarios %SC% --seeds %SEEDS% --workers %W_CPU% --out "%R%\k21_budget\random_b12"
python experiments\open_space\campaign.py --exp E8 --arms Random --budget 24 --scenarios %SC% --seeds %SEEDS% --workers %W_CPU% --out "%R%\k21_budget\random_b24"
echo === Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  goto failed
)
set "DSS_REQUIRE_GENAI=1"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --budget 12 --scenarios %SC% --seeds %SEEDS% --workers %W_LLM% --out "%R%\k21_budget\goda_b12"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --budget 24 --scenarios %SC% --seeds %SEEDS% --workers %W_LLM% --out "%R%\k21_budget\goda_b24"
echo ==== K21 done %errorlevel% %date% %time% >> "%R%\revision_K21.log"
pause
endlocal
exit /b 0
:failed
pause
endlocal
exit /b 1
