@echo off
REM K6 part C: application limits of the coating, planner and world alike.
REM   coat_stock_20ha  at most 20 ha coated per incident;  coat_delay_12min  a coating order acts one cycle later.
REM Step 1 closed+replay: CPU, 600 runs, about 40 min with 16 workers. Step 2 live: model calls, 300 runs, about 1 h with 6 workers.
REM Resumable: run again, finished runs are skipped; a run without the model is an error, not a result.
setlocal
cd /d "%~dp0..\..\.."
call ".venv\Scripts\activate.bat"
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "DSS_REQUIRE_GENAI=1"
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "R=experiments\benchmark\revision"
set "P=coat_stock_20ha,coat_delay_12min"
echo === 0. Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  pause
  exit /b 1
)
echo ==== K6 C started %date% %time% >> "%R%\revision_K6.log"
python %R%\k6_coating.py --perturb %P% --arms closed,replay --workers 16
echo K6 C step 1 finished %errorlevel% %time% >> "%R%\revision_K6.log"
python %R%\k6_coating.py --perturb %P% --arms live --workers 6
echo K6 C step 2 finished %errorlevel% %time% >> "%R%\revision_K6.log"
python %R%\revision_tables.py
echo ==== K6 C done %date% %time% >> "%R%\revision_K6.log"
pause
endlocal
