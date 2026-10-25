@echo off
REM K6 window B: VODA live under coat_half, coat_off, coat_price_x2. Model calls, 450 runs, about 1.5-2 h with 6 workers.
REM Resumable: run again to fill any run that failed on a model error (strict mode, a run without the model is an error, not a result).
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
echo === 0. Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  pause
  exit /b 1
)
echo ==== K6 B started %date% %time% >> "%R%\revision_K6.log"
python %R%\k6_coating.py --arms live --workers 6
echo K6 B finished %errorlevel% %time% >> "%R%\revision_K6.log"
python %R%\revision_tables.py
pause
endlocal
