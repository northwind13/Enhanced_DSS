@echo off
setlocal
REM K26. Information parity: the brief of the proposer written from the same
REM perturbed copy as the forecasts (150 worlds) and the S5 corruption applied
REM to the brief too (S5 only). LLM only. Two steps:
REM   1  closed  F5Ev under the perturbed world            CPU   ~10 min
REM   2  live    GODA, whole loop with model calls          LLM   ~1 h
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
set "DSS_K26_BREF=0.25"
set "R=experiments\benchmark\revision"
if "%W_CPU%"=="" set "W_CPU=16"
if "%W_LLM%"=="" set "W_LLM=6"
echo ==== K26 closed started %date% %time% >> "%R%\revision_K26.log"
set "DSS_REQUIRE_GENAI="
REM (no closed step: pairs with k19_perimeter closed and t1_reference closed)
echo === Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  goto failed
)
set "DSS_REQUIRE_GENAI=1"
echo ==== K26 live started %date% %time% >> "%R%\revision_K26.log"
python "%R%\k26_parity.py" --arms state --workers %W_LLM%
python "%R%\k26_parity.py" --arms s5 --scenarios S5 --workers %W_LLM%
echo ==== K26 done %errorlevel% %date% %time% >> "%R%\revision_K26.log"
pause
endlocal
exit /b 0
:failed
pause
endlocal
exit /b 1
