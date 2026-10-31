@echo off
setlocal
REM K28. Same search without a persistent doctrine: an admitted object is used
REM in the next cycle only and withdrawn (DSS_NO_PERSIST=1). LLM step only.
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
set "R=experiments\benchmark\revision"
if "%W_CPU%"=="" set "W_CPU=16"
if "%W_LLM%"=="" set "W_LLM=6"
echo ==== K28 closed started %date% %time% >> "%R%\revision_K28.log"
set "DSS_REQUIRE_GENAI="
REM (no closed step: the closed layer of t1_reference is the reference)
echo === Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  goto failed
)
set "DSS_REQUIRE_GENAI=1"
echo ==== K28 live started %date% %time% >> "%R%\revision_K28.log"
python "%R%\k28_no_persist.py" --arms live --workers %W_LLM%
echo ==== K28 done %errorlevel% %date% %time% >> "%R%\revision_K28.log"
pause
endlocal
exit /b 0
:failed
pause
endlocal
exit /b 1
