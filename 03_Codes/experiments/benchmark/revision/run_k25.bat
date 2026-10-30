@echo off
setlocal
REM K25. Gate ablation: selection on one future, admission on G5 only (G6 to
REM G8 off, no-harm guard on). One LLM step, then the object-removal replay
REM (CPU) for the harmful-admission rate. Two steps:
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
echo ==== K25 closed started %date% %time% >> "%R%\revision_K25.log"
set "DSS_REQUIRE_GENAI="
REM (no closed step: the closed layer of t1_reference is the reference)
echo === Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  goto failed
)
set "DSS_REQUIRE_GENAI=1"
echo ==== K25 live started %date% %time% >> "%R%\revision_K25.log"
python "%R%\k25_one_future.py" --arms live --workers %W_LLM%
python experiments\benchmark\loo.py --src "%R%\k25_one_future" --exp E9 --arm LLMScreen --out "%R%\k25_loo" --workers %W_CPU%
echo ==== K25 done %errorlevel% %date% %time% >> "%R%\revision_K25.log"
pause
endlocal
exit /b 0
:failed
pause
endlocal
exit /b 1
