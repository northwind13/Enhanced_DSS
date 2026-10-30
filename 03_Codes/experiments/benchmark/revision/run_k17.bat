@echo off
setlocal
REM K17. Structurally different spread model in the live world (cosine kernel,
REM short frequent spotting), planning on the nominal model. Two steps:
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
echo ==== K17 closed started %date% %time% >> "%R%\revision_K17.log"
set "DSS_REQUIRE_GENAI="
python "%R%\k17_kernel.py" --arms closed --workers 16
echo === Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  goto failed
)
set "DSS_REQUIRE_GENAI=1"
echo ==== K17 live started %date% %time% >> "%R%\revision_K17.log"
python "%R%\k17_kernel.py" --arms live --workers 6
echo ==== K17 done %errorlevel% %date% %time% >> "%R%\revision_K17.log"
pause
endlocal
exit /b 0
:failed
pause
endlocal
exit /b 1
