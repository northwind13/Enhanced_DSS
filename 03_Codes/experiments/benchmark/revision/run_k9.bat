@echo off
REM K9: recorded tactics against continued generation on unseen conditions (S6-S8) and a control (S1-S5 seeds 216-230).
REM Order: A and B novel (CPU), C control, C novel, D novel (model). About 2.5 h. Resumable: run again, finished runs are skipped.
setlocal
cd /d "%~dp0..\..\.."
call ".venv\Scripts\activate.bat"
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "DSS_LLM_SCREEN_FILL=1"
set "DSS_NO_CONCEPT=1"
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "R=experiments\benchmark\revision"
echo === 0. Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  pause
  exit /b 1
)
echo ==== K9 started %date% %time% >> "%R%\revision_K9.log"
set "DSS_REQUIRE_GENAI="
python %R%\k9_novelty.py --arm A --set novel --workers 16
python %R%\k9_novelty.py --arm B --set novel --workers 16
echo K9 A,B finished %errorlevel% %time% >> "%R%\revision_K9.log"
set "DSS_REQUIRE_GENAI=1"
python %R%\k9_novelty.py --arm C --set control --workers 6
echo K9 C control finished %errorlevel% %time% >> "%R%\revision_K9.log"
python %R%\k9_novelty.py --arm C --set novel --workers 6
echo K9 C novel finished %errorlevel% %time% >> "%R%\revision_K9.log"
python %R%\k9_novelty.py --arm D --set novel --workers 6
echo ==== K9 done %errorlevel% %date% %time% >> "%R%\revision_K9.log"
pause
endlocal
