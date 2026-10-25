@echo off
REM K7: wind different in the live world (x1.5, +30 deg, -30 deg), planning on the nominal wind. closed+replay CPU 900 runs ~1.5 h, live 450 runs ~2 h.
REM K8: VODA with the gates G5-G8 judging on two futures the pre-selection never saw (seeds 303, 404 instead of 101, 202). 150 live runs ~1 h.
REM Order: K8 live, K7 closed+replay, K7 live. Resumable: run again, finished runs are skipped.
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
echo ==== K7/K8 started %date% %time% >> "%R%\revision_K7K8.log"
echo === K8. gates on independent futures, seeds 303 and 404 ===
set "DSS_GATE_SEEDS=303,404"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers 6 --out %R%\k8_gate_seeds
set "DSS_GATE_SEEDS="
echo K8 finished %errorlevel% %time% >> "%R%\revision_K7K8.log"
echo === K7 step 1. closed layer and replay under wind error ===
set "DSS_REQUIRE_GENAI="
python %R%\k7_spread.py --arms closed,replay --workers 16
echo K7 step 1 finished %errorlevel% %time% >> "%R%\revision_K7K8.log"
echo === K7 step 2. VODA live under wind error ===
set "DSS_REQUIRE_GENAI=1"
python %R%\k7_spread.py --arms live --workers 6
echo K7 step 2 finished %errorlevel% %time% >> "%R%\revision_K7K8.log"
python %R%\revision_tables.py
echo ==== K7/K8 done %date% %time% >> "%R%\revision_K7K8.log"
pause
endlocal
