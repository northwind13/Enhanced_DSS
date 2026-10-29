@echo off
setlocal
REM K12u. Ungated control: the single-shot model proposer with the forecast
REM gates G5 to G8 switched off. The model is called; about one hour with
REM 6 workers for 150 worlds. Resumable: finished worlds are skipped.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_MODEL=claude-opus-5"
set "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_REQUIRE_GENAI=1"
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "DSS_NO_CONCEPT="
set "R=experiments\benchmark\revision"
echo === Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  goto failed
)
echo ==== K12u started %date% %time% >> "%R%\revision_K12u.log"
python "%R%\k12u_ungated.py" --workers 6
echo ==== K12u done %errorlevel% %date% %time% >> "%R%\revision_K12u.log"
pause
endlocal
exit /b 0
:failed
pause
endlocal
exit /b 1
