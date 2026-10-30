@echo off
setlocal
REM Runs K4 ellipse (seconds, CPU), then K18 and K19 side by side.
REM Window 1: K4, then K18 closed (8 workers) and K18 live (4 workers)
REM Window 2: K19 closed (8 workers) and K19 live (4 workers)
REM The two closed arms share the CPU; the two live arms together keep eight
REM model calls in flight, about the same load as one run with six workers
REM plus a little. If the model starts refusing (rate limit), stop one window
REM and rerun it afterwards; every step is resumable.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "R=experiments\benchmark\revision"
echo === K4 ellipse reference ===
python "%R%\k4_ellipse.py" > "%R%\k4_ellipse_out.txt" 2>&1
type "%R%\k4_ellipse_out.txt"
echo === Claude access test ===
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  pause
  exit /b 1
)
start "K19" cmd /k call "%R%\run_k_one.bat" k19_perimeter K19 "DSS_K19_DROP=0.25"
call "%R%\run_k_one.bat" k18_combined K18 ""
pause
endlocal
exit /b 0
