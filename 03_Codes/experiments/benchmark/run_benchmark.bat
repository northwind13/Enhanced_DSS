@echo off
REM DisasterAware benchmark: No DSS / F5 / F5+EvFIS / DisasterAware (LLM proposer)
REM Scenarios S1-S5 (thesis families), paired worlds, episodes to extinction or 12 h.
REM
REM   run_benchmark.bat            30 worlds per scenario (default)
REM   run_benchmark.bat 12         12 worlds per scenario (quick)
REM   run_benchmark.bat 30 S6      also run the extra families S6-S8 (second argument "S6")
REM
REM Resumable: run again after an interruption, finished runs are skipped.
REM Needs: python with numpy; claude CLI logged in via claude-login.bat (same config dir).
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "OUT=experiments\benchmark\out"
set "N=%~1"
if "%N%"=="" set "N=30"
set /a LAST=200+%N%
set "SEEDS=201-%LAST%"
set "SC=S1,S2,S3,S4,S5"
if /i "%~2"=="S6" set "SC=S1,S2,S3,S4,S5,S6,S7,S8"

echo === 0. Claude access test ===
claude -p "Reply with exactly this JSON and nothing else: {\"ok\": true}" --output-format json
if errorlevel 1 (
  echo claude -p failed; run claude-login.bat first.
  pause
  exit /b 1
)

echo === 1. Smoke test: one LLM run (S3, seed 201) ===
python experiments\open_space\campaign.py --exp E2 --arms LLM --scenarios S3 --seeds 201 --workers 1 --out %OUT%
if errorlevel 1 (
  echo smoke test failed.
  pause
  exit /b 1
)

echo === 2. Reference arms without the model: No DSS, F5, F5+EvFIS (%SC%, seeds %SEEDS%) ===
python experiments\open_space\campaign.py --exp E2 --arms T0,F5,F5Ev --scenarios %SC% --seeds %SEEDS% --workers 4 --out %OUT%

echo === 3. DisasterAware with the LLM proposer (%SC%, seeds %SEEDS%) ===
python experiments\open_space\campaign.py --exp E2 --arms LLM --scenarios %SC% --seeds %SEEDS% --workers 2 --out %OUT%

echo === 4. Tables and figures ===
python experiments\benchmark\benchmark_tables.py --out %OUT% --scenarios %SC%
echo.
echo Done. Tables: %OUT%\report\table1_outcome.md ... table4_adaptation.md, fig_burned.png, summary.txt
pause
endlocal
