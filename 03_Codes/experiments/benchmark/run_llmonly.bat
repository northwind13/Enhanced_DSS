@echo off
REM DisasterAware without evFIS: five seed rules + language-model proposer (stage 3 only)
REM on the same worlds as run_benchmark.bat (T0 and F5 rows are reused from out\runs.csv).
REM   run_llmonly.bat          30 worlds per scenario, S1-S5
REM   run_llmonly.bat 12       12 worlds (quick)
REM Resumable: finished runs are skipped.
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "OUT=experiments\benchmark\out"
set "N=%~1"
if "%N%"=="" set "N=30"
set /a LAST=200+%N%
set "SEEDS=201-%LAST%"
set "SC=S1,S2,S3,S4,S5"

echo === 0. Claude access test ===
claude -p "Reply with exactly this JSON and nothing else: {\"ok\": true}" --output-format json
if errorlevel 1 (
  echo claude -p failed; run claude-login.bat first.
  pause
  exit /b 1
)

echo === 1. Smoke test: one LLM-only run (S3, seed 201) ===
python experiments\open_space\campaign.py --exp E2 --arms LLMonly --scenarios S3 --seeds 201 --workers 1 --out %OUT%
if errorlevel 1 (
  echo smoke test failed.
  pause
  exit /b 1
)

echo === 2. DisasterAware without evFIS (%SC%, seeds %SEEDS%) ===
python experiments\open_space\campaign.py --exp E2 --arms LLMonly --scenarios %SC% --seeds %SEEDS% --workers 2 --out %OUT%

echo === 3. Tables: No DSS / F5 / DisasterAware(LLM only), plus the earlier four-arm report for comparison ===
python experiments\benchmark\benchmark_tables.py --out %OUT% --scenarios %SC% --da-arm LLMonly --arms T0,F5,DA --report report_llmonly
echo.
echo Done. Tables: %OUT%\report_llmonly\summary.txt
pause
endlocal
