@echo off
REM DisasterAware article rerun: VODA (LLM proposer) WITHOUT the actuator library.
REM Experiment E5 = S1-S5, arm LLM, library withheld (DSS_WITHHOLD_ACTUATORS via campaign E5).
REM The reference arms (No DSS / F5 / F5+EvFIS) are not rerun: they never see the library.
REM
REM   run_nolib.bat            30 worlds per scenario, seeds 201-230
REM   run_nolib.bat 12         quick run, seeds 201-212
REM
REM The language model is MANDATORY in this run:
REM   step 0  claude -p must answer, otherwise the script stops
REM   step 1  one smoke world; the script checks that proposals.jsonl holds "source": "llm" rows
REM   step 2  DSS_REQUIRE_GENAI=1: a failed or empty claude call aborts the run instead of
REM           silently continuing without the model (see dss/genai.py)
REM Resumable: run again after an interruption, finished runs are skipped.
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_REQUIRE_GENAI=1"
set "DSS_GENAI_TIMEOUT=180"
set "OUT=experiments\benchmark\out_nolib"
set "N=%~1"
if "%N%"=="" set "N=30"
set /a LAST=200+%N%
set "SEEDS=201-%LAST%"
set "SC=S1,S2,S3,S4,S5"
if not exist "%OUT%" mkdir "%OUT%"

echo === 0. Claude access test (config dir %CLAUDE_CONFIG_DIR%) ===
where claude >nul 2>nul
if errorlevel 1 (
  echo claude CLI is not on PATH. Install Claude Code and run claude-login.bat first.
  pause
  exit /b 1
)
claude -p "Reply with exactly this JSON and nothing else: {\"ok\": true}" --output-format json > "%OUT%\claude_check.json"
if errorlevel 1 (
  echo claude -p failed; run claude-login.bat first.
  pause
  exit /b 1
)
findstr /c:"ok" "%OUT%\claude_check.json" >nul
if errorlevel 1 (
  echo claude answered but not as expected; see %OUT%\claude_check.json
  pause
  exit /b 1
)
echo claude OK.

echo === 1. Smoke test: one LLM run without the library (S3, seed 201) ===
REM an errors.log left by an earlier, interrupted run must not stop this one
if exist "%OUT%\errors.log" (
  type "%OUT%\errors.log" >> "%OUT%\errors_previous.log"
  del "%OUT%\errors.log"
)
python experiments\open_space\campaign.py --exp E5 --arms LLM --scenarios S3 --seeds 201 --workers 1 --out %OUT%
if errorlevel 1 (
  echo smoke test failed.
  pause
  exit /b 1
)
if exist "%OUT%\errors.log" (
  echo The smoke run logged an error; see %OUT%\errors.log
  type "%OUT%\errors.log"
  pause
  exit /b 1
)
findstr /c:"\"source\": \"llm\"" "%OUT%\proposals.jsonl" >nul
if errorlevel 1 (
  echo The smoke run produced NO language-model proposals. The model did not run. Stopping.
  pause
  exit /b 1
)
for /f %%c in ('findstr /c:"\"source\": \"llm\"" "%OUT%\proposals.jsonl" ^| find /c /v ""') do echo smoke run: %%c language-model proposals recorded.
findstr /c:"retardant_drop" /c:"water_drafting" /c:"tactical_burn" "%OUT%\proposals.jsonl" >nul
if not errorlevel 1 (
  echo WARNING: a library actuator name appears in the smoke proposals; the library is NOT withheld. Stopping.
  pause
  exit /b 1
)
echo library withheld: OK.

echo === 2. Full run: VODA without the library (%SC%, seeds %SEEDS%) ===
python experiments\open_space\campaign.py --exp E5 --arms LLM --scenarios %SC% --seeds %SEEDS% --workers 2 --out %OUT%
if errorlevel 1 (
  echo campaign exited with an error.
  pause
  exit /b 1
)

echo === 3. Summary ===
python -c "import csv;r=[x for x in csv.DictReader(open(r'%OUT%\runs.csv',encoding='utf-8')) if x['exp']=='E5'];print('E5 LLM runs:',len(r));import collections;print(collections.Counter(x['scenario'] for x in r))"
python -c "import json;n=0;l=0;s=0;f=open(r'%OUT%\proposals.jsonl',encoding='utf-8');exec('for line in f:\n    d=json.loads(line);n+=1;l+=d.get(\"source\")==\"llm\";s+=bool(d.get(\"accepted\"))');print('proposals:',n,'from the model:',l,'admitted:',s)"
if exist "%OUT%\errors.log" (
  echo Some runs failed and were NOT written; see %OUT%\errors.log and run this script again to complete them.
)
echo.
echo Done. Output: %OUT%\runs.csv, curves.csv, proposals.jsonl, admitted.jsonl
pause
endlocal
