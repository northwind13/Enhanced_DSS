@echo off
REM FINAL campaign of article v52. Three steps, about 4.5 h in all. Resumable: if it stops, run it again.
REM   1  E10 RandBlind   one unpriced random candidate per call, pairs with E5 (LLM)          CPU  ~20 min
REM   2  ACCT  replay of the E5 admissions with retardant and ignition accounting           CPU  ~20 min
REM   3  E9  LLMScreen   model candidates screened with the E8 forecast budget, S1-S5 x 30    LLM  ~3-4 h
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "SEEDS=201-230"
set "B=experiments\benchmark"
set "LOG=%B%\final.log"
REM Workers are sized from this machine: logical processors minus two, capped by free memory
REM at about 0.5 GB per worker. The model step runs 6 worlds at once (the limit is the model, not the CPU).
set /a "CPUW=%NUMBER_OF_PROCESSORS%-2"
for /f %%m in ('powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/524288)"') do set "MEMW=%%m"
if defined MEMW if %MEMW% LSS %CPUW% set "CPUW=%MEMW%"
if %CPUW% LSS 2 set "CPUW=2"
set "LLMW=6"
for /f %%g in ('powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1048576,1)"') do set "FREEGB=%%g"
echo Machine: %NUMBER_OF_PROCESSORS% logical processors, %FREEGB% GB free memory. CPU workers: %CPUW%, model workers: %LLMW%
echo Machine: %NUMBER_OF_PROCESSORS% threads, %FREEGB% GB free, CPU workers %CPUW% >> "%LOG%"
echo ==== final campaign started %date% %time% >> "%LOG%"
echo === 0. Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  pause
  exit /b 1
)
echo === 1. E10 random, unscreened ===
set "DSS_REQUIRE_GENAI="
python experiments\open_space\campaign.py --exp E10 --arms RandBlind --scenarios S1,S2,S3,S4,S5 --seeds %SEEDS% --workers %CPUW% --out %B%\out_randblind
echo 1 finished %errorlevel% %time% >> "%LOG%"
echo === 2. Retardant and ignition accounting of the E5 admissions ===
python %B%\loo.py --src %B%\out_nolib --exp E5 --arm LLM --out %B%\out_acct_e5 --controls-only --workers %CPUW%
echo 2 finished %errorlevel% %time% >> "%LOG%"
echo === 3. E9 model candidates, screened ===
set "DSS_REQUIRE_GENAI=1"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios S3 --seeds 201 --workers 1 --out %B%\out_v52_main
python -c "import json;P=[json.loads(l) for l in open(r'%B%\out_v52_main\proposals.jsonl',encoding='utf-8')];m=[(p.get('search') or {}).get('model_candidates') for p in P];print('smoke: model candidates per call',m[:10])"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios S1,S2,S3,S4,S5 --seeds %SEEDS% --workers %LLMW% --out %B%\out_v52_main
echo 3 finished %errorlevel% %time% >> "%LOG%"
echo === Summary ===
for %%d in (out_randblind out_v52_main) do python -c "import csv,os,collections;p=r'%B%\%%d\runs.csv';print('%%d', len(list(csv.DictReader(open(p,encoding='utf-8')))) if os.path.exists(p) else 'missing')"
python -c "import os;p=r'%B%\out_acct_e5';print('out_acct_e5', [f for f in os.listdir(p)] if os.path.exists(p) else 'missing')"
for %%d in (out_randblind out_acct_e5 out_v52_main) do if exist "%B%\%%d\errors.log" echo %%d has errors.log, run this script again to complete it.
echo ==== final campaign finished %date% %time% >> "%LOG%"
echo Done.
pause
endlocal
