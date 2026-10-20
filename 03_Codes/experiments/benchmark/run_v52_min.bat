@echo off
REM Article v52, reduced campaign: the two cells missing from the proposer table.
REM   1  E10 RandBlind   one unpriced random candidate per call, pairs with E5 (LLM)        CPU  ~20 min
REM   2  E9  LLMScreen   model candidates screened with the E8 forecast budget, S1-S5 x 30  LLM  ~3-4 h
REM Main results (E5), E7, LOO and the replicate are NOT rerun. Resumable.
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "DSS_WITHHOLD_FAMILY_NAME="
set "SEEDS=201-230"
set "B=experiments\benchmark"
set "LOG=%B%\v52.log"
echo ==== v52 reduced campaign started %date% %time% >> "%LOG%"
echo === 0. Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  pause
  exit /b 1
)
echo === 1. E10 random, unscreened ===
set "DSS_REQUIRE_GENAI="
python experiments\open_space\campaign.py --exp E10 --arms RandBlind --scenarios S1,S2,S3,S4,S5 --seeds %SEEDS% --workers 10 --out %B%\out_randblind
echo 1 finished %errorlevel% %time% >> "%LOG%"
echo === 2. E9 model candidates, screened ===
set "DSS_REQUIRE_GENAI=1"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios S3 --seeds 201 --workers 1 --out %B%\out_v52_main
python -c "import json;P=[json.loads(l) for l in open(r'%B%\out_v52_main\proposals.jsonl',encoding='utf-8')];m=[(p.get('search') or {}).get('model_candidates') for p in P];print('smoke: model candidates per call',m[:10])"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios S1,S2,S3,S4,S5 --seeds %SEEDS% --workers 4 --out %B%\out_v52_main
echo 2 finished %errorlevel% %time% >> "%LOG%"
echo === Summary ===
for %%d in (out_randblind out_v52_main) do python -c "import csv,os,collections;p=r'%B%\%%d\runs.csv';print('%%d', dict(collections.Counter(x['scenario'] for x in csv.DictReader(open(p,encoding='utf-8')))) if os.path.exists(p) else 'missing')"
for %%d in (out_randblind out_v52_main) do if exist "%B%\%%d\errors.log" echo %%d has errors.log, run this script again to complete it.
echo ==== v52 reduced campaign finished %date% %time% >> "%LOG%"
echo Done.
pause
endlocal
