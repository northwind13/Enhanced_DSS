@echo off
REM Article v52 campaign. VODA is redefined: the language model writes several candidates per
REM call, they are screened on the shadow simulator with the forecast budget of E8 (36), and the
REM best goes to the gates. Closed-layer references (E2, E6, E7 closed arms) are unchanged and are
REM not rerun. Every step is resumable: run the script again and finished worlds are skipped.
REM   1  E10 RandBlind   one unpriced random candidate per call (pairs with E5 LLM)   CPU   ~20 min
REM   2  E9  LLMScreen   VODA v52, S1-S5 x 30                                           LLM   ~3-4 h
REM   3  E7  LLMScreen-nofam, containment line withheld, S1-S5                         LLM   ~3-4 h
REM   4  E7  LLMScreen-nofam, evacuation withheld, S4-S5                               LLM   ~1.5 h
REM   5  LOO replay of the E9 admissions                                                CPU   ~1 h
REM   6  E9  second replicate (proposer variance)                                       LLM   ~3-4 h
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "SEEDS=201-230"
set "CPUW=10"
set "LLMW=4"
set "B=experiments\benchmark"
set "LOG=%B%\v52.log"
echo ==== v52 campaign started %date% %time% >> "%LOG%"

echo === 0. Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  pause
  exit /b 1
)

echo === 1. E10 random, unscreened ===
set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
python experiments\open_space\campaign.py --exp E10 --arms RandBlind --scenarios S1,S2,S3,S4,S5 --seeds %SEEDS% --workers %CPUW% --out %B%\out_randblind
echo 1 finished %errorlevel% %time% >> "%LOG%"

echo === 2. E9 VODA v52, smoke world then all ===
set "DSS_REQUIRE_GENAI=1"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios S3 --seeds 201 --workers 1 --out %B%\out_v52_main
python -c "import json;P=[json.loads(l) for l in open(r'%B%\out_v52_main\proposals.jsonl',encoding='utf-8')];m=[(p.get('search') or {}).get('model_candidates') for p in P];print('smoke: attempts',len(P),'model candidates per call',m[:10],'admitted',sum(bool(p.get('accepted')) for p in P))"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios S1,S2,S3,S4,S5 --seeds %SEEDS% --workers %LLMW% --out %B%\out_v52_main
echo 2 finished %errorlevel% %time% >> "%LOG%"

echo === 3. E7 containment line withheld, VODA v52 ===
set "DSS_WITHHOLD_FAMILY_NAME=containment_line"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios S1,S2,S3,S4,S5 --seeds %SEEDS% --workers %LLMW% --out %B%\out_v52_line
echo 3 finished %errorlevel% %time% >> "%LOG%"

echo === 4. E7 evacuation withheld, VODA v52 ===
set "DSS_WITHHOLD_FAMILY_NAME=evacuation"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios S4,S5 --seeds %SEEDS% --workers %LLMW% --out %B%\out_v52_evac
echo 4 finished %errorlevel% %time% >> "%LOG%"
set "DSS_WITHHOLD_FAMILY_NAME="

echo === 5. LOO replay of the E9 admissions ===
set "DSS_REQUIRE_GENAI="
python %B%\loo.py --src %B%\out_v52_main --exp E9 --arm LLMScreen --out %B%\out_v52_loo --workers %CPUW%
echo 5 finished %errorlevel% %time% >> "%LOG%"

echo === 6. E9 second replicate ===
set "DSS_REQUIRE_GENAI=1"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios S1,S2,S3,S4,S5 --seeds %SEEDS% --workers %LLMW% --out %B%\out_v52_main_r2
echo 6 finished %errorlevel% %time% >> "%LOG%"

echo === Summary ===
for %%d in (out_randblind out_v52_main out_v52_line out_v52_evac out_v52_main_r2) do python -c "import csv,os,collections;p=r'%B%\%%d\runs.csv';print('%%d', dict(collections.Counter(x['arm'] for x in csv.DictReader(open(p,encoding='utf-8')))) if os.path.exists(p) else 'missing')"
for %%d in (out_randblind out_v52_main out_v52_line out_v52_evac out_v52_main_r2) do if exist "%B%\%%d\errors.log" echo %%d has errors.log, run this script again to complete it.
echo ==== v52 campaign finished %date% %time% >> "%LOG%"
echo Done.
pause
endlocal
