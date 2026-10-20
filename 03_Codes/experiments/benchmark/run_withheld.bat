@echo off
REM Withheld-knowledge test (article v51 -> v52). One doctrine family is removed from the
REM seed rules; the family stays in the vocabulary, so the layers may write it back.
REM   A  containment_line withheld, S1-S5 x 30 worlds, closed layer then VODA  -> out_withheld_line  (~4.5 h)
REM   B  evacuation withheld, S4-S5 x 30 worlds, closed layer then VODA        -> out_withheld_evac  (~1.5 h)
REM Compare with E2 F5Ev (closed, full doctrine) and E5 out_nolib (VODA, full doctrine).
REM Resumable; nothing pauses until the end. Closed-layer worlds already in runs.csv are kept;
REM Set DROP_LLM=1 below ONLY when an earlier run had completed VODA worlds without a working
REM proposer (launch failure); it discards every VODA world and redoes them. Worlds that
REM failed with an error are absent from runs.csv and are redone on the next start anyway.
REM A model outage (consecutive timeouts) no longer fails the world: the call waits up to
REM DSS_GENAI_OUTAGE_MAX_S (default 1800 s) for the model to answer again.
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_TIMEOUT=180"
set "SEEDS=201-230"
set "B=experiments\benchmark"
set "LOG=%B%\withheld.log"
set "DROP_LLM=0"
echo ==== withheld-family batch started %date% %time% >> "%LOG%"

echo === 0. Claude access test (same launch path as the campaign) ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','mode','reply','latency_ms','error','cmd')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer through the campaign launch path. Run claude-login.bat, then this script again.
  echo Step 0 failed at %time% >> "%LOG%"
  pause
  exit /b 1
)

if "%DROP_LLM%"=="1" (
  for %%d in (out_withheld_line out_withheld_evac) do if exist "%B%\%%d\runs.csv" python %B%\drop_arm_rows.py %B%\%%d LLM-nofam
)

echo === A. containment_line withheld, S1-S5 ===
set "DSS_WITHHOLD_FAMILY_NAME=containment_line"
set "OUT=%B%\out_withheld_line"
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\errors.log" ( type "%OUT%\errors.log" >> "%OUT%\errors_previous.log" & del "%OUT%\errors.log" )
set "DSS_REQUIRE_GENAI="
python experiments\open_space\campaign.py --exp E7 --arms F5Ev-nofam --scenarios S1,S2,S3,S4,S5 --seeds %SEEDS% --workers 2 --out %OUT%
echo A closed finished with code %errorlevel% at %time% >> "%LOG%"
set "DSS_REQUIRE_GENAI=1"
python experiments\open_space\campaign.py --exp E7 --arms LLM-nofam --scenarios S1,S2,S3,S4,S5 --seeds %SEEDS% --workers 2 --out %OUT%
echo A VODA finished with code %errorlevel% at %time% >> "%LOG%"

echo === B. evacuation withheld, S4-S5 ===
set "DSS_WITHHOLD_FAMILY_NAME=evacuation"
set "OUT=%B%\out_withheld_evac"
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\errors.log" ( type "%OUT%\errors.log" >> "%OUT%\errors_previous.log" & del "%OUT%\errors.log" )
set "DSS_REQUIRE_GENAI="
python experiments\open_space\campaign.py --exp E7 --arms F5Ev-nofam --scenarios S4,S5 --seeds %SEEDS% --workers 2 --out %OUT%
echo B closed finished with code %errorlevel% at %time% >> "%LOG%"
set "DSS_REQUIRE_GENAI=1"
python experiments\open_space\campaign.py --exp E7 --arms LLM-nofam --scenarios S4,S5 --seeds %SEEDS% --workers 2 --out %OUT%
echo B VODA finished with code %errorlevel% at %time% >> "%LOG%"

echo === C. Summary ===
python -c "import csv,os,collections;B=r'%B%';[print(n, collections.Counter((r['arm'],r['scenario']) for r in csv.DictReader(open(os.path.join(B,n,'runs.csv'),encoding='utf-8')))) if os.path.exists(os.path.join(B,n,'runs.csv')) else print(n,'missing') for n in ('out_withheld_line','out_withheld_evac')]"
for %%d in (out_withheld_line out_withheld_evac) do if exist "%B%\%%d\errors.log" echo %%d has errors.log, run this script again to complete it.
echo ==== withheld-family batch finished %date% %time% >> "%LOG%"
echo Done.
pause
endlocal
