@echo off
REM ======================================================================================
REM  RERUN OF THE WORLDS WHERE THE OPEN LAYER WAS RETIRED BY MISTAKE (article v52).
REM  A call whose candidates all failed the cheap gates returned None, which stage 3 read
REM  as "model unreachable" and the open layer was switched off for the rest of the world
REM  (loop.py _genai_dead). Fixed in proposers.py (SearchProposer.propose). The rows of the
REM  affected worlds were moved to <step>\dead_worlds\ by rerun_dead_worlds.py; this script
REM  reruns exactly those worlds, then redoes the leave-one-out replays. Resumable.
REM  Steps and worlds: T6 27, T6b 25, T9 22, T9b 32, T8 open 26, T8b 32, T12 open 42, T12b 42.
REM ======================================================================================
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "DSS_NO_COORD="
set "DSS_NO_CONCEPT=1"
set "DSS_LLM_SCREEN_FILL=1"
set "DSS_WITHHOLD_FAMILY_NAME="
set "SEEDS=201-230"
set "SC=S1,S2,S3,S4,S5"
set "B=experiments\benchmark"
set "P=%B%\final"
set "LOG=%B%\final_priced.log"
set /a "CPUW=%NUMBER_OF_PROCESSORS%-2"
if %CPUW% LSS 2 set "CPUW=2"
set "LLMW=6"
echo ==== rerun of retired worlds started %date% %time% >> "%LOG%"

echo === 0. Checks ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  pause
  exit /b 1
)
python -c "import sys; from dss import adapt; s=adapt._genai_schema(); sys.exit(1 if 'new_concept' in s else 0)"
if errorlevel 1 ( echo DSS_NO_CONCEPT is not in effect. Stopping. & pause & exit /b 1 )
python -c "import sys,inspect; from dss import proposers as P; src=inspect.getsource(P.SearchProposer.propose); sys.exit(0 if 'no candidate passed the cheap gates' in src else 1)"
if errorlevel 1 ( echo proposers.py does not carry the fix. Stopping. & pause & exit /b 1 )

set "DSS_REQUIRE_GENAI=1"
echo === T6 VODA, retired worlds ===
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t6_voda
echo T6 rerun %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t6_voda --model
if errorlevel 1 goto :stop
echo === T6b VODA-M, retired worlds ===
set "DSS_LLM_SCREEN_FILL=0"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t6b_voda_model_only
set "DSS_LLM_SCREEN_FILL=1"
echo T6b rerun %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t6b_voda_model_only --model --no-mutation
if errorlevel 1 goto :stop
echo === T9 REPEAT, retired worlds ===
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t9_repeat
echo T9 rerun %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t9_repeat --model
if errorlevel 1 goto :stop
echo === T9b REPEAT-M, retired worlds ===
set "DSS_LLM_SCREEN_FILL=0"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t9b_repeat_model_only
set "DSS_LLM_SCREEN_FILL=1"
echo T9b rerun %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t9b_repeat_model_only --model --no-mutation
if errorlevel 1 goto :stop
echo === T8 open, line withheld, retired worlds ===
set "DSS_WITHHOLD_FAMILY_NAME=containment_line"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t8_doctrine_line
echo T8 open rerun %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t8_doctrine_line --model
if errorlevel 1 goto :stop
echo === T8b, line withheld, model only, retired worlds ===
set "DSS_LLM_SCREEN_FILL=0"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t8b_doctrine_line_model_only
set "DSS_LLM_SCREEN_FILL=1"
echo T8b rerun %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t8b_doctrine_line_model_only --model --no-mutation
if errorlevel 1 goto :stop
echo === T12 open, resource_deployment withheld, retired worlds ===
set "DSS_WITHHOLD_FAMILY_NAME=resource_deployment"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t12_open_resource_deployment
echo T12 open rerun %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t12_open_resource_deployment --model
if errorlevel 1 goto :stop
echo === T12b, resource_deployment withheld, model only, retired worlds ===
set "DSS_LLM_SCREEN_FILL=0"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t12b_open_resource_deployment_model_only
set "DSS_LLM_SCREEN_FILL=1"
set "DSS_WITHHOLD_FAMILY_NAME="
echo T12b rerun %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t12b_open_resource_deployment_model_only --model --no-mutation
if errorlevel 1 goto :stop

set "DSS_REQUIRE_GENAI="
echo === T10 BENEFIT, full leave-one-out on the completed T6 ===
python %B%\loo.py --src %P%\t6_voda --exp E9 --arm LLMScreen --out %P%\t10_benefit --workers %CPUW%
echo T10 rerun %errorlevel% %time% >> "%LOG%"
echo === T10b BENEFIT-M ===
python %B%\loo.py --src %P%\t6b_voda_model_only --exp E9 --arm LLMScreen --out %P%\t10b_benefit_model_only --workers %CPUW%
echo T10b rerun %errorlevel% %time% >> "%LOG%"

echo === Summary ===
python -c "import csv,os,collections;P=r'%P%';[print(d, dict(collections.Counter(r['arm'] for r in csv.DictReader(open(os.path.join(P,d,'runs.csv'),encoding='utf-8'))))) for d in sorted(os.listdir(P)) if os.path.exists(os.path.join(P,d,'runs.csv'))]"
for /d %%d in (%P%\*) do if exist "%%d\errors.log" echo %%d has errors.log, run this script again to complete it.
echo ==== rerun of retired worlds finished %date% %time% >> "%LOG%"
echo Done.
pause
endlocal
exit /b 0

:stop
echo ==== rerun stopped by a check %date% %time% >> "%LOG%"
echo The script stopped: see the line above. Nothing after this step was run.
pause
endlocal
exit /b 1
