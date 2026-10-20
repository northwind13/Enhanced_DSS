@echo off
REM ======================================================================================
REM  FINAL PRICED CAMPAIGN, article v52. Same 150 worlds (S1-S5 x seeds 201-230) for every
REM  test. Closed space = the 5 seed rules with self-tuning. Open space (VODA) = closed space
REM  plus the LLM proposer whose candidates are screened on the simulator before the gates.
REM  Every order is priced (capacity, retardant, ignition, warning). Resumable.
REM
REM  T1  REFERENCE   no action (T0), 5 static rules (F5), closed space (F5Ev)          CPU
REM  T2  SELECTOR    closed space with a fixed order instead of the stage selector     CPU
REM  T3  AGENTS      closed space with 1 and 16 regional agents (T1 has 4)              CPU
REM  T4  PROPOSER-R1 random proposer, single candidate                                 CPU
REM  T5  PROPOSER-R2 random proposer, screened, 36 forecasts per call (same as T6)     CPU
REM  T5b PROPOSER-R2s random proposer, screened, small budget (same as T6b)             CPU
REM  T6  VODA        open space, model candidates + variations, 36 forecasts per call  LLM
REM  T6b VODA-M      open space, model candidates only (DSS_LLM_SCREEN_FILL=0)         LLM
REM  T7  PROPOSER-L1 LLM proposer, single candidate                                    LLM
REM  T8  DOCTRINE    containment line withheld from the 5 rules, closed and open      CPU+LLM
REM  T8b DOCTRINE-M  as T8, open arm with model candidates only                        LLM
REM  T9  REPEAT      VODA second run on the same worlds                                LLM
REM  T9b REPEAT-M    T6b second run on the same worlds                                 LLM
REM  T10 BENEFIT     leave-one-out replay of every object VODA admitted                CPU
REM  T10b BENEFIT-M  leave-one-out replay of every object T6b admitted                 CPU
REM  T11 REALTIME    decision-cycle time against agents and grid size, one process    CPU
REM  T12 DOCTRINE-2  closed layer with suppression_effort, then resource_deployment      CPU
REM                  withheld; choose_family.py keeps the family whose removal costs the
REM                  closed layer most (final\t12_choice.txt); open arm on that family LLM
REM  T12b DOCTRINE-2M as T12, open arm with model candidates only                       LLM
REM  Every run also records per-cycle decision time and how often the coordinator derated a region.
REM  DSS_NO_CONCEPT=1: the open layer proposes rules, macros and composed interventions only, no
REM  new decision concepts. After every open-layer step check_noconcept.py stops the script if a
REM  concept proposal was recorded. The b steps run VODA with the model candidates only, so the
REM  two definitions of VODA can be compared on the same worlds before one goes into the article.
REM ======================================================================================
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "DSS_NO_COORD="
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_CONCEPT=1"
set "DSS_LLM_SCREEN_FILL=1"
set "SEEDS=201-230"
set "SC=S1,S2,S3,S4,S5"
set "B=experiments\benchmark"
set "P=%B%\final"
set "LOG=%B%\final_priced.log"
set /a "CPUW=%NUMBER_OF_PROCESSORS%-2"
for /f %%m in ('powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/524288)"') do set "MEMW=%%m"
if defined MEMW if %MEMW% LSS %CPUW% set "CPUW=%MEMW%"
if %CPUW% LSS 2 set "CPUW=2"
set "LLMW=6"
if not exist "%P%" mkdir "%P%"
echo Machine: %NUMBER_OF_PROCESSORS% logical processors. CPU workers: %CPUW%, model workers: %LLMW%
echo ==== final priced campaign started %date% %time%, CPU workers %CPUW% >> "%LOG%"

echo === 0. Checks ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  pause
  exit /b 1
)
python -c "from disaster_phyengine.config import CostParams as C; c=C(); print('prices: coat',c.coat_capacity_steps,'ignite',c.ignite_capacity_steps,'warn',c.warn_capacity_per_step)"
python -c "import sys; from dss import adapt; s=adapt._genai_schema(); print('new concepts offered to the model:', 'new_concept' in s); sys.exit(1 if 'new_concept' in s else 0)"
if errorlevel 1 (
  echo DSS_NO_CONCEPT is not in effect. Stopping.
  pause
  exit /b 1
)

set "DSS_REQUIRE_GENAI="
echo === T1 REFERENCE ===
python experiments\open_space\campaign.py --exp E2P --arms T0,F5,F5Ev --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\t1_reference
echo T1 %errorlevel% %time% >> "%LOG%"
echo === T2 SELECTOR ===
python experiments\open_space\campaign.py --exp E6 --arms F5Ev-fixed --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\t2_selector
echo T2 %errorlevel% %time% >> "%LOG%"
echo === T3 AGENTS ===
python experiments\open_space\campaign.py --exp E11 --arms F5Ev-r1,F5Ev-r16 --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\t3_agents
echo T3 %errorlevel% %time% >> "%LOG%"
echo === T4 PROPOSER-R1 ===
python experiments\open_space\campaign.py --exp E10 --arms RandBlind --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\t4_random_single
echo T4 %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t4_random_single
if errorlevel 1 goto :stop
echo === T5 PROPOSER-R2 ===
python experiments\open_space\campaign.py --exp E8 --arms Random --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\t5_random_screened
echo T5 %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t5_random_screened
if errorlevel 1 goto :stop
echo === T5b PROPOSER-R2s, small screening budget ===
python experiments\open_space\campaign.py --exp E8 --arms Random --budget 6 --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\t5b_random_screened_small
echo T5b %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t5b_random_screened_small
if errorlevel 1 goto :stop
echo === T8 DOCTRINE, closed arms ===
set "DSS_WITHHOLD_FAMILY_NAME=containment_line"
python experiments\open_space\campaign.py --exp E7 --arms F5Ev-nofam --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\t8_doctrine_line
set "DSS_WITHHOLD_FAMILY_NAME="
echo T8 closed %errorlevel% %time% >> "%LOG%"
echo === T12 DOCTRINE-2, closed arms ===
set "DSS_WITHHOLD_FAMILY_NAME=suppression_effort"
python experiments\open_space\campaign.py --exp E7 --arms F5Ev-nofam --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\t12_closed_suppression_effort
set "DSS_WITHHOLD_FAMILY_NAME=resource_deployment"
python experiments\open_space\campaign.py --exp E7 --arms F5Ev-nofam --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\t12_closed_resource_deployment
set "DSS_WITHHOLD_FAMILY_NAME="
echo T12 closed %errorlevel% %time% >> "%LOG%"
set "FAM="
for /f %%f in ('python %B%\choose_family.py %P% suppression_effort resource_deployment') do set "FAM=%%f"
if not defined FAM (
  echo choose_family.py found no family. See %P%\t12_choice.txt
  goto :stop
)
type "%P%\t12_choice.txt"
type "%P%\t12_choice.txt" >> "%LOG%"

set "DSS_REQUIRE_GENAI=1"
echo === T6 VODA, one smoke world first ===
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios S3 --seeds 201 --workers 1 --out %P%\t6_voda
python %B%\check_noconcept.py %P%\t6_voda --model
if errorlevel 1 goto :stop
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t6_voda
echo T6 %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t6_voda --model
if errorlevel 1 goto :stop
echo === T6b VODA-M, model candidates only ===
set "DSS_LLM_SCREEN_FILL=0"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t6b_voda_model_only
set "DSS_LLM_SCREEN_FILL=1"
echo T6b %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t6b_voda_model_only --model --no-mutation
if errorlevel 1 goto :stop
echo === T7 PROPOSER-L1 ===
python experiments\open_space\campaign.py --exp E5 --arms LLM --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t7_llm_single
echo T7 %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t7_llm_single --model
if errorlevel 1 goto :stop
echo === T8 DOCTRINE, open arms ===
set "DSS_WITHHOLD_FAMILY_NAME=containment_line"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t8_doctrine_line
set "DSS_WITHHOLD_FAMILY_NAME="
echo T8 open %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t8_doctrine_line --model
if errorlevel 1 goto :stop
echo === T8b DOCTRINE-M, open arm with model candidates only ===
set "DSS_WITHHOLD_FAMILY_NAME=containment_line"
set "DSS_LLM_SCREEN_FILL=0"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t8b_doctrine_line_model_only
set "DSS_LLM_SCREEN_FILL=1"
set "DSS_WITHHOLD_FAMILY_NAME="
echo T8b %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t8b_doctrine_line_model_only --model --no-mutation
if errorlevel 1 goto :stop
echo === T9 REPEAT ===
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t9_repeat
echo T9 %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t9_repeat --model
if errorlevel 1 goto :stop
echo === T9b REPEAT-M ===
set "DSS_LLM_SCREEN_FILL=0"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t9b_repeat_model_only
set "DSS_LLM_SCREEN_FILL=1"
echo T9b %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t9b_repeat_model_only --model --no-mutation
if errorlevel 1 goto :stop
echo === T12 DOCTRINE-2, open arm, %FAM% withheld, model + variations ===
set "DSS_WITHHOLD_FAMILY_NAME=%FAM%"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t12_open_%FAM%
echo T12 open %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t12_open_%FAM% --model
if errorlevel 1 goto :stop
echo === T12b DOCTRINE-2M, open arm, %FAM% withheld, model candidates only ===
set "DSS_LLM_SCREEN_FILL=0"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\t12b_open_%FAM%_model_only
set "DSS_LLM_SCREEN_FILL=1"
set "DSS_WITHHOLD_FAMILY_NAME="
echo T12b open %errorlevel% %time% >> "%LOG%"
python %B%\check_noconcept.py %P%\t12b_open_%FAM%_model_only --model --no-mutation
if errorlevel 1 goto :stop

set "DSS_REQUIRE_GENAI="
echo === T10 BENEFIT ===
python %B%\loo.py --src %P%\t6_voda --exp E9 --arm LLMScreen --out %P%\t10_benefit --workers %CPUW%
echo T10 %errorlevel% %time% >> "%LOG%"
echo === T10b BENEFIT-M ===
python %B%\loo.py --src %P%\t6b_voda_model_only --exp E9 --arm LLMScreen --out %P%\t10b_benefit_model_only --workers %CPUW%
echo T10b %errorlevel% %time% >> "%LOG%"

echo === T11 REALTIME (single process, nothing else running) ===
python experiments\cost_probe.py --phase latency
python experiments\cost_probe.py --phase scale
if not exist "%P%\t11_realtime" mkdir "%P%\t11_realtime"
copy /y experiments\out\cost_latency.csv "%P%\t11_realtime\" >nul
copy /y experiments\out\cost_scale.csv "%P%\t11_realtime\" >nul
echo T11 %errorlevel% %time% >> "%LOG%"

echo === Summary ===
python -c "import csv,os,collections;P=r'%P%';[print(d, dict(collections.Counter(r['arm'] for r in csv.DictReader(open(os.path.join(P,d,'runs.csv'),encoding='utf-8'))))) for d in sorted(os.listdir(P)) if os.path.exists(os.path.join(P,d,'runs.csv'))]"
for /d %%d in (%P%\*) do if exist "%%d\errors.log" echo %%d has errors.log, run this script again to complete it.
echo ==== final priced campaign finished %date% %time% >> "%LOG%"
echo Done.
pause
endlocal
exit /b 0

:stop
echo ==== stopped by a check %date% %time% >> "%LOG%"
echo The script stopped: see the check line above. Nothing after this step was run.
pause
endlocal
exit /b 1
