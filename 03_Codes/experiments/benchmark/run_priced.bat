@echo off
REM PRICED CAMPAIGN (article v52). Every order now carries a response cost: fielded capacity as
REM before, plus retardant coating, tactical ignition and warnings (prices in CostParams). The
REM adaptation trigger reads the total cost, so every configuration is rerun on the same 150 worlds.
REM Resumable: if it stops, run it again and finished worlds are skipped. About 9-10 hours.
REM   CPU  1  E2 T0, F5, F5Ev  2 E6 fixed order  3 E7 closed arms  4 E10 random single  5 E8 random screened
REM   LLM  6 E5 LLM single  7 E9 LLM screened  8 E7 LLM arms  9 E5 second replicate
REM   CPU 10 leave-one-out replay of step 6 (with retardant accounting)
REM   LLM 11 E7 with the screened LLM  12 E9 second replicate
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "DSS_NO_COORD="
set "SEEDS=201-230"
set "SC=S1,S2,S3,S4,S5"
set "B=experiments\benchmark"
set "P=%B%\priced"
set "LOG=%B%\priced.log"
set /a "CPUW=%NUMBER_OF_PROCESSORS%-2"
for /f %%m in ('powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/524288)"') do set "MEMW=%%m"
if defined MEMW if %MEMW% LSS %CPUW% set "CPUW=%MEMW%"
if %CPUW% LSS 2 set "CPUW=2"
set "LLMW=6"
if not exist "%P%" mkdir "%P%"
echo Machine: %NUMBER_OF_PROCESSORS% logical processors. CPU workers: %CPUW%, model workers: %LLMW%
echo ==== priced campaign started %date% %time%, CPU workers %CPUW% >> "%LOG%"

echo === 0. Claude access test ===
python -c "import json,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','latency_ms','error')})); sys.exit(0 if r.get('ok') else 1)"
if errorlevel 1 (
  echo The model did not answer. Run claude-login.bat, then this script again.
  pause
  exit /b 1
)
python -c "from disaster_phyengine.config import CostParams as C; c=C(); print('prices: coat',c.coat_capacity_steps,'ignite',c.ignite_capacity_steps,'warn',c.warn_capacity_per_step)"

set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
echo === 1. E2 reference arms ===
python experiments\open_space\campaign.py --exp E2P --arms T0,F5,F5Ev --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\e2
echo 1 %errorlevel% %time% >> "%LOG%"
echo === 2. E6 fixed order ===
python experiments\open_space\campaign.py --exp E6 --arms F5Ev-fixed --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\e6
echo 2 %errorlevel% %time% >> "%LOG%"
echo === 3. E7 closed arms ===
set "DSS_WITHHOLD_FAMILY_NAME=containment_line"
python experiments\open_space\campaign.py --exp E7 --arms F5Ev-nofam --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\e7_line
set "DSS_WITHHOLD_FAMILY_NAME=evacuation"
python experiments\open_space\campaign.py --exp E7 --arms F5Ev-nofam --scenarios S4,S5 --seeds %SEEDS% --workers %CPUW% --out %P%\e7_evac
set "DSS_WITHHOLD_FAMILY_NAME="
echo 3 %errorlevel% %time% >> "%LOG%"
echo === 4. E10 random, single candidate ===
python experiments\open_space\campaign.py --exp E10 --arms RandBlind --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\e10
echo 4 %errorlevel% %time% >> "%LOG%"
echo === 5. E8 random, screened ===
python experiments\open_space\campaign.py --exp E8 --arms Random --scenarios %SC% --seeds %SEEDS% --workers %CPUW% --out %P%\e8
echo 5 %errorlevel% %time% >> "%LOG%"

set "DSS_REQUIRE_GENAI=1"
echo === 6. E5 LLM, single candidate ===
python experiments\open_space\campaign.py --exp E5 --arms LLM --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\e5
echo 6 %errorlevel% %time% >> "%LOG%"
echo === 7. E9 LLM, screened ===
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\e9
echo 7 %errorlevel% %time% >> "%LOG%"
echo === 8. E7 LLM arms ===
set "DSS_WITHHOLD_FAMILY_NAME=containment_line"
python experiments\open_space\campaign.py --exp E7 --arms LLM-nofam --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\e7_line
set "DSS_WITHHOLD_FAMILY_NAME=evacuation"
python experiments\open_space\campaign.py --exp E7 --arms LLM-nofam --scenarios S4,S5 --seeds %SEEDS% --workers %LLMW% --out %P%\e7_evac
set "DSS_WITHHOLD_FAMILY_NAME="
echo 8 %errorlevel% %time% >> "%LOG%"
echo === 9. E5 second replicate ===
python experiments\open_space\campaign.py --exp E5 --arms LLM --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\e5_r2
echo 9 %errorlevel% %time% >> "%LOG%"

set "DSS_REQUIRE_GENAI="
echo === 10. Leave-one-out replay of step 6 ===
python %B%\loo.py --src %P%\e5 --exp E5 --arm LLM --out %P%\loo --workers %CPUW%
echo 10 %errorlevel% %time% >> "%LOG%"

set "DSS_REQUIRE_GENAI=1"
echo === 11. E7 with the screened LLM ===
set "DSS_WITHHOLD_FAMILY_NAME=containment_line"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\e7_line
set "DSS_WITHHOLD_FAMILY_NAME=evacuation"
python experiments\open_space\campaign.py --exp E7 --arms LLMScreen-nofam --scenarios S4,S5 --seeds %SEEDS% --workers %LLMW% --out %P%\e7_evac
set "DSS_WITHHOLD_FAMILY_NAME="
echo 11 %errorlevel% %time% >> "%LOG%"
echo === 12. E9 second replicate ===
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios %SC% --seeds %SEEDS% --workers %LLMW% --out %P%\e9_r2
echo 12 %errorlevel% %time% >> "%LOG%"
set "DSS_REQUIRE_GENAI="

echo === Summary ===
python -c "import csv,os,collections;P=r'%P%';[print(d, dict(collections.Counter(r['arm'] for r in csv.DictReader(open(os.path.join(P,d,'runs.csv'),encoding='utf-8'))))) for d in sorted(os.listdir(P)) if os.path.exists(os.path.join(P,d,'runs.csv'))]"
for /d %%d in (%P%\*) do if exist "%%d\errors.log" echo %%d has errors.log, run this script again to complete it.
echo ==== priced campaign finished %date% %time% >> "%LOG%"
echo Done.
pause
endlocal
