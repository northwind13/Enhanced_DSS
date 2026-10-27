@echo off
setlocal
REM Resume K10 only: existing completed worlds are skipped, never overwritten.
REM Prior Opus rows are documented in k10_model_switch_20261027.json.
REM Sonnet trial rows were excluded and backed up in k10_excluded_sonnet_* .
REM All remaining live worlds use the explicitly pinned Opus model.
cd /d "%~dp0..\..\.."
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto failed
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_MODEL=claude-opus-5"
set "CLAUDE_CODE_EFFORT_LEVEL=high"
set "FORCE_PROMPT_CACHING_5M=1"
set "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "DSS_LLM_SCREEN_FILL=1"
set "DSS_NO_CONCEPT=1"
set "DSS_REQUIRE_GENAI=1"
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "R=experiments\benchmark\revision"
set "B=experiments\benchmark"
set "LOG=%R%\revision_K10.log"

echo === Completed worlds will be preserved. Only missing worlds will run. ===
python "%R%\k10_resume_status.py"
if errorlevel 1 goto failed
REM A free preview: --status makes no model calls.
if /I "%~1"=="--status" exit /b 0
python "%R%\k10_resume_status.py" --check all --quiet
if errorlevel 2 goto failed
if not errorlevel 1 goto done

echo === Waiting for K9 to finish ===
:wait
findstr /C:"==== K9 done" "%R%\revision_K9.log" >nul 2>&1
if errorlevel 1 (
  timeout /t 300 /nobreak >nul
  goto wait
)

echo === Verify the requested Opus model before starting missing worlds ===
python -c "import json,os,sys; from dss import adapt; r=adapt.genai_probe(); print(json.dumps({k:r.get(k) for k in ('ok','model','reported_model','latency_ms','error')})); sys.exit(0 if r.get('ok') and r.get('reported_model','').startswith(os.environ['DSS_GENAI_MODEL']) else 1)"
if errorlevel 1 goto failed
echo ==== K10 resumed %date% %time%, model %DSS_GENAI_MODEL%, cache 5m >> "%LOG%"

python "%R%\k10_resume_status.py" --check k8 --quiet
if errorlevel 2 goto failed
if not errorlevel 1 goto k8_check
echo === K8: missing worlds only; independent gate futures 303 and 404 ===
set "DSS_GATE_SEEDS=303,404"
python experiments\open_space\campaign.py --exp E9 --arms LLMScreen --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers 6 --out "%R%\k8_gate_seeds_nc"
if errorlevel 1 goto failed
set "DSS_GATE_SEEDS="
:k8_check
python "%R%\k10_resume_status.py" --check k8
if errorlevel 1 goto failed
python "%B%\check_noconcept.py" "%R%\k8_gate_seeds_nc" --model
if errorlevel 1 goto failed

python "%R%\k10_resume_status.py" --check k6 --quiet
if errorlevel 2 goto failed
if not errorlevel 1 goto k7
echo === K6: missing coating worlds only, Opus 5 ===
python "%R%\k6_coating.py" --perturb coat_half,coat_off,coat_price_x2,coat_stock_20ha,coat_delay_12min --arms live --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers 6 --out "%R%\k6_coating_nc"
if errorlevel 1 goto failed
python "%R%\k10_resume_status.py" --check k6
if errorlevel 1 goto failed
echo K6 complete %date% %time% >> "%LOG%"

:k7
python "%R%\k10_resume_status.py" --check k7 --quiet
if errorlevel 2 goto failed
if not errorlevel 1 goto done
echo === K7: missing wind-error worlds only, Opus 5 ===
python "%R%\k7_spread.py" --arms live --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers 6 --out "%R%\k7_spread_nc"
if errorlevel 1 goto failed
python "%R%\k10_resume_status.py" --check k7
if errorlevel 1 goto failed
echo K7 complete %date% %time% >> "%LOG%"

:done
python "%R%\k10_resume_status.py" --check all
if errorlevel 1 goto failed
echo ==== K10 done %date% %time% >> "%LOG%"
echo K10 complete. Existing and newly completed results are saved.
if not defined K10_NO_PAUSE pause
exit /b 0

:failed
echo K10 stopped: model verification, checkpoint validation, or a phase failed.
echo Completed results are preserved. Re-run this batch to resume missing worlds.
if not defined K10_NO_PAUSE pause
exit /b 1
