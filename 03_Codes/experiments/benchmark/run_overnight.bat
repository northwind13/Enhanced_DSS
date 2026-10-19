@echo off
REM DisasterAware article, overnight batch (v49). Three campaigns in a row, each resumable:
REM   A  E5 replicate 2   VODA (LLM) without the library, same 150 worlds   -> out_nolib_r2   (~3.5 h)
REM   B  E6 bandit ablation  closed layer with the fixed escalation order  -> out_fixed      (~1 h)
REM   C  LOO replay       leave-one-out of the admitted objects of out_nolib -> out_loo       (~2 h)
REM A failure in one campaign does not stop the next; nothing pauses until the end.
REM Run again after an interruption: finished runs are skipped.
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_TIMEOUT=180"
set "SC=S1,S2,S3,S4,S5"
set "SEEDS=201-230"
set "B=experiments\benchmark"
for %%d in (out_loo_smoke out_loo_smoke2 _old_loo_smoke out_fixed_smoke) do if exist "%B%\%%d" rmdir /s /q "%B%\%%d"
set "LOG=%B%\overnight.log"
echo ==== overnight batch started %date% %time% >> "%LOG%"

echo === A. E5 replicate 2: VODA without the library (%SC%, seeds %SEEDS%) ===
set "OUT=%B%\out_nolib_r2"
if not exist "%OUT%" mkdir "%OUT%"
where claude >nul 2>nul
if errorlevel 1 (
  echo claude CLI not on PATH, campaign A skipped >> "%LOG%"
  goto :B
)
claude -p "Reply with exactly this JSON and nothing else: {\"ok\": true}" --output-format json > "%OUT%\claude_check.json"
findstr /c:"ok" "%OUT%\claude_check.json" >nul
if errorlevel 1 (
  echo claude did not answer, campaign A skipped >> "%LOG%"
  goto :B
)
if exist "%OUT%\errors.log" (
  type "%OUT%\errors.log" >> "%OUT%\errors_previous.log"
  del "%OUT%\errors.log"
)
set "DSS_REQUIRE_GENAI=1"
python experiments\open_space\campaign.py --exp E5 --arms LLM --scenarios %SC% --seeds %SEEDS% --workers 2 --out %OUT%
echo A finished with code %errorlevel% at %time% >> "%LOG%"
set "DSS_REQUIRE_GENAI="

:B
echo === B. E6 bandit ablation: closed layer with the fixed stage order ===
set "OUT=%B%\out_fixed"
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\errors.log" (
  type "%OUT%\errors.log" >> "%OUT%\errors_previous.log"
  del "%OUT%\errors.log"
)
python experiments\open_space\campaign.py --exp E6 --arms F5Ev-fixed --scenarios %SC% --seeds %SEEDS% --workers 2 --out %OUT%
echo B finished with code %errorlevel% at %time% >> "%LOG%"

:C
echo === C. Leave-one-out replay of the admitted objects (source out_nolib) ===
set "OUT=%B%\out_loo"
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\errors.log" (
  type "%OUT%\errors.log" >> "%OUT%\errors_previous.log"
  del "%OUT%\errors.log"
)
python %B%\loo.py --src %B%\out_nolib --out %OUT% --workers 2
echo C finished with code %errorlevel% at %time% >> "%LOG%"

echo === D. Summary ===
python -c "import csv,os,collections;B=r'%B%';[print(n, len(list(csv.DictReader(open(os.path.join(B,n,f),encoding='utf-8')))), 'rows') if os.path.exists(os.path.join(B,n,f)) else print(n,'missing') for n,f in (('out_nolib_r2','runs.csv'),('out_fixed','runs.csv'),('out_loo','loo.csv'))]"
for %%d in (out_nolib_r2 out_fixed out_loo) do if exist "%B%\%%d\errors.log" echo %%d has errors.log, run this script again to complete it.
echo ==== overnight batch finished %date% %time% >> "%LOG%"
echo.
echo Done. Log: %LOG%
pause
endlocal
