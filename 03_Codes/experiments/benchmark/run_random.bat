@echo off
REM E8, proposer control (article v51 -> v52). The E5 setting with the language model
REM replaced by a random search over the same grammar and the same gates. No model calls,
REM CPU only. Pairs with out_nolib (E5 LLM). Resumable: run again to finish missing worlds.
setlocal
cd /d "%~dp0..\.."
set "B=experiments\benchmark"
set "OUT=%B%\out_random"
set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\errors.log" ( type "%OUT%\errors.log" >> "%OUT%\errors_previous.log" & del "%OUT%\errors.log" )
echo ==== E8 random proposer started %date% %time% >> "%B%\random.log"
python experiments\open_space\campaign.py --exp E8 --arms Random --scenarios S1,S2,S3,S4,S5 --seeds 201-230 --workers 8 --out %OUT%
echo E8 finished with code %errorlevel% at %time% >> "%B%\random.log"
python -c "import csv,collections;r=list(csv.DictReader(open(r'%OUT%\runs.csv',encoding='utf-8')));print('E8 rows',len(r),dict(collections.Counter(x['scenario'] for x in r)))"
if exist "%OUT%\errors.log" echo out_random has errors.log, run this script again to complete it.
echo Done.
pause
endlocal
