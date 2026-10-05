@echo off
REM LLM kolu (F5 + EvFIS + LLM) kampanyasi. 03_Codes klasorunden calistirilir.
REM Oncesinde claude-login.bat ile giris yapilmis olmali (ayni config dizini).
setlocal
cd /d "%~dp0..\.."
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "OUT=experiments\open_space\out_llm"

echo === 0. Claude erisim testi ===
claude -p "Reply with exactly this JSON and nothing else: {\"ok\": true}" --output-format json
if errorlevel 1 (
  echo claude -p calismadi; once claude-login.bat ile giris yapin.
  pause
  exit /b 1
)

echo === 1. Tek koslu duman testi (S3, tohum 201) ===
python experiments\open_space\campaign.py --exp E2 --arms LLM --scenarios S3 --seeds 201 --workers 1 --out %OUT%
if errorlevel 1 (
  echo Duman testi basarisiz.
  pause
  exit /b 1
)

echo === 2. E2, LLM kolu, S1-S5, tohum 201-212 (60 kosu) ===
python experiments\open_space\campaign.py --exp E2 --arms LLM --scenarios S1,S2,S3,S4,S5 --seeds 201-212 --workers 2 --out %OUT%

echo === 3. E2, LLM kolu, S6-S8, tohum 201-212 (36 kosu, kutuphane acik) ===
python experiments\open_space\campaign.py --exp E2 --arms LLM --scenarios S6,S7,S8 --seeds 201-212 --workers 2 --out %OUT%

echo === 4. E1, LLM kolu, S6-S8, uc taktik geri cekilmis, tohum 201-230 (90 kosu, yeniden kesif) ===
python experiments\open_space\campaign.py --exp E1 --arms LLM --seeds 201-230 --workers 2 --out %OUT%

echo === 5. Rapor ===
python experiments\open_space\report.py --out %OUT%
echo Bitti. Ciktilar: %OUT%\runs.csv, proposals.jsonl, admitted.jsonl, summary.txt
pause
endlocal
