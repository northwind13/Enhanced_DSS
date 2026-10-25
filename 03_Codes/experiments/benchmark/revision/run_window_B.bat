@echo off
REM Window B: K4 hindcast, 4 fires x 50 seeds, first-12-h footprint, area-matched stop, no wind correction, one process. Resumable per seed: run again to continue.
setlocal
cd /d "%~dp0..\..\.."
REM the project environment (rasterio, numpy, ...) lives in 03_Codes\.venv
call ".venv\Scripts\activate.bat"
set "R=experiments\benchmark\revision"
set "LOG=%R%\revision_B.log"
echo ==== window B started %date% %time% >> "%LOG%"
for %%c in (canakkale2023 manavgat2021 marmaris2021 milas2021) do (
  echo === K4. first 12 h footprint, area-matched stop, no wind correction, %%c, 50 seeds ===
  python %R%\k4_hindcast.py --protocol window12 --stop area --seeds 50 --cases %%c
  echo K4 window12 %%c finished %time% >> "%LOG%"
)
echo === summary, K1-K3 may still be running in window A ===
python %R%\revision_tables.py
echo ==== window B finished %date% %time% >> "%LOG%"
echo Window B done. If window A has finished too, run: python %R%\revision_tables.py
pause
endlocal
