@echo off
setlocal
REM K4. Hindcast rerun with a 5 min physical step, so the area-matched stop
REM lands within 5 min of the observed area (the 30 min step overshot F1 by
REM 37%), then the equal-area disk and wind-aligned ellipse references with
REM POD, FAR and Sorensen for Table V. CPU only, uses the cached FIRMS and
REM ERA5 inputs under validation\cache. About 1 h.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "R=experiments\benchmark\revision"
REM first run on a machine without validation\cache: set FIRMS_MAP_KEY=<key>
if not "%FIRMS_MAP_KEY%"=="" set "KEYARG=--firms-key %FIRMS_MAP_KEY%"
echo ==== K4 step5 started %date% %time% >> "%R%\revision_K4.log"
python "%R%\k4_hindcast.py" --stop area --seeds 50 --step 5 --out "%R%\k4_hindcast_step5" %KEYARG%
python "%R%\k4_ellipse.py" --ckpt "%R%\k4_hindcast_step5\window12_stop_area"
echo ==== K4 done %errorlevel% %date% %time% >> "%R%\revision_K4.log"
pause
endlocal
