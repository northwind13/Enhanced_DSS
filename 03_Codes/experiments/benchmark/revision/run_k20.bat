@echo off
setlocal
REM K20. Cost-function sensitivity: evacuee weight x5, reference burned fraction
REM 0.25, capacity charges x2, response and delay weights 1.0; closed layer and
REM replay of the admitted objects under each setting. CPU only, ~40 min.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "DSS_NO_CONCEPT=1"
set "R=experiments\benchmark\revision"
echo ==== K20 started %date% %time% >> "%R%\revision_K20.log"
python "%R%\k20_cost_weights.py" --workers 12
echo ==== K20 done %errorlevel% %date% %time% >> "%R%\revision_K20.log"
pause
endlocal
