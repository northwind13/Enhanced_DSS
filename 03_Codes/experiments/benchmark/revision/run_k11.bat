@echo off
setlocal
REM K11. Hand-written doctrine over the seven effects, closed layer only.
REM No language model is called. About one hour with 16 workers.
cd /d "%~dp0..\..\.."
call ".venv\Scripts\activate.bat"
set "DSS_REQUIRE_GENAI="
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "R=experiments\benchmark\revision"
echo ==== K11 started %date% %time% >> "%R%\revision_K11.log"
python %R%\k11_expert_doctrine.py --workers 16
echo ==== K11 done %errorlevel% %date% %time% >> "%R%\revision_K11.log"
pause
endlocal
