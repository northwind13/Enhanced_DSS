@echo off
REM One combined-error pipeline: closed arm (CPU, 8 workers) then live arm
REM (model calls, 4 workers). Arguments: script stem, label, extra "VAR=value".
setlocal
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "SCRIPT=%~1"
set "LABEL=%~2"
if not "%~3"=="" set "%~3"
set "CLAUDE_CONFIG_DIR=%CD%\.claude-config"
set "DSS_GENAI_MODEL=claude-opus-5"
set "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
set "DSS_GENAI_TIMEOUT=240"
set "DSS_LLM_SCREEN_N=6"
set "DSS_LLM_SCREEN_FILL=1"
set "DSS_NO_CONCEPT=1"
set "DSS_WITHHOLD_FAMILY_NAME="
set "DSS_NO_COORD="
set "DSS_GATE_SEEDS="
set "R=experiments\benchmark\revision"
echo ==== %LABEL% closed started %date% %time% >> "%R%\revision_%LABEL%.log"
set "DSS_REQUIRE_GENAI="
python "%R%\%SCRIPT%.py" --arms closed --workers 8
set "DSS_REQUIRE_GENAI=1"
echo ==== %LABEL% live started %date% %time% >> "%R%\revision_%LABEL%.log"
python "%R%\%SCRIPT%.py" --arms live --workers 4
echo ==== %LABEL% done %errorlevel% %date% %time% >> "%R%\revision_%LABEL%.log"
echo %LABEL% finished.
endlocal
exit /b 0
