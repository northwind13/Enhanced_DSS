@echo off
setlocal
REM Choose the language model of the batch runs: provider (claude / codex),
REM model, reasoning effort and fallback. Writes llm.cfg in this folder.
cd /d "%~dp0"
set "CLAUDE_CONFIG_DIR=%~dp0.claude-config"
set "CODEX_HOME=%~dp0.codex-config"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m dss.llm select
endlocal
pause
