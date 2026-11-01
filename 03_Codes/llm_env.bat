@echo off
REM Shared language-model settings for every run script. Called with
REM   call "%CD%\llm_env.bat"
REM from 03_Codes. Values come from llm.cfg (written by llm-select.bat or by
REM the dashboard button "Use this choice in batch runs"). Without llm.cfg the
REM default is Claude Opus 5 through Claude Code, the setting of the reported
REM campaigns. No setlocal here: the variables must reach the caller.
set "CLAUDE_CONFIG_DIR=%~dp0.claude-config"
set "CODEX_HOME=%~dp0.codex-config"
if exist "%~dp0llm.cfg" for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~dp0llm.cfg") do set "%%A=%%B"
if not defined DSS_GENAI_PROVIDER set "DSS_GENAI_PROVIDER=claude"
if /i "%DSS_GENAI_PROVIDER%"=="claude" if not defined DSS_GENAI_MODEL set "DSS_GENAI_MODEL=claude-opus-5"
echo [llm] provider=%DSS_GENAI_PROVIDER% model=%DSS_GENAI_MODEL% effort=%DSS_GENAI_EFFORT% fallback=%DSS_GENAI_FALLBACK%
exit /b 0
