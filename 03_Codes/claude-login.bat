@echo off
setlocal
set "CLAUDE_CONFIG_DIR=%~dp0.claude-config"
if not exist "%CLAUDE_CONFIG_DIR%" mkdir "%CLAUDE_CONFIG_DIR%"
echo Config dizini: %CLAUDE_CONFIG_DIR%
call claude auth login %*
echo.
call claude auth status --text
endlocal
pause