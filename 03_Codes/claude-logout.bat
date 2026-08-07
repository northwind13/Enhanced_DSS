@echo off
set "CLAUDE_CONFIG_DIR=%~dp0.claude-config"
echo Config dizini: %CLAUDE_CONFIG_DIR%
claude auth logout
echo.
claude auth status --text
pause