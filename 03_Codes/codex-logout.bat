@echo off
setlocal
set "CODEX_HOME=%~dp0.codex-config"
echo Config dizini: %CODEX_HOME%
call codex logout
call codex login status
endlocal
pause
