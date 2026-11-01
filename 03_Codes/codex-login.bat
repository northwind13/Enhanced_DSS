@echo off
setlocal
REM ChatGPT login for the GenAI stage (OpenAI Codex CLI, ChatGPT Plus/Pro).
REM Same idea as claude-login.bat: the session is written to .codex-config in
REM this folder, and every run script and the dashboard point CODEX_HOME there.
set "CODEX_HOME=%~dp0.codex-config"
if not exist "%CODEX_HOME%" mkdir "%CODEX_HOME%"
echo Config dizini: %CODEX_HOME%
where codex >nul 2>nul
if not errorlevel 1 goto have_codex
echo [codex] Codex CLI bulunamadi. Kuruluyor ^(Node.js gerekli^)...
where npm >nul 2>nul
if errorlevel 1 (
  echo [codex] npm yok. Once Node.js LTS kurun:  winget install OpenJS.NodeJS.LTS
  echo [codex] ya da https://nodejs.org  Sonra YENI bir pencerede bu dosyayi tekrar calistirin.
  goto end
)
call npm i -g @openai/codex
where codex >nul 2>nul
if errorlevel 1 (
  echo [codex] Kuruldu ama bu pencerede PATH'te degil. Yeni pencere acip tekrar calistirin.
  goto end
)
:have_codex
call codex --version
REM browser sign-in with the ChatGPT account; on a machine without a
REM browser use:  codex-login.bat --device-auth
call codex login %*
echo.
call codex login status
:end
endlocal
pause
