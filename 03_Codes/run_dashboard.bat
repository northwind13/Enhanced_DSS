@echo off
REM DisasterAware simulator - dashboard launcher
setlocal
cd /d "%~dp0"

REM Claude Code oturumu claude-login.bat ile bu klasordeki .claude-config
REM dizinine yaziliyor. Ayni dizini BURADA da vermek sart: verilmezse
REM dashboard'un cagirdigi `claude -p` varsayilan %%USERPROFILE%%\.claude
REM dizinine bakar, oturumu bulamaz ve "baglanti yok / not logged in" der.
set "CLAUDE_CONFIG_DIR=%~dp0.claude-config"

where py >nul 2>nul && (set "PY=py") || (set "PY=python")

if not exist ".venv\Scripts\activate.bat" (
    echo [setup] Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [error] Could not create virtual environment. Is Python installed?
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
REM stale .pyc files have caused "old code running" bugs after updates:
REM never write bytecode and clear any leftover caches on every start
set PYTHONDONTWRITEBYTECODE=1
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
echo [setup] Installing dependencies...
python -m pip install --upgrade pip >nul 2>nul
pip install -r requirements.txt
if errorlevel 1 (
    echo [error] Dependency installation failed.
    pause
    exit /b 1
)

REM --- Claude Code CLI (optional): powers the GenAI stage on your Claude
REM subscription. We AUTO-INSTALL the tool if it is missing and OFFER to run
REM `claude login` here (it waits until you finish), but the sign-in itself
REM is the official browser flow - the launcher never sees your credentials.
where claude >nul 2>nul
if not errorlevel 1 goto claude_ready
echo [genai] Installing Claude Code CLI ^(one time, no Node.js needed^)...
curl -fsSL https://claude.ai/install.cmd -o "%TEMP%\claude_install.cmd" 2>nul
if errorlevel 1 (
    echo [genai] Could not download the installer ^(no internet or curl^).
    echo [genai] Stage 3 stays off; the DSS still runs on stages 1 and 2.
    goto claude_done
)
call "%TEMP%\claude_install.cmd"
del "%TEMP%\claude_install.cmd" >nul 2>nul
where claude >nul 2>nul
if errorlevel 1 (
    echo [genai] Claude Code was installed but is not on PATH for THIS
    echo [genai] window yet. Close this window, open a NEW one, run
    echo [genai]     claude-login.bat
    echo [genai] once with your Pro/Max plan, then start run_dashboard.bat
    echo [genai] again. ^(Manual login - the launcher never sees your creds.^)
    goto claude_done
)
:claude_ready
echo [genai] Claude Code is available. Config dir: %CLAUDE_CONFIG_DIR%
echo [genai] Login status:
call claude auth status --text 2>nul
echo [genai] Not logged in? Run  claude-login.bat  ONCE (Pro/Max plan), then
echo [genai] start run_dashboard.bat again. The launcher never handles your
echo [genai] credentials and does NOT block on it; the dashboard starts below.
:claude_done

echo [run] Starting dashboard. Close this window or press Ctrl+C to stop.
streamlit run app\streamlit_app.py
pause
