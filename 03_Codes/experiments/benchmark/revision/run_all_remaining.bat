@echo off
REM Starts every remaining revision run in its own window, all resumable
REM (finished rows are skipped). Worker counts are kept low so that the five
REM runs fit in memory together: W_CPU processes per CPU step, W_LLM per
REM model step. Raise them (set W_CPU=8) if memory allows.
REM   K18  combined model error            closed then live
REM   K19  situation-picture error         closed then live
REM   K20  cost-function sensitivity       CPU only
REM   K22  larger domain, 30 worlds        closed then live
REM   K21  forecast-budget sweep           random (CPU) then GODA (LLM)
cd /d "%~dp0"
if "%W_CPU%"=="" set "W_CPU=5"
if "%W_LLM%"=="" set "W_LLM=3"
start "K18 combined" cmd /c run_k18.bat
start "K19 perimeter" cmd /c run_k19.bat
start "K20 cost weights" cmd /c run_k20.bat
start "K22 domain" cmd /c run_k22_domain.bat
start "K21 budget" cmd /c run_k21_budget.bat
