@echo off
REM Starts K24 (CPU), K25, K26 (LLM) and K27 (CPU then LLM) in four windows.
REM All resumable. W_CPU and W_LLM set the workers per step.
cd /d "%~dp0"
if "%W_CPU%"=="" set "W_CPU=6"
if "%W_LLM%"=="" set "W_LLM=4"
start "K24 effects" cmd /c run_k24.bat
start "K25 one future" cmd /c run_k25.bat
start "K26 parity" cmd /c run_k26.bat
start "K27 coat parity" cmd /c run_k27.bat
