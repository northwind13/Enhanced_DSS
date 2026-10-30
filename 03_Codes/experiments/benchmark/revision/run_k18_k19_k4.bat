@echo off
REM Starts the three remaining revision runs in parallel windows:
REM   K4   hindcast rerun (5 min step) and equal-area references   CPU   ~1 h
REM   K18  combined model error, closed then live                  CPU + LLM
REM   K19  situation-picture error, closed then live               CPU + LLM
REM K18 and K19 each call the model with 6 workers once their closed arm is
REM done; K20 (cost weights, CPU only, ~40 min) runs alongside.
cd /d "%~dp0"
start "K4 hindcast" cmd /c run_k4_ellipse.bat
start "K18 combined" cmd /c run_k18.bat
start "K19 perimeter" cmd /c run_k19.bat
start "K20 cost weights" cmd /c run_k20.bat
