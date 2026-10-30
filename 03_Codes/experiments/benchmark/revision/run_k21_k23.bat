@echo off
REM Starts the two remaining revision runs in their own windows (resumable).
REM Meant for a scheduled start, e.g. after a model outage:
REM   schtasks /create /tn GODA_K21_K23 /tr "D:\repo_github\Enhanced_DSS\03_Codes\experiments\benchmark\revision\run_k21_k23.bat" /sc once /st 02:00 /it
cd /d "%~dp0"
set "W_CPU=8"
set "W_LLM=6"
start "K21 budget" cmd /c run_k21_budget.bat
set "W_LLM=4"
start "K23 bref live" cmd /c run_k23.bat
