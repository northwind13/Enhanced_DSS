@echo off
REM Launcher for the round-4 CPU runs, one after the other: K30 (short),
REM K29 (two settings), K31 (long). Set W_CPU before calling to change the
REM worker count (default 16). All resumable.
cd /d "%~dp0"
call run_k30_expert_tuned.bat
call run_k29_evo_configs.bat
call run_k31_coating_sweep.bat
pause
