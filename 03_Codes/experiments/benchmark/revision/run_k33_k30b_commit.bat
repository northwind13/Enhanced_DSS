@echo off
setlocal
REM Runs K33 and K30b one after the other, then commits the result folders
REM and pushes to origin/main. Set W_CPU before calling to change the worker
REM count (default 16). Both runs are resumable.
cd /d "%~dp0"
call run_k33_evo_fam.bat
call run_k30b_expert_tuned.bat
cd /d "%~dp0..\..\..\.."
git add -A 03_Codes/experiments/benchmark/revision/k33_evo_fam 03_Codes/experiments/benchmark/revision/k30b_expert_tuned 03_Codes/experiments/benchmark/revision/revision_K33.log 03_Codes/experiments/benchmark/revision/revision_K30b.log
git commit -m "K33 Evo-fam and K30b results"
git pull --rebase origin main
git push origin main
echo ==== committed and pushed %date% %time%
pause
endlocal
