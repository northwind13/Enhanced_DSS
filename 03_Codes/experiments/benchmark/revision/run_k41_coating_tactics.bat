@echo off
setlocal
REM K41: hand-written head retardant line vs recorded head coating, each alone in the closed layer, and with trigger and band swapped. CPU only, then commit and push.
cd /d "%~dp0..\..\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python experiments\benchmark\revision\k41_coating_tactics.py experiments\benchmark\revision\k41_coating_tactics\runs.csv written,recorded,w_trig_r_band,r_trig_w_band
git add experiments\benchmark\revision\k41_coating_tactics
git commit -m "K41 coating tactics: written vs recorded, trigger and band swapped"
git pull --rebase
git push
pause
endlocal
