@echo off
rem ===================================================================
rem  Section 5.5 in one command: run the sensitivity study, draw its
rem  five figures, and write the section into the thesis as tracked
rem  changes.
rem
rem    run_sensitivity.bat
rem    run_sensitivity.bat ..\01_Thesis\OTHER.docx
rem
rem  The study runs in two phases and the first decides where the
rem  second runs. A sweep can only show what its operating point
rem  allows: where the fire is beaten under every setting, or lost
rem  under every setting, no parameter can move the result. Phase 1
rem  therefore maps fire load against resource level, with the free
rem  burn of the same world as the reference, and picks the setting at
rem  which the decision layer is neither winning nor losing outright.
rem  Phase 2 sweeps everything else there.
rem
rem  Resumable: finished runs are skipped, so an interrupted study
rem  continues where it stopped. To start clean, delete
rem  experiments\out\sens_runs.csv and experiments\out\sens_point.json.
rem ===================================================================
setlocal
cd /d "%~dp0"

set THESIS=%1
if "%THESIS%"=="" set THESIS=..\01_Thesis\DISASTERAWARE_PhDThesis_Fin1.docx

for /f "tokens=1-3 delims=/.- " %%a in ("%DATE%") do set D=%%c%%b%%a
set STAMP=%D%_%TIME:~0,2%%TIME:~3,2%
set STAMP=%STAMP: =0%

echo.
echo ===================================================================
echo  DisasterAware - Section 5.5, sensitivity analysis
echo  thesis: %THESIS%
echo ===================================================================

echo.
echo [1/4] calibration: where can a parameter be seen at all
python experiments\sensitivity2.py --phase calibrate
if errorlevel 1 goto :failed

echo.
echo [2/4] sweeps: environment, decision-layer tuning, cost weights
python experiments\sensitivity2.py --phase sweep
if errorlevel 1 goto :failed

echo.
echo [3/4] Figures 5.12-5.16
python experiments\plot_sensitivity.py
if errorlevel 1 goto :failed

echo.
echo [4/4] writing Section 5.5 into the thesis (tracked changes)
python experiments\fill_sensitivity.py "%THESIS%" "%~dp0..\01_Thesis\DISASTERAWARE_PhDThesis_Sensitivity_%STAMP%.docx"
if errorlevel 1 goto :failed

echo.
echo ===================================================================
echo  DONE
echo.
echo  figures : 01_Thesis\figures
echo    fig5_12_calibration   where the sweeps were run, and why there
echo    fig5_13_ranking       what governs the outcome
echo    fig5_14_capacity      where the system breaks, static vs adaptive
echo    fig5_15_thresholds    the decision layer's own parameters
echo    fig5_16_eta           what the quality gate admits
echo.
echo  results : experiments\out\sens_runs.csv, sens_point.json
echo  thesis  : 01_Thesis\DISASTERAWARE_PhDThesis_Sensitivity_%STAMP%.docx
echo ===================================================================
goto :eof

:failed
echo.
echo *** STOPPED: the step above failed. Nothing after it has run.
exit /b 1
