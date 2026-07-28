@echo off
rem ===================================================================
rem  Chapter 5 in one command: run the campaign, fill the tables, and
rem  draw the three figures the chapter cites.
rem
rem    run_ch5.bat            10 seeds, fills DISASTERAWARE_PhDThesis_Fin1
rem    run_ch5.bat 50         50 seeds
rem    run_ch5.bat 50 ..\01_Thesis\OTHER.docx
rem
rem  Figures produced, in 01_Thesis\figures:
rem    fig5_06_outcome_average  burned area, burned forest, affected
rem                             population, averaged over S1..S5, as a
rem                             share of the no-DSS run T_0
rem    fig5_07_Jphys_average    physical decision cost, averaged
rem    fig5_08_Jtotal_average   total decision cost, averaged
rem
rem  The campaign is resumable: a second run continues where the first
rem  stopped. To start clean, delete experiments\out\ladder_*.csv and
rem  experiments\out\ladder_products.jsonl first, or re-run one arm with
rem  run_full_campaign.py --redo F5EvAI.
rem ===================================================================
setlocal
cd /d "%~dp0"

set SEEDS=%1
if "%SEEDS%"=="" set SEEDS=10

set THESIS=%2
if "%THESIS%"=="" set THESIS=..\01_Thesis\DISASTERAWARE_PhDThesis_Fin1.docx

set FIG=..\01_Thesis\figures
set T58=validation\figures\table58

echo.
echo ===================================================================
echo  DisasterAware - Chapter 5
echo  seeds : %SEEDS%
echo  thesis: %THESIS%
echo ===================================================================

rem --- 1. the campaign itself, plus Tables 5.8/5.9 and the tracked fill
echo.
echo [1/4] campaign, tables, tracked-changes thesis fill
python experiments\run_full_campaign.py --seeds %SEEDS% --docx "%THESIS%"
if errorlevel 1 goto :failed

rem --- 2. Figure 5.6, from the run that just finished rather than from
rem        the document it has not reached yet.
rem        --relative: every value as a share of the averaged no-DSS run,
rem        so hectares and people sit on one linear axis and the bars read
rem        as "how much damage was left". The average is taken FIRST and
rem        normalised after, so a small scenario does not weigh as much as
rem        a large one.
echo.
echo [2/4] Figure 5.6  (average physical outcome, relative to T0)
python validation\plot_table58.py --csv experiments\out\table58_phys.csv --combined --relative --only-average
if errorlevel 1 goto :failed

rem --- 3. Figures 5.7 and 5.8
echo.
echo [3/4] Figures 5.7-5.8  (average physical and total decision cost)
python experiments\plot_J_ladder.py --only-average
if errorlevel 1 goto :failed

rem --- 4. one name per caption. The generators name their output after
rem        what it CONTAINS, which is right for them and useless when you
rem        are placing figures in a chapter; these copies carry the number
rem        the caption uses.
echo.
echo [4/4] filing the figures under their thesis numbers
if not exist "%FIG%" mkdir "%FIG%"
rem  clear the numbered copies of an earlier run: a figure that is no
rem  longer produced must not sit in the folder pretending it is current
del /q "%FIG%\fig5_0*.png" "%FIG%\fig5_1*.png" "%FIG%\fig5_2*.png" 2>nul

copy /y "%T58%\table58_combined_avg_relative.png" "%FIG%\fig5_06_outcome_average.png" >nul
copy /y "%FIG%\fig_Jphy_Avg.png"                  "%FIG%\fig5_07_Jphys_average.png"   >nul
copy /y "%FIG%\fig_Jtot_Avg.png"                  "%FIG%\fig5_08_Jtotal_average.png"  >nul

echo.
echo ===================================================================
echo  DONE
echo.
echo  figures : 01_Thesis\figures
echo    fig5_06_outcome_average  burned area / forest / affected people
echo    fig5_07_Jphys_average    physical decision cost
echo    fig5_08_Jtotal_average   total decision cost
echo.
echo  tables  : experiments\out\table58_phys.csv, table59_cost.csv
echo  thesis  : 01_Thesis\*_Ch5_filled_^<date^>.docx  (tracked changes)
echo ===================================================================
goto :eof

:failed
echo.
echo *** STOPPED: the step above failed. Nothing after it has run.
exit /b 1
