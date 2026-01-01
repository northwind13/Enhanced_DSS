@echo off
REM --------------------------------------------------
REM Move to the directory where this batch file exists
REM This ensures relative paths work correctly
REM --------------------------------------------------
cd /d "%~dp0"

REM --------------------------------------------------
REM Define output directory for membership function plots
REM --------------------------------------------------
set OUTDIR=level1_plots

REM --------------------------------------------------
REM Create output directory if it does not exist
REM --------------------------------------------------
if not exist "%OUTDIR%" (
    mkdir "%OUTDIR%"
)

REM --------------------------------------------------
REM Run the Python script with required arguments
REM --------------------------------------------------
python generate_mf_plots.py ^
    --excel "..\02_Docs\DSS_Tables.xlsx" ^
    --sheet "Level1" ^
    --outdir "%OUTDIR%"

REM --------------------------------------------------
REM Keep the window open to show errors or messages
REM --------------------------------------------------
pause
