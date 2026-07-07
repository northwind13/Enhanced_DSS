@echo off
REM ============================================================================
REM  DisasterAware - automatic hindcast validation against a REAL fire
REM
REM  WHAT IT DOES (all automatic):
REM   1) downloads the real terrain   (Copernicus GLO-30 DEM, AWS open data)
REM   2) downloads the real fuel map  (ESA WorldCover 10 m -> fuel classes)
REM   3) downloads the real weather   (hourly ERA5 wind/T/RH via open-meteo)
REM   4) downloads the real fire      (NASA FIRMS satellite detections:
REM      ignition = first detection, ground truth = burned footprint)
REM   5) runs the simulator BLIND for the documented duration, several seeds
REM   6) scores simulated vs observed burn: Dice, Jaccard, hit rate, false
REM      alarm, front error -> validation_report.json + agreement map .png
REM
REM  ONE-TIME SETUP: get a free NASA FIRMS key (1 minute):
REM      https://firms.modaps.eosdis.nasa.gov/api/map_key/
REM  then either  set FIRMS_MAP_KEY=yourkey  or pass it below.
REM
REM  Usage:
REM      run_validation.bat YOURKEY                (Manavgat 2021, default)
REM      run_validation.bat YOURKEY marmaris2021   (other case)
REM      run_validation.bat demo                   (offline self-test, no key)
REM ============================================================================
cd /d %~dp0
call .venv\Scripts\activate
pip install --quiet rasterio requests pillow

if "%1"=="demo" (
    python examples\auto_validate.py --offline-demo
    goto done
)
set KEY=%1
if "%KEY%"=="" set KEY=440e03b9ea7530a07a74d0973a87082e
if "%KEY%"=="" (
    echo Missing FIRMS MAP_KEY. Get one free at:
    echo   https://firms.modaps.eosdis.nasa.gov/api/map_key/
    echo then run:  run_validation.bat YOURKEY
    goto done
)
set CASE=%2
if "%CASE%"=="" set CASE=manavgat2021
python examples\auto_validate.py --case %CASE% --firms-key %KEY% --seeds 3
if exist validation_report.png start validation_report.png
:done
pause
