@echo off
rem ===================================================================
rem  Does the geometry diagnosis change what the generative stage
rem  proposes? Two arms, same worlds, same gates, one line of
rem  difference: the diagnosis raised, or suppressed.
rem
rem    run_geometry.bat            5 seeds per arm, 24-minute horizon
rem    run_geometry.bat 5 60       5 seeds per arm, 60-minute horizon
rem    run_geometry.bat 5 24 8 30  patience 8, stage 3 held back 30 steps
rem    run_geometry.bat 5 24 8 0 4 four seeds at once
rem    run_geometry.bat 5 24 8 0 4 fresh   start this tag over
rem
rem  A RESUMABLE CAMPAIGN THAT FINDS ITS WORK DONE EXITS AT ONCE, which
rem  reads like a failure to start. The header now names the files, the
rem  settings, what is already on file and what is left. Pass "fresh" as
rem  the sixth argument to archive the existing slice under a timestamp
rem  and begin again; the old files are renamed, never deleted, because
rem  a finished slice records a state of the code that is gone.
rem
rem  THE CAMPAIGN WAITS ON MODEL CALLS, not on the processor, so workers
rem  cost almost nothing and buy almost everything: four of them finish
rem  in close to a quarter of the time. Each owns whole seeds and writes
rem  its own shard, which is merged when they all return.
rem
rem  THE HORIZON IS THE SECOND EXPERIMENT. Every clause actuator of the
rem  24-minute slice was a preventive tactic and every one died at G5,
rem  because a coated belt shows no gain before the fire reaches it. A
rem  longer horizon asks whether the gate rejects bad tactics or slow
rem  ones. Results land in geometry_campaign_h60.csv, kept apart so the
rem  two horizons are never averaged together.
rem
rem  THIS MUST RUN HERE, not in a sandbox. Stage 3 calls the model
rem  through the Claude Code CLI, so the campaign needs the `claude`
rem  command on the PATH and a signed-in session. Check it first with
rem  the Test button on the dashboard, or:  claude -p "reply OK"
rem
rem  G5b IS NOW IN THE CHAIN and the file names carry it. G5 weighs the
rem  whole package against no package, so a rule that orders an ordinary
rem  intervention beside its new object could be admitted on the strength
rem  of that ordinary order. The previous slice showed it: two different
rem  clause geometries on the same seed returned bit-identical rollouts,
rem  because what moved the forecast was the resource deployment both
rem  carried. G5b re-runs the same two futures with the new object struck
rem  out and the ordinary orders left standing, and admits the object only
rem  when the cost rises without it. A slice measured with G5b must never
rem  be merged with one measured without it, so results land in
rem  geometry_campaign_g5b.csv.
rem
rem  Results, both resumable:
rem    experiments\out\geometry_campaign_g5b.csv   one row per (arm, seed)
rem    experiments\out\geometry_proposals_g5b.jsonl  every proposal, with
rem                                            its kind and its gate
rem ===================================================================
setlocal
cd /d "%~dp0"

set SEEDS=%1
if "%SEEDS%"=="" set SEEDS=5
set HORIZON=%2
if "%HORIZON%"=="" set HORIZON=24
set PATIENCE=%3
if "%PATIENCE%"=="" set PATIENCE=3
set EARLIEST=%4
if "%EARLIEST%"=="" set EARLIEST=0
set WORKERS=%5
if "%WORKERS%"=="" set WORKERS=1
set FRESH=
if /i "%6"=="fresh" set FRESH=--fresh

echo.
echo ===================================================================
echo  DisasterAware - geometry diagnosis campaign
echo  seeds per arm : %SEEDS%
echo  no-harm horizon: %HORIZON% min
echo  stage-3 patience: %PATIENCE%   held back until step: %EARLIEST%
echo  workers: %WORKERS%
echo  pool 0.25, 4 ignitions, 240 min, 4 local agents
echo ===================================================================
echo.

python experiments\geometry_campaign.py --seeds %SEEDS% --horizon %HORIZON% --patience %PATIENCE% --earliest %EARLIEST% --workers %WORKERS% %FRESH%
if errorlevel 1 goto :failed

echo.
echo ===================================================================
echo  DONE
echo.
echo  What to read in the CSV:
echo    diagnosed / cycles   how often the pool was spent while the
echo                         fire still grew
echo    clause               clause actuators PROPOSED (the question)
echo    clause_accepted      how many survived the gate chain
echo    rej_G3               proposed and lost on the forecast: a
echo                         different result from never proposing
echo    rej_G5b              the package helped, but its OWN rule
echo                         stripped of the new object helped just as
echo                         much, so the object earned nothing
echo    genai_retired        stage 3 gave up mid-run (two rejections)
echo ===================================================================
goto :eof

:failed
echo.
echo *** STOPPED: the campaign failed. Nothing after it has run.
exit /b 1
