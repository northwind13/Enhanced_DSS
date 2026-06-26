# DisasterAware Dashboard

Interactive, thesis-faithful wildfire decision-support application. It runs a live
grid wildfire simulation (hybrid Rothermel-type spread) over a procedural land-use
world, lets you start fires by clicking the map, and shows side by side what happens
with **no intervention** (left) versus what the **concept-based DSS** does (right),
including which action it takes on the forest, the farm, or the buildings.

## Run (Windows)

Double-click **`run.bat`**. It finds Python, installs `flask` and `numpy` only if
missing, starts the local server and opens `http://127.0.0.1:5000`.

Manual:

    pip install -r requirements.txt
    python app.py

## What you can do

- **Configure the world:** grid size (30–90), seed, fuel density, zoom.
- **Land-use map with icons:** forest, grove, agriculture, livestock, residential,
  city, critical facility, and a river that acts as a natural barrier. Switch the
  background to fuel / value / terrain.
- **Start fires** by clicking either map, as many as you like, with a chosen radius.
- **Realistic spread** grid by grid (wind-anisotropic Rothermel-type rate of spread,
  fuel consumption, suppression coupling) per the thesis Chapter 4 equations.
- **Timeline transport:** play, pause, step, fast-forward, rewind, jump to start/latest,
  and scrub to any past step.
- **Baseline vs DSS** in parallel, plus a decision panel that names the action the DSS
  takes in each cell (direct suppression / preventive fuel reduction / asset protection),
  broken down by land use, with the decision-quality breakdown.
- **What is burning:** per-category burned counts (is the forest burning or the town?).
- **Tooltips on every control** and a built-in **Help & guide**.

## Thesis fidelity

- Simulation Core (`core/firesim.py`): grid CA, per-cell state (burning, fuel,
  intensity, ignition time), hybrid transition, Rothermel-type ROS, suppression.
- Land use / value (`core/landuse.py`): value priority V = 0.20 building + 0.40 critical
  + 0.25 population + 0.15 evacuation (thesis 4.2.4).
- DSS Core (`core/dss.py`): six features, five-term fuzzification, four concepts, three
  intervention types m1/m2/m3 with priority weights alpha = (1.0, 0.7, 0.9) and
  priority-weighted aggregation (thesis eq. 70), evaluation weights (0.35, 0.30, 0.20,
  0.15), confidence gating, satisficing acceptance with graduated fail-safe.

## Files

    app.py                  Flask backend (REST API + page)
    core/firesim.py         Simulation Core
    core/landuse.py         Land-use world + value layer
    core/dss.py             DSS Core (M=3 intervention types)
    core/engine.py          twin-sim session + timeline history
    templates/index.html    Web UI Dashboard
    static/app.js           UI logic (canvas, icons, timeline, decision panel)
    static/style.css        styling
    run.bat                 Windows launcher
    requirements.txt        dependencies
    USER_GUIDE.md           full manual (every control explained)
    DSS_DECISION_RATIONALE.md   how the DSS makes correct decisions (thesis mapping)
    REQUIREMENTS.md         requirements + gap analysis

## REST API

    POST /api/reset    rebuild world (grid size, seed, fuel, wind, DSS params)
    POST /api/ignite   {x,y,radius} start a fire
    POST /api/step     {n} advance n steps -> snapshot
    POST /api/goto     {i} view a past step (timeline)
    POST /api/params   live update wind / capacity / eta / eps / DSS on-off
    POST /api/clear_fire ; GET /api/layers ; GET /api/state

See `USER_GUIDE.md` to get started and `DSS_DECISION_RATIONALE.md` for the method.
