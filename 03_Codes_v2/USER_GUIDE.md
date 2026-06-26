# DisasterAware Dashboard — User Guide

## 1. Start it

Double-click `run.bat` (Windows). It finds Python, installs `flask` and `numpy`
only if missing, starts the server, and opens `http://127.0.0.1:5000`.
Manual: `pip install -r requirements.txt` then `python app.py`.

## 2. The screen

- **Left map – Baseline:** the fire with no intervention. This is "what happens if
  I do nothing".
- **Right map – DisasterAware:** the same fire, same world, with the DSS acting every
  step. Compare the two to see the value of the decision support.
- **Control panel (left):** scenario, display, meteorology and DSS settings. Every
  control has a small `?` with a tooltip.
- **Transport bar + timeline:** play, pause, step, fast-forward, rewind, jump to
  start/latest, and a scrubber to review any past step.
- **DSS decision panel:** the colour legend of the three actions plus a sentence
  describing exactly what the DSS did this step and the quality breakdown.
- **Charts:** burned area and asset loss over time for both panels.

## 3. A first run

1. Leave the defaults. Click on the right map next to the town (top-right cluster).
2. Press **Play**. Watch the fire spread on the left and the DSS cut a containment
   line on the right.
3. Open **Help & guide** (top-right) for the in-app summary.
4. Drag the **timeline** back and forth to study any moment.

## 4. Every control explained

### Scenario

### Resources & responders (new)
- **Map size** — now 40 to 200 cells per side; zoom 2–24 and scroll to explore.
- **Forest amount** — how much woodland the world has.
- **First responders (FR)** — number of 🚒 stations on the map. Suppression is
  dispatched from them, so cells far from any station are harder to defend
  (thesis resource-accessibility feature). More FR = more capacity and reach.
- **Show regions grid** — draws the N regions (each has its own local DSS agent).
- The **resource bar** under the maps shows FR count, capacity used / total, and a
  clear **ACCEPTED / ATTENUATED** status for the acceptance threshold.
- The **wind arrow** now points the way the fire is pushed (it matches the spread).

- **Grid size** — cells per side (30–90). Larger = finer and slower. Rebuilds the world.
- **World seed** — chooses the procedural land-use map (forests, groves, farms, town,
  critical facility, river). Same seed = same world.
- **Fuel density** — scales combustible load everywhere. Higher = faster, larger fires.
- **Rebuild world** — apply the scenario settings.

### Display
- **Zoom** — pixel size of each cell. Zoom in to see the land-use icons.
- **Background layer** — Land use (coloured by category), Fuel, Value, or Terrain.
- **Show icons** — draw land-use glyphs (forest, homes, city, critical facility…).
- **Show DSS actions** — colour the right map by the intervention the DSS applies
  (cyan = direct suppression, blue = fuel reduction, magenta = asset protection).
  Turn off to see the raw suppression intensity instead.

### Meteorology
- **Wind speed** — 0–1, drives the Rothermel rate of spread. Stronger = faster.
- **Wind direction** — the heading the wind blows toward; fire runs downwind.

### Decision Support
- **DSS active** — turn the right-panel decision layer on or off.
- **Resource capacity** — total suppression effort per step. If the fire demands more,
  effort is shared out (resource normalisation) and containment may fail.
- **Acceptance threshold eta** — the minimum decision quality Q (0–1) for the DSS to
  apply its plan unchanged. Below it, a fail-safe attenuates the action. Higher eta =
  more cautious; very high eta makes the DSS hesitate and the fire grows.
- **Observation noise epsilon** — how blurred the DSS view of the fire is. Higher = the
  DSS mis-locates the front; confidence gating leans on memory to stay stable.
- **Regions N** — distributed-inference granularity. The coordinated decision is the
  same for any N; this only affects how the computation is split.
- **Ignition radius** — size of the fire each click creates.

## 5. Reading the result

- **Metrics** under each map: burned %, asset loss (value-weighted), active cells,
  and for the DSS panel the decision quality Q and resources used / capacity.
- **What is burning:** the chips under each map count burned cells per land-use
  category (Forest, Agriculture, Residential, City, Critical…), so you can see whether
  the fire is in the forest or reaching buildings.
- **Decision panel:** the sentence states how many cells got each action and in which
  land use, and the Q breakdown (spread / asset / resource / timeliness) explains why
  the plan was accepted or attenuated.

## 6. Things to try

- Start a fire upwind of the town and watch the DSS protect the houses and critical
  facility while the baseline loses them.
- Lower **Resource capacity** until the DSS can no longer contain the fire.
- Raise **Acceptance threshold** until the DSS hesitates and performs worse.
- Raise **Observation noise** and see graceful degradation.
- Change **Wind direction** mid-run and watch both the fire and the DSS adapt.

For the method behind the decisions, see `DSS_DECISION_RATIONALE.md`.
For requirements and the thesis mapping, see `REQUIREMENTS.md`.
