# DisasterAware Software Requirements and Gap Analysis

Requirements extracted from the thesis (Chapter 4: DisasterAware Simulation
Framework) and the decision architecture (Section III of the article). Each item
is marked: [DONE] implemented in this software, [PARTIAL] implemented in reduced
form, [GAP] not yet implemented (candidate future work).

## Architecture (thesis Fig. 4.1, four layers)
The thesis specifies four layers with strict separation and a Web UI Dashboard.
This software realizes all four:
- External Data Sources  -> `core/firesim.py` world generator (meteorology, terrain, fuel, value, resources)
- Simulation Core        -> `core/firesim.py` hybrid transition operator
- DSS Core               -> `core/dss.py` concept-based fuzzy decision layer
- Web UI Dashboard       -> `app.py` + `templates/` + `static/`

## Functional requirements

FR1  Grid-based discrete-time simulation over a spatial domain, per-cell state
     s = (burning status, fuel load, intensity proxy, ignition time).            [DONE]
FR2  Hybrid transition operator: burning-status evolution (persistence +
     propagation + ignition injection), fuel-mass evolution (combustion +
     suppression), intensity proxy, ignition-time memory.                        [DONE]
FR3  Rate of spread from a Rothermel-type formulation (fuel, wind, slope) with
     wind-anisotropic propagation over the 8-connected neighbourhood.            [DONE]
FR4  External data layers: meteorology (wind speed and direction), terrain
     (slope), fuel load, asset value, resource capacity.                         [DONE]
FR5  Ignition injection from the UI at one or several points (lightning / human
     / scenario), with selectable radius.                                        [DONE]
FR6  DSS Core: bounded-noise observation, six features, five-term fuzzification,
     four situational concepts, concept rule bases for four intervention types,
     resource-constrained coordination (normalisation + projection), confidence
     gating, satisficing evaluation with graduated fail-safe.                    [DONE]
FR7  Closed feedback loop: DSS observes the state, emits a suppression field,
     the suppression modifies fuel, the next state is re-observed and the
     actions are updated every step.                                            [DONE]
FR8  State immutability and layer separation: the UI changes only admissible
     inputs (wind, fuel, resources, ignition, DSS parameters); it never writes
     to the wildfire state (thesis Remark 4.1).                                  [DONE]
FR9  Counterfactual comparison: run "untouched" baseline and "DSS-managed" twin
     from the same world and ignitions, side by side.                           [DONE]
FR10 Web UI Dashboard: play / pause / step / clear / rebuild, live parameter
     control, layer visualisation (fuel / value / terrain), per-panel metrics,
     DSS status (quality, acceptance, resource use), time-series charts.         [DONE]
FR11 Scenario / what-if control: change meteorology and DSS parameters live and
     observe the effect on future steps.                                        [DONE]

## Partial and missing requirements (gap analysis)

FR12 Land-use / asset map with icons (forest, grove, agriculture, livestock,
     residential, city, critical facility, water barrier); value priority per
     thesis weights; per-category burned readout.                                  [DONE]
FR13 Timeline transport: play / pause / step / fast-forward / rewind / scrub.     [DONE]
FR14 DSS decision explanation: per-cell intervention type (m1/m2/m3), colour
     legend, decision text by land-use category, Q-criteria breakdown.             [DONE]
FR15 Tooltips on every control and a built-in Help & guide; written manual and
     decision-rationale documents.                                                 [DONE]

## Remaining gaps

FR16 Cell inspector + concept overlays + savings panel (explainability).          [DONE]
FR17 Fuel moisture/humidity, ember spotting, event-triggered DSS, real units.     [DONE]
FR18 Scenario presets, checkpoint/restore, CSV/JSON/PNG export, shortcuts, pan.    [DONE]
FR19 pytest invariant suite, central config, logging.                             [DONE]

## Remaining gaps

G1  Real GIS/weather data ingestion (synthetic world generator for now).          [GAP]
G2  Fuel-type vegetation classes with class-dependent combustion.                 [PARTIAL]
G3  Full meteorology fields: temperature, relative humidity, wind gust and
    spatio-temporal fuel-moisture dynamics. Currently wind speed and direction
    plus a static fuel field.                                                    [PARTIAL]
G4  Fuel-type / vegetation classes and class-dependent combustion. Currently a
    single continuous fuel-load field.                                           [PARTIAL]
G5  Stochastic spotting (ember jumps ahead of the front).                        [GAP]
G6  WebSocket streaming + delta payloads for very large grids (polling for now). [GAP]
G7  Per-cell resource accessibility / travel-time and explicit logistics in the
    resource layer. Currently capacity plus a simple availability decay.         [PARTIAL]
G8  Socio-economic value categories in the value layer. Currently a single
    aggregated value field.                                                      [PARTIAL]
G9  Save / load of scenarios and export of run results.                          [GAP]
G10 Multi-user access, authentication and deployment hardening.                  [GAP]

These gaps do not affect the core demonstration (FR1-FR11); they are the natural
extensions toward a field-deployable product and align with the future-work
section of the article.
