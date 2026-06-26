# DisasterAware — changelog

## v2.3 (current)
Realism (A): fuel moisture + humidity (damps spread), ember **spotting** across
firebreaks, **event-triggered** DSS activation, **real units** (30 m cells, minutes,
hectares, ROS in m/min).
Explainability (B): **cell inspector** (click a cell -> its 6 features, 4 concepts,
the fuzzy rules that fired, and the chosen action), **concept overlays** (threat /
feasibility / exposure / urgency heatmaps), **savings panel** (value & area protected,
% loss cut).
UX (C): **scenario presets** (lightning storm, arson near town, dry & windy),
**checkpoint / restore** (replay a what-if), **export** run as CSV, scenario as JSON,
view as PNG; **keyboard shortcuts** (Space, arrows); **drag-to-pan** + fit-to-view.
Engineering (D): **pytest** invariant suite (8 tests), central `core/config.py`,
logging. 

## v2.2
Emoji icons; maps up to 200x200; zoom 2-24; more forest; first-responder stations
with resource accessibility; region grid overlay; wind arrow matched to fire; 0.25x/0.5x
speeds; ACCEPTED/ATTENUATED acceptance status.

## v2.1
Land-use world (forest/grove/agri/livestock/residential/city/critical/water); thesis
M=3 intervention types; timeline transport; per-category burn readout; tooltips + help;
USER_GUIDE and DSS_DECISION_RATIONALE.

## v2.0
Flask web dashboard: baseline vs DSS twins, click-to-ignite, live charts.
