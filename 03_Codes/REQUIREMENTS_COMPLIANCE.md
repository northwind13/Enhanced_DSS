# Requirements Compliance (Simulation Core)

Traceability of `DisasterAware_Simulator_Requirements.xlsx` to the code in
`03_Codes/`. Status: Met = implemented and tested; Fixed = corrected to match
the authoritative value in this pass; Interface = boundary hook for the DSS
(Chapters 5 to 6), out of the Simulation Core scope.

## Changes made in this pass
- Eq. 126 / REQ-ROS-04: wind scaling constant `w0` default set to 10 m/s
  (`config.py`, was 5.0).
- Eq. 137 / REQ-INT-02: intensity is now
  `I = B * tanh(beta * (F~ + gamma_W*W~ + gamma_S*S~))`, i.e. `beta` multiplies
  the whole sum (`intensity.py`).
- Eq. 135 / REQ-SUP-06: suppression reduction capped literally as
  `F_red = min(F_load, F_red_raw)` (`core.py`).
- Eq. 56 / REQ-DSS-01: read only observation function `O_k = h(S_k, eps_k)`
  added in `observation.py` (`observe`), with optional epistemic noise and a
  region window; it never mutates the state.

## A. Architecture and layering
- REQ-ARC-01..05: Met. Layers are separate modules: external data
  (`layers.py`, `world.py`), Simulation Core (`core.py`), DSS as interface only
  (`core.step(resource_override, extra_ignition)`), UI (`app/streamlit_app.py`).
- REQ-ARC-03 invariants (0<=F_load<=1, B in {0,1}, I in [0,1], tau>=0): enforced
  in `core.step`; covered by `tests/test_core.py`.
- REQ-ARC-04 state immutability: only `Phi` writes state; external influence
  enters via inputs. `interaction.py` never touches `S_k`.

## B. Spatial grid and state
- REQ-STA-01..04: Met. Grid in `config.SimConfig`/`world.World`; state
  `s=(B,Fload,I,tau)` in `state.SimulationState`; 8-neighbourhood in
  `spread._OFFSETS` with zero-fill boundary handling.

## C. External inputs and data layers
- REQ-IN-01..10: Met. `F_in,k` = meteo/topo/fuel/ignition/DSS in `layers.py`;
  `F_DSS,k` = value+resource kept separate (not fed to `Phi`). `V_prio` Eq. 55
  in `layers.ValueLayer.priority` with weights 0.40/0.25/0.20/0.15.
- REQ-IN-11 data acquisition: GIS raster adapter in `gis.py` (Could).

## D. Transition operator Phi
- REQ-TR-01..14: Met. `core.step` implements Eq. 43 to 52: B_pers, B_prop
  (Psi>Theta_ign), external ignition; fuel mass Eq. 49 with combustion gated by
  B_k and non-negativity; intensity Eq. 51 using current fuel and excluding I_k;
  tau Eq. 52 using the just-computed B_{k+1}. ROS excludes F_load and I_k
  (REQ-TR-08) by construction in `spread.rate_of_spread`.

## E. Rate of spread (Appendix A)
- REQ-ROS-01..07: Met (w0 Fixed). `spread.py`, Eq. 123 to 128; per-fuel
  r_base/m_ext/a_w/a_s/a_asp from Table A.1 in `config.FUEL_MODELS`.

## F. Burn and suppression (Appendix B)
- REQ-FB-01..02, REQ-SUP-01..08: Met. `F_burn` Eq. 129 in `core.step`;
  suppression map Eq. 130 to 135 in `suppression.fuel_reduction`; fuel-bounded so
  F_red never exceeds available fuel; suppression touches only fuel, never B.

## G. Fire intensity (Appendix C)
- REQ-INT-01..03: Met (formula Fixed). `intensity.py`, Eq. 136 to 137; beta=2.0,
  gamma_W=0.5, gamma_S=0.3, W_ws,max=20, slope max configurable.

## H. Interaction and scenario control
- REQ-UI-01..08: Met / Should. `interaction.InteractionOperator` applies the
  Table 4.1 admissible transforms (ignition, meteo, resource, value, availability)
  and logs each action; geo/fuel pass through. Dashboard provides pause/step/run/
  reset/inject-ignition/resource edit; ignition can be injected before, at start
  of and during a run.

## I. DSS interface (boundary only)
- REQ-DSS-02 (input-level intervention): Met via `step(resource_override=...,
  extra_ignition=...)`.
- REQ-DSS-01 observation `O_k=h(S_k,eps_k)`: Met. `observation.observe` returns
  the observable projection with epistemic noise injected only at this stage;
  read only, never mutates `S_k`.
- REQ-DSS-03 regional aggregation, REQ-DSS-04 event-triggered activation:
  Interface (Could). Belong to the DSS (Chapters 5 to 6); the core already
  supports per-region observation windows and event-driven stepping.

## J. Non-functional / numerical
- REQ-NF-01..06: Met / Should. Bounded factors (tanh/exp/clip), fuel
  non-negativity with eps_fuel=1e-4, deterministic transition (seed only affects
  exogenous fields), configurable dx/dt; per-cell vectorized NumPy updates.

## Verification
`pytest -q` -> 15 passed. Dashboard runs under Streamlit AppTest without
exceptions. Intensity stays in [0,1]; fuel stays non-negative; wind drives
anisotropic spread; suppression and firebreaks reduce burned area.
