"""Simulation Core: the discrete time state transition operator Phi.

This module implements the DisasterAware hybrid fire spread model,
combining burning status evolution, fuel mass evolution
, fire intensity evolution and
ignition time evolution into a single coupled update:

    S_{k+1} = Phi(S_k, F_in,k)

The operator is deterministic; stochastic behaviour enters only through the
exogenous input fields (for example wind gusts). External modules never write to
the state directly; suppression and ignition influence the state only through
the external input set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .config import FUEL_MODELS
from .state import SimulationState
from .spread import (rate_of_spread, propagation_influence, _fuel_param,
                     effective_spread_vector, directional_weights)
from .suppression import fuel_reduction
from .intensity import fire_intensity
from .world import World


@dataclass
class StepDiagnostics:
    """Per step physical diagnostics returned by the transition operator."""

    step: int
    n_burning: int
    n_burned_cumulative: int
    fuel_consumed_step: float
    fuel_suppressed_step: float
    ros_mean_active: float
    intensity_mean_active: float


class _MeteoView:
    """Meteo layer proxy with terrain-adjusted wind speed / direction."""

    def __init__(self, base, wws, wwd=None):
        self._base = base
        self.wws = wws
        if wwd is not None:
            self.wwd = wwd

    def __getattr__(self, name):
        return getattr(self._base, name)


class Simulator:
    """Stateful wrapper around the transition operator Phi."""

    def __init__(self, world: World):
        self.world = world
        self.cfg = world.config
        self.state = SimulationState.from_world(world)
        # cumulative bookkeeping for cost and metric accounting
        self.ever_burned = np.zeros(world.shape, dtype=bool)
        self.fuel_consumed_total = np.zeros(world.shape, dtype=float)
        self.fuel_suppressed_total = np.zeros(world.shape, dtype=float)
        # step index at which each cell first ignited (-1 = never), for the
        # time-to-burn propagation layer (Kose et al., 3D wildfire viz)
        self.first_ignition_step = np.full(world.shape, -1, dtype=int)
        # and the step at which it stopped burning (-1 = never burned, or
        # still alight). A cell's story is "lit at k, out at k'"; only the
        # first half was being recorded, so a burn scar could not say how
        # long it burned or when the front had passed.
        self.burnout_step = np.full(world.shape, -1, dtype=int)
        # ignition influence buildup A_k: time-integrated neighbour
        # influence; a cell ignites when the buildup crosses theta_ign
        self.ign_buildup = np.zeros(world.shape, dtype=float)
        self.history: List[StepDiagnostics] = []
        # time-integrated cost accumulators (per engine step):
        # persons inside actively burning cells, and committed
        # response capacity; both feed the J_pop / J_resp terms
        self.exposure_person_steps: float = 0.0
        self.response_capacity_steps: float = 0.0
        self.population_evacuated: float = 0.0
        # RETARDANT COATING: long-term chemical/soil cover laid by
        # aerial drops; unlike wetting it does not rinse out with the
        # moisture model, it decays on its own slow clock
        self.retard = np.zeros(world.shape, dtype=float)
        #: people who left ON THEIR OWN, kept apart from the ordered
        #: evacuation so the two can be told apart in the accounting
        self.population_self_evacuated = 0.0
        self.evacuated_person_steps = 0.0
        self._vpop0 = np.asarray(world.value.vpop,
                                 dtype=float).copy()
        # cumulative simulated time: step lengths may CHANGE mid-run
        # (Time panel), so the clock integrates the actual minutes of
        # every executed step instead of multiplying step x dt
        self.t_elapsed_min: float = 0.0
        self._b_base = _fuel_param(world.fuel.ftype, "b_base")
        self._rng = np.random.default_rng(self.cfg.rng_seed)
        # baseline fuel moisture: reset() must restore it, or the
        # wetting left by a previous run's suppression/rain makes the
        # NEXT run non-reproducible (same settings, different fire)
        self._fmoist0 = np.asarray(world.fuel.fmoist, dtype=float).copy()
        # ---- rewind support: automatic per-step state snapshots ----------
        self.record_states: bool = True
        self._snap_budget_bytes: int = 150 * 1024 * 1024
        self._snapshots: dict = {0: self._snapshot()}

    # ------------------------------------------------------------- snapshots
    def _snapshot(self) -> dict:
        s = self.state
        return {
            "burning": (s.burning > 0.5).astype(np.uint8),
            "fload": s.fload.astype(np.float32),
            "intensity": s.intensity.astype(np.float32),
            "tau": s.tau.astype(np.float32),
            "buildup": self.ign_buildup.astype(np.float32),
            "ever": self.ever_burned.copy(),
            "first": self.first_ignition_step.astype(np.int32),
            "out": self.burnout_step.astype(np.int32),
            "cons": self.fuel_consumed_total.astype(np.float32),
            "supp": self.fuel_suppressed_total.astype(np.float32),
            # in-place mutables outside SimState: suppression and rain
            # WET the fuel (fmoist), spotting consumes rng draws; both
            # must rewind or a counterfactual replay inherits the
            # soaked fuel of the factual run and burns far too little
            "fmoist": self.world.fuel.fmoist.astype(np.float32),
            "rng": self._rng.bit_generator.state,
            "exps": float(self.exposure_person_steps),
            "resps": float(self.response_capacity_steps),
            "tmin": float(self.t_elapsed_min),
            "vpop": self.world.value.vpop.astype(np.float32),
            "nevac": float(self.population_evacuated),
            "nself": float(self.population_self_evacuated),
            "evps": float(self.evacuated_person_steps),
            "retard": self.retard.astype(np.float32),
        }

    def _record_snapshot(self) -> None:
        if not self.record_states:
            return
        self._snapshots[self.state.step] = self._snapshot()
        per = sum(a.nbytes for a in
                  self._snapshots[self.state.step].values()
                  if hasattr(a, "nbytes"))
        cap = max(20, self._snap_budget_bytes // max(per, 1))
        while len(self._snapshots) > cap:
            # step 0 is the anchor of the "no orders at all"
            # counterfactual: it is never evicted, the budget thins
            # the oldest of the REST instead
            _ks = sorted(self._snapshots)
            self._snapshots.pop(_ks[1] if _ks[0] == 0 and len(_ks) > 1
                                else _ks[0])

    @property
    def rewindable_steps(self) -> list:
        return sorted(self._snapshots)

    def rewind(self, k: int) -> bool:
        """Restore the full simulation state at step k (if snapshotted) and
        drop everything after it, so the run can be replayed from k with
        modified conditions (wind, moisture, resources, ...)."""
        if k not in self._snapshots:
            return False
        snap = self._snapshots[k]
        s = self.state
        s.burning = snap["burning"].astype(float)
        s.fload = snap["fload"].astype(float)
        s.intensity = snap["intensity"].astype(float)
        s.tau = snap["tau"].astype(float)
        s.step = int(k)
        self.ign_buildup = snap["buildup"].astype(float)
        self.ever_burned = snap["ever"].copy()
        self.first_ignition_step = snap["first"].astype(int)
        if "out" in snap:
            self.burnout_step = snap["out"].astype(int)
        self.fuel_consumed_total = snap["cons"].astype(float)
        self.fuel_suppressed_total = snap["supp"].astype(float)
        self.world.fuel.fload = s.fload
        if "fmoist" in snap:
            self.world.fuel.fmoist = snap["fmoist"].astype(float)
        if "rng" in snap:
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = snap["rng"]
        self.exposure_person_steps = float(snap.get("exps", 0.0))
        self.response_capacity_steps = float(snap.get("resps", 0.0))
        if "vpop" in snap:
            self.world.value.vpop = snap["vpop"].astype(float)
        self.population_evacuated = float(snap.get("nevac", 0.0))
        self.population_self_evacuated = float(snap.get("nself", 0.0))
        self.evacuated_person_steps = float(snap.get("evps", 0.0))
        if "retard" in snap:
            self.retard = snap["retard"].astype(float)
        self.t_elapsed_min = float(snap.get(
            "tmin", k * float(getattr(self.cfg, "step_minutes", 30.0))))
        self.history = self.history[:k]
        self._snapshots = {kk: v for kk, v in self._snapshots.items() if kk <= k}
        return True

    # ----------------------------------------------------------------- control
    def reset(self) -> None:
        self.state = SimulationState.from_world(self.world)
        self.ever_burned[:] = False
        self.fuel_consumed_total[:] = 0.0
        self.fuel_suppressed_total[:] = 0.0
        self.first_ignition_step[:] = -1
        self.burnout_step[:] = -1
        self.ign_buildup[:] = 0.0
        self.history.clear()
        self.world.fuel.fload = self.world.fuel.fload0.copy()
        self.state.fload = self.world.fuel.fload0.copy()
        # a reset run must REPRODUCE the first run bit for bit: the
        # rng is reseeded (spotting draws) and the fuel moisture
        # returns to its baseline (undo suppression/rain wetting)
        self.world.fuel.fmoist = self._fmoist0.copy()
        self._rng = np.random.default_rng(self.cfg.rng_seed)
        self.exposure_person_steps = 0.0
        self.response_capacity_steps = 0.0
        self.population_evacuated = 0.0
        self.population_self_evacuated = 0.0
        self.evacuated_person_steps = 0.0
        self.retard[:] = 0.0
        self.world.value.vpop = self._vpop0.copy()
        self.t_elapsed_min = 0.0
        # THE ORDERS GO WITH THE FIRE. compute_costs reads this field for
        # the fielded capacity and the response DELAY, so leaving the last
        # step of the previous fire in place made a freshly reset map report
        # a response that was not happening: J_delay and the capacity
        # readout carried over while everything around them read zero.
        self.last_applied_resource = None
        self._snapshots = {0: self._snapshot()}

    # -------------------------------------------------------------- transition
    def step(self, resource_override=None,
             extra_ignition: Optional[np.ndarray] = None) -> StepDiagnostics:
        """Advance the state by one time step applying Phi.

        One step represents cfg.step_minutes of real time. Per-step rates are
        calibrated at a 30 min reference: with
        s = step_minutes / 30 the engine runs ceil(s) internal substeps of
        scale s / ceil(s), so long steps advance the front by up to ~s cells
        while 30 min steps reproduce the reference equations exactly.

        resource_override : an optional ResourceLayer supplied by a decision
            support system to replace the static suppression field for this step.
        extra_ignition : an optional binary field injected on top of the
            scheduled ignitions (used by the interactive what if controls).
        """
        cfg = self.cfg
        s = self.state
        world = self.world
        meteo, topo, fuel = world.meteo, world.topo, world.fuel
        resource = resource_override if resource_override is not None else world.resource
        self.last_applied_resource = resource   # cost model reads THIS
        # EVACUATION: an ordered evacuation physically MOVES people
        # out of the ordered cells (~5%/min at full tempo). The
        # evacuated are SAFE: they leave vpop, so the exposure and
        # J_pop terms stop counting them; the running total is kept
        # for reporting.
        # ---- SELF-EVACUATION: people leave without being told ----
        # Nobody stands in a burning street waiting for an order. The model
        # had no such term, so without an order the population sat where it
        # was until the flame reached it. Residents leave at a rate set by
        # how close the fire is, and only when they have somewhere to go:
        # flight needs at least one neighbouring direction that is not
        # alight, so a settlement the fire has surrounded does not quietly
        # empty itself.
        _se = getattr(cfg, "self_evac", None)
        if (_se is not None and getattr(_se, "enabled", False)
                and float(np.max(world.value.vpop)) > 1e-9):
            _B_s = s.burning > 0.5
            _adj = np.zeros_like(_B_s)
            _adj[1:, :] |= _B_s[:-1, :]
            _adj[:-1, :] |= _B_s[1:, :]
            _adj[:, 1:] |= _B_s[:, :-1]
            _adj[:, :-1] |= _B_s[:, 1:]
            # how far the fire is, in cells, out to the awareness range
            _near = _B_s.copy()
            _dist = np.full(_B_s.shape, np.inf)
            _dist[_B_s] = 0.0
            for _d in range(1, max(1, int(_se.awareness_cells)) + 1):
                _g = _near.copy()
                _g[1:, :] |= _near[:-1, :]
                _g[:-1, :] |= _near[1:, :]
                _g[:, 1:] |= _near[:, :-1]
                _g[:, :-1] |= _near[:, 1:]
                _new = _g & ~_near
                _dist[_new] = float(_d)
                _near = _g
            _R = int(_se.awareness_cells)
            _rate = np.where(
                _dist <= 0.0, float(_se.in_flame_per_min),
                np.where(_dist <= 1.0, float(_se.adjacent_per_min),
                         np.where(np.isfinite(_dist),
                                  float(_se.aware_per_min)
                                  * np.clip(1.0 - (_dist - 1.0)
                                            / max(_R - 1, 1), 0.0, 1.0),
                                  0.0)))
            # SOMEWHERE TO GO. A cell whose every neighbour is alight has no
            # open direction, so its people cannot flee on their own.
            _open = np.zeros_like(_B_s)
            _open[1:, :] |= ~_B_s[:-1, :]
            _open[:-1, :] |= ~_B_s[1:, :]
            _open[:, 1:] |= ~_B_s[:, :-1]
            _open[:, :-1] |= ~_B_s[:, 1:]
            _rate = _rate * _open
            if float(_rate.max()) > 0.0:
                _dtm_s = float(getattr(cfg, "step_minutes", 30.0))
                _vp_s = world.value.vpop
                _floor = (1.0 - float(_se.max_share)) * self._vpop0
                _room = np.maximum(_vp_s - _floor, 0.0)
                _gone = _room * np.clip(_rate * _dtm_s, 0.0, 1.0)
                _vp_s -= _gone
                self.population_self_evacuated += float(
                    _gone.sum() * (cfg.cell_area_ha / 100.0))
        _rev = getattr(resource, "revac", None)
        if _rev is not None and float(np.max(_rev)) > 1e-6:
            _vp = world.value.vpop
            _dtm_e = float(getattr(cfg, "step_minutes", 30.0))
            # ordered evacuation empties a cell at ~5%/min; people IN
            # or BESIDE actively burning cells flee at ~30%/min, so
            # the person-steps exposure collapses within a few
            # decision steps once the order lands
            _B_e = s.burning > 0.5
            _hot = _B_e.copy()
            _hot[1:, :] |= _B_e[:-1, :]
            _hot[:-1, :] |= _B_e[1:, :]
            _hot[:, 1:] |= _B_e[:, :-1]
            _hot[:, :-1] |= _B_e[:, 1:]
            _rate_e = np.where(_hot, 0.30 * _dtm_e, 0.05 * _dtm_e)
            # WARNED cells respond faster: a public warning issued
            # before (or with) the evacuation order primes the
            # population, roughly doubling the departure tempo at
            # full warning. The warning alone moves nobody.
            _rw = getattr(resource, "rwarn", None)
            if _rw is not None:
                _rate_e = _rate_e * (1.0 + 1.0 * np.clip(_rw, 0.0,
                                                         1.0))
            _fracv = np.clip(_rev, 0.0, 1.0) * np.minimum(0.9,
                                                          _rate_e)
            _moved = _vp * _fracv
            _vp -= _moved
            self.population_evacuated += float(
                _moved.sum() * (cfg.cell_area_ha / 100.0))
        # cost accumulators: exposure (persons in burning cells) and
        # committed capacity, integrated over the steps
        _cell_km2 = self.cfg.cell_area_ha / 100.0
        self.exposure_person_steps += float(
            (world.value.vpop * _cell_km2)[s.burning > 0.5].sum())
        # the ORDERED evacuees, integrated over time: they are displaced for
        # as long as the incident lasts, and that is what the small
        # evacuation weight in J_pop is charged against. Self-evacuation is
        # exogenous, not a decision, so it is not charged to the DSS.
        self.evacuated_person_steps += float(self.population_evacuated)
        self.response_capacity_steps += float(
            (resource.rcap * np.clip(resource.ravail, 0, 1)).sum())
        self.t_elapsed_min += float(getattr(cfg, "step_minutes", 30.0))

        tscale = float(getattr(cfg, "step_minutes", 30.0)) / 30.0
        # terrain-modified wind: ridges are exposed (faster), valleys are
        # sheltered (slower). Linearized mass-consistent adjustment on the
        # normalized elevation; the direction keeps the gradient wind and
        # the upslope pull enters via the effective spread vector below.
        twg = float(getattr(cfg.spread, "terrain_wind_gain", 0.0))
        if twg > 0.0:
            if not hasattr(self, "_elev_norm"):
                e = np.asarray(topo.elev, dtype=float)
                span = float(np.ptp(e))
                self._elev_norm = ((e - e.min()) / span if span > 1e-6
                                   else np.zeros_like(e))
                # valley channeling weight: on steep ground the flow veers
                # toward the local slope axis (drainage direction), so any
                # user-set uniform wind becomes a terrain-following field
                self._chan = 0.5 * np.clip(topo.slope / 0.6, 0.0, 1.0)
            factor = np.clip(1.0 + twg * (self._elev_norm - 0.5) * 2.0,
                             0.4, 1.8)
            c = self._chan
            wwd_eff = np.arctan2(
                (1.0 - c) * np.sin(meteo.wwd) + c * np.sin(topo.aspect),
                (1.0 - c) * np.cos(meteo.wwd) + c * np.cos(topo.aspect))
            meteo = _MeteoView(meteo, meteo.wws * factor, wwd_eff)
        # ---- DRYING, the counterpart of every wetting term below ----
        # Rain, retardant and suppression all RAISE the moisture field. With
        # nothing lowering it the field was monotonically non-decreasing over
        # a run: a cell burned to ash kept its ambient moisture, the front
        # never dried the fuel it was about to reach, and a cell wetted once
        # stayed wet forever, so a line held once held itself for free.
        #
        # This runs BEFORE the wetting terms on purpose. A cell that is being
        # rained on or actively sprayed ends the step wet, because those
        # terms are applied after; a cell the crews have LEFT starts drying
        # back toward what the air can hold.
        _dry = getattr(cfg, "drying", None)
        if _dry is not None and getattr(_dry, "enabled", False):
            _dt_h = float(getattr(cfg, "step_minutes", 30.0)) / 60.0
            # THE AMBIENT TARGET IS THE SCENARIO'S OWN MOISTURE FIELD.
            # This simulator treats dead fuel moisture as an EXOGENOUS
            # field (fuel_moisture.py says so in as many words): the
            # scenario declares what the landscape holds, and the
            # equilibrium model is an optional tool the dashboard applies
            # when the weather is edited. So the recovery term restores
            # that declared state after the model's own wetting, and
            # nothing more. Relaxing toward the air's equilibrium instead
            # would silently re-baseline every scenario: measured, it dried
            # the grass test world far below its declared value until the
            # spread rose enough to drive the adaptive substepping to its
            # cap of 200, and it emptied a deliberately soaked landscape
            # whose entire premise was that fuel above extinction does not
            # carry fire. Only the FIRE may go below this level, through
            # the preheating and combustion terms below, because that
            # drying is caused by the fire and not by the air.
            _meq = self._fmoist0
            _fm = fuel.fmoist
            _alight = s.burning > 0.5
            # COMBUSTION DRYING APPLIES TO THE CHAR, NOT TO THE FLAME.
            # In this engine the rate of spread is a property of the SOURCE
            # cell, so drying a cell while it is alight raises its own rate
            # and pushes the fire into neighbours that its own moisture
            # should have stopped: measured, that carried a single ignition
            # across a landscape soaked to 0.9 until all 400 cells burned.
            # A cell that has gone OUT is ash, and ash is dry, which is
            # what the map should report; because it is no longer a source,
            # that costs the spread model nothing.
            #
            # ASH ONLY, NOT EVERY CELL THAT STOPPED BURNING. A cell can also
            # stop burning because the crews QUENCHED it, and that cell is
            # wet on purpose and still holds its fuel. Drying it undid the
            # very wetting that had just saved it, and it re-lit: measured
            # on the end-to-end test the fire stopped being extinguishable
            # at all (273 cells still alight at the horizon, against 77
            # cells and out in 42 minutes). So the char is identified by
            # its FUEL being spent, not merely by its flame being gone.
            # the engine's OWN exhaustion test, so the two cannot disagree:
            # `has_fuel = Fload > eps_fuel` is what stops a cell burning
            _spent = np.asarray(fuel.fload) <= float(cfg.spread.eps_fuel)
            _burn = (self.first_ignition_step >= 0) & (~_alight) & _spent
            # neighbour flame intensity: what preheats a cell is the fire
            # NEXT to it, so the four-neighbour maximum drives the pull
            _inb = np.zeros_like(_fm)
            _ii = np.asarray(s.intensity, dtype=float) * _alight
            _inb[1:, :] = np.maximum(_inb[1:, :], _ii[:-1, :])
            _inb[:-1, :] = np.maximum(_inb[:-1, :], _ii[1:, :])
            _inb[:, 1:] = np.maximum(_inb[:, 1:], _ii[:, :-1])
            _inb[:, :-1] = np.maximum(_inb[:, :-1], _ii[:, 1:])
            _inb = np.clip(_inb, 0.0, 1.0) * (~_alight) \
                * (self.first_ignition_step < 0)
            # target: ambient equilibrium, pulled DOWN where the front is
            # radiating onto the cell, and collapsed to the char residual
            # inside the flame itself
            # PREHEATING MAY NOT MAKE UNBURNABLE FUEL BURNABLE. Radiant
            # heat from the front does dry the fuel it reaches, but a flame
            # in soaked fuel is weak and loses heat to its surroundings: it
            # does not dry a wet landscape into carrying fire. Left
            # unbounded it did exactly that, walking a single ignition
            # across a deliberately soaked map cell by cell until the whole
            # thing burned. So preheating acts only on fuel that could
            # ALREADY carry fire, which is where it belongs: it makes a
            # receptive fuel bed drier and faster, and it does not resurrect
            # one that is out of the running.
            _mext = np.asarray(
                [FUEL_MODELS[i].m_ext if i in FUEL_MODELS else 0.3
                 for i in range(int(np.max(fuel.ftype)) + 1)],
                dtype=float)[np.asarray(fuel.ftype, dtype=int)]
            _inb = _inb * (self._fmoist0 < _mext)
            _tgt = _meq * (1.0 - float(_dry.preheat_depth) * _inb)
            _tgt = np.where(_burn, float(_dry.burn_floor), _tgt)
            # response time: the 1-hour timelag class, shortened by the
            # preheating and collapsed while the cell is alight
            _tau = np.full_like(_fm, max(1e-3, float(_dry.timelag_h)))
            _tau = _tau / (1.0 + float(_dry.preheat_gain) * _inb)
            _tau = np.where(_burn,
                            max(1e-3, float(_dry.burn_timelag_min) / 60.0),
                            _tau)
            _k = 1.0 - np.exp(-_dt_h / _tau)
            # DRYING ONLY. Absorption from humid air is slower and weaker
            # than the wetting terms already modelled, and letting it run
            # here would re-baseline every scenario that starts drier than
            # its own equilibrium.
            _fm += np.where(_fm > _tgt, (_tgt - _fm) * _k, 0.0)
            np.clip(_fm, 0.0, 1.0, out=_fm)
        # precipitation wets the dead fuel: while it rains the moisture
        # relaxes toward at least 0.35 (above every extinction threshold)
        # with a ~30 min time constant scaled by rain intensity, so
        # sustained rain stalls the spread and lets burning cells exhaust
        # themselves. This runs for EVERY weather source (manual sliders,
        # diurnal driver, real-case series).
        _pr = np.asarray(meteo.prec, dtype=float)
        if float(_pr.max()) > 0.05:
            _fm = fuel.fmoist
            _k = np.clip(_pr / 2.0, 0.0, 1.0) * min(1.0, tscale)
            _fm += (np.maximum(_fm, 0.35) - _fm) * _k
        # grid scaling: the fuel table calibrates r_base in m/min, i.e. in
        # cells per 30 min AT 30 m cells. On a coarser/finer grid one cell is
        # a different distance, so the cell rate is scaled by 30 m / cell so
        # the METRIC speed is independent of the grid resolution.
        cell_scale = 30.0 / float(cfg.cell_size_m)
        # adaptive substepping: the substep count also covers the fastest
        # local rate of spread, so a fire running at R_spread cells per step
        # is resolved without violating the one-cell-per-substep limit; the
        # 99.5th percentile keeps single extreme cells (cliffs) from
        # inflating the substep count
        ros_ref = rate_of_spread(fuel, topo, meteo, cfg.spread) * cell_scale
        # directionality comes from the wind PLUS the upslope pull; the
        # eight directional weight fields are constant within the step, so
        # they are computed once and reused by every substep
        weff_ws, weff_wd = effective_spread_vector(topo, meteo, cfg.spread)
        dir_w = directional_weights(weff_wd, cfg.spread, wws=weff_ws)
        ros_peak = float(np.percentile(ros_ref, 99.5))
        n_sub = max(1, int(np.ceil(tscale - 1e-9)),
                    min(200, int(np.ceil(tscale * ros_peak - 1e-9))))
        # SHADOW-RUN FIDELITY CAP: forecast clones set _substep_cap so
        # a 45-min lookahead does not pay hundreds of substeps on fine
        # grids; all candidates of a comparison share the same cap, so
        # the RELATIVE ranking (which is all a forecast is used for)
        # is preserved while the live run keeps full fidelity.
        _scap = getattr(self, "_substep_cap", None)
        if _scap:
            n_sub = min(n_sub, int(_scap))
        sub = tscale / n_sub

        B_start = s.burning.copy()
        ign0 = world.ignition_field(s.step)
        if extra_ignition is not None:
            ign0 = np.maximum(ign0, extra_ignition)
        # TACTICAL BURN (counter-fire): ordered cells are set alight by
        # the firing crew. A real ignition with the real spread physics
        # follows, so the burnt-out strip is genuinely burnt on the map
        # and a badly judged order genuinely backfires; only the
        # decision layer's forecast gates keep it safe.
        _rb_ord = getattr(resource, "rburn", None)
        if _rb_ord is not None and float(np.max(_rb_ord)) > 1e-6:
            _lit = ((_rb_ord > 0.5) & (s.burning < 0.5)
                    & (s.fload > 0.05)
                    & (fuel.fmoist < 0.30))
            ign0 = np.maximum(ign0, _lit.astype(float))
        # RETARDANT/SOIL DROP: ordered cells gain coating; the field
        # decays at ~0.3%/min (hours of protection, not days)
        _rr_ord = getattr(resource, "rret", None)
        if _rr_ord is not None and float(np.max(_rr_ord)) > 1e-6:
            self.retard = np.maximum(self.retard,
                                     np.clip(_rr_ord, 0.0, 1.0))
        if float(self.retard.max()) > 1e-6:
            _dtm_r = float(getattr(cfg, "step_minutes", 30.0))
            self.retard *= max(0.0, 1.0 - 0.003 * _dtm_r)
            # the coating behaves like moisture that does NOT dry: the
            # spread model reads fuel.fmoist, so the coated cells are
            # floored every step for as long as the coating lasts
            # (0.45 sits above every fuel's extinction moisture)
            fuel.fmoist = np.maximum(
                fuel.fmoist, 0.45 * np.clip(self.retard, 0.0, 1.0))

        comb_tot = np.zeros_like(s.fload)
        red_tot = np.zeros_like(s.fload)
        burned_any = s.burning > 0.5
        ros = None
        for isub in range(n_sub):
            B = s.burning
            Fload = s.fload
            I = s.intensity

            # 1. rate of spread and propagation influence,
            #    scaled to the substep length
            ros = ros_ref * sub
            psi = propagation_influence(B, ros, weff_wd, cfg.spread,
                                        wws=weff_ws, weights=dir_w)

            # 2. burning status update: the influence builds
            #    up over time (with a small leak) and the cell ignites when
            #    the buildup crosses theta_ign; external ignition is
            #    injected on the first substep only
            has_fuel = (Fload > cfg.spread.eps_fuel).astype(float)
            b_pers = B * has_fuel
            # rain knockdown: under sustained rain (>= 3 mm/h) a burning
            # cell is quenched once the rain has driven its fuel moisture
            # up to ~0.35 (the wetting relaxation above supplies the ~30
            # min time constant, so the front dies out gradually, not in
            # one frame)
            if float(_pr.max()) >= 3.0:
                _quench = (_pr >= 3.0) & (fuel.fmoist >= 0.35)
                b_pers = b_pers * (~_quench)
            leak = min(1.0, cfg.spread.buildup_leak * sub)
            self.ign_buildup *= (1.0 - leak)
            self.ign_buildup += psi
            b_prop = ((self.ign_buildup > cfg.spread.theta_ign)
                      .astype(float) * has_fuel)
            if isub == 0:
                ign = ign0 * has_fuel
                B_next = np.maximum.reduce([b_pers, b_prop, ign])
            else:
                B_next = np.maximum(b_pers, b_prop)

            # 2b. optional ember spotting (default off)
            if cfg.spread.spotting:
                ny, nx = B_next.shape
                # embers need wind to loft and travel: no spotting in calm air
                hot = ((B_next > 0.5) & (I > cfg.spread.spot_intensity_min)
                       & (meteo.wws > 3.0) & (meteo.prec < 1.0))
                if hot.any():
                    # spot_prob is defined per REFERENCE step; compound it to
                    # the substep so the expected ember count is independent
                    # of the substep resolution
                    p_eff = 1.0 - (1.0 - min(cfg.spread.spot_prob, 1.0)) ** sub
                    hy, hx = np.where(hot)
                    throw = self._rng.random(hx.size) < p_eff
                    hy, hx = hy[throw], hx[throw]
                    if hx.size:
                        wd = meteo.wwd[hy, hx]
                        d = cfg.spread.spot_distance
                        tx = hx + np.round(d * np.cos(wd)).astype(int)
                        ty = hy - np.round(d * np.sin(wd)).astype(int)
                        ok = (tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny)
                        tx, ty = tx[ok], ty[ok]
                        if tx.size:
                            # an ember only takes where fuel remains AND the
                            # fuel is drier than its extinction moisture
                            m_ext = _fuel_param(fuel.ftype, "m_ext")
                            fuelok = ((Fload[ty, tx] > cfg.spread.eps_fuel)
                                      & (fuel.fmoist[ty, tx]
                                         < m_ext[ty, tx] - 1e-9))
                            B_next[ty[fuelok], tx[fuelok]] = 1.0

            # 3. fuel mass update, substep compounded
            f_burn = np.clip(self._b_base * (1.0 - fuel.fmoist)
                             * (1.0 - 0.8 * np.clip(self.retard,
                                                    0.0, 1.0)),
                             0.0, 1.0)
            _fb_ref = f_burn                     # reference-step scale
            if sub != 1.0:
                f_burn = 1.0 - (1.0 - f_burn) ** sub
            combustion = Fload * B * f_burn
            # AERIAL OPS WEATHER GATE: helicopters and tankers fly
            # freely below 8 m/s, are derated linearly above, and are
            # grounded at 20 m/s (a per-cell field, so a sheltered
            # valley can still be flown while the ridge is not)
            _airs = np.clip(1.0 - (np.asarray(meteo.wws, dtype=float)
                                   - 8.0) / 12.0, 0.0, 1.0)
            f_red_raw = fuel_reduction(resource, topo, I,
                                       cfg.suppression, air_scale=_airs)
            # suppression KNOCKDOWN: crews put flames out, they do not
            # only remove fuel. A burning cell is quenched when the
            # suppression PRESSURE (the eta product of, without
            # the fuel-removal gain alpha_s) exceeds the knockdown
            # threshold scaled by how fiercely the cell burns relative
            # to a calm 0.10/step reference. Running heads in cured
            # grass stay unquenchable (as in reality); moderate surface
            # fire within reach of committed capacity goes out.
            _press = f_red_raw / max(float(cfg.suppression.alpha_s),
                                     1e-6)
            _kn = float(getattr(cfg.suppression, "knockdown_ratio", 0.15))
            if _kn > 0.0:
                # fierceness scales with fuel burn rate AND the local
                # fire intensity: a low-intensity smolder (mop-up) is
                # easy to put out even in fast fuel, while a running
                # high-intensity head stays unquenchable
                _fierce = np.maximum(_fb_ref / 0.10, 0.2) * (
                    0.3 + 0.7 * np.clip(I, 0.0, 1.0))
                _quench = _press > _kn * _fierce
                # dedicated-commitment knockdown: a burning cell with
                # NEAR-FULL committed, available capacity on it is
                # extinguished within the step (crews and bucket
                # drops finish what they fully engage, regardless of
                # road access) UNLESS the cell can still RUN: the
                # local rate of spread is the discriminator, so the
                # operational sequence is engage -> wet (ROS dies)
                # -> extinguish, and a dry wind-driven head stays
                # immune to direct attack until it is wetted
                _quench |= ((resource.rcap
                             > 0.7 * cfg.suppression.rcap_max)
                            & (np.clip(resource.ravail, 0, 1) > 0.6)
                            & (ros_ref < 1.0))
                # PARAMETER CONSISTENCY: the wetting chain drives the
                # fuel toward 0.35 moisture; a committed, SOAKED cell
                # (>=0.30) cannot sustain flame regardless of its
                # nominal ROS, so it goes out instead of burning to
                # black while surrounded by water
                _quench |= ((resource.rcap
                             > 0.5 * cfg.suppression.rcap_max)
                            & (np.clip(resource.ravail, 0, 1) > 0.5)
                            & (fuel.fmoist >= 0.30))
                B_next = B_next * (~_quench)
            # WETTING: suppression is water/retardant, so engaged cells
            # get WETTER, exactly like rain: moisture relaxes toward
            # 0.35 at a rate set by the suppression pressure. This is
            # what makes a held line HOLD (moist fuel refuses ignition,
            # g_moist -> 0) and what makes even a grass fire quenchable
            # once its fuel is wetted (its fierceness f_burn drops).
            _wg = float(getattr(cfg.suppression, "wet_gain", 1.0))
            if _wg > 0.0 and _press.max() > 1e-6:
                _fm = fuel.fmoist
                # COMMITTED FLOOR: on low-access ground (remote
                # forest) the pressure product collapses through
                # eta_reach x eta_eff and pure pressure-driven
                # wetting stalls near the ambient moisture, so a
                # contained fire smolders forever. Crews that are
                # FULLY COMMITTED on a cell (near-full capacity,
                # high availability) keep applying water no matter
                # how bad the road is; their wetting rate is floored
                # (scaled down by access, never to zero) so the
                # engage -> wet -> ROS dies -> extinguish chain
                # always completes on a contained fire.
                # the WETTING floor engages at half commitment: a
                # containment band ordered at moderate strength must
                # still wet up before the front arrives, else a big
                # open-terrain fire outruns its own line (the QUENCH
                # threshold above stays at the stricter 0.7/0.6)
                _cmt = ((resource.rcap
                         > 0.5 * cfg.suppression.rcap_max)
                        & (np.clip(resource.ravail, 0, 1) > 0.5))
                _acc_w = np.clip(topo.access, 0.0, 1.0)
                _raw_ = getattr(resource, "rair", None)
                if _raw_ is not None:
                    _acc_w = np.maximum(
                        _acc_w, np.clip(_raw_, 0.0, 1.0) * _airs)
                # floor calibrated to OPERATIONS, not to the 30-min
                # reference: a committed crew hoses a 30 m cell wet
                # in ~15 min (0.9 per reference step), scaled by the
                # (air-aware) access. The old 0.12/ref was 0.4%/min:
                # a running head outgrew it hopelessly.
                _floor = 0.9 * _cmt * np.clip(
                    2.0 * _acc_w, 0.2, 1.0)
                _rate = np.maximum(np.clip(_press, 0.0, 1.0) * _wg,
                                   _floor) * min(1.0, sub)
                _fm += (np.maximum(_fm, 0.35) - _fm) * _rate
            # COMMITTED LINE-BUILDING: crews/dozers fully committed
            # to an UNBURNING cell clear its fuel at a floor rate no
            # matter how weak the pressure product is (low access,
            # long dispatch): the ordered containment band really
            # gets DUG, so the break exists before the front arrives
            # (the has_fuel gate then makes it unburnable ground)
            _cut = None
            _rc_ord = getattr(resource, "rcut", None)
            if _kn > 0.0 and _rc_ord is not None:
                _cmt_l = ((resource.rcap
                           > 0.5 * cfg.suppression.rcap_max)
                          & (np.clip(resource.ravail, 0, 1) > 0.5))
                # dig ONLY where the LINE is ordered: protection
                # rings and deployments must never scrape the ground
                _cut = _cmt_l & (B < 0.5) & (_rc_ord > 0.5)
            if _cut is not None:
                # a committed dozer/handcrew clears a 30 m cell in
                # ~30 min (0.6 per reference step); the old 0.05/ref
                # needed 5+ hours per cell and no line ever closed
                f_red_raw = np.where(_cut,
                                     np.maximum(f_red_raw, 0.6),
                                     f_red_raw)
            if sub != 1.0:
                f_red_raw = 1.0 - (1.0 - f_red_raw) ** sub
            f_red = np.minimum(f_red_raw, Fload)
            Fload_next = np.maximum(0.0, Fload - combustion - f_red)

            # 4. fire intensity update; uses current fuel 
            I_next = fire_intensity(B_next, Fload, topo, meteo, cfg.intensity)

            # commit the substep; burning cells carry no buildup
            self.ign_buildup[B_next > 0.5] = 0.0
            burned_any |= B_next > 0.5
            s.burning = B_next
            s.fload = Fload_next
            s.intensity = I_next
            comb_tot += combustion
            red_tot += f_red

        # 5. ignition time update, once per outer step
        cont = (s.burning > 0.5) & (B_start > 0.5)
        # THE RESIDENCE CLOCK IS IN MINUTES, which is what (34) and
        # every reader of tau assume. It used to advance by cfg.dt,
        # which counts STEPS, so on any configuration whose step is
        # not one minute the clock silently ran at the wrong rate.
        s.tau = np.where(cont, s.tau + float(cfg.step_minutes), 0.0)
        s.step += 1
        world.fuel.fload = s.fload  # keep the layer in sync for observation

        # 6. bookkeeping and diagnostics; cells that ignited and burned out
        # within the substeps of this very step still count as burned
        active = s.burning > 0.5
        newly = burned_any & (self.first_ignition_step < 0)
        self.first_ignition_step[newly] = s.step
        # a cell that WAS alight and no longer is has burned out at this
        # step; recorded once, so a re-ignition does not rewrite the first
        # time the front passed through
        _went_out = (self.first_ignition_step >= 0) & (~active) \
            & (self.burnout_step < 0)
        self.burnout_step[_went_out] = s.step
        self.ever_burned |= burned_any
        self.fuel_consumed_total += comb_tot
        self.fuel_suppressed_total += red_tot

        diag = StepDiagnostics(
            step=s.step,
            n_burning=int(active.sum()),
            n_burned_cumulative=int(self.ever_burned.sum()),
            fuel_consumed_step=float(comb_tot.sum()),
            fuel_suppressed_step=float(red_tot.sum()),
            ros_mean_active=float(ros[active].mean()) if active.any() else 0.0,
            intensity_mean_active=float(s.intensity[active].mean())
            if active.any() else 0.0,
        )
        self.history.append(diag)
        self._record_snapshot()
        return diag

    # --------------------------------------------------------------------- run
    def run(self, n_steps: Optional[int] = None,
            callback: Optional[Callable[["Simulator", StepDiagnostics], None]] = None,
            stop_when_quiescent: bool = True) -> List[StepDiagnostics]:
        """Run the simulation for n_steps, or until the fire dies out."""
        limit = n_steps if n_steps is not None else self.cfg.max_steps
        for _ in range(limit):
            diag = self.step()
            if callback is not None:
                callback(self, diag)
            if stop_when_quiescent and diag.n_burning == 0 and self.state.step > 1:
                # only stop once a fire has actually started and finished
                if self.ever_burned.any():
                    break
        return self.history


    # ----------------------------------------------------------------- helpers
    @property
    def burned_mask(self) -> np.ndarray:
        return self.ever_burned

    def is_quiescent(self) -> bool:
        return bool((self.state.burning > 0.5).sum() == 0)
