"""Simulation Core: the discrete time state transition operator Phi."""

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


class Simulator:
    """Stateful wrapper around the transition operator Phi."""

    def __init__(self, world: World):
        self.world = world
        self.cfg = world.config
        self.state = SimulationState.from_world(world)
        self.ever_burned = np.zeros(world.shape, dtype=bool)
        self.fuel_consumed_total = np.zeros(world.shape, dtype=float)
        self.fuel_suppressed_total = np.zeros(world.shape, dtype=float)
        # cumulative person-steps exposed inside the burned footprint
        self.exposure_person_steps: float = 0.0
        self.first_ignition_step = np.full(world.shape, -1, dtype=int)
        self.ign_buildup = np.zeros(world.shape, dtype=float)
        self.history: List[StepDiagnostics] = []
        self._b_base = _fuel_param(world.fuel.ftype, "b_base")
        self._rng = np.random.default_rng(self.cfg.rng_seed)
        self.record_states: bool = True
        self._snap_budget_bytes: int = 150 * 1024 * 1024
        self._snapshots: dict = {0: self._snapshot()}

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
            "cons": self.fuel_consumed_total.astype(np.float32),
            "supp": self.fuel_suppressed_total.astype(np.float32),
            "exp_steps": float(self.exposure_person_steps),
        }

    def _record_snapshot(self) -> None:
        if not self.record_states:
            return
        self._snapshots[self.state.step] = self._snapshot()
        per = sum(a.nbytes for a in self._snapshots[self.state.step].values()
                  if hasattr(a, "nbytes"))
        cap = max(20, self._snap_budget_bytes // max(per, 1))
        while len(self._snapshots) > cap:
            self._snapshots.pop(min(self._snapshots))

    @property
    def rewindable_steps(self) -> list:
        return sorted(self._snapshots)

    def rewind(self, k: int) -> bool:
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
        self.fuel_consumed_total = snap["cons"].astype(float)
        self.fuel_suppressed_total = snap["supp"].astype(float)
        self.exposure_person_steps = float(snap.get("exp_steps", 0.0))
        self.world.fuel.fload = s.fload
        self.history = self.history[:k]
        self._snapshots = {kk: v for kk, v in self._snapshots.items() if kk <= k}
        return True

    def reset(self) -> None:
        self.state = SimulationState.from_world(self.world)
        self.ever_burned[:] = False
        self.fuel_consumed_total[:] = 0.0
        self.fuel_suppressed_total[:] = 0.0
        self.exposure_person_steps = 0.0
        self.first_ignition_step[:] = -1
        self.ign_buildup[:] = 0.0
        self.history.clear()
        self._snapshots = {0: self._snapshot()}
        self.world.fuel.fload = self.world.fuel.fload0.copy()
        self.state.fload = self.world.fuel.fload0.copy()

    def step(self, resource_override=None,
             extra_ignition: Optional[np.ndarray] = None) -> StepDiagnostics:
        cfg = self.cfg
        s = self.state
        world = self.world
        meteo, topo, fuel = world.meteo, world.topo, world.fuel
        resource = resource_override if resource_override is not None else world.resource

        tscale = float(getattr(cfg, "step_minutes", 30.0)) / 30.0
        cell_scale = 30.0 / float(cfg.cell_size_m)
        ros_ref = rate_of_spread(fuel, topo, meteo, cfg.spread) * cell_scale
        weff_ws, weff_wd = effective_spread_vector(topo, meteo, cfg.spread)
        dir_w = directional_weights(weff_wd, cfg.spread, wws=weff_ws)
        ros_peak = float(np.percentile(ros_ref, 99.5))
        n_sub = max(1, int(np.ceil(tscale - 1e-9)),
                    min(200, int(np.ceil(tscale * ros_peak - 1e-9))))
        sub = tscale / n_sub

        B_start = s.burning.copy()
        ign0 = world.ignition_field(s.step)
        if extra_ignition is not None:
            ign0 = np.maximum(ign0, extra_ignition)

        comb_tot = np.zeros_like(s.fload)
        red_tot = np.zeros_like(s.fload)
        burned_any = s.burning > 0.5
        ros = None
        for isub in range(n_sub):
            B = s.burning
            Fload = s.fload
            I = s.intensity

            ros = ros_ref * sub
            psi = propagation_influence(B, ros, weff_wd, cfg.spread,
                                        wws=weff_ws, weights=dir_w)

            has_fuel = (Fload > cfg.spread.eps_fuel).astype(float)
            b_pers = B * has_fuel
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

            if cfg.spread.spotting:
                ny, nx = B_next.shape
                hot = ((B_next > 0.5) & (I > cfg.spread.spot_intensity_min)
                       & (meteo.wws > 3.0))
                if hot.any():
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
                            m_ext = _fuel_param(fuel.ftype, "m_ext")
                            fuelok = ((Fload[ty, tx] > cfg.spread.eps_fuel)
                                      & (fuel.fmoist[ty, tx]
                                         < m_ext[ty, tx] - 1e-9))
                            B_next[ty[fuelok], tx[fuelok]] = 1.0

            f_burn = np.clip(self._b_base * (1.0 - fuel.fmoist), 0.0, 1.0)
            if sub != 1.0:
                f_burn = 1.0 - (1.0 - f_burn) ** sub
            combustion = Fload * B * f_burn
            f_red_raw = fuel_reduction(resource, topo, I, cfg.suppression)
            if sub != 1.0:
                f_red_raw = 1.0 - (1.0 - f_red_raw) ** sub
            f_red = np.minimum(f_red_raw, Fload)
            Fload_next = np.maximum(0.0, Fload - combustion - f_red)

            I_next = fire_intensity(B_next, Fload, topo, meteo, cfg.intensity)

            self.ign_buildup[B_next > 0.5] = 0.0
            burned_any |= B_next > 0.5
            s.burning = B_next
            s.fload = Fload_next
            s.intensity = I_next
            comb_tot += combustion
            red_tot += f_red

        cont = (s.burning > 0.5) & (B_start > 0.5)
        s.tau = np.where(cont, s.tau + cfg.dt, 0.0)
        s.step += 1
        world.fuel.fload = s.fload

        active = s.burning > 0.5
        newly = burned_any & (self.first_ignition_step < 0)
        self.first_ignition_step[newly] = s.step
        self.ever_burned |= burned_any
        self.fuel_consumed_total += comb_tot
        self.fuel_suppressed_total += red_tot
        # cumulative population-steps of exposure inside the burned footprint,
        # used by the normalized population term of the decision cost
        cell_km2 = self.cfg.cell_area_ha / 100.0
        self.exposure_person_steps += float(
            np.sum(self.world.value.vpop[self.ever_burned])) * cell_km2

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
                if self.ever_burned.any():
                    break
        return self.history

    @property
    def burned_mask(self) -> np.ndarray:
        return self.ever_burned

    def is_quiescent(self) -> bool:
        return bool((self.state.burning > 0.5).sum() == 0)
