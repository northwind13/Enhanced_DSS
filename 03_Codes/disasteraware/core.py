"""Simulation Core: the discrete time state transition operator Phi.

This module implements the hybrid fire spread model of thesis Section 4.2.3,
combining burning status evolution (Eq. 43 to 48), fuel mass evolution
(Eq. 49 to 50, 68 to 69), fire intensity evolution (Eq. 51, 136 to 137) and
ignition time evolution (Eq. 52) into a single coupled update:

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
from .spread import rate_of_spread, propagation_influence, _fuel_param
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
        # cumulative bookkeeping for cost and metric accounting
        self.ever_burned = np.zeros(world.shape, dtype=bool)
        self.fuel_consumed_total = np.zeros(world.shape, dtype=float)
        self.fuel_suppressed_total = np.zeros(world.shape, dtype=float)
        self.history: List[StepDiagnostics] = []
        self._b_base = _fuel_param(world.fuel.ftype, "b_base")

    # ----------------------------------------------------------------- control
    def reset(self) -> None:
        self.state = SimulationState.from_world(self.world)
        self.ever_burned[:] = False
        self.fuel_consumed_total[:] = 0.0
        self.fuel_suppressed_total[:] = 0.0
        self.history.clear()
        self.world.fuel.fload = self.world.fuel.fload0.copy()
        self.state.fload = self.world.fuel.fload0.copy()

    # -------------------------------------------------------------- transition
    def step(self, resource_override=None,
             extra_ignition: Optional[np.ndarray] = None) -> StepDiagnostics:
        """Advance the state by one time step applying Phi.

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

        B = s.burning
        Fload = s.fload
        I = s.intensity

        # 1. rate of spread and accumulated propagation influence (Eq. 123, 46)
        ros = rate_of_spread(fuel, topo, meteo, cfg.spread)
        psi = propagation_influence(B, ros, meteo.wwd, cfg.spread)

        # 2. burning status update (Eq. 43 to 45)
        has_fuel = (Fload > cfg.spread.eps_fuel).astype(float)
        b_pers = B * has_fuel
        # propagation can only ignite a cell that still holds combustible fuel,
        # so depleting fuel (suppression or preventive reduction) creates a
        # firebreak once a cell drops below the extinction threshold (Eq. 44)
        b_prop = (psi > cfg.spread.theta_ign).astype(float) * has_fuel
        ign = world.ignition_field(s.step)
        if extra_ignition is not None:
            ign = np.maximum(ign, extra_ignition)
        ign = ign * has_fuel
        B_next = np.maximum.reduce([b_pers, b_prop, ign])

        # 3. fuel mass update (Eq. 68 to 69, 129)
        f_burn = np.clip(self._b_base * (1.0 - fuel.fmoist), 0.0, 1.0)
        combustion = Fload * B * f_burn
        f_red_raw = fuel_reduction(resource, topo, I, cfg.suppression)
        # Eq. 135: suppression reduction cannot exceed available fuel
        f_red = np.minimum(f_red_raw, Fload)
        Fload_next = np.maximum(0.0, Fload - combustion - f_red)

        # 4. fire intensity update (Eq. 137); uses current fuel per Eq. 51, 136
        I_next = fire_intensity(B_next, Fload, topo, meteo, cfg.intensity)

        # 5. ignition time update (Eq. 52)
        cont = (B_next > 0.5) & (B > 0.5)
        tau_next = np.where(cont, s.tau + cfg.dt, 0.0)

        # 6. commit state and bookkeeping
        s.burning = B_next
        s.fload = Fload_next
        s.intensity = I_next
        s.tau = tau_next
        s.step += 1
        world.fuel.fload = Fload_next  # keep the layer in sync for observation

        self.ever_burned |= (B_next > 0.5)
        self.fuel_consumed_total += combustion
        self.fuel_suppressed_total += f_red

        active = B_next > 0.5
        diag = StepDiagnostics(
            step=s.step,
            n_burning=int(active.sum()),
            n_burned_cumulative=int(self.ever_burned.sum()),
            fuel_consumed_step=float(combustion.sum()),
            fuel_suppressed_step=float(f_red.sum()),
            ros_mean_active=float(ros[active].mean()) if active.any() else 0.0,
            intensity_mean_active=float(I_next[active].mean()) if active.any() else 0.0,
        )
        self.history.append(diag)
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
