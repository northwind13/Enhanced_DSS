"""Interaction and scenario control operator (,).

The user interaction operator Theta_UI applies admissible, bounded modifications
to the external input set and the decision context set without ever touching the
authoritative state S_k (state immutability, Remark 4.1). Each admissible
transformation follows

This module provides a thin, traceable API that the dashboard calls so that every
operator action is expressed as an input level transformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class InteractionLog:
    """Audit record of one operator action (supports traceability)."""

    step: int
    kind: str
    detail: str


class InteractionOperator:
    """Applies admissible transformations to a World between time steps."""

    def __init__(self, world):
        self.world = world
        self.log: List[InteractionLog] = []

    def _record(self, step: int, kind: str, detail: str) -> None:
        self.log.append(InteractionLog(step=step, kind=kind, detail=detail))

    # ---- ignition injection: U_Ign,k* = U_Ign,k + delta_Ign, clipped to [0,1]
    def inject_ignition(self, x: int, y: int, step: int, radius: int = 0) -> np.ndarray:
        self.world.add_ignition(x, y, step=step, radius=radius)
        self._record(step, "ignition", f"inject at ({x},{y}) r={radius}")
        return self.world.ignition_field(step)

    # ---- meteorological perturbation: U_Meteo,k* = U_Meteo,k + delta_Meteo
    def perturb_wind(self, step: int, d_speed: float = 0.0, d_dir: float = 0.0) -> None:
        self.world.meteo.wws = np.clip(self.world.meteo.wws + d_speed, 0.0, 60.0)
        self.world.meteo.wwd = (self.world.meteo.wwd + d_dir) % (2 * np.pi)
        self._record(step, "wind", f"d_speed={d_speed:+.2f} d_dir={d_dir:+.2f}")

    def set_moisture(self, step: int, value: float) -> None:
        self.world.fuel.fmoist[:] = np.clip(value, 0.0, 1.0)
        self._record(step, "moisture", f"fmoist={value:.3f}")

    # ---- resource override: U_DSS,k* = U_DSS,k + delta_DSS, clipped to [0,1]
    def deploy_resource(self, step: int, region: Tuple[int, int, int, int],
                        rcap: float = 1.0, ravail: float = 1.0,
                        reff: float = 0.8, rtime: float = 1.0) -> None:
        self.world.set_resource_field(rcap=rcap, ravail=ravail, reff=reff,
                                      rtime=rtime, region=region)
        self._record(step, "resource", f"deploy {region} cap={rcap} eff={reff}")

    # ---- value context: U_Val,k* = sigma_Val * U_Val,k
    def scale_value(self, step: int, sigma: float) -> None:
        sigma = float(np.clip(sigma, 0.0, 1.0))
        self.world.value.vcrit *= sigma
        self.world.value.vbld *= sigma
        self._record(step, "value", f"sigma={sigma:.2f}")
