"""Wildfire state representation (Eq. 41, 61).

The authoritative state is the per cell vector s = (B, Fload, I, tau):

    B      burning status in {0, 1}
    Fload  available combustible mass in [0, 1]
    I      local fire intensity proxy in [0, 1]
    tau    time since ignition

The state is stored as four aligned 2D arrays. Only the Simulation Core writes
to it; observation and decision layers read derived quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationState:
    burning: np.ndarray     # B_k
    fload: np.ndarray       # Fload_k
    intensity: np.ndarray   # I_k
    tau: np.ndarray         # tau_k
    step: int = 0

    @classmethod
    def from_world(cls, world) -> "SimulationState":
        ny, nx = world.shape
        return cls(
            burning=np.zeros((ny, nx), dtype=float),
            fload=world.fuel.fload.copy(),
            intensity=np.zeros((ny, nx), dtype=float),
            tau=np.zeros((ny, nx), dtype=float),
            step=0,
        )

    def copy(self) -> "SimulationState":
        return SimulationState(
            burning=self.burning.copy(), fload=self.fload.copy(),
            intensity=self.intensity.copy(), tau=self.tau.copy(), step=self.step)
