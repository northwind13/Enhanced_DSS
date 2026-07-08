"""Optional dead fuel moisture dynamics.

By default the simulator treats fuel moisture as a static exogenous field (the
base assumption). This module provides an optional equilibrium moisture
content (EMC) model (Simard 1968) so that moisture can respond to temperature
and relative humidity, as operational tools do. It is only applied when the
dashboard toggle is on."""

from __future__ import annotations

import numpy as np


def equilibrium_moisture(temp_c, rh_pct) -> np.ndarray:
    """Simard (1968) equilibrium moisture content as a fraction (0..1)."""
    T = np.asarray(temp_c, dtype=float)
    H = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 100.0)
    emc = np.where(
        H < 10.0,
        0.03229 + 0.281073 * H - 0.000578 * H * T,
        np.where(H < 50.0,
                 2.22749 + 0.160107 * H - 0.014784 * T,
                 21.0606 + 0.005565 * H * H - 0.00035 * H * T - 0.483199 * H))
    return np.clip(emc / 100.0, 0.01, 0.6)


def update_dead_fuel_moisture(world) -> None:
    """Set the fuel moisture field from the current temperature and humidity."""
    world.fuel.fmoist[:] = equilibrium_moisture(world.meteo.temp, world.meteo.rh)
