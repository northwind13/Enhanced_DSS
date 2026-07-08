"""Derived fire behaviour outputs (FARSITE / FlamMap style).

These are additive diagnostic fields computed from the simulator state. They do
not change the state transition; they interpret it, the way FARSITE/FlamMap
report rate of spread, fireline intensity, flame length and crown fire activity.
"""

from __future__ import annotations

import numpy as np

from .config import FUEL_MODELS
from .spread import rate_of_spread


def rate_of_spread_field(world) -> np.ndarray:
    """Per cell rate of spread (grid cells per step) from the current inputs."""
    return rate_of_spread(world.fuel, world.topo, world.meteo, world.config.spread)


def rate_of_spread_m_per_min(world) -> np.ndarray:
    cfg = world.config
    return rate_of_spread_field(world) * cfg.cell_size_m / max(cfg.dt, 1e-6)


def fireline_intensity(sim) -> np.ndarray:
    """Byram fireline intensity I = H * w * R (kW/m), non zero on the fire front.

    H  heat of combustion (kJ/kg), w available fuel per area (kg/m2),
    R  rate of spread (m/s). Uses the simulator's current fuel and spread."""
    world = sim.world
    cfg = world.config
    H = cfg.intensity.heat_content
    w = np.clip(world.fuel.fload, 0.0, 1.0) * cfg.intensity.biomass_ref
    r_m_s = rate_of_spread_m_per_min(world) / 60.0
    fli = H * w * r_m_s                      # kW/m
    active = sim.state.burning > 0.5
    return np.where(active, fli, 0.0)


def flame_length(fli: np.ndarray) -> np.ndarray:
    """Byram flame length L = 0.0775 * I^0.46 (I in kW/m, L in metres)."""
    fli = np.clip(fli, 0.0, None)
    return 0.0775 * np.power(fli + 1e-9, 0.46)


def flame_length_field(sim) -> np.ndarray:
    return flame_length(fireline_intensity(sim))


def _forest_mask(ftype) -> np.ndarray:
    out = np.zeros(ftype.shape, dtype=bool)
    for fid, m in FUEL_MODELS.items():
        if m.is_forest:
            out |= (ftype == fid)
    return out


def crown_fire_mask(sim) -> np.ndarray:
    """Cells undergoing crown fire: burning forest above the intensity threshold
    (a simple Van Wagner style flag; does not feed back into spread)."""
    cfg = sim.world.config
    forest = _forest_mask(sim.world.fuel.ftype)
    active = sim.state.burning > 0.5
    return forest & active & (sim.state.intensity > cfg.intensity.crown_fire_threshold)


def perimeter_mask(sim) -> np.ndarray:
    """Boundary cells of the burned area (fire perimeter)."""
    burned = sim.ever_burned
    if not burned.any():
        return np.zeros_like(burned)
    p = np.pad(burned, 1)
    neigh = (p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return burned & ~neigh
