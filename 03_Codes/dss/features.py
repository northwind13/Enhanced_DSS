"""Layer 2 perception: the ten bounded features (article Eq. 3, Table A.III).

The feature-extraction map converts the sensed observation and the known
external inputs into ten normalized indicators, each a clamped scalar field
in [0, 1]:

    fire_intensity, spread_potential, weather_severity, ignition_proximity,
    fuel_load, asset_exposure, resource_accessibility, access_road_status,
    suppression_availability, temporal_urgency

The hidden wildfire state reaches this layer only through the sensing
composite (dss.sensing.SensorNetwork); the external data layers (meteo,
terrain, fuel class, values, resources) are known inputs and are read
directly. The per-cell observation confidence kappa produced by the sensor
network is retained as a first-class quantity and later gates the concepts
(Eq. 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from disasteraware.config import FUEL_MODELS

FEATURE_NAMES = (
    "fire_intensity",
    "spread_potential",
    "weather_severity",
    "ignition_proximity",
    "fuel_load",
    "asset_exposure",
    "resource_accessibility",
    "access_road_status",
    "suppression_availability",
    "temporal_urgency",
)

# calibration defaults (feature normalization)
TEMP_REF_LOW, TEMP_REF_HIGH = 5.0, 45.0
WWS_REF = 20.0
PRECIP_DAMP = 0.5
PROX_DECAY_CELLS = 8.0
CONF_EPS_REF = 0.5


def _fuel_param(ftype: np.ndarray, name: str) -> np.ndarray:
    out = np.zeros(ftype.shape, dtype=float)
    for fid, model in FUEL_MODELS.items():
        out[ftype == fid] = getattr(model, name)
    return out


def _chebyshev_distance_to(mask: np.ndarray, max_iter: int = 64) -> np.ndarray:
    """Chebyshev distance (cells) to the nearest True cell of mask."""
    if not mask.any():
        return np.full(mask.shape, float(max_iter))
    try:
        from scipy.ndimage import distance_transform_cdt
        return distance_transform_cdt(~mask, metric="chessboard").astype(float)
    except Exception:
        dist = np.full(mask.shape, float(max_iter))
        frontier = mask.copy()
        dist[frontier] = 0.0
        for d in range(1, int(max_iter)):
            grown = frontier.copy()
            grown[1:, :] |= frontier[:-1, :]
            grown[:-1, :] |= frontier[1:, :]
            grown[:, 1:] |= frontier[:, :-1]
            grown[:, :-1] |= frontier[:, 1:]
            grown[1:, 1:] |= frontier[:-1, :-1]
            grown[1:, :-1] |= frontier[:-1, 1:]
            grown[:-1, 1:] |= frontier[1:, :-1]
            grown[:-1, :-1] |= frontier[1:, 1:]
            newly = grown & ~frontier
            if not newly.any():
                break
            dist[newly] = float(d)
            frontier = grown
        return dist


@dataclass
class FeatureSet:
    """Ten bounded feature fields plus the per-cell observation confidence."""

    values: Dict[str, np.ndarray] = field(default_factory=dict)
    confidence: Optional[np.ndarray] = None
    step: int = 0

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[name]


def observation_confidence(shape, epsilon: float = 0.0,
                           region_mask: Optional[np.ndarray] = None
                           ) -> np.ndarray:
    """Fallback confidence when no sensor network is used: zero outside the
    observed region, degraded by the disturbance magnitude inside it."""
    kappa_noise = float(np.clip(1.0 - epsilon / CONF_EPS_REF, 0.0, 1.0))
    kappa = np.full(shape, kappa_noise, dtype=float)
    if region_mask is not None:
        kappa = np.where(region_mask, kappa, 0.0)
    return kappa


def extract_features(obs, world, epsilon: float = 0.0,
                     region_mask: Optional[np.ndarray] = None,
                     kappa: Optional[np.ndarray] = None) -> FeatureSet:
    """Feature extraction map F: (O_k, U_ext) -> [0, 1]^10 per cell (Eq. 3).

    kappa : per-cell confidence from the sensor network. When omitted, the
        fallback epsilon-based confidence is used (ideal-observer runs).
    """
    meteo, topo, fuel = world.meteo, world.topo, world.fuel
    res = world.resource
    cfg = world.config

    burning = obs.burning > 0.5
    intensity = np.clip(obs.intensity, 0.0, 1.0)
    fload = np.clip(obs.fload, 0.0, 1.0)

    wind_n = np.clip(meteo.wws / WWS_REF, 0.0, 1.0)
    temp_n = np.clip((meteo.temp - TEMP_REF_LOW) / (TEMP_REF_HIGH - TEMP_REF_LOW),
                     0.0, 1.0)
    dry_n = np.clip(1.0 - meteo.rh / 100.0, 0.0, 1.0)
    weather_severity = np.clip(
        (0.40 * wind_n + 0.35 * dry_n + 0.25 * temp_n)
        * np.exp(-PRECIP_DAMP * np.clip(meteo.prec, 0.0, None)), 0.0, 1.0)

    r_base = _fuel_param(fuel.ftype, "r_base")
    m_ext = np.maximum(_fuel_param(fuel.ftype, "m_ext"), 1e-6)
    moist_damp = np.clip(1.0 - fuel.fmoist / m_ext, 0.0, 1.0)
    slope_n = np.clip(topo.slope / 0.7854, 0.0, 1.0)
    r_base_n = np.clip(r_base / max(m.r_base for m in FUEL_MODELS.values()),
                       0.0, 1.0)
    has_fuel = (fload > cfg.spread.eps_fuel).astype(float)
    spread_potential = np.clip(
        r_base_n * moist_damp * (0.4 + 0.4 * wind_n + 0.2 * slope_n) * has_fuel,
        0.0, 1.0)

    dist = _chebyshev_distance_to(burning)
    ignition_proximity = np.where(burning, 1.0,
                                  np.exp(-dist / PROX_DECAY_CELLS))
    if not burning.any():
        ignition_proximity = np.zeros_like(dist)

    asset_exposure = np.clip(world.value.priority(cfg.value_weights), 0.0, 1.0)

    reach = np.exp(-cfg.suppression.beta_t * np.clip(res.rtime, 0.0, None))
    resource_accessibility = np.clip(reach * np.clip(topo.access, 0.0, 1.0),
                                     0.0, 1.0)

    access_road_status = np.clip(topo.access, 0.0, 1.0)
    if world.roads is not None:
        access_road_status = np.maximum(access_road_status,
                                        world.roads.astype(float))

    rcap_n = np.clip(res.rcap / max(cfg.suppression.rcap_max, 1e-6), 0.0, 1.0)
    suppression_availability = np.clip(
        0.5 * np.clip(res.ravail, 0.0, 1.0) + 0.5 * rcap_n, 0.0, 1.0)

    temporal_urgency = np.clip(
        ignition_proximity * (0.3 + 0.7 * spread_potential), 0.0, 1.0)

    values = {
        "fire_intensity": intensity,
        "spread_potential": spread_potential,
        "weather_severity": weather_severity,
        "ignition_proximity": ignition_proximity,
        "fuel_load": fload,
        "asset_exposure": asset_exposure,
        "resource_accessibility": resource_accessibility,
        "access_road_status": access_road_status,
        "suppression_availability": suppression_availability,
        "temporal_urgency": temporal_urgency,
    }
    if region_mask is not None:
        values = {k: np.where(region_mask, v, 0.0) for k, v in values.items()}

    if kappa is None:
        kappa = observation_confidence(intensity.shape, epsilon, region_mask)
    else:
        kappa = np.clip(kappa, 0.0, 1.0)
        if region_mask is not None:
            kappa = np.where(region_mask, kappa, 0.0)
    return FeatureSet(values=values, confidence=kappa, step=obs.step)
