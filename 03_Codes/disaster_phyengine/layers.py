"""External data layers that drive the simulation.

These containers instantiate the external input structure of thesis Section 4.2.1
and the external data sources of Section 4.3:

    MeteoLayer     -> U_Meteo,k   (Table 4.2)
    TopoLayer      -> U_Geo       (Table 4.3)
    FuelLayer      -> U_Fuel,k    (Table 4.4)
    ValueLayer     -> U_Val,k     (Table 4.5)  decisional context
    ResourceLayer  -> U_Res,k / U_DSS,k (Table 4.6)  decisional context

Every field is stored as a 2D array of shape (ny, nx) using the convention
array[y, x]. Helper builders allow a field to be created either as a uniform
constant or from an explicit array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def uniform(ny: int, nx: int, value: float) -> np.ndarray:
    return np.full((ny, nx), float(value), dtype=float)


def as_field(ny: int, nx: int, value) -> np.ndarray:
    """Coerce a scalar or array like into a (ny, nx) float field."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return uniform(ny, nx, float(arr))
    if arr.shape != (ny, nx):
        raise ValueError(f"field shape {arr.shape} does not match grid ({ny}, {nx})")
    return arr.astype(float)


@dataclass
class MeteoLayer:
    """Spatio temporal atmospheric drivers (Table 4.2)."""

    temp: np.ndarray        # air temperature, deg C
    rh: np.ndarray          # relative humidity, percent
    wws: np.ndarray         # wind speed, m/s
    wwd: np.ndarray         # wind direction, rad (math convention, 0 = +x, blows toward)
    gust: np.ndarray        # wind gust speed, m/s
    prec: np.ndarray        # precipitation, mm/h

    @classmethod
    def uniform(cls, ny: int, nx: int, temp=28.0, rh=30.0, wws=6.0,
                wwd=0.0, gust=8.0, prec=0.0) -> "MeteoLayer":
        return cls(
            temp=uniform(ny, nx, temp), rh=uniform(ny, nx, rh),
            wws=uniform(ny, nx, wws), wwd=uniform(ny, nx, wwd),
            gust=uniform(ny, nx, gust), prec=uniform(ny, nx, prec),
        )


@dataclass
class TopoLayer:
    """Static terrain layer (Table 4.3)."""

    elev: np.ndarray        # elevation, m
    slope: np.ndarray       # slope, rad
    aspect: np.ndarray      # aspect (slope orientation), rad
    access: np.ndarray      # accessibility index in [0, 1]

    @classmethod
    def flat(cls, ny: int, nx: int, access=1.0) -> "TopoLayer":
        return cls(
            elev=uniform(ny, nx, 0.0), slope=uniform(ny, nx, 0.0),
            aspect=uniform(ny, nx, 0.0), access=uniform(ny, nx, access),
        )


@dataclass
class FuelLayer:
    """Combustible material layer (Table 4.4).

    ftype holds integer fuel class ids referencing config.FUEL_MODELS.
    fload is the available fuel mass normalized to [0, 1]; fload0 stores the
    initial value for burned area accounting.
    """

    ftype: np.ndarray       # integer fuel class ids
    fload: np.ndarray       # available fuel load in [0, 1]
    fmoist: np.ndarray      # fuel moisture fraction
    fload0: Optional[np.ndarray] = None

    def __post_init__(self):
        self.ftype = self.ftype.astype(int)
        if self.fload0 is None:
            self.fload0 = self.fload.copy()

    @classmethod
    def uniform(cls, ny: int, nx: int, ftype=1, fload=1.0, fmoist=0.08) -> "FuelLayer":
        return cls(
            ftype=np.full((ny, nx), int(ftype), dtype=int),
            fload=uniform(ny, nx, fload),
            fmoist=uniform(ny, nx, fmoist),
        )


@dataclass
class ValueLayer:
    """Values at risk layer (Table 4.5). Inputs are spatial and static during a
    short horizon run; the aggregated priority V_prio is recomputed on demand.
    """

    vbld: np.ndarray        # building footprint presence in [0, 1]
    vcrit: np.ndarray       # critical facility index in [0, 1]
    vpop: np.ndarray        # population density, person/km^2
    vevac: np.ndarray       # distance to evacuation route, m

    @classmethod
    def empty(cls, ny: int, nx: int) -> "ValueLayer":
        return cls(
            vbld=uniform(ny, nx, 0.0), vcrit=uniform(ny, nx, 0.0),
            vpop=uniform(ny, nx, 0.0), vevac=uniform(ny, nx, 0.0),
        )

    def priority(self, weights) -> np.ndarray:
        """Protection priority score V_prio (Eq. 55) as a normalized weighted sum."""
        w = weights.normalized()
        vpop_n = _minmax(self.vpop)
        # evacuation distance is inverted: closer to a route means higher priority
        vevac_n = 1.0 - _minmax(self.vevac)
        vprio = (w.w_bld * np.clip(self.vbld, 0, 1)
                 + w.w_crit * np.clip(self.vcrit, 0, 1)
                 + w.w_pop * vpop_n
                 + w.w_evac * vevac_n)
        return np.clip(vprio, 0.0, 1.0)


@dataclass
class ResourceLayer:
    """Operational suppression resource layer (Table 4.6).

    These fields feed both the suppression mapping (as U_DSS,k) and the
    decisional context (as U_Res,k). When no decision support system is active,
    a static suppression field can be supplied directly to study what if cases.
    """

    rcap: np.ndarray        # suppression capacity (water equivalent / step)
    ravail: np.ndarray      # availability in [0, 1]
    reff: np.ndarray        # suppression efficiency in [0, 1]
    rtime: np.ndarray       # travel time to cell (min or h)

    @classmethod
    def none(cls, ny: int, nx: int) -> "ResourceLayer":
        return cls(
            rcap=uniform(ny, nx, 0.0), ravail=uniform(ny, nx, 0.0),
            reff=uniform(ny, nx, 0.0), rtime=uniform(ny, nx, 0.0),
        )


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
