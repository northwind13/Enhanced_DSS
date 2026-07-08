"""Sensor network: the observation side of the DSS.

Implements the observation model with the heterogeneous source families of
the framework. The observed state components are the state vector channels
j in {B, F, I, tau}; fuel moisture and weather come from the meteorological
service and static maps (terrain, fuel type, values, own resources) are
known priors.

Per-component observation confidence (min-aggregation of four independent
degradation factors), cell confidence and the bounded disturbance:

    conf_{j,k}^i(x,y) = min( theta_{j,k}^i(x,y),          observability
                             rho_k^i(x,y),                 coverage density
                             exp(-lambda_conf * dt_rep),   freshness decay
                             gamma_k^i )                   source reliability
    conf_k^i(x,y)     = min_j conf_{j,k}^i(x,y)
    |eps_{j,k}^i|    <= (1 - conf_{j,k}^i) * eps_bar_j

Sensor families:
    satellite     satellite imagery (B, I): whole map, infrequent, latent
    aerial        aerial reconnaissance / thermal IR (B, I): narrow, frequent
    in_situ       in-situ ground sensors + field survey (F): point area
    field_report  crew reports / event log (B, tau): sparse, delayed, noisy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# state-component channels j in {B, F, I, tau}
CHANNELS = ("burning", "fload", "intensity", "tau")

# type -> footprint radius (m; None = whole map), revisit (min),
#         latency (min), observed channels, bounded noise amplitude eps_bar
SENSOR_CATALOG: Dict[str, dict] = {
    "satellite": dict(radius_m=None, revisit_min=360.0, latency_min=20.0,
                      channels=("burning", "intensity"), eps=0.05,
                      label="Satellite imagery (whole map, 6 h revisit)"),
    "aerial": dict(radius_m=2500.0, revisit_min=15.0, latency_min=2.0,
                   channels=("burning", "intensity"), eps=0.03,
                   label="Aerial recon / thermal IR (2.5 km, 15 min)"),
    "in_situ": dict(radius_m=1500.0, revisit_min=5.0, latency_min=0.0,
                    channels=("fload",), eps=0.02,
                    label="In-situ ground sensors (fuel state, 1.5 km)"),
    "field_report": dict(radius_m=1000.0, revisit_min=30.0,
                         latency_min=10.0, channels=("burning", "tau"),
                         eps=0.10,
                         label="Field reports / event log (1 km, 30 min)"),
}

LAMBDA_CONF = float(np.log(2.0) / 90.0)   # freshness halves in 90 min
AGE_INIT_MIN = 1e6                        # "never observed"


@dataclass
class Sensor:
    kind: str
    x: int
    y: int
    name: str = ""

    def spec(self) -> dict:
        return SENSOR_CATALOG[self.kind]


@dataclass
class SensorNetwork:
    """Maintains the fused observation O_k^i and its confidence factors."""

    sensors: List[Sensor]
    ny: int
    nx: int
    cell_m: float
    seed: int = 0
    obs: Dict[str, np.ndarray] = field(default_factory=dict)
    age: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        shape = (self.ny, self.nx)
        for ch in CHANNELS:
            self.obs[ch] = np.zeros(shape, dtype=float)
            self.age[ch] = np.full(shape, AGE_INIT_MIN, dtype=float)
        self._timers = [0.0 for _ in self.sensors]
        self._rng = np.random.default_rng(self.seed)
        self._primed = False
        # structural factors, fixed by the fleet layout:
        # theta_j: is component j observable AT ALL at this cell
        # rho: normalized coverage density (assets covering the cell / 2)
        # gamma_j: best source reliability (1 - eps_bar) covering the cell
        self.theta = {ch: np.zeros(shape, dtype=float) for ch in CHANNELS}
        self.gamma = {ch: np.zeros(shape, dtype=float) for ch in CHANNELS}
        nsens = {ch: np.zeros(shape, dtype=float) for ch in CHANNELS}
        for s in self.sensors:
            fp = self._footprint(s)
            spec = s.spec()
            for ch in spec["channels"]:
                nsens[ch][fp] += 1.0
                self.theta[ch][fp] = 1.0
                self.gamma[ch][fp] = np.maximum(self.gamma[ch][fp],
                                                1.0 - spec["eps"])
        # the PRE-FIRE FUEL MAP is itself a source for F (pre-fire fuel
        # maps, field survey): observable everywhere, with
        # survey-grade reliability, counting as half a sensing asset
        self.theta["fload"][:] = 1.0
        self.gamma["fload"] = np.maximum(self.gamma["fload"], 0.7)
        nsens["fload"] += 0.5
        self.rho = {ch: np.clip(nsens[ch] / 2.0, 0.0, 1.0)
                    for ch in CHANNELS}

    # ------------------------------------------------------------- helpers
    def _footprint(self, s: Sensor) -> np.ndarray:
        spec = s.spec()
        if spec["radius_m"] is None:
            return np.ones((self.ny, self.nx), dtype=bool)
        r = max(1, int(round(spec["radius_m"] / self.cell_m)))
        yy, xx = np.mgrid[0:self.ny, 0:self.nx]
        return (xx - s.x) ** 2 + (yy - s.y) ** 2 <= r * r

    def _true_field(self, sim, ch: str) -> np.ndarray:
        return {"burning": sim.state.burning,
                "intensity": sim.state.intensity,
                "fload": sim.state.fload,
                "tau": sim.state.tau}[ch]

    # -------------------------------------------------------------- update
    def update(self, sim, dt_min: float) -> None:
        """Advance the network by one simulation step of dt_min minutes."""
        if not self._primed:
            # the pre-fire fuel map is a known prior (pre-fire fuel
            # maps), with aged confidence
            self.obs["fload"] = np.asarray(sim.world.fuel.fload0,
                                           dtype=float).copy()
            self.age["fload"][:] = 3.0 * 90.0
            self._primed = True
        for ch in CHANNELS:
            self.age[ch] += dt_min
        for i, s in enumerate(self.sensors):
            self._timers[i] -= dt_min
            if self._timers[i] > 0:
                continue
            spec = s.spec()
            self._timers[i] += max(spec["revisit_min"], dt_min)
            fp = self._footprint(s)
            for ch in spec["channels"]:
                true = np.asarray(self._true_field(sim, ch), dtype=float)
                # bounded disturbance, |eps| <= (1 - conf) * eps_bar
                conf = self.conf_channel(ch)
                amp = spec["eps"] * (1.0 - conf)
                noise = amp * (2.0 * self._rng.random(true.shape) - 1.0)
                if ch == "tau":
                    val = np.maximum(true * (1.0 + noise), 0.0)
                else:
                    val = np.clip(true + noise, 0.0, 1.0)
                if ch == "burning":
                    val = (val > 0.5).astype(float)
                self.obs[ch][fp] = val[fp]
                self.age[ch][fp] = spec["latency_min"]

    # ---------------------------------------------------------- confidence
    def conf_channel(self, ch: str) -> np.ndarray:
        """conf_{j,k}(x,y) - min of the four factors."""
        fresh = np.exp(-LAMBDA_CONF * self.age[ch])
        return np.minimum.reduce([self.theta[ch], self.rho[ch], fresh,
                                  self.gamma[ch]])

    def conf_cell(self) -> np.ndarray:
        """conf_k(x,y) = min_j conf_{j,k}(x,y).

        This conservative minimum is the MODEL value used for gating and
        for the disturbance bound."""
        return np.minimum.reduce([self.conf_channel(ch) for ch in CHANNELS])

    def region_conf(self, region) -> float:
        """Region-level DISPLAY scalar: mean over components of the mean
        channel confidence in the region. The cell-level minimum stays the
        cell-level model value (conf_cell); a component with no source at
        all (e.g. tau without field reports) would otherwise pin every
        region to zero and hide all information the sensors do deliver."""
        sy, sx = region.slices()
        vals = [float(self.conf_channel(ch)[sy, sx].mean())
                for ch in CHANNELS]
        return float(np.mean(vals))

    def region_conf_components(self, region) -> dict:
        sy, sx = region.slices()
        return {ch: float(self.conf_channel(ch)[sy, sx].mean())
                for ch in CHANNELS}

    def coverage_note(self, region) -> str:
        sy, sx = region.slices()
        worst = min(CHANNELS,
                    key=lambda c: float(self.conf_channel(c)[sy, sx].mean()))
        a = self.age[worst][sy, sx]
        fresh = float((a < 30.0).mean())
        return (f"weakest component: {worst} · "
                f"{fresh:.0%} of cells observed in the last 30 min")
