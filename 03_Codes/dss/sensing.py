"""Sensing layer: the observation side of the input space (article Layer 2).

The DSS never reads the hidden wildfire state directly. A network of sensor
assets samples the state channels (burning, fuel load, intensity, ignition
time) and the DSS reasons on the composite of the latest measurements. The
sensor set follows Section II.A of the article: satellite imagery, aerial
imagery (UAV), in-situ sensors, and field reports. The external data layers
(meteorology, terrain, fuel class, values, resources) are known inputs and
are not sensed.

Per-cell observation confidence follows the article: it is formed from
sensor coverage, sensor density, and the age of the most recent report, and
is taken conservatively as the weakest channel rather than an average.
Uncovered or stale cells keep the last known measurement (persistence) with
decayed confidence, so the gating of Eq. (6) falls back to the prior instead
of trusting a blind read.

Sensors are droppable assets: position, coverage radius, noise, update
period, report latency; a field layout is composed per scenario and later
placed by drag and drop on the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from disasteraware.observation import Observation

CHANNELS = ("burning", "fload", "intensity", "tau")
REQUIRED_CHANNELS = ("burning", "fload", "intensity")

AGE_DECAY = 0.85          # per-step confidence decay of a stale measurement

SENSOR_PRESETS = {
    # kind: (radius, channels, noise, period, latency)
    "satellite": (None, ("burning", "intensity"), 0.15, 12, 2),
    "aerial": (6, ("burning", "intensity", "fload", "tau"), 0.05, 1, 0),
    "in_situ": (2, ("burning", "intensity", "fload"), 0.02, 1, 0),
    "field_report": (4, ("burning",), 0.30, 6, 3),
}


@dataclass
class Sensor:
    """A droppable sensing asset. radius=None covers the whole grid."""

    kind: str
    x: int = 0
    y: int = 0
    radius: Optional[int] = None
    channels: Tuple[str, ...] = CHANNELS
    noise: float = 0.1
    period: int = 1
    latency: int = 0
    availability: float = 1.0
    sensor_id: str = ""

    @classmethod
    def preset(cls, kind: str, x: int = 0, y: int = 0,
               sensor_id: str = "", **overrides) -> "Sensor":
        if kind not in SENSOR_PRESETS:
            raise ValueError(f"unknown sensor kind: {kind!r}")
        radius, channels, noise, period, latency = SENSOR_PRESETS[kind]
        params = dict(kind=kind, x=x, y=y, radius=radius, channels=channels,
                      noise=noise, period=period, latency=latency,
                      sensor_id=sensor_id or f"{kind}_{x}_{y}")
        params.update(overrides)
        return cls(**params)

    def coverage(self, shape) -> np.ndarray:
        if self.radius is None:
            return np.ones(shape, dtype=bool)
        ny, nx = shape
        yy, xx = np.ogrid[:ny, :nx]
        return (xx - self.x) ** 2 + (yy - self.y) ** 2 <= self.radius ** 2


class SensorNetwork:
    """Composite observation of the hidden state from a set of sensors."""

    def __init__(self, sensors: List[Sensor], shape: Tuple[int, int],
                 seed: Optional[int] = None):
        self.sensors = list(sensors)
        self.shape = shape
        self._rng = np.random.default_rng(seed)
        self.last: Dict[str, np.ndarray] = {c: np.zeros(shape) for c in CHANNELS}
        self.conf: Dict[str, np.ndarray] = {c: np.zeros(shape) for c in CHANNELS}

    # ------------------------------------------------------------- sampling
    def sample(self, sim, k: int) -> None:
        """Let every due sensor measure the true state at step k."""
        state = sim.state
        truth = {"burning": state.burning, "fload": state.fload,
                 "intensity": state.intensity, "tau": state.tau}

        for c in CHANNELS:                      # age term: staleness decay
            self.conf[c] *= AGE_DECAY

        for s in self.sensors:
            if s.availability <= 0 or k % max(s.period, 1) != 0:
                continue
            cov = s.coverage(self.shape)
            # latency discounts the freshness of the reading (age at arrival)
            c_new = (1.0 - s.noise) * s.availability * (AGE_DECAY ** s.latency)
            for ch in s.channels:
                meas = truth[ch].copy()
                if s.noise > 0 and ch == "burning":
                    flip = self._rng.random(self.shape) < (s.noise * 0.2)
                    meas = np.where(flip, 1.0 - (meas > 0.5), meas)
                elif s.noise > 0:
                    meas = np.clip(meas + self._rng.uniform(
                        -s.noise, s.noise, self.shape), 0.0, None)
                better = cov & (c_new > self.conf[ch])
                self.last[ch][better] = meas[better]
                # density: overlapping coverage compounds confidence
                dens = cov & ~better
                self.conf[ch][dens] = 1.0 - (1.0 - self.conf[ch][dens]) \
                    * (1.0 - 0.5 * c_new)
                self.conf[ch][better] = c_new

    # ------------------------------------------------------------ composite
    def composite(self, step: int,
                  region_mask: Optional[np.ndarray] = None
                  ) -> Tuple[Observation, np.ndarray]:
        """Latest composite observation and per-cell confidence kappa.

        kappa is the minimum over the required channels (weakest channel),
        zero where nothing has ever been sensed.
        """
        def cut(a):
            return np.where(region_mask, a, 0.0) if region_mask is not None \
                else a.copy()

        obs = Observation(
            burning=cut(self.last["burning"]),
            fload=cut(np.clip(self.last["fload"], 0.0, 1.0)),
            intensity=cut(np.clip(self.last["intensity"], 0.0, 1.0)),
            tau=cut(self.last["tau"]),
            step=step,
        )
        kappa = np.minimum.reduce([self.conf[c] for c in REQUIRED_CHANNELS])
        kappa = np.clip(kappa, 0.0, 1.0)
        if region_mask is not None:
            kappa = np.where(region_mask, kappa, 0.0)
        return obs, kappa

    @classmethod
    def ideal(cls, shape) -> "SensorNetwork":
        """Full-coverage, noise-free network (baselines and tests)."""
        return cls([Sensor(kind="ideal", radius=None, channels=CHANNELS,
                           noise=0.0, period=1, latency=0,
                           sensor_id="ideal")], shape)
