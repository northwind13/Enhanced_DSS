"""Sensor network: the observation side of the DSS.

Implements the observation model of the DSS (see System Description
Sec. 11). The observed state components are the
state vector channels j in {B, F, I, tau}; fuel moisture and weather
come from the meteorological service and static maps (terrain, fuel type,
values, own resources) are known priors.

Per-component observation confidence (min-aggregation of
four independent degradation factors), cell confidence (Eq. 69) and the
bounded disturbance (Eqs. 67, 70):

    conf_{j,k}^i(x,y) = min( theta_{j,k}^i(x,y),          observability
                             rho_k^i(x,y),                 coverage density
                             exp(-lambda_conf * dt_rep),   freshness decay
                             gamma_k^i )                   source reliability
    conf_k^i(x,y)     = min_j conf_{j,k}^i(x,y)
    |eps_{j,k}^i|    <= (1 - conf_{j,k}^i) * eps_bar_j

In the architecture figure the four factors are written
gamma_obs (= theta), gamma_cov (= rho), gamma_fre (= freshness decay) and
gamma_rel (= gamma); epistemic uncertainty enters the system exclusively
through this partial-observation channel.

Sensor families (Layer 1, sensed sources):
    satellite      satellite imagery + hot-spot detection (B, I):
                   whole map, infrequent, latent
    aerial         UAV / aerial thermal recon (B, I): narrow, frequent
    ground_camera  fixed lookout camera, smoke/flame (B, I): wide ring,
                   continuous, line-of-sight quality
    in_situ        environmental ground sensors + field survey (F)
    field_report   first-responder field data (B, tau): sparse, delayed
    public_report  public reports / emergency calls (B): only where
                   people are, slow and unreliable

Layer 1 sources that are NOT sensors here: terrain DEM, fuel/vegetation
map, road/access network, value-at-risk, own resources and suppression
sources are KNOWN PRIORS (they enter U_Geo / U_Fuel / U_Val / U_Res
directly); meteorological stations and forecasts drive U_Meteo. The
pre-fire fuel map is folded into the F channel as an aged prior below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# state-component channels j in {B, F, I, tau}
CHANNELS = ("burning", "fload", "intensity", "tau")

# type -> footprint radius (m; None = whole map), revisit (min;
#         every source reads once per simulation minute, the
#         software's minimum step, so no source lags the sim),
#         latency (min), observed channels, bounded noise amplitude eps_bar
SENSOR_CATALOG: Dict[str, dict] = {
    "satellite": dict(radius_m=None, revisit_min=1.0, latency_min=20.0,
                      channels=("burning", "intensity"), eps=0.05,
                      label="Satellite imagery (whole map)"),
    "aerial": dict(radius_m=2500.0, revisit_min=1.0, latency_min=2.0,
                   channels=("burning", "intensity"), eps=0.03,
                   label="UAV / aerial thermal recon (2.5 km)"),
    "ground_camera": dict(radius_m=4000.0, revisit_min=1.0,
                          latency_min=1.0,
                          channels=("burning", "intensity"), eps=0.06,
                          label="Fixed lookout camera (smoke/flame, 4 km)"),
    "in_situ": dict(radius_m=1500.0, revisit_min=1.0, latency_min=0.0,
                    channels=("fload",), eps=0.02,
                    label="Environmental ground sensors (fuel, 1.5 km)"),
    "field_report": dict(radius_m=1000.0, revisit_min=1.0,
                         latency_min=10.0, channels=("burning", "tau"),
                         eps=0.10,
                         label="First-responder field data (1 km)"),
    "public_report": dict(radius_m=1200.0, revisit_min=1.0,
                          latency_min=15.0, channels=("burning",), eps=0.20,
                          label="Public reports / calls (1.2 km)"),
}

CHANNEL_SYMBOL = {"burning": "B", "fload": "F",
                  "intensity": "I", "tau": "\u03c4"}

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
        # first passes are STAGGERED within each family (a fleet does not
        # report in lockstep) and every report is DELIVERED only after the
        # source latency has elapsed (the data describe sampling time)
        kind_seen: Dict[str, int] = {}
        self._timers = []
        for s_ in self.sensors:
            k = kind_seen.get(s_.kind, 0)
            kind_seen[s_.kind] = k + 1
            rev = SENSOR_CATALOG[s_.kind]["revisit_min"]
            self._timers.append((k % 4) * rev / 4.0)
        self._pending: list = []      # [due_min, sensor_idx, fp, {ch: vals}]
        self._last_report = [AGE_INIT_MIN for _ in self.sensors]
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
        # the PRE-FIRE FUEL MAP is itself a source for F (known prior:
        # "Pre-fire fuel maps, field survey"): observable everywhere, with
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
            # the pre-fire fuel map is a known prior (:
            # "pre-fire fuel maps"), with aged confidence
            self.obs["fload"] = np.asarray(sim.world.fuel.fload0,
                                           dtype=float).copy()
            self.age["fload"][:] = 3.0 * 90.0
            self._primed = True
        for ch in CHANNELS:
            self.age[ch] += dt_min
        for i in range(len(self._last_report)):
            self._last_report[i] += dt_min
        # deliver the reports whose latency has elapsed: the fields written
        # are the SAMPLED ones, so the picture arriving now is latency old
        still = []
        for rec in self._pending:
            rec[0] -= dt_min
            due, si, fp, vals = rec
            if due <= 0.0:
                lat = self.sensors[si].spec()["latency_min"]
                for ch, v in vals.items():
                    self.obs[ch][fp] = v
                    self.age[ch][fp] = lat - due
                self._last_report[si] = lat - due
            else:
                still.append(rec)
        self._pending = still
        for i, s in enumerate(self.sensors):
            self._timers[i] -= dt_min
            if self._timers[i] > 0:
                continue
            spec = s.spec()
            self._timers[i] += max(spec["revisit_min"], dt_min)
            fp = self._footprint(s)
            vals = {}
            for ch in spec["channels"]:
                true = np.asarray(self._true_field(sim, ch), dtype=float)
                # bounded disturbance, |eps| <= (1 - conf) * eps_bar (Eq. 70)
                conf = self.conf_channel(ch)
                amp = spec["eps"] * (1.0 - conf)
                noise = amp * (2.0 * self._rng.random(true.shape) - 1.0)
                if ch == "tau":
                    val = np.maximum(true * (1.0 + noise), 0.0)
                else:
                    val = np.clip(true + noise, 0.0, 1.0)
                if ch == "burning":
                    val = (val > 0.5).astype(float)
                vals[ch] = val[fp]
            lat = float(spec["latency_min"])
            if lat <= 0.0:
                for ch, v in vals.items():
                    self.obs[ch][fp] = v
                    self.age[ch][fp] = 0.0
                self._last_report[i] = 0.0
            else:
                self._pending.append([lat, i, fp, vals])

    # ------------------------------------------------------------- status
    def status(self) -> List[dict]:
        """Per-sensor timing: next pass, reports en route, last data age."""
        out = []
        for i, s in enumerate(self.sensors):
            spec = s.spec()
            lr = self._last_report[i]
            out.append(dict(
                kind=s.kind, x=s.x, y=s.y,
                next_pass_min=max(0.0, float(self._timers[i])),
                latency_min=float(spec["latency_min"]),
                in_transit=sum(1 for r in self._pending if r[1] == i),
                last_report_min=(None if lr >= AGE_INIT_MIN / 2
                                 else float(lr))))
        return out

    def region_age(self, region) -> Dict[str, float]:
        """Median data age (min) per channel over the region's cells."""
        sy, sx = region.slices()
        return {ch: float(np.median(self.age[ch][sy, sx]))
                for ch in CHANNELS}

    # ---------------------------------------------------------- confidence
    def conf_channel(self, ch: str) -> np.ndarray:
        """conf_{j,k}(x,y): min of the four confidence factors."""
        fresh = np.exp(-LAMBDA_CONF * self.age[ch])
        return np.minimum.reduce([self.theta[ch], self.rho[ch], fresh,
                                  self.gamma[ch]])

    def conf_cell(self) -> np.ndarray:
        """conf_k(x,y) = min_j conf_{j,k}(x,y).

        This conservative minimum is the MODEL value used for gating and
        for the disturbance bound (Eq. 70)."""
        return np.minimum.reduce([self.conf_channel(ch) for ch in CHANNELS])

    def region_conf(self, region) -> float:
        """Region-level DISPLAY scalar: mean over components of the mean
        channel confidence in the region. The Eq. 69 minimum stays the
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


def suggest_network(world, budget: Dict[str, int] | None = None):
    """Optimization-based field deployment (greedy maximum weighted
    coverage).

    Objective: choose positions that maximize sum_cells risk(x,y) *
    coverage(x,y), where risk combines spread danger (rate of spread),
    protection priority (values at risk) and available fuel, and each
    placed asset lifts the coverage of the cells in its footprint by its
    sensing quality w = 0.9 * |channels| / 4 (coverages compose as
    1 - prod(1 - w)). Greedy placement of one asset at a time is the
    standard (1 - 1/e)-approximation for this submodular objective.

    Constraints per family: ground hardware only on land; lookout
    cameras prefer high ground (line of sight); field-report posts sit
    on the road network (crews report from reachable ground); public
    reports are pinned to the settlements (calls come from people, they
    are not placed by the planner); one satellite is always tasked.

    Returns (placements, rationale_lines)."""
    from disaster_phyengine.behavior import rate_of_spread_field
    from disaster_phyengine.config import FUEL_NAME_TO_ID
    cfg = world.config
    ny, nx = cfg.ny, cfg.nx
    cell = float(cfg.cell_size_m)
    budget = dict(budget or {"aerial": 2, "ground_camera": 2,
                             "in_situ": 3, "field_report": 2})

    ros = np.asarray(rate_of_spread_field(world), dtype=float)
    ros = ros / (ros.max() + 1e-9)
    pri = np.asarray(world.priority_field(), dtype=float)
    pri = pri / (pri.max() + 1e-9)
    fload = np.clip(np.asarray(world.fuel.fload, dtype=float), 0.0, 1.0)
    risk = 0.45 * ros + 0.35 * pri + 0.20 * fload
    water = world.fuel.ftype == FUEL_NAME_TO_ID["water"]
    risk[water] = 0.0
    land = ~water

    e = np.asarray(world.topo.elev, dtype=float)
    span = float(np.ptp(e))
    elev_n = (e - e.min()) / span if span > 1e-6 else np.zeros_like(e)
    roads = getattr(world, "roads", None)
    roads = (np.asarray(roads, dtype=bool)
             if roads is not None else np.zeros((ny, nx), dtype=bool))

    # candidate lattice (keeps the greedy scan cheap)
    step = max(2, min(nx, ny) // 28)
    cand = [(x, y) for y in range(step // 2, ny, step)
            for x in range(step // 2, nx, step) if land[y, x]]

    yy, xx = np.mgrid[0:ny, 0:nx]
    cov = np.full((ny, nx), 0.30)          # satellite baseline (infrequent)
    placements = [dict(kind="satellite", x=0, y=0)]
    lines = ["satellite: always tasked (whole-map B, I baseline)"]

    def _quality(kind):
        return 0.9 * len(SENSOR_CATALOG[kind]["channels"]) / len(CHANNELS)

    order = [k for k in ("aerial", "ground_camera", "in_situ",
                         "field_report") for _ in range(budget.get(k, 0))]
    same_kind: Dict[str, list] = {}
    for kind in order:
        spec = SENSOR_CATALOG[kind]
        r = max(1, int(round(spec["radius_m"] / cell)))
        w = _quality(kind)
        best, best_gain = None, -1.0
        pool = cand
        if kind == "field_report" and roads.any():
            ry, rx = np.where(roads)
            idx = np.linspace(0, rx.size - 1,
                              min(rx.size, 250)).astype(int)
            pool = list(zip(rx[idx].tolist(), ry[idx].tolist()))
        for (x, y) in pool:
            # same-kind assets keep at least one footprint radius apart,
            # otherwise the greedy piles the fleet on a single hot spot
            if any((x - px) ** 2 + (y - py) ** 2 < r * r
                   for px, py in same_kind.get(kind, [])):
                continue
            m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
            gain = float((risk[m] * (1.0 - cov[m])).sum()) * w
            if kind == "ground_camera":
                gain *= 0.5 + 0.5 * float(elev_n[y, x])   # line of sight
            if gain > best_gain:
                best_gain, best = gain, (x, y)
        if best is None:
            continue
        same_kind.setdefault(kind, []).append(best)
        x, y = best
        m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        cov[m] = 1.0 - (1.0 - cov[m]) * (1.0 - w)
        placements.append(dict(kind=kind, x=int(x), y=int(y)))
        why = {"aerial": "max marginal risk coverage",
               "ground_camera": "max risk coverage x line of sight",
               "in_situ": "max uncovered fuel/value risk",
               "field_report": "max risk coverage on the road network"}
        lines.append(f"{kind} @ ({x},{y}): {why[kind]} "
                     f"(gain {best_gain:.1f})")

    # public reports live where the people are
    pops = sorted((a for a in getattr(world, "assets", [])
                   if getattr(a, "kind", "") == "population"),
                  key=lambda a: -float(getattr(a, "population", 0) or 0))
    for a in pops[:2]:
        placements.append(dict(kind="public_report",
                               x=int(a.x), y=int(a.y)))
        lines.append(f"public_report @ ({a.x},{a.y}): pinned to "
                     f"settlement '{a.name}' (calls come from people)")
    return placements, lines
