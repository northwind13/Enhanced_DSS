"""World model: the editable container that holds every external data layer plus
the editing API used by the dashboard and by scenario scripts.

The World groups the simulation configuration with the five external data layers
and the ignition schedule. It exposes high level editing operations so that a
user can paint forest patches, drop assets and schedule ignitions on the grid
without touching the raw arrays. Everything serializes to a plain dict for YAML
or JSON storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SimConfig, FUEL_NAME_TO_ID
from .layers import (MeteoLayer, TopoLayer, FuelLayer, ValueLayer, ResourceLayer)


@dataclass
class Asset:
    """A point or area asset placed on the map for impact accounting."""

    name: str
    kind: str                 # one of: building, critical, population, evac_route
    x: int
    y: int
    radius: int = 0           # 0 means a single cell
    value: float = 1.0        # intensity of the asset attribute (e.g. Vcrit level)
    population: float = 0.0   # persons, for population assets


@dataclass
class IgnitionEvent:
    """A scheduled ignition injection (U_Ign,k)."""

    x: int
    y: int
    step: int = 0
    radius: int = 0


@dataclass
class World:
    config: SimConfig
    meteo: MeteoLayer
    topo: TopoLayer
    fuel: FuelLayer
    value: ValueLayer
    resource: ResourceLayer
    assets: List[Asset] = field(default_factory=list)
    ignitions: List[IgnitionEvent] = field(default_factory=list)
    roads: Optional[np.ndarray] = None   # boolean mask of road / access corridors

    # ------------------------------------------------------------------ build
    @classmethod
    def blank(cls, config: Optional[SimConfig] = None,
              default_fuel: str = "grass", default_load: float = 1.0,
              default_moisture: float = 0.08) -> "World":
        """Create an empty world with uniform default fields."""
        cfg = config or SimConfig()
        ny, nx = cfg.ny, cfg.nx
        fid = FUEL_NAME_TO_ID.get(default_fuel, 1)
        return cls(
            config=cfg,
            meteo=MeteoLayer.uniform(ny, nx),
            topo=TopoLayer.flat(ny, nx),
            fuel=FuelLayer.uniform(ny, nx, ftype=fid, fload=default_load,
                                   fmoist=default_moisture),
            value=ValueLayer.empty(ny, nx),
            resource=ResourceLayer.none(ny, nx),
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.config.ny, self.config.nx

    # ---------------------------------------------------------------- editing
    def _mask(self, x0: int, y0: int, x1: int, y1: int) -> Tuple[slice, slice]:
        ny, nx = self.shape
        xa, xb = sorted((int(np.clip(x0, 0, nx - 1)), int(np.clip(x1, 0, nx - 1))))
        ya, yb = sorted((int(np.clip(y0, 0, ny - 1)), int(np.clip(y1, 0, ny - 1))))
        return slice(ya, yb + 1), slice(xa, xb + 1)

    def _disk(self, x: int, y: int, radius: int) -> np.ndarray:
        ny, nx = self.shape
        yy, xx = np.ogrid[:ny, :nx]
        return (xx - x) ** 2 + (yy - y) ** 2 <= max(radius, 0) ** 2

    def add_forest_patch(self, x0: int, y0: int, x1: int, y1: int,
                         fuel_type: str = "pine_litter", load: float = 1.0,
                         moisture: float = 0.08) -> None:
        """Paint a rectangular forest (or any fuel) patch onto the grid."""
        ys, xs = self._mask(x0, y0, x1, y1)
        fid = FUEL_NAME_TO_ID.get(fuel_type, 3)
        self.fuel.ftype[ys, xs] = fid
        self.fuel.fload[ys, xs] = load
        self.fuel.fload0[ys, xs] = load
        self.fuel.fmoist[ys, xs] = moisture

    def add_forest_disk(self, x: int, y: int, radius: int,
                        fuel_type: str = "pine_litter", load: float = 1.0,
                        moisture: float = 0.08) -> None:
        m = self._disk(x, y, radius)
        fid = FUEL_NAME_TO_ID.get(fuel_type, 3)
        self.fuel.ftype[m] = fid
        self.fuel.fload[m] = load
        self.fuel.fload0[m] = load
        self.fuel.fmoist[m] = moisture

    def clear_fuel(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Convert a region to non burnable (firebreak, water, bare ground)."""
        ys, xs = self._mask(x0, y0, x1, y1)
        self.fuel.ftype[ys, xs] = 0
        self.fuel.fload[ys, xs] = 0.0
        self.fuel.fload0[ys, xs] = 0.0

    def clear_fuel_disk(self, x: int, y: int, radius: int = 1) -> None:
        """Circular firebreak: convert a disk to non burnable."""
        m = self._disk(x, y, max(radius, 0))
        self.fuel.ftype[m] = 0
        self.fuel.fload[m] = 0.0
        self.fuel.fload0[m] = 0.0

    def paint_rect(self, x0: int, y0: int, x1: int, y1: int, ftype: int,
                   load: float = 0.0, moisture: float = 0.08) -> None:
        """Set a rectangular region to an arbitrary fuel class (water=5, bare=0,
        vegetation 1-4). Used by the Fuel and Firebreak tools."""
        ys, xs = self._mask(x0, y0, x1, y1)
        self.fuel.ftype[ys, xs] = int(ftype)
        self.fuel.fload[ys, xs] = load
        self.fuel.fload0[ys, xs] = load
        self.fuel.fmoist[ys, xs] = moisture

    def paint_disk(self, x: int, y: int, radius: int, ftype: int,
                   load: float = 0.0, moisture: float = 0.08) -> None:
        m = self._disk(x, y, max(radius, 0))
        self.fuel.ftype[m] = int(ftype)
        self.fuel.fload[m] = load
        self.fuel.fload0[m] = load
        self.fuel.fmoist[m] = moisture

    def add_asset(self, asset: Asset) -> None:
        """Place an asset and write its contribution into the value layers."""
        self.assets.append(asset)
        m = self._disk(asset.x, asset.y, asset.radius)
        if asset.kind == "building":
            self.value.vbld[m] = np.maximum(self.value.vbld[m], asset.value)
        elif asset.kind == "critical":
            self.value.vcrit[m] = np.maximum(self.value.vcrit[m], asset.value)
        elif asset.kind == "population":
            area_km2 = m.sum() * self.config.cell_area_ha / 100.0
            density = asset.population / area_km2 if area_km2 > 0 else asset.population
            self.value.vpop[m] = np.maximum(self.value.vpop[m], density)
        elif asset.kind == "evac_route":
            self._stamp_evac_distance(asset.x, asset.y)

    def _stamp_evac_distance(self, x: int, y: int) -> None:
        ny, nx = self.shape
        yy, xx = np.ogrid[:ny, :nx]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) * self.config.cell_size_m
        self.value.vevac = np.minimum(
            np.where(self.value.vevac > 0, self.value.vevac, dist), dist)

    def add_ignition(self, x: int, y: int, step: int = 0, radius: int = 0) -> None:
        self.ignitions.append(IgnitionEvent(x=int(x), y=int(y), step=int(step),
                                            radius=int(radius)))

    def set_uniform_wind(self, speed: float, direction_rad: float) -> None:
        self.meteo.wws[:] = speed
        self.meteo.wwd[:] = direction_rad

    def set_resource_field(self, rcap=0.0, ravail=0.0, reff=0.0, rtime=0.0,
                           region: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Set a static suppression resource field, optionally over a region."""
        if region is None:
            sel = (slice(None), slice(None))
        else:
            sel = self._mask(*region)
        self.resource.rcap[sel] = rcap
        self.resource.ravail[sel] = ravail
        self.resource.reff[sel] = reff
        self.resource.rtime[sel] = rtime

    # -------------------------------------------------------------------- roads
    def _ensure_roads(self) -> np.ndarray:
        if self.roads is None or self.roads.shape != self.shape:
            self.roads = np.zeros(self.shape, dtype=bool)
        return self.roads

    def add_road_disk(self, x: int, y: int, radius: int = 1) -> None:
        """Stamp a circular road / access patch: marks roads and sets access=1."""
        roads = self._ensure_roads()
        m = self._disk(x, y, max(radius, 0))
        roads[m] = True
        self.topo.access[m] = 1.0
        self.fuel.ftype[m] = 0            # paved road is non flammable
        self.fuel.fload[m] = 0.0
        self.fuel.fload0[m] = 0.0

    def add_road_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        roads = self._ensure_roads()
        ys, xs = self._mask(x0, y0, x1, y1)
        roads[ys, xs] = True
        self.topo.access[ys, xs] = 1.0
        self.fuel.ftype[ys, xs] = 0       # paved road is non flammable
        self.fuel.fload[ys, xs] = 0.0
        self.fuel.fload0[ys, xs] = 0.0

    def add_road_segment(self, x0: int, y0: int, x1: int, y1: int,
                         width: int = 1) -> None:
        """Rasterize a straight road segment and mark roads + access along it."""
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        n = max(abs(x1 - x0), abs(y1 - y0)) + 1
        xs = np.linspace(x0, x1, n).round().astype(int)
        ys = np.linspace(y0, y1, n).round().astype(int)
        for x, y in zip(xs, ys):
            self.add_road_disk(int(x), int(y), width)

    # ----------------------------------------------------------- elevation
    def bump_terrain(self, x: int, y: int, radius: int = 3,
                     delta: float = 40.0, recompute: bool = True) -> None:
        """Raise (delta > 0) or lower (delta < 0) the ground with a smooth
        Gaussian bump centred on (x, y). Slope and aspect can be refreshed here
        or once after a whole brush stroke via recompute_slope_aspect()."""
        ny, nx = self.shape
        yy, xx = np.ogrid[:ny, :nx]
        sigma = max(float(radius), 1.0) / 2.0
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
        elev = np.asarray(self.topo.elev, dtype=float) + float(delta) * g
        self.topo.elev = np.clip(elev, 0.0, None)
        if recompute:
            self.recompute_slope_aspect()

    def recompute_slope_aspect(self) -> None:
        """Refresh slope and aspect from the current elevation (Horn 3x3)."""
        from .gis import slope_aspect_from_dem
        slope, aspect = slope_aspect_from_dem(
            np.asarray(self.topo.elev, dtype=float), self.config.cell_size_m)
        self.topo.slope = slope
        self.topo.aspect = aspect

    def priority_field(self) -> np.ndarray:
        return self.value.priority(self.config.value_weights)

    # ------------------------------------------------------------ ignition map
    def ignition_field(self, step: int) -> np.ndarray:
        """Binary ignition injection field for the given step (U_Ign,k)."""
        ny, nx = self.shape
        ign = np.zeros((ny, nx), dtype=float)
        for ev in self.ignitions:
            if ev.step == step:
                if ev.radius > 0:
                    ign[self._disk(ev.x, ev.y, ev.radius)] = 1.0
                else:
                    if 0 <= ev.y < ny and 0 <= ev.x < nx:
                        ign[ev.y, ev.x] = 1.0
        return ign

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        def f(arr):
            return np.asarray(arr).tolist()
        return {
            "config": self.config.to_dict(),
            "meteo": {k: f(getattr(self.meteo, k)) for k in
                      ["temp", "rh", "wws", "wwd", "gust", "prec"]},
            "topo": {k: f(getattr(self.topo, k)) for k in
                     ["elev", "slope", "aspect", "access"]},
            "fuel": {"ftype": f(self.fuel.ftype), "fload": f(self.fuel.fload),
                     "fmoist": f(self.fuel.fmoist), "fload0": f(self.fuel.fload0)},
            "value": {k: f(getattr(self.value, k)) for k in
                      ["vbld", "vcrit", "vpop", "vevac"]},
            "resource": {k: f(getattr(self.resource, k)) for k in
                         ["rcap", "ravail", "reff", "rtime"]},
            "assets": [a.__dict__ for a in self.assets],
            "ignitions": [e.__dict__ for e in self.ignitions],
            "roads": (np.asarray(self.roads).astype(int).tolist()
                      if self.roads is not None else None),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "World":
        cfg = SimConfig.from_dict(data["config"])
        a = lambda d, k: np.asarray(d[k], dtype=float)
        meteo = MeteoLayer(**{k: a(data["meteo"], k) for k in
                              ["temp", "rh", "wws", "wwd", "gust", "prec"]})
        topo = TopoLayer(**{k: a(data["topo"], k) for k in
                            ["elev", "slope", "aspect", "access"]})
        fuel = FuelLayer(ftype=np.asarray(data["fuel"]["ftype"], dtype=int),
                         fload=a(data["fuel"], "fload"),
                         fmoist=a(data["fuel"], "fmoist"),
                         fload0=a(data["fuel"], "fload0"))
        value = ValueLayer(**{k: a(data["value"], k) for k in
                              ["vbld", "vcrit", "vpop", "vevac"]})
        resource = ResourceLayer(**{k: a(data["resource"], k) for k in
                                    ["rcap", "ravail", "reff", "rtime"]})
        assets = [Asset(**d) for d in data.get("assets", [])]
        ignitions = [IgnitionEvent(**d) for d in data.get("ignitions", [])]
        roads = data.get("roads")
        roads = (np.asarray(roads, dtype=bool) if roads is not None else None)
        return cls(config=cfg, meteo=meteo, topo=topo, fuel=fuel, value=value,
                   resource=resource, assets=assets, ignitions=ignitions,
                   roads=roads)
