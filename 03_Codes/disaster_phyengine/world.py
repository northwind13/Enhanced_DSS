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
    # THE SETTLEMENT THIS BELONGS TO. A town is not one asset: it is a
    # painted block of built-up ground, a street grid, its residents and
    # its civic facilities. Without a name tying those together the editor
    # could only delete them one at a time and the block of urban fuel
    # stayed behind, so the town went on burning as a town with nobody in
    # it. Empty for assets placed on their own.
    group: str = ""


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

    #: value of one cell of general built-up ground, against 1.0 for a
    #: designated structure asset. A block of houses is worth less per cell
    #: than a hospital, and more than nothing.
    BUILTUP_VALUE = 0.5

    def seed_builtup_value(self, value: float | None = None) -> int:
        """Give every BUILT-UP cell the structure value it plainly has.

        The value layers used to be written only by add_asset, inside the
        radius of a placed Asset, while the built-up LAND COVER was painted
        across a much wider footprint. On a generated landscape that left
        90% of what the map draws and labels as "urban / built-up" carrying
        no structure value at all: a fire could burn straight through the
        town and the asset loss stayed at zero, which is what the map's own
        legend says should not happen.

        A hand-placed asset still wins where it overlaps, because a named
        hospital is worth more than the block around it.

        Returns the number of cells given a value.
        """
        v = float(self.BUILTUP_VALUE if value is None else value)
        m = np.asarray(self.fuel.ftype) == FUEL_NAME_TO_ID["urban"]
        if not m.any() or v <= 0.0:
            return 0
        self.value.vbld[m] = np.maximum(self.value.vbld[m], v)
        return int(m.sum())

    def spread_population_over_builtup(self) -> int:
        """Put a settlement's people across the settlement, not on a disc.

        add_asset writes the population into a circle of the asset's radius
        around its centre. The map meanwhile paints the town as BLOCKS
        separated by streets, and the two do not coincide: measured on a
        generated landscape the population covered 61% of the built-up
        footprint and spilled onto ground the map does not call a town at
        all, so a fire could burn most of what looks like a city while the
        population cost barely moved.

        Every built-up cell is assigned to the nearest population marker and
        each marker's people are spread evenly over the cells that fall to
        it. Connectivity is deliberately not used: a town is a cluster of
        blocks with roads between them, so its cells are not one connected
        component and treating them as such covers a fraction of it.

        The head count is preserved exactly. Returns the number of markers
        whose people were redistributed.
        """
        pops = [a for a in self.assets
                if getattr(a, "kind", "") == "population"]
        if not pops:
            return 0
        m = np.asarray(self.fuel.ftype) == FUEL_NAME_TO_ID["urban"]
        if not m.any():
            return 0
        ys, xs = np.where(m)
        px = np.array([float(a.x) for a in pops])
        py = np.array([float(a.y) for a in pops])
        # nearest marker for each built-up cell
        d2 = ((xs[:, None] - px[None, :]) ** 2
              + (ys[:, None] - py[None, :]) ** 2)
        owner = np.argmin(d2, axis=1)
        cell_km2 = self.config.cell_area_ha / 100.0
        for a in pops:
            # the disc this marker wrote is cleared first, so people are not
            # left standing in the fields beside the town
            self.value.vpop[self._disk(a.x, a.y,
                                       getattr(a, "radius", 0))] = 0.0
        for i, a in enumerate(pops):
            sel = owner == i
            if not sel.any():
                continue
            # THE TOWN, UNIONED WITH THE GROUND THE MARKER ALREADY HELD.
            # Built-up cells alone would do for coverage, but where the
            # painted footprint is smaller than the marker's disc that
            # CONCENTRATES the same people into fewer cells, and the
            # protection priority reads density: measured, it pulled the
            # allocator off the flame front and onto the town it now
            # thought was three times as dense, and the fire escaped. The
            # union keeps every built-up cell populated while the density
            # can only fall, never spike.
            terr = np.zeros_like(m)
            terr[ys[sel], xs[sel]] = True
            disc = (self._disk(a.x, a.y, getattr(a, "radius", 0))
                    & self.buildable_mask())
            # only the part of the disc that is not another town's ground
            terr |= disc & ~(m & ~terr)
            n_cells = int(terr.sum())
            if n_cells == 0:
                continue
            dens = float(getattr(a, "population", 0.0)) / (n_cells * cell_km2)
            self.value.vpop[terr] = np.maximum(self.value.vpop[terr], dens)
        return len(pops)

    def rebuild_value_layers(self, builtup_value: bool = True,
                             spread_population: bool = True) -> None:
        """Recompute vbld / vcrit / vpop / vevac from the asset list.

        add_asset WRITES into the value layers and nothing takes a written
        value back out, because np.maximum cannot be undone: two assets may
        have contributed to the same cell and there is no record of which.
        So renaming is harmless but MOVING or DELETING an asset needs the
        layers rebuilt from scratch, or a deleted hospital goes on being
        worth protecting at the place it used to stand.

        The derived layers are re-applied afterwards in the same order the
        generator uses, so an edited map and a freshly generated one end up
        described the same way.
        """
        self.value.vbld[:] = 0.0
        self.value.vcrit[:] = 0.0
        self.value.vpop[:] = 0.0
        self.value.vevac[:] = 0.0
        kept = list(self.assets)
        self.assets = []
        for a in kept:
            self.add_asset(a)          # re-appends and re-stamps the layers
        if builtup_value:
            self.seed_builtup_value()
        if spread_population:
            self.spread_population_over_builtup()

    def buildable_mask(self) -> np.ndarray:
        """Ground that can hold structures and people.

        Nothing stands on open water, and the road and bare-ground class is
        the corridor between things rather than a place with something in
        it. The value layers used to be written over a plain disc with no
        regard for what was underneath, so buildings and residents ended up
        on lakes and roads: measured on a generated landscape, 19% of all
        asset value sat on ground that cannot burn, which is both physically
        absurd and put a ceiling under the loss term that no fire could
        reach.
        """
        ft = np.asarray(self.fuel.ftype)
        return (ft != FUEL_NAME_TO_ID["water"]) \
            & (ft != FUEL_NAME_TO_ID["non_fuel"])

    def add_asset(self, asset: Asset) -> None:
        """Place an asset and write its contribution into the value layers."""
        self.assets.append(asset)
        m = self._disk(asset.x, asset.y, asset.radius) & self.buildable_mask()
        if not m.any():
            # the marker itself sits on water or bare ground: keep the point
            # so the map still shows it, but write no value onto the water
            if asset.kind == "evac_route":
                self._stamp_evac_distance(asset.x, asset.y)
            return
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
        """Stamp a circular road / access patch: marks roads and sets access=1.
        Over water the road is a bridge: the roads/access layers are set
        but the cell itself stays water (never drained, never flammable)."""
        from .config import FUEL_NAME_TO_ID
        roads = self._ensure_roads()
        m = self._disk(x, y, max(radius, 0))
        # over water the road is a BRIDGE: it carries access but the cell
        # itself stays water (never drained, never flammable)
        land = m & (self.fuel.ftype != FUEL_NAME_TO_ID["water"])
        roads[m] = True
        self.topo.access[m] = 1.0
        self.fuel.ftype[land] = 0         # paved road is non flammable
        self.fuel.fload[land] = 0.0
        self.fuel.fload0[land] = 0.0

    def add_road_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Rectangular road strip; over water it acts as a bridge (see
        add_road_disk)."""
        from .config import FUEL_NAME_TO_ID
        roads = self._ensure_roads()
        ys, xs = self._mask(x0, y0, x1, y1)
        keep = self.fuel.ftype[ys, xs] != FUEL_NAME_TO_ID["water"]
        roads[ys, xs] = True                 # bridges allowed over water
        self.topo.access[ys, xs] = 1.0
        self.fuel.ftype[ys, xs][keep] = 0   # paved road is non flammable
        self.fuel.fload[ys, xs][keep] = 0.0
        self.fuel.fload0[ys, xs][keep] = 0.0

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
