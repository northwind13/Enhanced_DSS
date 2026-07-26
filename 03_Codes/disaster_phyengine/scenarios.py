"""Built in example scenarios used by the dashboard and the tests.

Each builder returns a ready to run World so that a user has a meaningful
starting point and can then edit forests, assets and parameters from there.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig, FUEL_NAME_TO_ID
from .world import World, Asset


def _synthetic_terrain(world: World, ridge: bool = True) -> None:
    """Add a simple ridge so that slope and aspect drive directional behaviour."""
    ny, nx = world.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    if ridge:
        elev = 400.0 * np.exp(-((xx - nx * 0.5) ** 2) / (2 * (nx * 0.18) ** 2))
        world.topo.elev = elev
        gy, gx = np.gradient(elev, world.config.cell_size_m)
        world.topo.slope = np.clip(np.arctan(np.hypot(gx, gy)), 0.0, 1.2)
        world.topo.aspect = np.arctan2(-gy, -gx)


def wui_interface(seed: int = 42) -> World:
    """Wildland urban interface: a forest belt next to a town with assets."""
    cfg = SimConfig(nx=100, ny=70, cell_size_m=30.0, max_steps=400, rng_seed=seed)
    w = World.blank(cfg, default_fuel="grass", default_load=0.6, default_moisture=0.10)
    _synthetic_terrain(w)

    # dense forest belt on the western half
    w.add_forest_patch(5, 5, 45, 64, fuel_type="pine_litter", load=1.0, moisture=0.07)
    w.add_forest_patch(20, 10, 35, 40, fuel_type="hardwood", load=1.0, moisture=0.12)
    # shrub transition zone
    w.add_forest_patch(45, 5, 60, 64, fuel_type="shrub", load=0.8, moisture=0.09)
    # a river firebreak
    w.clear_fuel(62, 0, 64, 69)

    # town with assets on the eastern side
    w.add_asset(Asset("Town centre", "building", x=82, y=35, radius=8, value=1.0))
    w.add_asset(Asset("Hospital", "critical", x=80, y=28, radius=2, value=1.0))
    w.add_asset(Asset("Power station", "critical", x=88, y=42, radius=1, value=1.0))
    w.add_asset(Asset("Residents", "population", x=82, y=35, radius=8,
                      value=1.0, population=12000))
    w.add_asset(Asset("Evacuation route", "evac_route", x=99, y=35, radius=0))

    # prevailing wind blowing east toward the town
    w.set_uniform_wind(speed=9.0, direction_rad=0.0)

    # ignition deep in the forest
    w.add_ignition(x=12, y=34, step=0, radius=1)
    return w


def grassland_run(seed: int = 7) -> World:
    """Fast moving grassland fire on flat terrain, strong wind."""
    cfg = SimConfig(nx=120, ny=80, cell_size_m=25.0, max_steps=300, rng_seed=seed)
    w = World.blank(cfg, default_fuel="grass", default_load=1.0, default_moisture=0.06)
    w.add_forest_disk(x=90, y=40, radius=14, fuel_type="shrub", load=0.9, moisture=0.10)
    w.add_asset(Asset("Farmstead", "building", x=95, y=40, radius=3, value=1.0))
    w.add_asset(Asset("Residents", "population", x=95, y=40, radius=3,
                      value=1.0, population=300))
    w.set_uniform_wind(speed=14.0, direction_rad=0.0)
    w.add_ignition(x=10, y=40, step=0, radius=1)
    return w


def mountain_forest(seed: int = 13) -> World:
    """Forest fire climbing a ridge, illustrating slope driven acceleration."""
    cfg = SimConfig(nx=90, ny=90, cell_size_m=30.0, max_steps=450, rng_seed=seed)
    w = World.blank(cfg, default_fuel="shrub", default_load=0.7, default_moisture=0.12)
    _synthetic_terrain(w)
    w.add_forest_patch(10, 10, 80, 80, fuel_type="pine_litter", load=1.0, moisture=0.09)
    w.add_asset(Asset("Lodge", "building", x=45, y=10, radius=3, value=1.0))
    w.add_asset(Asset("Lookout", "critical", x=50, y=8, radius=1, value=0.7))
    w.set_uniform_wind(speed=7.0, direction_rad=np.pi / 2)  # wind blowing north, upslope
    w.add_ignition(x=45, y=80, step=0, radius=1)
    return w


def city_wui(seed: int = 21) -> World:
    """Ready made city / wildland-urban interface map: a street grid of built-up
    blocks surrounded by forest and shrub, with many building and critical
    assets. Fire started in the wildland can spread into the city."""
    cfg = SimConfig(nx=120, ny=90, cell_size_m=25.0, max_steps=400, rng_seed=seed)
    w = World.blank(cfg, default_fuel="grass", default_load=0.7, default_moisture=0.09)
    _synthetic_terrain(w, ridge=False)
    urban = FUEL_NAME_TO_ID["urban"]

    # surrounding wildland: forest belt on the west, shrub transition
    w.add_forest_patch(2, 2, 45, 88, fuel_type="pine_litter", load=1.0, moisture=0.07)
    w.add_forest_patch(45, 2, 60, 88, fuel_type="shrub", load=0.8, moisture=0.09)

    # city core: grid of built-up blocks separated by streets
    cx0, cy0, cx1, cy1 = 66, 18, 112, 74
    block = 7
    for by in range(cy0, cy1, block):
        for bx in range(cx0, cx1, block):
            w.paint_rect(bx, by, bx + block - 2, by + block - 2, urban,
                         load=0.6, moisture=0.06)          # a city block
    # major streets only (every 2 blocks) so fire can move block to block; the
    # wildland-facing edge is left open so a WUI fire can enter the city
    for gx in range(cx0 + 2 * block, cx1, 2 * block):
        w.add_road_rect(gx, cy0, gx, cy1)
    for gy in range(cy0 + 2 * block, cy1, 2 * block):
        w.add_road_rect(cx0, gy, cx1, gy)
    # an arterial from the wildland to the city (kept off the ignition row)
    w.add_road_segment(0, 38, cx0, 38, width=1)
    # embers let a wildfire jump streets into the built-up area (WUI behaviour)
    cfg.spread.spotting = True
    cfg.spread.spot_prob = 0.06
    cfg.spread.spot_distance = 3
    cfg.spread.spot_intensity_min = 0.4

    # assets across the city
    w.add_asset(Asset("City centre", "building", 88, 45, radius=10, value=1.0))
    w.add_asset(Asset("Residents", "population", 88, 45, radius=12,
                      value=1.0, population=45000))
    w.add_asset(Asset("Hospital", "critical", 78, 32, radius=2, value=1.0))
    w.add_asset(Asset("Power station", "critical", 104, 60, radius=2, value=1.0))
    w.add_asset(Asset("Water works", "critical", 100, 28, radius=1, value=0.9))
    w.add_asset(Asset("School", "building", 82, 60, radius=2, value=0.8))
    w.add_asset(Asset("Evacuation route", "evac_route", 119, 45, radius=0))

    w.set_uniform_wind(speed=10.0, direction_rad=0.0)      # wind toward the city
    w.add_ignition(x=15, y=58, step=0, radius=2)           # ignition in the forest
    return w


SCENARIOS = {
    "City / WUI": city_wui,
    "Wildland urban interface": wui_interface,
    "Grassland fire": grassland_run,
    "Mountain forest": mountain_forest,
}
