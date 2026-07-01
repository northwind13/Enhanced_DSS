"""Procedural realistic landscape generation.

Builds a believable terrain (fractal elevation), derives slope and aspect, and
assigns fuel classes by elevation bands plus clustered forest patches. The
output is a ready to run World at any resolution, so the dashboard can create
realistic maps without external GIS data.

Implementation uses NumPy only. Gaussian smoothing is done in the frequency
domain so there is no SciPy dependency.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig, FUEL_NAME_TO_ID
from .world import World, Asset
from .layers import TopoLayer, FuelLayer


def _gaussian_smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur via FFT (separable, periodic)."""
    if sigma <= 0:
        return field
    ny, nx = field.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    kernel = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (kx ** 2 + ky ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(field) * kernel))


def fractal_noise(ny: int, nx: int, rng: np.random.Generator,
                  octaves: int = 5, persistence: float = 0.55) -> np.ndarray:
    """Sum of smoothed random octaves, normalized to roughly [0, 1]."""
    out = np.zeros((ny, nx), dtype=float)
    amp, total = 1.0, 0.0
    base = max(nx, ny)
    for o in range(octaves):
        sigma = base / (2 ** (o + 1)) / 3.0
        layer = _gaussian_smooth(rng.standard_normal((ny, nx)), sigma)
        std = layer.std()
        if std > 1e-9:
            layer /= std
        out += amp * layer
        total += amp
        amp *= persistence
    out /= max(total, 1e-9)
    out -= out.min()
    rng_span = out.max() - out.min()
    return out / rng_span if rng_span > 1e-9 else out


def _smooth1d(a, k):
    k = max(1, int(k))
    if k <= 1:
        return a
    ker = np.ones(k) / k
    return np.convolve(a, ker, mode="same")


def _valley_road_y(elev):
    """Column-wise lowest row, smoothed, giving a natural curvy valley road."""
    yroad = np.argmin(elev, axis=0).astype(float)
    yroad = _smooth1d(yroad, max(3, elev.shape[1] // 12))
    return yroad.round().astype(int)


PRESETS = {
    "Mountain forest": dict(relief_m=750.0, forest_density=0.62, water_level=0.03,
                            coast=False, river=False, base_moisture=0.09),
    "River valley":    dict(relief_m=320.0, forest_density=0.5, water_level=0.05,
                            coast=False, river=True, base_moisture=0.10),
    "Mediterranean coast": dict(relief_m=380.0, forest_density=0.42, water_level=0.0,
                                coast=True, river=False, base_moisture=0.06),
    "Rolling hills":   dict(relief_m=200.0, forest_density=0.35, water_level=0.04,
                            coast=False, river=False, base_moisture=0.08),
    "Flat grassland":  dict(relief_m=70.0, forest_density=0.16, water_level=0.02,
                            coast=False, river=False, base_moisture=0.07),
}


def generate_landscape(config: SimConfig | None = None,
                       seed: int = 42,
                       relief_m: float = 450.0,
                       forest_density: float = 0.45,
                       base_moisture: float = 0.08,
                       water_level: float = 0.06,
                       wind_speed: float = 8.0,
                       wind_dir_rad: float = 0.0,
                       with_assets: bool = True,
                       with_roads: bool = True,
                       preset: str | None = None,
                       coast: bool = False,
                       river: bool = False) -> World:
    """Generate a realistic World.

    preset          optional named landscape type (see PRESETS); overrides the
                    relief / forest / water / coast / river arguments
    relief_m        peak to valley elevation difference in meters
    forest_density  fraction of land covered by forest clusters (0 to 1)
    water_level     fraction of lowest cells turned into water (non fuel)
    coast           add a sea along the eastern edge with a wavy coastline
    river           carve a meandering river along the valley
    """
    if preset and preset in PRESETS:
        pp = PRESETS[preset]
        relief_m = pp["relief_m"]; forest_density = pp["forest_density"]
        water_level = pp["water_level"]; base_moisture = pp["base_moisture"]
        coast = pp["coast"]; river = pp["river"]

    cfg = config or SimConfig()
    ny, nx = cfg.ny, cfg.nx
    rng = np.random.default_rng(seed)

    elev = fractal_noise(ny, nx, rng, octaves=6, persistence=0.58) * relief_m
    gy, gx = np.gradient(elev, cfg.cell_size_m)
    slope = np.clip(np.arctan(np.hypot(gx, gy)), 0.0, 1.3)
    aspect = np.arctan2(-gy, -gx)
    elev_norm = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-9)

    grass = FUEL_NAME_TO_ID["grass"]
    shrub = FUEL_NAME_TO_ID["shrub"]
    pine = FUEL_NAME_TO_ID["pine_litter"]
    hard = FUEL_NAME_TO_ID["hardwood"]

    ftype = np.full((ny, nx), grass, dtype=int)
    veg = fractal_noise(ny, nx, rng, octaves=5, persistence=0.6)
    ftype[(veg > 0.45) & (elev_norm > 0.25)] = shrub
    forest_threshold = np.quantile(veg, 1.0 - np.clip(forest_density, 0.0, 0.95))
    forest = veg >= forest_threshold
    ftype[forest & (elev_norm <= 0.55)] = hard
    ftype[forest & (elev_norm > 0.55)] = pine

    if water_level > 0:
        thr = np.quantile(elev_norm, np.clip(water_level, 0.0, 0.3))
        ftype[elev_norm <= thr] = 0

    # coastline: sea along the eastern edge with a wavy boundary
    if coast:
        xx = np.arange(nx)[None, :]
        yy = np.arange(ny)[:, None]
        wave = 0.10 * nx * np.sin(2 * np.pi * yy / max(ny, 1) * 1.5)
        coastline = 0.82 * nx + wave
        sea = xx > coastline
        ftype[sea] = 0
        elev[sea] = 0.0
        elev_norm = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-9)

    # river: meandering non-fuel watercourse along the valley
    yroad = _valley_road_y(elev)
    if river:
        for x in range(nx):
            y = int(np.clip(yroad[x], 0, ny - 1))
            ftype[max(0, y - 1):y + 2, x] = 0

    fload = np.where(ftype > 0,
                     0.55 + 0.45 * fractal_noise(ny, nx, rng, octaves=4), 0.0)
    fload = np.clip(fload, 0.0, 1.0)
    moisture = base_moisture + 0.10 * (1.0 - elev_norm)
    moisture = np.clip(moisture, 0.02, 0.5)

    world = World.blank(cfg)
    world.topo = TopoLayer(elev=elev, slope=slope, aspect=aspect,
                           access=np.clip(1.0 - slope / 1.3, 0.1, 1.0))
    world.fuel = FuelLayer(ftype=ftype, fload=fload, fmoist=moisture)
    world.set_uniform_wind(wind_speed, wind_dir_rad)

    # town on a flat, accessible spot
    tx, ty = int(nx * 0.74), int(np.clip(yroad[int(nx * 0.74)], 4, ny - 5))

    if with_roads:
        # a single curvy road following the valley, plus a short spur to the town
        rw = 1
        prev = None
        for x in range(nx):
            y = int(np.clip(yroad[x], 0, ny - 1))
            if prev is not None:
                world.add_road_segment(prev[0], prev[1], x, y, width=rw)
            prev = (x, y)
        world.add_road_segment(tx, ty, tx, int(np.clip(yroad[tx], 0, ny - 1)),
                               width=rw)

    if with_assets:
        world.clear_fuel(tx - 5, ty - 4, tx + 5, ty + 4)
        world.add_asset(Asset("Town", "building", tx, ty, radius=4, value=1.0))
        world.add_asset(Asset("Hospital", "critical", tx - 3, ty - 2,
                              radius=1, value=1.0))
        world.add_asset(Asset("Power substation", "critical", tx + 3, ty + 2,
                              radius=1, value=0.9))
        world.add_asset(Asset("Residents", "population", tx, ty, radius=4,
                              value=1.0, population=8000))
        world.add_asset(Asset("Evacuation route", "evac_route", nx - 1, ty,
                              radius=0))
    return world
