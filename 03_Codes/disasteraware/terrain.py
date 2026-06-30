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
from .world import World
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


def generate_landscape(config: SimConfig | None = None,
                       seed: int = 42,
                       relief_m: float = 450.0,
                       forest_density: float = 0.45,
                       base_moisture: float = 0.08,
                       water_level: float = 0.06,
                       wind_speed: float = 8.0,
                       wind_dir_rad: float = 0.0) -> World:
    """Generate a realistic World.

    relief_m        peak to valley elevation difference in meters
    forest_density  fraction of land covered by forest clusters (0 to 1)
    water_level     fraction of lowest cells turned into water (non fuel)
    """
    cfg = config or SimConfig()
    ny, nx = cfg.ny, cfg.nx
    rng = np.random.default_rng(seed)

    # elevation and derived terrain
    elev = fractal_noise(ny, nx, rng, octaves=6, persistence=0.58) * relief_m
    gy, gx = np.gradient(elev, cfg.cell_size_m)
    slope = np.clip(np.arctan(np.hypot(gx, gy)), 0.0, 1.3)
    aspect = np.arctan2(-gy, -gx)
    elev_norm = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-9)

    # fuel assignment: start from grass, layer shrub, forest clusters and water
    grass = FUEL_NAME_TO_ID["grass"]
    shrub = FUEL_NAME_TO_ID["shrub"]
    pine = FUEL_NAME_TO_ID["pine_litter"]
    hard = FUEL_NAME_TO_ID["hardwood"]

    ftype = np.full((ny, nx), grass, dtype=int)
    veg = fractal_noise(ny, nx, rng, octaves=5, persistence=0.6)

    # shrub on transition zones and gentle mid slopes
    ftype[(veg > 0.45) & (elev_norm > 0.25)] = shrub
    # forest clusters where the vegetation field is dense
    forest_threshold = np.quantile(veg, 1.0 - np.clip(forest_density, 0.0, 0.95))
    forest = veg >= forest_threshold
    # hardwood prefers lower, wetter slopes; pine prefers higher, drier ridges
    ftype[forest & (elev_norm <= 0.55)] = hard
    ftype[forest & (elev_norm > 0.55)] = pine

    # water bodies in the lowest basins (non fuel)
    if water_level > 0:
        thr = np.quantile(elev_norm, np.clip(water_level, 0.0, 0.3))
        ftype[elev_norm <= thr] = 0

    # fuel load and moisture vary spatially (drier on ridges, wetter near water)
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
    return world
