"""Optional GIS import: build a World from raster data sources.

Real elevation (DEM) and fuel rasters can be imported when rasterio is
installed. Slope and aspect are derived from the DEM. Fuel class rasters are
remapped to the internal fuel ids. If rasterio is not available, the functions
raise a clear error so the caller can fall back to the synthetic editor.

The import resamples every raster onto the simulation grid defined by the
provided SimConfig, so that real and synthetic scenarios share one engine.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import SimConfig
from .world import World
from .layers import MeteoLayer, TopoLayer, FuelLayer, ValueLayer, ResourceLayer


def _require_rasterio():
    try:
        import rasterio  # noqa: F401
        from rasterio.enums import Resampling  # noqa: F401
        return rasterio, Resampling
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "GIS import needs the optional dependency 'rasterio'. "
            "Install it with: pip install rasterio") from exc


def _read_resampled(path: str, ny: int, nx: int, nearest: bool = False) -> np.ndarray:
    rasterio, Resampling = _require_rasterio()
    method = Resampling.nearest if nearest else Resampling.bilinear
    with rasterio.open(path) as src:
        data = src.read(1, out_shape=(ny, nx), resampling=method)
    return np.asarray(data, dtype=float)


def slope_aspect_from_dem(dem: np.ndarray, cell_size_m: float):
    """Return (slope_rad, aspect_rad) derived from an elevation grid."""
    gy, gx = np.gradient(dem, cell_size_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, -gx)
    return np.clip(slope, 0.0, 1.4), aspect


def world_from_rasters(config: SimConfig,
                       dem_path: Optional[str] = None,
                       fuel_path: Optional[str] = None,
                       fuel_value_map: Optional[Dict[int, int]] = None,
                       moisture: float = 0.08,
                       default_load: float = 1.0) -> World:
    """Construct a World from a DEM raster and an optional fuel class raster."""
    ny, nx = config.ny, config.nx
    w = World.blank(config)

    if dem_path is not None:
        dem = _read_resampled(dem_path, ny, nx, nearest=False)
        slope, aspect = slope_aspect_from_dem(dem, config.cell_size_m)
        w.topo = TopoLayer(elev=dem, slope=slope, aspect=aspect,
                           access=np.ones((ny, nx)))

    if fuel_path is not None:
        raw = _read_resampled(fuel_path, ny, nx, nearest=True).astype(int)
        ftype = np.zeros((ny, nx), dtype=int)
        mapping = fuel_value_map or {}
        if mapping:
            for src_val, internal_id in mapping.items():
                ftype[raw == src_val] = internal_id
        else:
            # assume the raster already uses internal ids 0..4
            ftype = np.clip(raw, 0, 4)
        load = np.where(ftype > 0, default_load, 0.0)
        w.fuel = FuelLayer(ftype=ftype, fload=load,
                           fmoist=np.full((ny, nx), moisture))
    return w
