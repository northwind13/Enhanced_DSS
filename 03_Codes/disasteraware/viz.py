"""Rendering helpers shared by the dashboard and example scripts.

These functions turn simulator fields into RGB images (numpy arrays of shape
(ny, nx, 3) with values in [0, 1]) so they can be displayed by any backend.
"""

from __future__ import annotations

import numpy as np

from .config import FUEL_MODELS

# base colours per fuel id (unburned landscape)
_FUEL_COLORS = {
    0: (0.62, 0.71, 0.78),   # non fuel: pale blue grey
    1: (0.79, 0.84, 0.52),   # grass: light yellow green
    2: (0.55, 0.70, 0.40),   # shrub: olive green
    3: (0.25, 0.50, 0.30),   # pine litter: forest green
    4: (0.15, 0.38, 0.22),   # hardwood: dark green
}


def landscape_rgb(world) -> np.ndarray:
    ftype = world.fuel.ftype
    ny, nx = ftype.shape
    img = np.zeros((ny, nx, 3), dtype=float)
    for fid, color in _FUEL_COLORS.items():
        mask = ftype == fid
        for c in range(3):
            img[..., c][mask] = color[c]
    return img


def fire_state_rgb(sim, show_intensity: bool = True) -> np.ndarray:
    """Compose the landscape with burned scar and active fire overlays."""
    img = landscape_rgb(sim.world)
    burned = sim.ever_burned
    active = sim.state.burning > 0.5

    # burned scar: dark grey to black
    scar = np.array([0.18, 0.15, 0.13])
    img[burned] = scar

    # active fire: yellow (low intensity) to deep red (high intensity)
    if active.any():
        inten = np.clip(sim.state.intensity, 0.0, 1.0)
        if show_intensity:
            r = 0.95 * np.ones_like(inten)
            g = 0.85 * (1.0 - inten)
            b = 0.10 * (1.0 - inten)
        else:
            r = np.full_like(inten, 0.90)
            g = np.full_like(inten, 0.35)
            b = np.full_like(inten, 0.05)
        img[..., 0][active] = r[active]
        img[..., 1][active] = g[active]
        img[..., 2][active] = b[active]
    return img


def value_overlay_rgb(world, alpha: float = 0.55) -> np.ndarray:
    """Landscape with the protection priority field overlaid in magenta."""
    img = landscape_rgb(world)
    prio = world.priority_field()
    overlay = np.zeros_like(img)
    overlay[..., 0] = prio          # red
    overlay[..., 2] = prio          # blue -> magenta tint
    mask = prio > 0.02
    for c in range(3):
        img[..., c][mask] = (1 - alpha) * img[..., c][mask] + alpha * overlay[..., c][mask]
    return img


def legend_items():
    """Return (label, rgb) pairs describing the fire state colour scheme."""
    items = [(f"{m.name}", _FUEL_COLORS[i]) for i, m in FUEL_MODELS.items()]
    items += [("burned", (0.18, 0.15, 0.13)),
              ("active fire", (0.95, 0.30, 0.08))]
    return items
