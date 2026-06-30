"""Rendering helpers shared by the dashboard and example scripts.

Two output styles are provided:

* float RGB arrays (shape (ny, nx, 3), values in [0, 1]) for matplotlib and tests
* polished PIL images (with hillshade relief, asset markers, wind arrow and a
  scale bar) used as the live background of the interactive map editor

The PIL renderer is what gives the map its professional look and what the
drawable canvas draws on top of.
"""

from __future__ import annotations

import numpy as np

from .config import FUEL_MODELS

# base colours per fuel id (unburned landscape)
_FUEL_COLORS = {
    0: (0.40, 0.58, 0.75),   # non fuel / water: blue
    1: (0.80, 0.82, 0.45),   # grass: golden green
    2: (0.56, 0.68, 0.36),   # shrub: olive
    3: (0.20, 0.46, 0.26),   # pine litter: forest green
    4: (0.12, 0.34, 0.18),   # hardwood: dark green
}

_ASSET_STYLE = {
    "building":   {"color": (245, 245, 245), "shape": "square"},
    "critical":   {"color": (220, 40, 40),  "shape": "cross"},
    "population": {"color": (60, 130, 240),  "shape": "circle"},
    "evac_route": {"color": (40, 200, 120),  "shape": "diamond"},
}


# --------------------------------------------------------------- float arrays
def landscape_rgb(world) -> np.ndarray:
    ftype = world.fuel.ftype
    ny, nx = ftype.shape
    img = np.zeros((ny, nx, 3), dtype=float)
    for fid, color in _FUEL_COLORS.items():
        mask = ftype == fid
        for c in range(3):
            img[..., c][mask] = color[c]
    # modulate grass and shrub brightness by fuel load for a richer look
    load = np.clip(world.fuel.fload, 0.0, 1.0)
    veg = ftype > 0
    shade = (0.7 + 0.3 * load)[..., None]
    img = np.where(veg[..., None], img * shade, img)
    return np.clip(img, 0, 1)


def hillshade(elev: np.ndarray, cell_size_m: float = 30.0,
              azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    """Standard hillshade in [0, 1] from an elevation grid."""
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    gy, gx = np.gradient(elev, cell_size_m)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, -gx)
    sh = (np.sin(alt) * np.sin(slope)
          + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(sh, 0.0, 1.0)


def fire_state_rgb(sim, show_intensity: bool = True) -> np.ndarray:
    """Compose the landscape with burned scar and active fire overlays."""
    img = landscape_rgb(sim.world)
    burned = sim.ever_burned
    active = sim.state.burning > 0.5

    img[burned] = np.array([0.16, 0.13, 0.12])

    if active.any():
        inten = np.clip(sim.state.intensity, 0.0, 1.0)
        if show_intensity:
            r = 0.98 * np.ones_like(inten)
            g = 0.85 * (1.0 - 0.8 * inten)
            b = 0.10 * (1.0 - inten)
        else:
            r = np.full_like(inten, 0.95)
            g = np.full_like(inten, 0.40)
            b = np.full_like(inten, 0.05)
        img[..., 0][active] = r[active]
        img[..., 1][active] = g[active]
        img[..., 2][active] = b[active]
    return img


def value_overlay_rgb(world, alpha: float = 0.55) -> np.ndarray:
    img = landscape_rgb(world)
    prio = world.priority_field()
    mask = prio > 0.02
    overlay = np.zeros_like(img)
    overlay[..., 0] = prio
    overlay[..., 2] = prio
    for c in range(3):
        img[..., c][mask] = (1 - alpha) * img[..., c][mask] + alpha * overlay[..., c][mask]
    return img


# ------------------------------------------------------------------ PIL image
def _base_rgb(world, sim=None, show_fire=True, show_value=False,
              show_hillshade=True) -> np.ndarray:
    if show_value:
        img = value_overlay_rgb(world)
        if sim is not None and show_fire:
            img[sim.ever_burned] = [0.16, 0.13, 0.12]
            img[sim.state.burning > 0.5] = [0.98, 0.35, 0.06]
    elif sim is not None and show_fire:
        img = fire_state_rgb(sim)
    else:
        img = landscape_rgb(world)

    if show_hillshade and float(np.ptp(world.topo.elev)) > 1.0:
        hs = hillshade(world.topo.elev, world.config.cell_size_m)
        img = img * (0.55 + 0.55 * hs[..., None])
    return np.clip(img, 0, 1)


def render_pil(world, sim=None, scale: int = 8, show_fire: bool = True,
               show_assets: bool = True, show_value: bool = False,
               show_hillshade: bool = True, show_wind: bool = True,
               show_ignitions: bool = True, show_grid: bool = False,
               show_labels: bool = False):
    """Render the map to a polished PIL image of size (nx*scale, ny*scale)."""
    from PIL import Image, ImageDraw

    rgb = _base_rgb(world, sim, show_fire, show_value, show_hillshade)
    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    ny, nx = world.shape
    img = img.resize((nx * scale, ny * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img, "RGBA")

    # optional cell grid for precise placement
    if show_grid and scale >= 6:
        step = scale * (5 if nx > 60 else 2)
        for gx in range(0, nx * scale + 1, step):
            draw.line([gx, 0, gx, ny * scale], fill=(255, 255, 255, 40), width=1)
        for gy in range(0, ny * scale + 1, step):
            draw.line([0, gy, nx * scale, gy], fill=(255, 255, 255, 40), width=1)

    # scheduled ignition markers
    if show_ignitions:
        for ev in world.ignitions:
            cx, cy = ev.x * scale + scale // 2, ev.y * scale + scale // 2
            r = max(4, scale)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         outline=(255, 90, 0, 255), width=2)
            draw.line([cx - r, cy, cx + r, cy], fill=(255, 90, 0, 255), width=2)
            draw.line([cx, cy - r, cx, cy + r], fill=(255, 90, 0, 255), width=2)

    # asset markers
    if show_assets:
        for a in world.assets:
            style = _ASSET_STYLE.get(a.kind, {"color": (255, 255, 0), "shape": "circle"})
            col = style["color"] + (255,)
            cx, cy = a.x * scale + scale // 2, a.y * scale + scale // 2
            r = max(5, scale)
            shape = style["shape"]
            if shape == "square":
                draw.rectangle([cx - r, cy - r, cx + r, cy + r],
                               outline=(0, 0, 0, 255), fill=col, width=1)
            elif shape == "cross":
                draw.line([cx - r, cy, cx + r, cy], fill=col, width=3)
                draw.line([cx, cy - r, cx, cy + r], fill=col, width=3)
            elif shape == "diamond":
                draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
                             outline=(0, 0, 0, 255), fill=col)
            else:
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             outline=(0, 0, 0, 255), fill=col, width=1)
            if show_labels and getattr(a, "name", ""):
                draw.text((cx + r + 2, cy - 6), str(a.name),
                          fill=(255, 255, 255, 255))

    # wind arrow (top right)
    if show_wind:
        ws = float(world.meteo.wws.mean())
        wd = float(world.meteo.wwd.mean())
        cx, cy = nx * scale - 55, 45
        length = 18 + min(40, ws * 2.5)
        dx, dy = np.cos(wd) * length, -np.sin(wd) * length
        draw.ellipse([cx - 42, cy - 42, cx + 42, cy + 42], fill=(0, 0, 0, 90))
        draw.line([cx, cy, cx + dx, cy + dy], fill=(255, 255, 255, 255), width=3)
        # arrow head
        ang = np.arctan2(dy, dx)
        for da in (2.6, -2.6):
            hx = cx + dx - 9 * np.cos(ang + da)
            hy = cy + dy - 9 * np.sin(ang + da)
            draw.line([cx + dx, cy + dy, hx, hy], fill=(255, 255, 255, 255), width=3)
        draw.text((cx - 18, cy + 30), f"{ws:.0f} m/s", fill=(255, 255, 255, 255))

    # north arrow (top left)
    nx0, ny0 = 26, 18
    draw.ellipse([nx0 - 16, ny0 - 4, nx0 + 16, ny0 + 52], fill=(0, 0, 0, 90))
    draw.line([nx0, ny0 + 46, nx0, ny0 + 6], fill=(255, 255, 255, 255), width=3)
    draw.polygon([(nx0, ny0), (nx0 - 6, ny0 + 12), (nx0 + 6, ny0 + 12)],
                 fill=(255, 255, 255, 255))
    draw.text((nx0 - 4, ny0 + 50), "N", fill=(255, 255, 255, 255))

    # scale bar (bottom left)
    bar_cells = max(1, nx // 6)
    bx0, by0 = 12, ny * scale - 22
    draw.rectangle([bx0, by0, bx0 + bar_cells * scale, by0 + 6],
                   fill=(255, 255, 255, 230), outline=(0, 0, 0, 255))
    meters = bar_cells * world.config.cell_size_m
    draw.text((bx0, by0 - 14), f"{meters:.0f} m", fill=(255, 255, 255, 255))
    return img


def legend_items():
    items = [(f"{m.name}", _FUEL_COLORS[i]) for i, m in FUEL_MODELS.items()]
    items += [("burned", (0.16, 0.13, 0.12)), ("active fire", (0.98, 0.35, 0.06))]
    return items
