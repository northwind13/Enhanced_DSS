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
    0: (0.66, 0.63, 0.56),   # bare ground / non fuel: grey tan
    1: (0.80, 0.82, 0.45),   # grass: golden green
    2: (0.56, 0.68, 0.36),   # shrub: olive
    3: (0.20, 0.46, 0.26),   # pine litter: forest green
    4: (0.12, 0.34, 0.18),   # hardwood: dark green
    5: (0.28, 0.52, 0.75),   # water: blue
    6: (0.55, 0.47, 0.44),   # urban / built-up: brownish grey
}

_ASSET_STYLE = {
    "building":   {"color": (150, 60, 40),  "shape": "square"},
    "critical":   {"color": (210, 30, 30),  "shape": "cross"},
    "population": {"color": (60, 130, 240),  "shape": "circle"},
    "evac_route": {"color": (40, 170, 90),  "shape": "diamond"},
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


_HS_CACHE = {}


def hillshade(elev: np.ndarray, cell_size_m: float = 30.0,
              azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    """Standard hillshade in [0, 1] from an elevation grid. Cached by the
    elevation array identity so the animation loop does not recompute it."""
    key = (id(elev), elev.shape, round(cell_size_m, 3))
    hit = _HS_CACHE.get(key)
    if hit is not None:
        return hit
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    gy, gx = np.gradient(elev, cell_size_m)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, -gx)
    sh = (np.sin(alt) * np.sin(slope)
          + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    out = np.clip(sh, 0.0, 1.0)
    if len(_HS_CACHE) > 8:
        _HS_CACHE.clear()
    _HS_CACHE[key] = out
    return out


_BURN_SCAR = np.array([0.14, 0.11, 0.10])


def apply_fire(img: np.ndarray, sim) -> np.ndarray:
    """Overlay burned scar and RED to ORANGE active fire on an RGB image.

    Fire stays red dominant (deep red -> orange) and never turns yellow, so the
    colour is the same whether or not other overlays are active."""
    burned = sim.ever_burned
    active = sim.state.burning > 0.5
    img[burned] = _BURN_SCAR
    if active.any():
        inten = np.clip(sim.state.intensity, 0.0, 1.0)
        img[..., 0][active] = 1.0                       # red channel full
        img[..., 1][active] = (0.10 + 0.32 * inten)[active]   # slight orange when hot
        img[..., 2][active] = 0.02
    return img


def fire_state_rgb(sim, show_intensity: bool = True) -> np.ndarray:
    """Compose the landscape with burned scar and active fire overlays."""
    return apply_fire(landscape_rgb(sim.world), sim)


def _dilate_mask(mask, r: int = 1):
    out = np.array(mask, dtype=bool)
    for _ in range(int(r)):
        d = out.copy()
        d[:-1, :] |= out[1:, :]; d[1:, :] |= out[:-1, :]
        d[:, :-1] |= out[:, 1:]; d[:, 1:] |= out[:, :-1]
        out = d
    return out


def value_overlay_rgb(world) -> np.ndarray:
    """Landscape with the protection priority shown clearly on and around the
    cells that hold value (buildings, critical facilities, population). Core
    cells are strongly tinted (cyan = low, magenta = high) and a soft halo makes
    even small assets easy to spot, without turning the whole map into a heat
    map."""
    img = landscape_rgb(world)
    v = world.value
    core = (v.vbld > 0.02) | (v.vcrit > 0.02) | (v.vpop > 0.0)
    if not core.any():
        return img
    halo = _dilate_mask(core, 2) & ~core
    prio = np.clip(world.priority_field(), 0.0, 1.0)
    prio = 0.25 + 0.75 * prio          # keep a visible tint even at low priority
    try:
        from matplotlib import colormaps
        ramp = np.asarray(colormaps["RdPu"](prio))[..., :3]   # pink -> purple
        # (distinct from water blue, fire orange and vegetation greens)
    except Exception:
        ramp = np.stack([0.99 - 0.55 * prio, 0.75 - 0.65 * prio,
                         0.85 - 0.30 * prio], axis=-1)
    for c in range(3):
        img[..., c][core] = 0.10 * img[..., c][core] + 0.90 * ramp[..., c][core]
        img[..., c][halo] = 0.55 * img[..., c][halo] + 0.45 * ramp[..., c][halo]
    return img


# ------------------------------------------------------------------ PIL image
def _base_rgb(world, sim=None, show_fire=True, show_value=False,
              show_hillshade=True) -> np.ndarray:
    if show_value:
        img = value_overlay_rgb(world)
        if sim is not None and show_fire:
            img = apply_fire(img, sim)
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
               show_labels: bool = False, show_roads: bool = True,
               show_perimeter: bool = False, show_spread_arrows: bool = False,
               sim_for_behavior=None, night_factor: float = 1.0,
               clock_text=None, region_box=None, region_label=None,
               regions=None, sensors=None):
    """Render the map to a polished PIL image of size (nx*scale, ny*scale)."""
    from PIL import Image, ImageDraw

    rgb = _base_rgb(world, sim, show_fire, show_value, show_hillshade)
    roads = getattr(world, 'roads', None)
    if show_roads and roads is not None:
        protect = np.zeros(rgb.shape[:2], dtype=bool)
        if sim is not None:
            protect = (sim.ever_burned | (sim.state.burning > 0.5))
        rmask = np.asarray(roads, dtype=bool) & ~protect
        rgb[rmask] = [0.82, 0.78, 0.66]
    # day/night lighting: dim the landscape at night but keep the fire
    # glowing so a night run reads like a night fire
    nf = float(np.clip(night_factor, 0.3, 1.0))
    if nf < 0.999:
        if sim is not None and show_fire:
            hot = (sim.state.burning > 0.5) | sim.ever_burned
            dim = np.where(hot[..., None], 1.0 - (1.0 - nf) * 0.3, nf)
        else:
            dim = nf
        rgb = rgb * dim
        rgb[..., 2] = np.clip(rgb[..., 2] * (1.0 + (1.0 - nf) * 0.25), 0, 1)
    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    ny, nx = world.shape
    img = img.resize((nx * scale, ny * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img, "RGBA")

    # optional cell grid for precise placement (dark, always visible)
    if show_grid:
        gstep = 10 if nx > 80 else 5
        for gc in range(0, nx + 1, gstep):
            gx = gc * scale
            draw.line([gx, 0, gx, ny * scale], fill=(15, 15, 15, 120), width=1)
        for gc in range(0, ny + 1, gstep):
            gy = gc * scale
            draw.line([0, gy, nx * scale, gy], fill=(15, 15, 15, 120), width=1)

    # scheduled ignition markers
    if show_ignitions:
        for ev in world.ignitions:
            cx, cy = ev.x * scale + scale // 2, ev.y * scale + scale // 2
            r = max(4, scale)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         outline=(162, 0, 222, 255), width=2)
            draw.line([cx - r, cy, cx + r, cy], fill=(162, 0, 222, 255), width=2)
            draw.line([cx, cy - r, cx, cy + r], fill=(162, 0, 222, 255), width=2)

    # asset markers: recognizable icons
    if show_assets:
        for a in world.assets:
            style = _ASSET_STYLE.get(a.kind, {"color": (255, 255, 0)})
            base = style["color"]
            cx, cy = a.x * scale + scale // 2, a.y * scale + scale // 2
            if getattr(a, "radius", 0) and a.radius > 0:
                fr = a.radius * scale
                draw.ellipse([cx - fr, cy - fr, cx + fr, cy + fr],
                             fill=base + (55,), outline=base + (150,), width=1)
            r = max(7, int(scale * 1.1))
            black = (0, 0, 0, 255)
            if a.kind == "building":
                # house: body + roof
                draw.rectangle([cx - r, cy - r // 3, cx + r, cy + r], fill=(238, 232, 220, 255), outline=black, width=1)
                draw.polygon([(cx - r - 1, cy - r // 3), (cx, cy - r - 2), (cx + r + 1, cy - r // 3)], fill=(150, 60, 40, 255), outline=black)
            elif a.kind == "critical":
                # generic critical infrastructure marker (hospital, power,
                # water, fuel, ...): red-bordered white square with a bold
                # red exclamation mark
                draw.rectangle([cx - r, cy - r, cx + r, cy + r],
                               fill=(250, 250, 250, 255),
                               outline=(180, 20, 20, 255), width=2)
                bw = max(2, (2 * r) // 5)
                draw.rectangle([cx - bw // 2, cy - r + 3,
                                cx + bw // 2, cy + r - bw - 4],
                               fill=(200, 25, 25, 255))
                draw.ellipse([cx - bw // 2, cy + r - 2 - bw,
                              cx + bw // 2, cy + r - 2],
                             fill=(200, 25, 25, 255))
            elif a.kind == "population":
                # people: three dots
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 130, 240, 210), outline=black, width=1)
                for ox in (-r // 2, 0, r // 2):
                    draw.ellipse([cx + ox - 2, cy - 3, cx + ox + 2, cy + 1], fill=(255, 255, 255, 255))
            elif a.kind == "evac_route":
                # exit: green square with arrow
                draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(40, 170, 90, 255), outline=black, width=1)
                draw.polygon([(cx - r // 2, cy - r // 2), (cx + r // 2, cy), (cx - r // 2, cy + r // 2)], fill=(255, 255, 255, 255))
            else:
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=base + (255,), outline=black, width=1)
            if show_labels and getattr(a, "name", ""):
                tx0 = cx + r + 3
                draw.rectangle([tx0 - 1, cy - 7, tx0 + 6 * len(str(a.name)) + 1, cy + 6], fill=(0, 0, 0, 120))
                draw.text((tx0, cy - 6), str(a.name), fill=(255, 255, 255, 255))

    # all DSS agent regions: thin borders + small labels. Entries may carry
    # a 6th element (attended flag): the coordinator's attended regions are
    # drawn hot and filled, ignored ones stay dim.
    if regions:
        for entry in regions:
            rx0, ry0, rx1, ry1, lab = entry[:5]
            attended = bool(entry[5]) if len(entry) > 5 else None
            px0, py0 = int(rx0) * scale, int(ry0) * scale
            px1, py1 = int(rx1) * scale - 1, int(ry1) * scale - 1
            if attended:
                draw.rectangle([px0, py0, px1, py1],
                               fill=(255, 120, 40, 28),
                               outline=(255, 120, 40, 230), width=3)
                txt = f"{lab} \u25cf"
                col = (255, 190, 130, 255)
            elif attended is False:
                draw.rectangle([px0, py0, px1, py1],
                               outline=(180, 180, 180, 90), width=1)
                txt, col = str(lab), (200, 200, 200, 140)
            else:
                draw.rectangle([px0, py0, px1, py1],
                               outline=(255, 210, 40, 120), width=2)
                txt, col = str(lab), (255, 230, 120, 200)
            draw.text((px0 + 6, py1 - 16), txt, fill=col)

    # selected DSS agent region highlight (translucent fill + border + label)
    if region_box is not None:
        rx0, ry0, rx1, ry1 = [int(v) for v in region_box]
        px0, py0 = rx0 * scale, ry0 * scale
        px1, py1 = rx1 * scale - 1, ry1 * scale - 1
        draw.rectangle([px0, py0, px1, py1], fill=(255, 220, 60, 45),
                       outline=(255, 210, 40, 230), width=3)
        if region_label:
            draw.rectangle([px0 + 4, py0 + 4,
                            px0 + 10 + 7 * len(str(region_label)), py0 + 22],
                           fill=(0, 0, 0, 160))
            draw.text((px0 + 8, py0 + 8), str(region_label),
                      fill=(255, 230, 120, 255))

    # DSS sensors: coverage ring + type marker + label
    if sensors:
        _sat_i = 0
        for sx_, sy_, r_c, kind, lab in sensors:
            if r_c is None:      # satellite: whole-map footprint
                bx = nx * scale - 60
                by = 70 + 26 * _sat_i
                _sat_i += 1
                draw.ellipse([bx - 7, by - 5, bx + 7, by + 5],
                             outline=(120, 200, 255, 230), width=2)
                draw.line([bx - 10, by + 7, bx + 10, by - 7],
                          fill=(120, 200, 255, 230), width=2)
                draw.text((bx - 24, by + 8), str(lab),
                          fill=(150, 210, 255, 220))
                continue
            cx, cy = sx_ * scale + scale // 2, sy_ * scale + scale // 2
            rr = int(r_c * scale)
            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                         outline=(120, 200, 255, 150), width=2)
            if kind == "tower":
                draw.polygon([(cx, cy - 7), (cx - 6, cy + 6),
                              (cx + 6, cy + 6)],
                             fill=(120, 200, 255, 240),
                             outline=(0, 0, 0, 200))
            elif kind == "uav":
                draw.polygon([(cx - 7, cy), (cx, cy - 4), (cx + 7, cy),
                              (cx, cy + 4)], fill=(120, 200, 255, 240),
                             outline=(0, 0, 0, 200))
            else:                # station
                draw.rectangle([cx - 5, cy - 5, cx + 5, cy + 5],
                               fill=(120, 200, 255, 240),
                               outline=(0, 0, 0, 200))
            draw.text((cx + 8, cy - 6), str(lab),
                      fill=(150, 210, 255, 230))

    # clock badge (top left)
    if clock_text:
        pad = 6
        tw = 7 * len(str(clock_text)) + 2 * pad
        draw.rectangle([8, 8, 8 + tw, 30], fill=(0, 0, 0, 150))
        draw.text((8 + pad, 13), str(clock_text), fill=(255, 255, 255, 255))

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

    _sim = sim_for_behavior if sim_for_behavior is not None else sim
    # fire perimeter outline (FARSITE style)
    if show_perimeter and _sim is not None:
        from . import behavior
        per = behavior.perimeter_mask(_sim)
        ys, xs = np.where(per)
        for py, px in zip(ys, xs):
            draw.rectangle([px * scale, py * scale,
                            px * scale + scale, py * scale + scale],
                           outline=(255, 60, 40, 255), width=max(1, scale // 4))
    # spread direction arrows at the active front (wind aligned)
    if show_spread_arrows and _sim is not None:
        act = _sim.state.burning > 0.5
        ys, xs = np.where(act)
        if xs.size:
            stride = max(1, xs.size // 60)
            for py, px in zip(ys[::stride], xs[::stride]):
                wd = float(world.meteo.wwd[py, px])
                cx, cy = px * scale + scale // 2, py * scale + scale // 2
                L = scale * 2.0
                ex, ey = cx + L * np.cos(wd), cy - L * np.sin(wd)
                draw.line([cx, cy, ex, ey], fill=(30, 30, 30, 230), width=2)
                ang = np.arctan2(ey - cy, ex - cx)
                for da in (2.6, -2.6):
                    draw.line([ex, ey, ex - 5 * np.cos(ang + da),
                               ey - 5 * np.sin(ang + da)],
                              fill=(30, 30, 30, 230), width=2)
    return img


def legend_items():
    items = [(f"{m.name}", _FUEL_COLORS[i]) for i, m in FUEL_MODELS.items()]
    items += [("burned", (0.16, 0.13, 0.12)), ("active fire", (0.98, 0.35, 0.06))]
    return items


def _hex(rgb01):
    r, g, b = [int(round(255 * c)) for c in rgb01]
    return f"#{r:02x}{g:02x}{b:02x}"


def legend_entries():
    """Full legend as (group, label, hex_color) for the dashboard side panel."""
    out = []
    for i, m in FUEL_MODELS.items():
        label = {"non_fuel": "bare ground / rock", "water": "water",
                 "grass": "grass / crops", "shrub": "shrub / maquis",
                 "pine_litter": "conifer forest (pine litter)",
                 "hardwood": "broadleaf forest (hardwood)",
                 "urban": "urban / built-up"}.get(m.name, m.name.replace("_", " "))
        out.append(("Land cover", label, _hex(_FUEL_COLORS[i])))
    out.append(("Fire", "burned scar", _hex((0.16, 0.13, 0.12))))
    out.append(("Fire", "active fire", _hex((0.98, 0.35, 0.06))))
    out.append(("Infrastructure", "road / access", _hex((0.82, 0.78, 0.66))))
    out.append(("Markers", "ignition point", "#a200de"))
    asset_labels = {"building": "building", "critical": "critical facility",
                    "population": "population", "evac_route": "evacuation route"}
    for kind, lab in asset_labels.items():
        c = _ASSET_STYLE[kind]["color"]
        out.append(("Assets", lab, f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"))
    return out


def render_compass(direction_rad: float, speed: float = 0.0, size: int = 150):
    """Render a compass rose with the wind direction arrow as a PIL image.

    The arrow points in the direction the wind blows toward, using the same
    math convention as the simulator (0 rad = +x / East, increasing counter
    clockwise)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = size // 2
    r = size // 2 - 12
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(245, 247, 250, 255),
              outline=(60, 60, 60, 255), width=2)
    for lab, ang in [("N", 90), ("E", 0), ("S", 270), ("W", 180)]:
        ax = cx + (r - 10) * np.cos(np.radians(ang))
        ay = cy - (r - 10) * np.sin(np.radians(ang))
        d.text((ax - 4, ay - 6), lab, fill=(90, 90, 90, 255))
    dx = np.cos(direction_rad) * (r - 18)
    dy = -np.sin(direction_rad) * (r - 18)
    d.line([cx, cy, cx + dx, cy + dy], fill=(200, 60, 30, 255), width=4)
    ang = np.arctan2(dy, dx)
    for da in (2.6, -2.6):
        hx = cx + dx - 12 * np.cos(ang + da)
        hy = cy + dy - 12 * np.sin(ang + da)
        d.line([cx + dx, cy + dy, hx, hy], fill=(200, 60, 30, 255), width=4)
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(60, 60, 60, 255))
    return img


def terrain_pil(world, scale: int = 6):
    """Clear standalone 2D terrain map: elevation colour ramp shaded by relief."""
    from PIL import Image
    try:
        from matplotlib import colormaps
        cmap = colormaps["terrain"]
    except Exception:                       # very old matplotlib
        import matplotlib.cm as cm
        cmap = cm.get_cmap("terrain")
    elev = np.asarray(world.topo.elev, dtype=float)
    span = float(np.ptp(elev))
    en = (elev - elev.min()) / span if span > 1e-9 else np.zeros_like(elev)
    base = np.asarray(cmap(en))[..., :3]
    if span > 1.0:
        hs = hillshade(elev, world.config.cell_size_m)
        base = base * (0.45 + 0.6 * hs[..., None])
    arr = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    ny, nx = elev.shape
    return Image.fromarray(arr, "RGB").resize((nx * scale, ny * scale), Image.BILINEAR)


# --------------------------------------------------------------- 3D surface
def _state_code_field(world, sim=None):
    """Categorical field for 3D colouring: water/bare/grass/shrub/pine/hardwood
    then burned and active fire on top."""
    ftype = np.asarray(world.fuel.ftype)
    code = np.zeros(ftype.shape, dtype=float)
    code[ftype == 0] = 1     # bare
    code[ftype == 1] = 2     # grass
    code[ftype == 2] = 3     # shrub
    code[ftype == 3] = 4     # pine
    code[ftype == 4] = 5     # hardwood
    code[ftype == 5] = 0     # water
    code[ftype == 6] = 1     # urban -> bare-like tone in 3D
    if sim is not None:
        code[sim.ever_burned] = 6
        code[sim.state.burning > 0.5] = 7
    return code


def fire_surface_figure(world, sim=None, max_cells: int = 150,
                        relief_frac: float = 0.28, pick: bool = False,
                        pick_label: str = "click to place ignition"):
    """Live 3D terrain with the same content as the 2D map: land cover, water,
    roads, buildings and flame coloured active fire draped on the relief.

    Returns a plotly Figure (drag to rotate, scroll to zoom)."""
    import plotly.graph_objects as go
    ny, nx = world.shape
    step = max(1, max(nx, ny) // max_cells)
    elev_full = np.asarray(world.topo.elev, dtype=float)
    span = float(np.ptp(elev_full))
    zfull = ((elev_full - elev_full.min()) / span if span > 1e-9
             else np.zeros_like(elev_full)) * (relief_frac * max(nx, ny))

    elev = elev_full[::step, ::step]
    code = _state_code_field(world, sim)[::step, ::step]
    # roads on the surface colour (bare-like tan) so the road shows in 3D
    roads = getattr(world, "roads", None)
    if roads is not None:
        rr = np.asarray(roads, dtype=bool)[::step, ::step]
        code = np.where(rr & (code < 6), 1, code)
    sy, sx = elev.shape
    xs = np.arange(sx) * step
    ys = np.arange(sy) * step
    z = zfull[::step, ::step]

    cats = [(0.28, 0.52, 0.75), (0.66, 0.63, 0.56), (0.80, 0.82, 0.45),
            (0.56, 0.68, 0.36), (0.20, 0.46, 0.26), (0.12, 0.34, 0.18),
            (0.16, 0.13, 0.12), (0.98, 0.35, 0.06)]
    n = len(cats)
    colorscale = []
    for i, c in enumerate(cats):
        rgb = f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
        colorscale.append([i / n, rgb]); colorscale.append([(i + 1) / n, rgb])

    data = [go.Surface(x=xs, y=ys, z=z, surfacecolor=code, colorscale=colorscale,
                       cmin=0, cmax=n, showscale=False,
                       lighting=dict(ambient=0.65, diffuse=0.75, specular=0.1),
                       hoverinfo="skip")]

    # asset markers (buildings, hospital, ...) as 3D points on the terrain
    sym = {"building": "square", "critical": "x",
           "population": "circle", "evac_route": "diamond"}
    col = {"building": "rgb(240,240,240)", "critical": "rgb(220,40,40)",
           "population": "rgb(60,130,240)", "evac_route": "rgb(40,200,120)"}
    if world.assets:
        for kind in sym:
            pts = [a for a in world.assets if a.kind == kind]
            if not pts:
                continue
            data.append(go.Scatter3d(
                x=[a.x for a in pts], y=[a.y for a in pts],
                z=[zfull[int(np.clip(a.y, 0, ny - 1)), int(np.clip(a.x, 0, nx - 1))]
                   + 0.6 for a in pts],
                mode="markers+text", text=[a.name for a in pts],
                textposition="top center", textfont=dict(size=9, color="white"),
                marker=dict(size=6, symbol=sym[kind], color=col[kind],
                            line=dict(width=1, color="black")),
                name=kind, hoverinfo="text"))

    # active fire: bright markers sitting on the surface (the surface itself is
    # already coloured orange where burning, matching the 2D map)
    if sim is not None:
        act = sim.state.burning > 0.5
        if act.any():
            ay, ax = np.where(act)
            if ax.size > 3000:
                idx = np.random.default_rng(0).choice(ax.size, 3000, replace=False)
                ax, ay = ax[idx], ay[idx]
            inten = np.clip(sim.state.intensity[ay, ax], 0, 1)
            data.append(go.Scatter3d(
                x=ax, y=ay, z=zfull[ay, ax] + 0.3, mode="markers",
                marker=dict(size=4, color=inten,
                            colorscale=[[0, "rgb(210,30,10)"], [0.5, "rgb(255,140,0)"],
                                        [1, "rgb(255,235,120)"]],
                            cmin=0, cmax=1, opacity=0.95),
                name="fire", hoverinfo="skip"))

    if pick:
        # a light grid of clickable points; a single click selects the nearest
        # one and its (x, y) are the exact grid cell for the ignition
        pstep = max(1, int(np.ceil(max(nx, ny) / 110.0)))
        pj, pi = np.mgrid[0:ny:pstep, 0:nx:pstep]
        pj = pj.ravel(); pi = pi.ravel()
        data.append(go.Scatter3d(
            x=pi, y=pj, z=zfull[pj, pi] + 0.6, mode="markers",
            marker=dict(size=5, color="rgba(255,255,255,0.55)",
                        line=dict(width=0)),
            name="place", customdata=elev_full[pj, pi],
            hovertemplate="x=%{x}, y=%{y}, elev=%{customdata:.0f} m"
                          f"<extra>{pick_label}</extra>"))

    fig = go.Figure(data=data)
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=0, b=0),
                      showlegend=False, uirevision="keep",
                      scene=dict(aspectmode="data", uirevision="keep",
                                 xaxis=dict(visible=False),
                                 # row index grows SOUTH on the 2D map;
                                 # reversing y makes the 3D orientation match
                                 # the 2D view instead of mirroring it
                                 yaxis=dict(visible=False,
                                            autorange="reversed"),
                                 zaxis=dict(visible=False)))
    # note: no explicit camera -> uirevision keeps the user's rotation and zoom
    # across steps instead of resetting every frame
    return fig


def map_figure_2d(world, sim=None, scale: int = 6, **flags):
    """2D map as a plotly image so it supports scroll zoom and pan like the 3D
    view. Draws the same content as render_pil (land cover, roads, assets,
    flame fire)."""
    import plotly.graph_objects as go
    pil = render_pil(world, sim=sim, scale=scale, show_labels=True, **flags)
    arr = np.asarray(pil)
    fig = go.Figure(go.Image(z=arr))
    ny, nx = arr.shape[0], arr.shape[1]
    fig.update_xaxes(visible=False, range=[0, nx], constrain="domain")
    fig.update_yaxes(visible=False, range=[ny, 0], scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=560,
                      dragmode="pan", uirevision="keep")
    return fig

def ignition_time_pil(sim, scale: int = 6):
    """2D time-to-burn map: cells coloured by the step they first ignited
    (red = earliest, blue = latest). Unburned cells keep the landscape colour.
    After Kose et al., '3D Wildfire Simulation System'."""
    from PIL import Image
    world = sim.world
    img = landscape_rgb(world)
    fis = sim.first_ignition_step
    burned = fis >= 0
    if burned.any():
        t = fis.astype(float)
        lo, hi = float(t[burned].min()), float(t[burned].max())
        tn = (t - lo) / (hi - lo) if hi > lo else np.zeros_like(t)
        try:
            from matplotlib import colormaps
            ramp = np.asarray(colormaps["RdYlBu"](tn))[..., :3]
        except Exception:
            ramp = np.stack([1 - tn, tn * 0.6, tn], axis=-1)
        for c in range(3):
            img[..., c][burned] = ramp[..., c][burned]
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ny, nx = arr.shape[:2]
    return Image.fromarray(arr, "RGB").resize((nx * scale, ny * scale), Image.NEAREST)


def flame_length_norm(intensity):
    """Stylized normalized flame length from the intensity proxy (higher
    intensity -> taller flame)."""
    return np.sqrt(np.clip(intensity, 0.0, 1.0))

def behavior_pil(sim, kind: str = "fireline_intensity", scale: int = 6):
    """Render a FARSITE style behaviour field over the landscape.

    kind in {fireline_intensity, flame_length, rate_of_spread, crown_fire}."""
    from PIL import Image
    from . import behavior
    world = sim.world
    img = landscape_rgb(world)
    img[sim.ever_burned] = [0.16, 0.13, 0.12]
    if kind == "crown_fire":
        mask = behavior.crown_fire_mask(sim)
        img[sim.state.burning > 0.5] = [0.98, 0.45, 0.10]
        img[mask] = [1.0, 0.15, 0.0]
    else:
        if kind == "flame_length":
            f = behavior.flame_length_field(sim); norm = np.clip(f / 8.0, 0, 1); mask = f > 0.01
        elif kind == "rate_of_spread":
            f = behavior.rate_of_spread_field(world)
            m = float(f.max()); norm = f / m if m > 1e-9 else f
            mask = (sim.state.burning > 0.5)
        else:
            f = behavior.fireline_intensity(sim)
            m = float(f.max()); norm = f / m if m > 1e-9 else f
            mask = f > 0
        try:
            from matplotlib import colormaps
            ramp = np.asarray(colormaps["inferno"](norm))[..., :3]
        except Exception:
            ramp = np.stack([norm, norm * 0.4, np.zeros_like(norm)], axis=-1)
        for c in range(3):
            img[..., c][mask] = ramp[..., c][mask]
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ny, nx = arr.shape[:2]
    return Image.fromarray(arr, "RGB").resize((nx * scale, ny * scale), Image.NEAREST)
