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
            # deep red (cool) -> orange -> yellow/white (hot core)
            r = np.ones_like(inten)
            g = 0.18 + 0.77 * inten
            b = 0.04 + 0.45 * inten * inten
        else:
            r = np.full_like(inten, 1.0)
            g = np.full_like(inten, 0.45)
            b = np.full_like(inten, 0.08)
        img[..., 0][active] = r[active]
        img[..., 1][active] = np.clip(g[active], 0, 1)
        img[..., 2][active] = np.clip(b[active], 0, 1)
    return img


def value_overlay_rgb(world, alpha: float = 0.65) -> np.ndarray:
    """Landscape with the protection priority draped as a green->yellow->red
    heat ramp (higher priority = more red)."""
    img = landscape_rgb(world)
    prio = np.clip(world.priority_field(), 0.0, 1.0)
    mask = prio > 0.02
    try:
        from matplotlib import colormaps
        ramp = np.asarray(colormaps["RdYlGn_r"](prio))[..., :3]
    except Exception:
        ramp = np.stack([prio, 1.0 - prio, np.zeros_like(prio)], axis=-1)
    for c in range(3):
        img[..., c][mask] = (1 - alpha) * img[..., c][mask] + alpha * ramp[..., c][mask]
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
               show_labels: bool = False, show_roads: bool = True):
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
                # hospital: white square with red cross
                draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(250, 250, 250, 255), outline=black, width=1)
                t = max(2, r // 3)
                draw.rectangle([cx - t, cy - r + 2, cx + t, cy + r - 2], fill=(210, 30, 30, 255))
                draw.rectangle([cx - r + 2, cy - t, cx + r - 2, cy + t], fill=(210, 30, 30, 255))
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


def _hex(rgb01):
    r, g, b = [int(round(255 * c)) for c in rgb01]
    return f"#{r:02x}{g:02x}{b:02x}"


def legend_entries():
    """Full legend as (group, label, hex_color) for the dashboard side panel."""
    out = []
    for i, m in FUEL_MODELS.items():
        label = {"non_fuel": "bare ground", "water": "water"}.get(m.name, m.name.replace("_", " "))
        out.append(("Land cover", label, _hex(_FUEL_COLORS[i])))
    out.append(("Fire", "burned scar", _hex((0.16, 0.13, 0.12))))
    out.append(("Fire", "active fire", _hex((0.98, 0.35, 0.06))))
    out.append(("Infrastructure", "road / access", _hex((0.82, 0.78, 0.66))))
    out.append(("Markers", "ignition point", "#ff5a00"))
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
    if sim is not None:
        code[sim.ever_burned] = 6
        code[sim.state.burning > 0.5] = 7
    return code


def fire_surface_figure(world, sim=None, max_cells: int = 150,
                        relief_frac: float = 0.28):
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
    sym = {"building": "square", "critical": "cross",
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

    # active fire as bright flame coloured points
    if sim is not None:
        act = sim.state.burning > 0.5
        if act.any():
            ay, ax = np.where(act)
            if ax.size > 4000:                      # subsample very large fires
                idx = np.random.default_rng(0).choice(ax.size, 4000, replace=False)
                ax, ay = ax[idx], ay[idx]
            inten = np.clip(sim.state.intensity[ay, ax], 0, 1)
            data.append(go.Scatter3d(
                x=ax, y=ay, z=zfull[ay, ax] + 0.4, mode="markers",
                marker=dict(size=3.5, color=inten, colorscale=[[0, "rgb(180,20,10)"],
                            [0.5, "rgb(255,120,0)"], [1, "rgb(255,240,120)"]],
                            cmin=0, cmax=1, opacity=0.9),
                name="fire", hoverinfo="skip"))

    fig = go.Figure(data=data)
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=0, b=0),
                      showlegend=False,
                      scene=dict(aspectmode="data",
                                 xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False),
                                 camera=dict(eye=dict(x=1.4, y=1.4, z=1.0))))
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
                      dragmode="pan")
    return fig
