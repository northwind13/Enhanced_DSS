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
    # BURN SEVERITY SPECTRUM: how much of the cell's fuel actually
    # burned tells the story of the suppression. A cell knocked down
    # early keeps most of its fuel: the terrain stays visible under a
    # light ash-grey veil. A cell that burned out completely goes to
    # the black scar. The map therefore SHOWS the success of the
    # attack, not just the perimeter it reached.
    _cons = getattr(sim, "fuel_consumed_total", None)
    _fl0 = getattr(getattr(sim, "world", None), "fuel", None)
    _fl0 = getattr(_fl0, "fload0", None)
    if _cons is not None and _fl0 is not None and burned.any():
        sev = np.clip(np.asarray(_cons)
                      / np.maximum(np.asarray(_fl0), 1e-6),
                      0.0, 1.0)
        # light singe (s=0): terrain + thin grey-white veil;
        # full burn (s=1): the black scar
        _veil = np.array([0.82, 0.82, 0.80])
        _mixv = 0.30 + 0.20 * sev[..., None]     # veil opacity
        _lightly = (img * (1.0 - _mixv) + _veil * _mixv)
        _sc = np.array(_BURN_SCAR)
        _t = np.clip((sev[..., None] - 0.25) / 0.75, 0.0, 1.0) ** 0.8
        _shaded = _lightly * (1.0 - _t) + _sc * _t
        img[burned] = _shaded[burned]
    else:
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
               regions=None, sensors=None, depots=None, alloc=None,
               actions=None):
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
        _lbl_boxes = []          # drawn label rectangles, for de-cluttering
        for a in world.assets:
            # the evacuation route is a routing hint for the E intervention,
            # not a value asset; it is no longer drawn as an asset marker
            if a.kind == "evac_route":
                continue
            style = _ASSET_STYLE.get(a.kind, {"color": (255, 255, 0)})
            base = style["color"]
            cx, cy = a.x * scale + scale // 2, a.y * scale + scale // 2
            # NOTE: no translucent extent circle is drawn for assets any
            # more. Those halos looked like coverage / resource rings in a
            # town; assets now show only their point icon (house / facility
            # square / population dots). Value still comes from the value
            # layers, not from a drawn ring.
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
            # labels only for named NON-population assets (population sits on
            # the same spot as its building, so it would double every label),
            # and only when the label box does not collide with one already
            # drawn: this keeps a dense town from turning into a wall of
            # overlapping text.
            if (show_labels and getattr(a, "name", "")
                    and a.kind != "population"):
                tx0 = cx + r + 3
                _bx = [tx0 - 1, cy - 7,
                       tx0 + 6 * len(str(a.name)) + 1, cy + 6]
                _hit = any(not (_bx[2] < ob[0] or _bx[0] > ob[2]
                                or _bx[3] < ob[1] or _bx[1] > ob[3])
                           for ob in _lbl_boxes)
                if not _hit:
                    draw.rectangle(_bx, fill=(0, 0, 0, 120))
                    draw.text((tx0, cy - 6), str(a.name),
                              fill=(255, 255, 255, 255))
                    _lbl_boxes.append(_bx)

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

    # DSS sensors: every family has its OWN color, glyph and a FILLED
    # coverage footprint. Colors come from the single source of truth
    # (dss.sensors.SENSOR_CATALOG) so the map and the legend never drift.
    try:
        from dss.sensors import SENSOR_CATALOG as _SCAT
        SENSOR_COLORS = {k: tuple(v.get("color", (120, 200, 255)))
                         for k, v in _SCAT.items()}
    except Exception:
        SENSOR_COLORS = {"satellite": (170, 120, 255),
                         "aerial": (0, 220, 255),
                         "ground_camera": (255, 255, 255),
                         "in_situ": (255, 220, 0),
                         "field_report": (255, 150, 60),
                         "public_report": (255, 105, 180)}
    if sensors:
        _sat_i = 0
        for sx_, sy_, r_c, kind, lab in sensors:
            col = SENSOR_COLORS.get(kind, (120, 200, 255))
            if r_c is None:      # satellite: whole-map capability badge
                bx = nx * scale - 60
                by = 70 + 26 * _sat_i
                _sat_i += 1
                draw.ellipse([bx - 7, by - 5, bx + 7, by + 5],
                             outline=(*col, 230), width=2)
                draw.line([bx - 10, by + 7, bx + 10, by - 7],
                          fill=(*col, 230), width=2)
                draw.text((bx - 24, by + 8), str(lab), fill=(*col, 220))
                continue
            cx, cy = sx_ * scale + scale // 2, sy_ * scale + scale // 2
            rr = int(r_c * scale)
            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                         fill=(*col, 26), outline=(*col, 170), width=2)
            if kind == "aerial":                 # drone diamond
                draw.polygon([(cx - 8, cy), (cx, cy - 5), (cx + 8, cy),
                              (cx, cy + 5)], fill=(*col, 240),
                             outline=(0, 0, 0, 200))
            elif kind == "ground_camera":        # camera: body + lens
                draw.rectangle([cx - 7, cy - 4, cx + 5, cy + 5],
                               fill=(*col, 240),
                               outline=(0, 0, 0, 200))
                draw.ellipse([cx - 3, cy - 2, cx + 3, cy + 4],
                             outline=(0, 0, 0, 220), width=2)
            elif kind == "in_situ":              # ground probe: square+stem
                draw.rectangle([cx - 5, cy - 5, cx + 5, cy + 5],
                               fill=(*col, 240),
                               outline=(0, 0, 0, 200))
                draw.line([cx, cy + 5, cx, cy + 10],
                          fill=(*col, 240), width=2)
            elif kind == "field_report":         # responder: head + body
                draw.ellipse([cx - 3, cy - 9, cx + 3, cy - 3],
                             fill=(*col, 240), outline=(0, 0, 0, 200))
                draw.line([cx, cy - 3, cx, cy + 6],
                          fill=(*col, 240), width=3)
                draw.line([cx - 5, cy + 1, cx + 5, cy + 1],
                          fill=(*col, 240), width=2)
            elif kind == "public_report":        # phone
                draw.rectangle([cx - 4, cy - 7, cx + 4, cy + 7],
                               fill=(*col, 240),
                               outline=(0, 0, 0, 200))
                draw.ellipse([cx - 1, cy + 3, cx + 1, cy + 5],
                             fill=(0, 0, 0, 220))
            else:
                draw.rectangle([cx - 5, cy - 5, cx + 5, cy + 5],
                               fill=(*col, 240),
                               outline=(0, 0, 0, 200))
            draw.text((cx + 9, cy - 6), str(lab), fill=(*col, 235))

    # DSS intervention overlay: every intervention type has its OWN
    # visual so the viewer reads the operation at a glance:
    #   containment line  = brown-black cut squares (a line being built)
    #   suppression       = blue water dots on the engaged fire cells
    #   asset protection  = green shield rings around defended cells
    #   evacuation        = orange up-arrow + E at the region's people
    #   public warning    = yellow triangle at the region corner
    #   region badge      = compact S/D/C/P/E/W order intensities
    if actions:
        cont = actions.get("cont")
        if cont is not None:
            # containment order = the SAME brown diagonal dozer strokes as the
            # dug fuel break it produces, so order and result read identically
            ys_, xs_ = np.where(cont)
            for yy_, xx_ in zip(ys_.tolist(), xs_.tolist()):
                _x0c, _y0c = xx_ * scale, yy_ * scale
                _qc = max(2, scale // 4)
                draw.line([_x0c + 1, _y0c + scale - _qc,
                           _x0c + scale - _qc, _y0c + 1],
                          fill=(110, 70, 30, 255), width=2)
                draw.line([_x0c + _qc, _y0c + scale - 1,
                           _x0c + scale - 1, _y0c + _qc],
                          fill=(110, 70, 30, 255), width=2)
        supp = actions.get("supp")
        if supp is not None:
            ys_, xs_ = np.where(supp)
            for yy_, xx_ in zip(ys_.tolist(), xs_.tolist()):
                if (yy_ + xx_) % 2:
                    continue
                cx = xx_ * scale + scale // 2
                cy = yy_ * scale + scale // 2
                r_ = max(2, scale // 3)
                draw.ellipse([cx - r_, cy - r_, cx + r_, cy + r_],
                             fill=(40, 120, 255, 230),
                             outline=(255, 255, 255, 180))
        prot = actions.get("prot")
        if prot is not None:
            ys_, xs_ = np.where(prot)
            for yy_, xx_ in zip(ys_.tolist(), xs_.tolist()):
                if (yy_ + xx_) % 2:
                    continue
                cx = xx_ * scale + scale // 2
                cy = yy_ * scale + scale // 2
                r_ = max(3, scale // 2)
                draw.ellipse([cx - r_, cy - r_, cx + r_, cy + r_],
                             outline=(40, 220, 90, 230), width=2)
        for ro in actions.get("regions", []):
            u = ro["u"]
            if max(u.values()) <= 0.05:
                continue
            x0_, y0_, x1_, y1_ = ro["box"]
            # GenAI-generated order marker: a magenta "G" badge on the
            # region where a G# (generative) rule fired this cycle, so the
            # generated orders read distinctly from the base ones.
            _macdefs = actions.get("macros") or {}
            # A generated intervention that FIRED in this region is drawn as
            # its own chip, coloured by what it is made of and tagged with
            # its initials, so the map shows WHICH one acted. The plain "G"
            # badge stays as the fallback for a generated RULE that ordered
            # only base channels, where there is no macro to name.
            _fired_m = [(_mn, float(u.get(_mn, 0.0)))
                        for _mn in _macdefs
                        if float(u.get(_mn, 0.0)) > 0.05]
            _gcx = (x0_ + x1_) * scale // 2
            _gcy = (y0_ + y1_) * scale // 2
            _mcells_all = actions.get("macro_cells") or {}
            if _fired_m:
                _fired_m.sort(key=lambda t: -t[1])
                for _i, (_mn, _mv) in enumerate(_fired_m[:3]):
                    _mc, _msh = macro_style(_mn)
                    _fill = (_mc[0], _mc[1], _mc[2], 235)
                    _ink = (0, 0, 0, 220)
                    _cells = _mcells_all.get(_mn)
                    if _cells is not None and _cells.any():
                        # the symbol sits ON THE WORKED CELLS, sized
                        # to the cell, drawn sparsely so the terrain
                        # stays readable; the 2-letter tag marks the
                        # cluster centre for identity
                        _cys, _cxs = np.where(_cells)
                        _r = max(2, scale // 2 - 1)
                        for _yy2, _xx2 in zip(_cys.tolist(),
                                              _cxs.tolist()):
                            if (_yy2 * 7 + _xx2 * 3) % 3:
                                continue
                            _px = _xx2 * scale + scale // 2
                            _py = _yy2 * scale + scale // 2
                            _draw_macro_shape(draw, _px, _py, _r,
                                              _msh, _fill, _ink)
                        _tag = macro_tag(_mn)
                        _tx = int(_cxs.mean()) * scale
                        _ty = int(_cys.mean()) * scale
                        draw.text((_tx - 3 * len(_tag), _ty - 6),
                                  _tag, fill=(20, 20, 20, 255))
                        continue
                    # no recorded cells (older log replay): small
                    # badge at the region centre as before
                    _cy0 = _gcy + _i * 24 - (len(_fired_m[:3]) - 1) * 12
                    _r = 10 + int(5 * min(1.0, _mv))
                    _draw_macro_shape(draw, _gcx, _cy0, _r, _msh,
                                      (_mc[0], _mc[1], _mc[2], 255),
                                      (0, 0, 0, 255))
                    _tag = macro_tag(_mn)
                    draw.text((_gcx - 3 * len(_tag), _cy0 - 6), _tag,
                              fill=(20, 20, 20, 255))
            elif ro.get("name") in (actions.get("genai_regions") or set()):
                draw.ellipse([_gcx - 8, _gcy - 8, _gcx + 8, _gcy + 8],
                             fill=(192, 0, 255, 210), outline=(0, 0, 0, 220),
                             width=1)
                draw.text((_gcx - 4, _gcy - 6), "G",
                          fill=(255, 255, 255, 255))
            # evacuation: arrow + E at every populated asset inside
            if u["evacuation"] > 0.3 and getattr(world, "assets", None):
                for a in world.assets:
                    if getattr(a, "kind", "") != "population":
                        continue
                    if not (x0_ <= a.x < x1_ and y0_ <= a.y < y1_):
                        continue
                    cx = a.x * scale + scale // 2
                    cy = a.y * scale + scale // 2
                    draw.polygon([(cx, cy - 12), (cx - 6, cy - 2),
                                  (cx + 6, cy - 2)],
                                 fill=(255, 140, 0, 255),
                                 outline=(0, 0, 0, 200))
                    draw.rectangle([cx - 2, cy - 2, cx + 2, cy + 6],
                                   fill=(255, 140, 0, 255))
                    draw.text((cx + 8, cy - 10), "EVAC",
                              fill=(255, 160, 20, 255))
            # public warning: yellow triangle, region's top-right corner
            if u["public_warning"] > 0.3:
                tx_ = x1_ * scale - 18
                ty_ = y0_ * scale + 6
                draw.polygon([(tx_, ty_ + 12), (tx_ + 12, ty_ + 12),
                              (tx_ + 6, ty_)],
                             fill=(255, 220, 0, 240),
                             outline=(0, 0, 0, 220))
                draw.text((tx_ + 4, ty_ + 2), "!",
                          fill=(0, 0, 0, 255))

    # DSS allocation overlay (D = resource deployment): the STAGED capacity
    # ahead of the front glows a faint cyan. Burning / burned cells are
    # skipped (they already read as fire + the blue suppression dots), and
    # only cells carrying a meaningful share of the peak are drawn, so the
    # map is no longer washed blue.
    if alloc is not None:
        al = np.asarray(alloc, dtype=float)
        mx = float(al.max())
        if mx > 1e-9:
            _busy = None
            if sim is not None:
                try:
                    _busy = (np.asarray(sim.state.burning) > 0.5) \
                        | np.asarray(sim.ever_burned)
                except Exception:
                    _busy = np.asarray(sim.state.burning) > 0.5
            ys_, xs_ = np.where(al > 0.25 * mx)
            for yy_, xx_ in zip(ys_.tolist(), xs_.tolist()):
                if _busy is not None and _busy[yy_, xx_]:
                    continue
                a_ = int(35 + 80 * min(1.0, al[yy_, xx_] / mx))
                draw.rectangle([xx_ * scale, yy_ * scale,
                                (xx_ + 1) * scale - 1,
                                (yy_ + 1) * scale - 1],
                               outline=None, fill=(0, 210, 255, a_))

    # resource units: GROUND depot = green house marker; HELIBASE (aerial)
    # = cyan circle with an "H", so the two kinds read differently on the map
    # and match their own legend entries. Both carry the red service ring.
    if depots:
        for dx_, dy_, r_c, cap_, lab in depots:
            cx, cy = dx_ * scale + scale // 2, dy_ * scale + scale // 2
            _is_heli = "heli" in str(lab).lower()
            if r_c:
                rr = int(r_c * scale)
                draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                             fill=(255, 80, 80, 28),
                             outline=(255, 100, 100, 190), width=2)
            if _is_heli:
                draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7],
                             fill=(0, 190, 235, 240), outline=(0, 0, 0, 210),
                             width=1)
                draw.line([cx - 3, cy - 4, cx - 3, cy + 4],
                          fill=(0, 0, 0, 230), width=2)
                draw.line([cx + 3, cy - 4, cx + 3, cy + 4],
                          fill=(0, 0, 0, 230), width=2)
                draw.line([cx - 3, cy, cx + 3, cy], fill=(0, 0, 0, 230),
                          width=2)
                draw.text((cx + 9, cy - 6), str(lab),
                          fill=(150, 230, 255, 230))
            else:
                draw.rectangle([cx - 5, cy - 2, cx + 5, cy + 6],
                               fill=(60, 200, 120, 240),
                               outline=(0, 0, 0, 200))
                draw.polygon([(cx - 6, cy - 2), (cx, cy - 8),
                              (cx + 6, cy - 2)],
                             fill=(60, 200, 120, 240), outline=(0, 0, 0, 200))
                draw.text((cx + 8, cy - 6), str(lab),
                          fill=(150, 255, 190, 230))

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
    # DUG FUEL BREAKS: ground whose fuel the crews REMOVED (not
    # burned away) renders as scraped bare earth with a dark dashed
    # edge, exactly like a dozer line on an operations map, so the
    # viewer SEES the containment work, not only its order icons
    if _sim is not None:
        _cutm = ((getattr(_sim, "fuel_suppressed_total", None)
                  is not None)
                 and (_sim.fuel_suppressed_total > 0.08))
        if _cutm is not False and np.any(_cutm):
            _cutm = (_cutm & (world.fuel.fload < 0.08)
                     & ~(_sim.state.burning > 0.5)
                     & ~_sim.ever_burned)
            # ICON overlay, terrain stays readable underneath: two
            # short parallel 'dozer blade' strokes per cell (like a
            # hand-drawn fire line on an ops map), no solid fill
            ys_c, xs_c = np.where(_cutm)
            for cy_, cx_ in zip(ys_c.tolist(), xs_c.tolist()):
                x0_, y0_ = cx_ * scale, cy_ * scale
                q = max(2, scale // 4)
                draw.line([x0_ + 1, y0_ + scale - q,
                           x0_ + scale - q, y0_ + 1],
                          fill=(110, 70, 30, 255), width=2)
                draw.line([x0_ + q, y0_ + scale - 1,
                           x0_ + scale - 1, y0_ + q],
                          fill=(235, 205, 150, 255), width=2)

    # fire perimeter outline (FARSITE style)
    if show_perimeter and _sim is not None:
        from . import behavior
        per = behavior.perimeter_mask(_sim)
        ys, xs = np.where(per)
        for py, px in zip(ys, xs):
            draw.rectangle([px * scale, py * scale,
                            px * scale + scale, py * scale + scale],
                           outline=(255, 60, 40, 255), width=max(1, scale // 4))
    # (spread-direction arrows removed: not drawn on the map or the legend)
    return img


def legend_items():
    items = [(f"{m.name}", _FUEL_COLORS[i]) for i, m in FUEL_MODELS.items()]
    items += [("burned", (0.16, 0.13, 0.12)), ("active fire", (0.98, 0.35, 0.06))]
    return items


def _hex(rgb01):
    r, g, b = [int(round(255 * c)) for c in rgb01]
    return f"#{r:02x}{g:02x}{b:02x}"


# the six base channels, in the order the legend lists them, with the colour
# each one is drawn in. A generated macro is a weighted bundle of these, so
# its identity can be DERIVED from them instead of being an anonymous badge.
BASE_IV_RGB = {
    "suppression_effort": (40, 120, 255),
    "resource_deployment": (0, 230, 255),
    "containment_line": (110, 70, 30),
    "asset_protection": (40, 220, 90),
    "evacuation": (255, 140, 0),
    "public_warning": (255, 220, 0),
    "tactical_burn": (255, 90, 30),
    "water_drafting": (60, 170, 255),
    "retardant_drop": (200, 90, 200),
}
BASE_IV_LABEL = {
    "suppression_effort": "suppression",
    "resource_deployment": "deployment",
    "containment_line": "containment line",
    "asset_protection": "asset protection",
    "evacuation": "evacuation",
    "public_warning": "public warning",
    "tactical_burn": "tactical burn",
    "water_drafting": "water drafting",
    "retardant_drop": "retardant drop",
}


# A generated intervention needs to be told apart from the OTHER generated
# ones at a glance. Mixing its colour from its composition failed at exactly
# that: three macros built from containment plus suppression all came out the
# same blue-grey. So identity comes from a fixed palette and a shape, and the
# composition is carried in the legend TEXT where it can be read exactly.
#
# The palette deliberately avoids the six base-channel colours (blue, cyan,
# brown, green, orange, yellow) so a generated order never reads as a base one.
# Chosen by greedy maximum separation rather than by eye: the hand-picked
# list had violet next to lilac (118 apart, indistinguishable on a busy map).
# Every pair here is at least 150 apart in RGB manhattan distance and every
# entry is at least 171 from the nearest base-channel colour.
MACRO_PALETTE = [
    (204, 10, 204),    # magenta
    (124, 255, 38),    # spring green
    (10, 29, 204),     # deep blue
    (255, 114, 255),   # light pink
    (255, 38, 81),     # crimson
    (114, 255, 212),   # aquamarine
    (226, 255, 114),   # pale lime
    (204, 10, 10),     # dark red
]
MACRO_SHAPES = ("hex", "diamond", "pent", "star", "chip", "bars")


def _macro_key(name: str) -> int:
    """Stable across processes: Python's hash() is salted per run, which
    would repaint every macro on restart."""
    import hashlib
    return int(hashlib.md5(str(name).encode("utf-8")).hexdigest()[:8], 16)


def macro_style(name: str):
    """(rgb, shape) of a generated intervention, fixed by its NAME.

    Keyed on the name rather than on its position in a list, so adding a new
    macro never repaints the existing ones, and the legend swatch, the map
    badge and any figure in the thesis keep agreeing with each other."""
    k = _macro_key(name)
    return (MACRO_PALETTE[k % len(MACRO_PALETTE)],
            MACRO_SHAPES[(k // len(MACRO_PALETTE)) % len(MACRO_SHAPES)])


def macro_rgb(spec_or_name) -> tuple:
    """Colour of a generated intervention.

    Accepts the macro NAME (preferred) or its spec dict, so older callers
    that passed the spec keep working."""
    if isinstance(spec_or_name, str):
        return macro_style(spec_or_name)[0]
    return (192, 0, 255)


def _macro_polygon(cx, cy, r, shape):
    """Vertices of a macro badge. Colour alone is not enough to separate a
    dozen generated interventions, and it fails outright for a reader who
    cannot distinguish two hues, so the shape carries the identity too."""
    import math
    if shape == "diamond":
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    if shape in ("hex", "pent"):
        n = 6 if shape == "hex" else 5
        off = math.pi / 2
        return [(cx + r * math.cos(off + 2 * math.pi * i / n),
                 cy - r * math.sin(off + 2 * math.pi * i / n))
                for i in range(n)]
    if shape == "star":
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else r * 0.46
            a = math.pi / 2 + math.pi * i / 5
            pts.append((cx + rr * math.cos(a), cy - rr * math.sin(a)))
        return pts
    return None            # chip and bars are drawn as rectangles


def _draw_macro_shape(draw, cx, cy, r, shape, fill, ink):
    poly = _macro_polygon(cx, cy, r, shape)
    if poly is not None:
        draw.polygon(poly, fill=fill, outline=ink)
        return
    if shape == "bars":
        # two stacked bars: reads as a distinct silhouette next to a chip
        draw.rectangle([cx - r - 4, cy - r + 1, cx + r + 4, cy - 1],
                       fill=fill, outline=ink, width=1)
        draw.rectangle([cx - r - 4, cy + 1, cx + r + 4, cy + r - 1],
                       fill=fill, outline=ink, width=1)
        return
    draw.rounded_rectangle([cx - r - 4, cy - r + 2, cx + r + 4, cy + r - 2],
                           radius=5, fill=fill, outline=ink, width=1)


def macro_tag(name: str) -> str:
    """Two or three letters for the on-map badge: 'downwind backburn' -> DB."""
    parts = [p for p in str(name).replace("-", "_").split("_") if p]
    if len(parts) >= 2:
        return "".join(p[0] for p in parts[:3]).upper()
    return str(name)[:2].upper()


def macro_description(name: str, spec) -> str:
    """The legend text: what the generated intervention actually orders."""
    comp = (spec or {}).get("composition") or []
    bits = []
    for item in comp:
        if isinstance(item, dict):
            ch, wt = item.get("channel"), float(item.get("weight", 0.0))
        else:
            ch, wt = item[0], float(item[1])
        bits.append(f"{BASE_IV_LABEL.get(str(ch), str(ch))} {wt:.2f}")
    body = " + ".join(bits) if bits else "no resolved composition"
    return (f"{str(name).replace('_', ' ')} [{macro_tag(name)}] "
            f"— GenAI macro: {body}")


def legend_entries(macros=None):
    """Categorized legend as (group, label, hex_color, glyph).

    glyph = how the item appears ON THE MAP: "sq" filled square,
    "dot" filled circle, "ring" open circle, "box" open square,
    "tri" triangle, or a literal text badge. The dashboard renders
    the glyph in the item's color, so the legend shows the actual
    icon vocabulary of the map.

    macros: optional {name: {...}} of GenAI-created MACRO interventions
    (stage 3 vocabulary packages). Each is added under its own group so
    the operator sees the new decisions the generative stage introduced."""
    out = []
    for i, m in FUEL_MODELS.items():
        label = {"non_fuel": "bare ground / rock", "water": "water",
                 "grass": "grass / crops", "shrub": "shrub / maquis",
                 "pine_litter": "conifer forest (pine litter)",
                 "hardwood": "broadleaf forest (hardwood)",
                 "urban": "urban / built-up"}.get(m.name, m.name.replace("_", " "))
        out.append(("Land cover", label, _hex(_FUEL_COLORS[i]), "sq"))
    out.append(("Fire", "active fire", _hex((0.98, 0.35, 0.06)), "sq"))
    out.append(("Fire", "burn scar spectrum: light grey veil = "
                "knocked down early (fuel saved), black = burned out",
                _hex((0.16, 0.13, 0.12)), "sq"))
    # perimeter is drawn as RED cell outlines (not a white ring)
    out.append(("Fire", "fire perimeter", "#ff3c28", "box"))
    # Assets & infrastructure (merged): things on the map that carry value
    # and/or provide access. road/access is its own layer; the facilities
    # (power, water, transport, telecom, hospital, ...) are critical-facility
    # markers, named on the map; buildings and population carry structure /
    # life value. The evacuation route is NOT here (it is a DSS intervention).
    out.append(("Assets", "road / access",
                _hex((0.82, 0.78, 0.66)), "sq"))
    asset_labels = {"building": "building",
                    "critical": "critical facility (power, water, transport, "
                                "telecom, hospital, ...)",
                    "population": "population"}
    for kind, lab in asset_labels.items():
        c = _ASSET_STYLE[kind]["color"]
        # population is drawn as a filled circle, the others as squares
        _gl = "dot" if kind == "population" else "sq"
        out.append(("Assets", lab,
                    f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}", _gl))
    out.append(("Markers", "ignition point (ring + cross)", "#a200de", "ring"))
    # the on-map wind arrow is white (the red compass is a separate widget)
    out.append(("Markers", "wind arrow (blows toward)", "#ffffff", "tri"))
    # ---- DSS orders: the intervention icon vocabulary (all six base
    # interventions S D C P E W, in order) ----
    out.append(("DSS orders (base)", "S — suppression effort "
                "(water on engaged cells)", "#2878ff", "dot"))
    out.append(("DSS orders (base)", "D — resource deployment "
                "(staged capacity, cyan glow)", "#00e6ff", "sq"))
    # C (containment) and its physical footprint (the dug fuel break) are the
    # SAME operation, so they share ONE entry and ONE map glyph: brown diagonal
    # dozer strokes. The order and the dug result are drawn identically.
    out.append(("DSS orders (base)",
                "C — containment line / dug fuel break (diagonal strokes)",
                "#6e461e", "diag"))
    out.append(("DSS orders (base)", "P — asset protection (shield rings)",
                "#28dc5a", "ring"))
    out.append(("DSS orders (base)", "E — evacuation (arrow + EVAC at people)",
                "#ff8c00", "tri"))
    out.append(("DSS orders (base)", "W — public warning (region corner)",
                "#ffdc00", "tri"))
    # ---- GenAI (stage 3) GENERATED interventions: a SECOND type of DSS
    # order. Each macro the generative stage introduces gets its own entry;
    # the group is always shown so it is clear where generated orders appear.
    _macro_names = list((macros or {}).keys())
    if _macro_names:
        for _mn in _macro_names:
            _c, _shape = macro_style(_mn)
            out.append((
                "DSS orders (GenAI-generated)",
                macro_description(_mn, (macros or {}).get(_mn)),
                f"#{_c[0]:02x}{_c[1]:02x}{_c[2]:02x}", _shape))
    else:
        out.append(("DSS orders (GenAI-generated)",
                    "none yet — interventions the generative stage creates "
                    "appear here", "#c000ff", "dot"))
    # ---- sensors: label, color AND range all from the single source of
    # truth (dss.sensors.SENSOR_CATALOG), so the legend can never drift from
    # the Add dropdown or the map glyphs ----
    try:
        from dss.sensors import SENSOR_CATALOG as _SCATl
        for _k, _sp in _SCATl.items():
            _c = tuple(_sp.get("color", (120, 200, 255)))
            _rm = _sp.get("radius_m")
            _rng = ("whole map" if _rm is None
                    else f"{_rm / 1000.0:.1f} km range")
            out.append(("Sensors (+ coverage fill)",
                        f"{_sp.get('label', _k)} — {_rng}",
                        f"#{_c[0]:02x}{_c[1]:02x}{_c[2]:02x}", "dot"))
    except Exception:
        pass
    # ---- resources (kinds match dss.RESOURCE_KINDS) ----
    # two PLACEABLE units (Add dropdown) + two derived elements:
    out.append(("Resources", "ground depot — placeable (green house)",
                "#3cc878", "sq"))
    out.append(("Resources", "helibase / aerial — placeable (cyan H, "
                "map-wide reach)", "#00bee6", "dot"))
    out.append(("Resources", "service radius — the ring drawn AROUND each "
                "unit (not a separate type)", "#ff6464", "ring"))
    out.append(("Resources", "road corridor — AUTO thin capacity along "
                "roads (not addable)", _hex((0.82, 0.78, 0.66)), "sq"))
    # region outlines: yellow = normal/attended, grey = monitored (ignored),
    # orange filled = the coordinator's focused hotspot this cycle
    out.append(("Agents", "DSS region boundary + label", "#ffd228", "box"))
    out.append(("Agents", "focused (attended) region", "#ff7828", "box"))
    out.append(("Agents", "monitored region (not attended)", "#b4b4b4",
                "box"))
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
    uirev = str(flags.pop("uirevision", "keep"))
    pil = render_pil(world, sim=sim, scale=scale, show_labels=True, **flags)
    arr = np.asarray(pil)
    # map the raster onto GRID CELL coordinates (dx = dy = 1/scale) so the
    # axes and the hover read the same x,y the user types when placing a
    # sensor/asset (cells), NOT pixels (pixel = cell * scale, which made a
    # sensor at cell 400 hover as 1600).
    _sc = max(float(scale), 1e-6)
    fig = go.Figure(go.Image(z=arr, x0=0.5 / _sc, dx=1.0 / _sc,
                             y0=0.5 / _sc, dy=1.0 / _sc,
                             hovertemplate="x=%{x:.0f}, y=%{y:.0f}"
                                           "<extra></extra>"))
    # IMPORTANT: no explicit axis ranges. With a constant uirevision the
    # user's zoom/pan survives every step; re-setting ranges on every
    # rebuild would fight the preserved UI state and snap the view back.
    fig.update_xaxes(visible=False, constrain="domain", uirevision=uirev)
    fig.update_yaxes(visible=False, autorange="reversed",
                     scaleanchor="x", scaleratio=1, uirevision=uirev)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=560,
                      dragmode="pan", uirevision=uirev)
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
