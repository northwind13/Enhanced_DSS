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

from .config import FUEL_MODELS, FUEL_NAME_TO_ID, CROP_FUEL_LOADS

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
    "building":   {"color": (150, 60, 40),  "shape": "square",
                   "glyph": "house"},
    "critical":   {"color": (210, 30, 30),  "shape": "cross",
                   "glyph": "crit"},
    "population": {"color": (60, 130, 240),  "shape": "circle",
                   "glyph": "people"},
    "evac_route": {"color": (40, 170, 90),  "shape": "diamond",
                   "glyph": "exit"},
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
    _paint_farmland(img, world)
    return np.clip(img, 0, 1)


#: pale field colours: stubble, young crop, ploughed earth, fallow, pasture.
#: A real field mosaic is read from the air by COLOUR, not by outline, and
#: the parcels were all one dim shade of grass: a quilt of one colour is a
#: smudge. They stay pale so a settlement, a fire or an order still reads
#: over them.
#: user-tuned quilt: pale green / yellow / orange family only (the
#: old ploughed-red and fallow-grey read as bare rock on the relief)
_CROP_COLORS = [(0.93, 0.87, 0.50),      # wheat / stubble, pale yellow
                (0.68, 0.85, 0.50),      # young crop, pale green
                (0.95, 0.74, 0.46),      # ripening field, pale orange
                (0.81, 0.87, 0.55),      # pasture, yellow-green
                (0.91, 0.79, 0.50)]      # hay, pale ochre-orange

#: kept for callers that ask what a field looks like; the MASK is the exact
#: ladder in config, not this band (wild grass reaches down into it)
CROP_LOAD_LO, CROP_LOAD_HI = min(CROP_FUEL_LOADS), max(CROP_FUEL_LOADS)


def _paint_farmland(img, world) -> None:
    """Give every parcel its own pale colour, in place.

    The class carries no field of its own: the parcel colour is derived
    from the fuel load the generator wrote, which is drawn once per parcel.
    That is deliberate. A separate crop layer would have to be saved with
    the map, resampled on a resize and merged on a GIS import, and it would
    describe the same thing twice; deriving it means a saved map, a resized
    map and a hand-painted field all colour the same way.
    """
    ft = np.asarray(world.fuel.ftype)
    fl0 = np.asarray(getattr(world.fuel, "fload0", world.fuel.fload))
    grass = FUEL_NAME_TO_ID["grass"]
    for k, _lv in enumerate(CROP_FUEL_LOADS):
        m = (ft == grass) & (np.abs(fl0 - float(_lv)) < 1e-6)
        if not m.any():
            continue
        col = _CROP_COLORS[k % len(_CROP_COLORS)]
        for c in range(3):
            img[..., c][m] = col[c]


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
        # EVERY burned cell looks burnt: even a cell knocked down in
        # minutes is visibly scorched (ash-brown, terrain faintly
        # showing through), and the tone runs to charcoal black as
        # the consumption approaches total. The earlier version left
        # the light end nearly pristine, which read as "nothing
        # happened here" on a ground the fire had actually touched.
        _char0 = np.array([0.33, 0.27, 0.22])    # light char / ash brown
        _char1 = np.array([0.05, 0.045, 0.04])   # burned-out charcoal
        _t = np.clip(sev[..., None], 0.0, 1.0) ** 0.7
        _char = _char0 * (1.0 - _t) + _char1 * _t
        _mixv = 0.55 + 0.45 * _t                 # scorch opacity
        _shaded = img * (1.0 - _mixv) + _char * _mixv
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
    # NO FLOOR. The tint used to be lifted to 0.25 so a lone asset disc was
    # still visible; now that every built-up cell carries structure value
    # and people, the whole town is "core", and a floor made all of it
    # equally loud. Let low priority stay quiet so the tint says something.
    try:
        from matplotlib import colormaps
        ramp = np.asarray(colormaps["RdPu"](prio))[..., :3]   # pink -> purple
        # (distinct from water blue, fire orange and vegetation greens)
    except Exception:
        ramp = np.stack([0.99 - 0.55 * prio, 0.75 - 0.65 * prio,
                         0.85 - 0.30 * prio], axis=-1)
    # DO NOT PAINT OVER THE TOWN. The overlay used to fill every valued
    # cell at 90% opacity, which was tolerable while "valued" meant a few
    # small asset discs. Now that the whole built-up footprint carries
    # structure value and people, filling it buried the land cover, the
    # street grid and the markers under a flat wash: pink over the grey of
    # built-up ground reads as mud, and the reader loses both the fuel and
    # the town underneath it.
    #
    # The built-up class is already visible on its own. What the overlay has
    # to add is WHERE THE PRIORITY IS, so it draws a ring around the valued
    # ground and leaves the ground itself alone: valued cells keep a light
    # touch that only rises where the priority is genuinely high.
    _w = 0.10 + 0.30 * prio            # 0.10 at low priority, 0.40 at full
    for c in range(3):
        img[..., c][core] = ((1.0 - _w[core]) * img[..., c][core]
                             + _w[core] * ramp[..., c][core])
        img[..., c][halo] = (0.55 * img[..., c][halo]
                             + 0.45 * ramp[..., c][halo])
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


# ---------------------------------------------------------------- symbols
# ONE DEFINITION PER SYMBOL, used by the map AND by the legend. The legend
# swatches used to be hand-written CSS approximations kept in a different
# file from the PIL code that draws the map, so the two drifted apart by
# construction and the reader had to guess which blob answered which line.
# Every glyph below is drawn once, here, and the legend renders its icon by
# calling the same function at a smaller radius.

def draw_supp(draw, cx, cy, r, rgb=(40, 120, 255)):
    """S: water on an engaged cell."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=(*rgb, 230), outline=(255, 255, 255, 180))


def draw_deploy(draw, cx, cy, r, rgb=(216, 60, 255)):
    """D: staged capacity ahead of the front."""
    draw.rectangle([cx - r, cy - r, cx + r, cy + r],
                   fill=(*rgb, 120), outline=(*rgb, 220))


def draw_containment(draw, cx, cy, r, rgb=(110, 70, 30)):
    """C: the dozer strokes of a dug fuel break."""
    q = max(1, r // 2)
    draw.line([cx - r + 1, cy + r - q, cx + r - q, cy - r + 1],
              fill=(*rgb, 255), width=2)
    draw.line([cx - r + q, cy + r - 1, cx + r - 1, cy - r + q],
              fill=(*rgb, 255), width=2)


def draw_protect(draw, cx, cy, r, rgb=(120, 255, 150)):
    """P: a shield ring, with a dark casing so it reads on green terrain."""
    draw.ellipse([cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1],
                 outline=(10, 40, 20, 235), width=3)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(*rgb, 255), width=2)


def draw_evac(draw, cx, cy, r, rgb=(255, 140, 0)):
    """E: an arrow leading the population out."""
    draw.polygon([(cx, cy - 2 * r), (cx - r, cy - r), (cx + r, cy - r)],
                 fill=(*rgb, 255), outline=(15, 15, 15, 255))
    draw.rectangle([cx - max(1, r // 3), cy - r,
                    cx + max(1, r // 3), cy + r // 2],
                   fill=(*rgb, 255), outline=(15, 15, 15, 255))


def draw_warn(draw, cx, cy, r, rgb=(255, 220, 0)):
    """W: the warning triangle covering a whole region."""
    draw.polygon([(cx - r, cy + r), (cx + r, cy + r), (cx, cy - r)],
                 fill=(*rgb, 250), outline=(15, 15, 15, 255))
    draw.text((cx - 2, cy - 2), "!", fill=(0, 0, 0, 255))


def draw_ignition(draw, cx, cy, r, rgb=(162, 0, 222)):
    """The ignition marker: a ring THROUGH a cross, not a plain ring.

    The legend called it "ring + cross" while its swatch was an empty
    circle, so the one marker whose name says what it looks like was the
    one that did not.
    """
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(*rgb, 255), width=2)
    draw.line([cx - r, cy, cx + r, cy], fill=(*rgb, 255), width=2)
    draw.line([cx, cy - r, cx, cy + r], fill=(*rgb, 255), width=2)


def draw_fill(draw, cx, cy, r, rgb=(120, 120, 120)):
    """A flat filled cell: land cover, urban, water, burn scar."""
    draw.rectangle([cx - r, cy - r, cx + r, cy + r],
                   fill=(*rgb, 255), outline=(70, 70, 70, 200))


def draw_point(draw, cx, cy, r, rgb=(120, 120, 120)):
    """A point marker: a sensor, a depot, a population cluster."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=(*rgb, 255), outline=(30, 30, 30, 220))


def draw_outline(draw, cx, cy, r, rgb=(120, 120, 120)):
    """An outlined area: a fire perimeter, a region boundary."""
    draw.rectangle([cx - r, cy - r, cx + r, cy + r],
                   outline=(*rgb, 255), width=2)


def draw_radius(draw, cx, cy, r, rgb=(120, 120, 120)):
    """A service or coverage radius: the ring drawn AROUND a unit."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(*rgb, 255), width=2)


def draw_arrow(draw, cx, cy, r, rgb=(120, 120, 120)):
    """The wind arrow, pointing the way the wind blows toward."""
    draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx, cy + r // 3),
                  (cx + r, cy + r)],
                 fill=(*rgb, 255), outline=(30, 30, 30, 220))


def draw_house(draw, cx, cy, r, rgb=(150, 60, 40)):
    """A settlement / building: pale body under a coloured roof."""
    _k = (0, 0, 0, 255)
    draw.rectangle([cx - r, cy - r // 3, cx + r, cy + r],
                   fill=(238, 232, 220, 255), outline=_k, width=1)
    draw.polygon([(cx - r - 1, cy - r // 3), (cx, cy - r - 2),
                  (cx + r + 1, cy - r // 3)], fill=(*rgb, 255), outline=_k)


def draw_critical(draw, cx, cy, r, rgb=(200, 25, 25)):
    """A critical facility: bordered white plate with an exclamation mark."""
    draw.rectangle([cx - r, cy - r, cx + r, cy + r],
                   fill=(250, 250, 250, 255),
                   outline=(180, 20, 20, 255), width=2)
    bw = max(2, (2 * r) // 5)
    # THE MARK HAS TO FIT THE PLATE. At legend size the plate is 13 px, so
    # the stem's computed bottom landed above its top and PIL refused to
    # draw the rectangle at all: the whole figure script died on its own
    # legend. The parts are clamped to the plate instead.
    _top = cy - r + 3
    _bot = max(_top + 1, cy + r - bw - 4)
    draw.rectangle([cx - bw // 2, _top, cx + bw // 2, _bot], fill=(*rgb, 255))
    _dy1 = cy + r - 2
    _dy0 = min(_dy1 - 1, _dy1 - bw)
    draw.ellipse([cx - bw // 2, _dy0, cx + bw // 2, _dy1], fill=(*rgb, 255))


def draw_people(draw, cx, cy, r, rgb=(60, 130, 240)):
    """Population: a filled disc carrying three heads."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*rgb, 210),
                 outline=(0, 0, 0, 255), width=1)
    for ox in (-r // 2, 0, r // 2):
        draw.ellipse([cx + ox - 2, cy - 3, cx + ox + 2, cy + 1],
                     fill=(255, 255, 255, 255))


def draw_exit(draw, cx, cy, r, rgb=(40, 170, 90)):
    """The evacuation exit: a plate with the way-out arrow on it."""
    draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(*rgb, 255),
                   outline=(0, 0, 0, 255), width=1)
    draw.polygon([(cx - r // 2, cy - r // 2), (cx + r // 2, cy),
                  (cx - r // 2, cy + r // 2)], fill=(255, 255, 255, 255))


#: legend key -> the function that draws it, so a swatch cannot go stale
SYMBOL_DRAW = {
    "supp": draw_supp, "deploy": draw_deploy, "cont": draw_containment,
    "prot": draw_protect, "evac": draw_evac, "warn": draw_warn,
    "ignite": draw_ignition, "sq": draw_fill, "dot": draw_point,
    "box": draw_outline, "ring": draw_radius, "tri": draw_arrow,
    # THE ASSET GLYPHS ARE THE MAP'S OWN. The legend keyed buildings,
    # facilities and people to a plain square and a plain dot while the map
    # drew a house, a red exclamation plate and a disc of heads: a reader
    # comparing the two had to guess which line meant which marker.
    "house": draw_house, "crit": draw_critical, "people": draw_people,
    "exit": draw_exit,
}


#: asset kind -> the drawing function used on the map AND in the legend
ASSET_GLYPH_DRAW = {k: SYMBOL_DRAW[v["glyph"]]
                    for k, v in _ASSET_STYLE.items() if "glyph" in v}


def legend_icon_png(kind: str, rgb, px: int = 18) -> bytes:
    """The legend swatch, drawn by the MAP's own code.

    `kind` is either a base-order key from SYMBOL_DRAW or one of the macro
    shapes, which go through _draw_macro_shape exactly as they do on the map.
    """
    from PIL import Image, ImageDraw
    import io
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = px // 2
    rgb = tuple(int(v) for v in rgb[:3])
    fn = SYMBOL_DRAW.get(kind)
    if fn is not None:
        r = max(2, px // 2 - (4 if kind == "evac" else 2))
        # the arrow is drawn from its tip, so it is nudged down to sit
        # inside the tile the same way it sits above a settlement
        fn(d, c, c + (px // 4 if kind == "evac" else 0), r, rgb)
    else:
        _draw_macro_shape(d, c, c, max(3, px // 2 - 3), kind,
                          (*rgb, 255), (25, 25, 25, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_FONTS: dict = {}


def _font(size: int = 12):
    """Anti-aliased TrueType lettering for the map.

    PIL's built-in bitmap font has one size and no anti-aliasing, so
    every label drawn with it reads as debug output. DejaVu Sans ships
    inside matplotlib (already a dependency), so a real font is always
    available; Windows system fonts are preferred when present."""
    f = _FONTS.get(int(size))
    if f is not None:
        return f
    from PIL import ImageFont
    cands = ["C:/Windows/Fonts/segoeui.ttf",
             "C:/Windows/Fonts/arial.ttf"]
    try:
        import matplotlib.font_manager as _fm
        cands.append(_fm.findfont("DejaVu Sans"))
    except Exception:
        pass
    for c in cands:
        try:
            f = ImageFont.truetype(c, int(size))
            break
        except Exception:
            continue
    else:
        f = ImageFont.load_default()
    _FONTS[int(size)] = f
    return f


def _th(draw) -> int:
    """Measured line height of the draw's current font."""
    try:
        bb = draw.textbbox((0, 0), "Ag")
        return int(bb[3] - bb[1]) + 2
    except Exception:
        return 13


def _tw(draw, text) -> int:
    """Measured pixel width of `text` in the draw's current font."""
    try:
        return int(draw.textlength(str(text)))
    except Exception:
        return 6 * len(str(text))


def _badge(draw, cx, cy, text, rgba, ink=(10, 10, 10, 255)):
    """A short label with its own plate.

    Bare coloured glyphs disappear on a map that is already green, brown and
    orange: the DSS order symbols were all being drawn, and none of them
    could be told apart from terrain. A filled plate with a dark border
    behind the word fixes the contrast wherever the symbol lands.
    """
    _w = _tw(draw, text) + 10
    _h = _th(draw) + 6
    x0, y0 = int(cx - _w // 2), int(cy - _h // 2)
    draw.rounded_rectangle([x0, y0, x0 + _w, y0 + _h], radius=4,
                           fill=rgba, outline=(15, 15, 15, 245),
                           width=1)
    draw.text((x0 + 5, y0 + 3), str(text), fill=ink)


def render_pil(world, sim=None, scale: int = 8, show_fire: bool = True,
               show_assets: bool = True, show_value: bool = False,
               show_hillshade: bool = True, show_wind: bool = True,
               show_ignitions: bool = True, show_grid: bool = False,
               show_labels: bool = False, show_roads: bool = True,
               show_perimeter: bool = False, show_spread_arrows: bool = False,
               sim_for_behavior=None, night_factor: float = 1.0,
               clock_text=None, region_box=None, region_label=None,
               regions=None, sensors=None, depots=None, alloc=None,
               actions=None, pretty: bool = True,
               defer_text: bool = False, label_kinds=None):
    """Render the map to a polished PIL image of size (nx*scale, ny*scale).

    `pretty` is the aesthetic switch the panel exposes: TrueType
    anti-aliased lettering and a bilinear terrain upscale (smooth
    hills, ribbon-like roads). Off = raw nearest-neighbour cells, the
    honest view of the simulation grid.

    `defer_text` PLACES every label but does not draw it, reporting the
    positions in img.info["label_boxes"] instead. The plotly view scales
    the raster to the browser window, and lettering baked into the pixels
    goes to mush when it does; the same view can draw those labels as real
    text on top. The placement still happens here, because that is where
    the markers, the plates and the chrome are known."""
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
    # NEAREST, always: the smoothing experiment (bilinear, then
    # Lanczos) only traded crisp cells for fog. The terrain stays the
    # honest grid; the aesthetics live in the LETTERING, which is
    # always TrueType now.
    img = img.resize((nx * scale, ny * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img, "RGBA")
    # LARGER lettering: the plotly path shows the raster slightly
    # under 1:1, so a 12 px label lands at ~10 px on screen and dies.
    # Plates are measured from the text, so they scale with it.
    draw.font = _font(max(13, min(20, int(scale * 1.7))))

    # ---- ONE LABEL SYSTEM FOR THE WHOLE MAP -------------------------
    # Settlement and facility names were de-cluttered against each other
    # while every other label on the map - sensors, depots, order badges,
    # the whole-map sensor list - was written wherever its own marker
    # happened to be. So a town's name ran through a sensor's label and a
    # depot's name sat on a facility's, and the reader could not tell which
    # word belonged to which thing. Every text on the map goes through the
    # same placer now, and every ICON reserves its own box, so a label
    # never lands on a symbol either.
    _lbl_boxes = []
    #: the boxes of the TEXTS actually written, kept so the invariant "no
    #: two labels overlap" can be measured rather than eyeballed; returned
    #: on the image itself as img.info["label_boxes"]
    _text_boxes = []
    _W_img, _H_img = nx * scale, ny * scale

    def _reserve(box):
        _lbl_boxes.append([int(box[0]), int(box[1]),
                           int(box[2]), int(box[3])])
        return box

    def _overlap(box):
        """How much of `box` is already taken, in pixels squared."""
        return sum(max(0, min(box[2], ob[2]) - max(box[0], ob[0]))
                   * max(0, min(box[3], ob[3]) - max(box[1], ob[1]))
                   for ob in _lbl_boxes)

    def _place_text(cx, cy, r, text, fill=(255, 255, 255, 255),
                    plate=(15, 15, 15, 165), force=False, anchor_dx=0):
        """Write `text` near (cx, cy) where it collides with nothing.

        Eight positions are tried around the marker and each is clamped
        inside the image. `force` takes the least-crowded position when
        every side is taken; it is for things the reader cannot recover
        from the picture (a settlement name), while an ordinary label is
        given up rather than written across something else.
        """
        _t = str(text)
        if not _t:
            return None
        _w_l = _tw(draw, _t) + 8
        _h_l = _th(draw) + 5
        # EIGHT SIDES, THEN FURTHER OUT. Trying only the ring against the
        # marker meant that in a town centre - which is where the depots,
        # the facilities and the settlement name all are - every candidate
        # collided and a forced label was written over its neighbour
        # anyway. Stepping the ring outwards finds clear ground and a
        # leader is unnecessary: the label is still the nearest one.
        _best = None
        for _rad in (r, r + 16, r + 34, r + 56):
            _cands = [(cx + _rad + 3 + anchor_dx, cy - 6),
                      (cx - _rad - 3 - _w_l, cy - 6),
                      (cx - _w_l // 2, cy - _rad - 4 - _h_l),
                      (cx - _w_l // 2, cy + _rad + 4),
                      (cx + _rad + 3, cy - _rad - 4 - _h_l),
                      (cx - _rad - 3 - _w_l, cy - _rad - 4 - _h_l),
                      (cx + _rad + 3, cy + _rad + 4),
                      (cx - _rad - 3 - _w_l, cy + _rad + 4)]
            for _lx, _ly in _cands:
                _lx = int(min(max(2, _lx), max(2, _W_img - _w_l - 2)))
                _ly = int(min(max(2, _ly), max(2, _H_img - _h_l - 2)))
                _bx = [_lx - 2, _ly - 1, _lx + _w_l, _ly + _h_l]
                _ov = _overlap(_bx)
                if _best is None or _ov < _best[0]:
                    _best = (_ov, _lx, _ly, _bx)
                if _ov == 0:
                    break
            if _best is not None and _best[0] == 0:
                break
        if _best is None or (_best[0] > 0 and not force):
            return None
        _ov, _lx, _ly, _bx = _best
        if not defer_text:
            if plate:
                draw.rounded_rectangle(_bx, radius=3, fill=plate)
            draw.text((_lx + 2, _ly + 1), _t, fill=fill)
        _reserve(_bx)
        _text_boxes.append(dict(box=list(_bx), text=_t, fill=tuple(fill),
                                plate=tuple(plate) if plate else None))
        return _bx

    def _place_badge(cx, cy, text, rgba, ink=(10, 10, 10, 255), force=True):
        """A badge, moved off whatever is already there."""
        _t = str(text)
        _w_b = _tw(draw, _t) + 10
        _h_b = _th(draw) + 6
        _best = None
        for _dx, _dy in ((0, 0), (0, -_h_b - 3), (0, _h_b + 3),
                         (_w_b // 2 + 6, 0), (-_w_b // 2 - 6, 0),
                         (0, -2 * _h_b - 6), (0, 2 * _h_b + 6)):
            _x = int(min(max(_w_b // 2 + 2, cx + _dx),
                         max(_w_b // 2 + 2, _W_img - _w_b // 2 - 2)))
            _y = int(min(max(_h_b // 2 + 2, cy + _dy),
                         max(_h_b // 2 + 2, _H_img - _h_b // 2 - 2)))
            _bx = [_x - _w_b // 2, _y - _h_b // 2,
                   _x + _w_b // 2, _y + _h_b // 2]
            _ov = _overlap(_bx)
            if _best is None or _ov < _best[0]:
                _best = (_ov, _x, _y, _bx)
            if _ov == 0:
                break
        if _best is None or (_best[0] > 0 and not force):
            return None
        _ov, _x, _y, _bx = _best
        if not defer_text:
            _badge(draw, _x, _y, _t, rgba, ink)
        _reserve(_bx)
        _text_boxes.append(dict(box=list(_bx), text=_t, fill=tuple(ink),
                                plate=tuple(rgba)))
        return _bx

    # THE CHROME IS RESERVED FIRST, though it is drawn last. The clock,
    # the compass, the north arrow and the scale bar always sit in the same
    # corners, and labels were being written under them and then painted
    # over: reserving the corners up front moves the label instead.
    if clock_text:
        _reserve([6, 6, 10 + _tw(draw, str(clock_text)) + 14, 32])
    if show_wind:
        _reserve([_W_img - 100, 0, _W_img - 8, 92])
    _reserve([8, 12, 46, 76])                       # north arrow
    _reserve([10, _H_img - 40, 14 + max(1, nx // 6) * scale, _H_img - 8])

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
            # BIG ENOUGH TO FIND. The ignition is the one thing a scenario
            # figure has to show besides the ground itself, and at one cell
            # across it disappeared among the sensor footprints.
            draw_ignition(draw, cx, cy, max(7, int(scale * 1.4)))

    # asset markers: recognizable icons
    if show_assets:
        # de-cluttering: the boxes already occupied. MARKERS go in here too,
        # not only labels: checking label against label alone let a name be
        # written straight across a neighbouring facility's icon, which is
        # how a dense town ended up with clipped, unreadable text.
        # THE PEOPLE MARKER DOES NOT COVER THE TOWN. A settlement carries a
        # population asset on the SAME cell as its building, so the blue
        # disc was drawn straight over the house and every town on the map
        # read as a population cluster with no settlement under it. The head
        # count is already written on the settlement's own label, so the
        # disc is only drawn where population stands on its own.
        _built_at = {(int(a.x), int(a.y)) for a in world.assets
                     if getattr(a, "kind", "") == "building"}
        _draw_assets = [a for a in world.assets
                        if not (getattr(a, "kind", "") == "population"
                                and (int(a.x), int(a.y)) in _built_at)]
        for a in _draw_assets:
            _mx, _my = a.x * scale + scale // 2, a.y * scale + scale // 2
            _ar0 = float(getattr(a, "radius", 1) or 1)
            _mr = int(np.clip(scale * (0.75 + 0.30 * _ar0),
                              max(5, int(scale * 0.9)),
                              max(9, int(scale * 2.4))))
            _lbl_boxes.append([_mx - _mr, _my - _mr, _mx + _mr, _my + _mr])
        for a in _draw_assets:
            # THE EVACUATION EXIT IS DRAWN. It is a routing hint rather
            # than a value at risk, and it was skipped here on that ground
            # while the 3D view kept drawing it from its own symbol table:
            # the same world showed an exit in one view and not in the
            # other, and the marker's drawing code below sat unreachable.
            # It is where the E intervention sends people, so the reader
            # has to be able to see it.
            style = _ASSET_STYLE.get(a.kind, {"color": (255, 255, 0)})
            base = style["color"]
            cx, cy = a.x * scale + scale // 2, a.y * scale + scale // 2
            # NOTE: no translucent extent circle is drawn for assets any
            # more. Those halos looked like coverage / resource rings in a
            # town; assets now show only their point icon (house / facility
            # square / population dots). Value still comes from the value
            # layers, not from a drawn ring.
            # THE MARKER SHOULD SAY HOW BIG THE PLACE IS. It was a fixed
            # size, so a two-cell hamlet was drawn exactly as large as a
            # twelve-cell town and a map of small settlements read as a map
            # of cities. It grows with the asset's own footprint, with a
            # floor so the smallest is still legible and a ceiling so a
            # city does not swallow its neighbours.
            _ar = float(getattr(a, "radius", 1) or 1)
            r = int(np.clip(scale * (0.75 + 0.30 * _ar),
                            max(5, int(scale * 0.9)),
                            max(9, int(scale * 2.4))))
            black = (0, 0, 0, 255)
            # ONE DRAWING FUNCTION PER MARKER, shared with the legend. The
            # icons used to be written out here and described a second time
            # in the key, so the two drifted apart and the reader had to
            # match a house against a square by guesswork.
            _fn = ASSET_GLYPH_DRAW.get(a.kind)
            if _fn is not None:
                _fn(draw, cx, cy, r, base)
            else:
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             fill=base + (255,), outline=black, width=1)
            # labels only for named NON-population assets (population sits on
            # the same spot as its building, so it would double every label),
            # and only when the label box does not collide with one already
            # drawn: this keeps a dense town from turning into a wall of
            # overlapping text.
            # WHICH KINDS CARRY A NAME. On a fifteen-town map every civic
            # facility asked for a label and the picture became a wall of
            # text; a scenario figure needs the settlements and the exits
            # named and can leave the rest to the legend. Default: all of
            # them, as the dashboard has always drawn them.
            if (show_labels and getattr(a, "name", "")
                    and a.kind != "population"
                    and (label_kinds is None or a.kind in label_kinds)):
                # PLACE IT SOMEWHERE, DO NOT DROP IT. The label used to be
                # tried in ONE spot, to the right of the marker, and simply
                # abandoned when that collided or ran off the frame: in a
                # town holding several facilities the markers ended up
                # nameless, which is the one thing a label is for. Eight
                # positions are tried around the marker and the box is
                # clamped inside the image, so a name is only given up when
                # every side is genuinely taken.
                _nm = str(a.name)
                # A SETTLEMENT'S NAME AND SIZE, NOTHING ELSE. The label read
                # "Village 2 centre  4k": the word "centre" is the asset's
                # internal name and says nothing a reader needs, and the
                # head count is the one fact that separates a hamlet from a
                # city. It reads "Village 2  4K" now, and the population
                # marker that carries the number is not drawn separately
                # (it would double every label).
                _is_town = False
                if a.kind == "building":
                    _g = str(getattr(a, "group", "") or "")
                    _is_town = bool(_g)
                    if _g:
                        _nm = _g
                    elif _nm.lower().endswith(" centre"):
                        _nm = _nm[:-7]
                    _pop = 0.0
                    for _b in world.assets:
                        if (getattr(_b, "kind", "") == "population"
                                and abs(_b.x - a.x) <= 1
                                and abs(_b.y - a.y) <= 1):
                            _pop = max(_pop,
                                       float(getattr(_b, "population", 0.0)))
                    if _pop >= 1000:
                        _nm += f"  {_pop / 1000:.0f}K"
                    elif _pop >= 1.0:
                        _nm += f"  {_pop:.0f}"
                # A SETTLEMENT IS ALWAYS NAMED. A facility label may be
                # given up when every side of its marker is taken, but a
                # town without its name and head count is the one thing the
                # reader cannot recover from the picture, so those are
                # forced into the least-crowded position instead.
                _place_text(cx, cy, r, _nm, force=_is_town)

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
            _reserve([px0 + 4, py1 - 18, px0 + 10 + _tw(draw, txt), py1 - 2])
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
            _reserve([px0 + 4, py0 + 4,
                      px0 + 10 + 7 * len(str(region_label)), py0 + 22])
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
                _reserve([bx - 10, by - 8, bx + 10, by + 8])
                _place_text(bx, by, 12, str(lab), fill=(*col, 235),
                            force=True)
                continue
            cx, cy = sx_ * scale + scale // 2, sy_ * scale + scale // 2
            rr = int(r_c * scale)
            # A THIN WASH. Thirteen suggested sensors overlap on a small
            # map, and at the old alpha their fills stacked until the
            # terrain underneath went milky - the map lost the land cover
            # it exists to show. The ring still says where the footprint
            # ends; the fill only has to hint at the inside.
            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                         fill=(*col, 12), outline=(*col, 175), width=2)
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
            _reserve([cx - 8, cy - 9, cx + 8, cy + 9])
            _place_text(cx, cy, 9, str(lab), fill=(*col, 235))

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
                draw_containment(draw, _x0c + scale // 2,
                                 _y0c + scale // 2, max(2, scale // 2))
        supp = actions.get("supp")
        if supp is not None:
            ys_, xs_ = np.where(supp)
            for yy_, xx_ in zip(ys_.tolist(), xs_.tolist()):
                if (yy_ + xx_) % 2:
                    continue
                cx = xx_ * scale + scale // 2
                cy = yy_ * scale + scale // 2
                draw_supp(draw, cx, cy, max(2, scale // 3))
        prot = actions.get("prot")
        if prot is not None and np.asarray(prot).any():
            # A BRIGHT GREEN RING ON GREEN TERRAIN IS NOT A SYMBOL. The
            # shield rings were being drawn all along, they just sat in the
            # same hue family as the grass and the canopy, so a defended
            # town read as texture. Every ring now carries a dark casing
            # under the bright stroke, which separates it from any
            # background, and the cluster is named with a P badge so the
            # order is identifiable and not merely visible.
            prot = np.asarray(prot)
            ys_, xs_ = np.where(prot)
            for yy_, xx_ in zip(ys_.tolist(), xs_.tolist()):
                # one ring per ~4 cells: a full blanket of shields hid
                # the town it was protecting, labels included
                if (yy_ * 7 + xx_ * 3) % 4:
                    continue
                cx = xx_ * scale + scale // 2
                cy = yy_ * scale + scale // 2
                draw_protect(draw, cx, cy, max(3, scale // 2))
            _pbx = int(xs_.max()) * scale + scale
            _pby = int(ys_.mean()) * scale + scale // 2
            _place_badge(_pbx + 26, _pby, "P protect", (40, 200, 90, 245))
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
            # library actuators earn a chip exactly like runtime
            # macros: if it worked cells this cycle, it is drawn
            _actu = ("tactical_burn", "retardant_drop",
                     "water_drafting")
            _cand = list(_macdefs) + [a for a in _actu
                                      if a not in _macdefs]
            _fired_m = [(_mn, float(u.get(_mn, 0.0)))
                        for _mn in _cand
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
                        # identity comes from the legend entry and
                        # the cell readout; a floating text code over
                        # the terrain reads as map noise
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
            # EVACUATION. Drawn from the SAME population the physics acts
            # on. It used to read world.assets and look for kind ==
            # "population", but the order itself is applied to the vpop
            # raster, so on any map whose people came from a GIS import
            # rather than a hand-placed asset the evacuation changed the
            # simulation and drew absolutely nothing.
            if u["evacuation"] > 0.3:
                _pts = []
                for a in (getattr(world, "assets", None) or []):
                    if getattr(a, "kind", "") == "population" \
                            and x0_ <= a.x < x1_ and y0_ <= a.y < y1_:
                        _pts.append((int(a.x), int(a.y)))
                if not _pts:
                    _vp = getattr(getattr(world, "value", None), "vpop", None)
                    if _vp is not None:
                        _sub = np.asarray(_vp)[y0_:y1_, x0_:x1_]
                        _yy2, _xx2 = np.where(_sub > 1e-6)
                        if _yy2.size:
                            # one marker per populated cluster, not per cell:
                            # bin coarsely and keep the densest cell of each
                            _seen = {}
                            for _y3, _x3 in zip(_yy2.tolist(), _xx2.tolist()):
                                _k = (_y3 // 25, _x3 // 25)
                                _v = float(_sub[_y3, _x3])
                                if _v > _seen.get(_k, (0.0, 0, 0))[0]:
                                    _seen[_k] = (_v, _x3 + x0_, _y3 + y0_)
                            # the marker names the settlement, it does not
                            # map it: a dozen overlapping arrows hid the
                            # fire they were about
                            _pts = [(_x3, _y3) for _v, _x3, _y3
                                    in sorted(_seen.values(),
                                              key=lambda t: -t[0])[:2]]
                for _px, _py in _pts:
                    cx = _px * scale + scale // 2
                    cy = _py * scale + scale // 2
                    _s = max(9, scale + 4)          # scale with the zoom
                    draw_evac(draw, cx, cy, _s)
                    _place_badge(cx, cy + _s + 8, "EVAC",
                                 (255, 150, 20, 250))
            # PUBLIC WARNING. A 12 px triangle parked in the region corner
            # was the whole symbol, so the order was technically on screen
            # and unreadable. It is now a labelled plate of its own, still
            # at the region corner because the warning covers the WHOLE
            # region rather than any one cell.
            if u["public_warning"] > 0.3:
                # TOP CENTRE OF THE REGION, not the image corner. The corner
                # is where the compass rose and the scale bar live, so the
                # warning kept landing underneath them.
                tx_ = (x0_ + x1_) * scale // 2
                ty_ = y0_ * scale + 8
                _t = max(9, scale + 2)
                draw_warn(draw, tx_, ty_ + _t, _t)
                _place_badge(tx_, ty_ + 2 * _t + 12, "W warn",
                             (255, 225, 40, 250))

    # DSS allocation overlay (D = resource deployment): the STAGED capacity
    # ahead of the front glows a faint violet. Burning / burned cells are
    # skipped (they already read as fire + the blue suppression dots), and
    # only cells carrying a meaningful share of the peak are drawn, so the
    # map is no longer washed in one colour.
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
                               outline=None, fill=(205, 55, 245, a_))

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
                _reserve([cx - 9, cy - 9, cx + 9, cy + 9])
                # A BASE IS NAMED WHEREVER IT STANDS. Depots sit in towns,
                # which is exactly where the map is most crowded, so an
                # ordinary label would be given up there and the reader
                # would see an unexplained marker in the middle of the
                # settlement. Which unit is where drives the response.
                _place_text(cx, cy, 9, str(lab),
                            fill=(150, 230, 255, 235), force=True)
            else:
                draw.rectangle([cx - 5, cy - 2, cx + 5, cy + 6],
                               fill=(60, 200, 120, 240),
                               outline=(0, 0, 0, 200))
                draw.polygon([(cx - 6, cy - 2), (cx, cy - 8),
                              (cx + 6, cy - 2)],
                             fill=(60, 200, 120, 240), outline=(0, 0, 0, 200))
                _reserve([cx - 8, cy - 9, cx + 8, cy + 7])
                _place_text(cx, cy, 9, str(lab),
                            fill=(150, 255, 190, 235), force=True)

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
    # what was written where, so "no two labels overlap" is a measurement
    try:
        img.info["label_boxes"] = list(_text_boxes)
    except Exception:
        pass
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
    "resource_deployment": (216, 60, 255),
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


def legend_sheet(macros=None, groups=None, cols: int = 2,
                 px: int = 20, scale: int = 2, title: str | None = None):
    """The whole key as one picture, drawn by the MAP's own glyph code.

    The legend lived in the page as HTML: readable on screen and impossible
    to put in a document, so a figure had to be captioned by hand and the
    hand-written version drifted from what the map drew. This renders every
    entry with legend_icon_png - the same functions the renderer uses - so
    the sheet cannot claim a symbol the map does not draw.

    `groups` selects and orders the sections; None takes all of them, which
    is what "complete" means here. `scale` supersamples the whole sheet for
    print.
    """
    from PIL import Image, ImageDraw
    import io
    _all = legend_entries(macros)
    _order = list(groups) if groups else []
    for grp, _l, _c, _g in _all:
        if grp not in _order:
            _order.append(grp)
    _pad, _lh, _hh = 12, px + 8, px + 14
    _fw = max(7, int(px * 0.62))                  # rough glyph width
    _wrap = 58

    def _lines(text):
        """NOTHING IS CUT OFF. Long entries used to end in an ellipsis, so
        the sheet stopped saying what half its own symbols meant."""
        out, cur = [], ""
        for word in str(text).split():
            if cur and len(cur) + 1 + len(word) > _wrap:
                out.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        if cur:
            out.append(cur)
        return out or [""]

    rows = {g: [(_lines(l), c, k) for gg, l, c, k in _all if gg == g]
            for g in _order}
    rows = {g: r for g, r in rows.items() if r}

    # column heights: sections are kept whole, so a group never straddles
    _blocks = [(g, sum(len(t) for t, _c, _k in rows[g]))
               for g in rows]
    _total = sum(_hh + n * _lh + _pad for g, n in _blocks)
    _target = _total / max(1, int(cols))
    _colsets, _cur, _h = [], [], 0
    for g, n in _blocks:
        _bh = _hh + n * _lh + _pad
        if _cur and _h + _bh > _target and len(_colsets) < int(cols) - 1:
            _colsets.append(_cur)
            _cur, _h = [], 0
        _cur.append(g)
        _h += _bh
    if _cur:
        _colsets.append(_cur)

    _colw = _pad * 2 + px + 10 + _fw * _wrap
    _heights = [sum(_hh + sum(len(t) for t, _c, _k in rows[g]) * _lh + _pad
                    for g in cs) for cs in _colsets]
    _W = _colw * len(_colsets)
    _H = int(max(_heights) + _pad * 2 + (_hh if title else 0))
    img = Image.new("RGB", (_W, _H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.font = _font(px - 4)
    _y0 = _pad
    if title:
        d.text((_pad, _pad), str(title), fill=(10, 10, 10))
        _y0 += _hh
    for ci, cs in enumerate(_colsets):
        x = ci * _colw + _pad
        y = _y0
        for g in cs:
            d.text((x, y), str(g).upper(), fill=(10, 10, 10))
            d.line([x, y + px + 2, x + _colw - 2 * _pad, y + px + 2],
                   fill=(150, 150, 150))
            y += _hh
            for _txt, hexc, glyph in rows[g]:
                h = str(hexc).lstrip("#")
                rgb = tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))
                ic = Image.open(io.BytesIO(legend_icon_png(glyph, rgb,
                                                           px=px)))
                img.paste(ic, (x, y + 2), ic)
                for _i, _ln in enumerate(_txt):
                    d.text((x + px + 8, y + 3), _ln, fill=(25, 25, 25))
                    y += _lh
            y += _pad
    if int(scale) > 1:
        img = img.resize((img.width * int(scale), img.height * int(scale)),
                         Image.LANCZOS)
    return img


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
    out.append(("Fire", "burn scar spectrum: ash brown = knocked "
                "down early (fuel saved), charcoal black = burned "
                "out", _hex((0.16, 0.13, 0.12)), "sq"))
    # perimeter is drawn as RED cell outlines (not a white ring)
    out.append(("Fire", "fire perimeter", "#ff3c28", "box"))
    # Assets & infrastructure (merged): things on the map that carry value
    # and/or provide access. road/access is its own layer; the facilities
    # (power, water, transport, telecom, hospital, ...) are critical-facility
    # markers, named on the map; buildings and population carry structure /
    # life value. The evacuation exit IS here now: the map draws it, so the
    # key has to name it, and a symbol on the map with no line in the legend
    # is exactly the mismatch this legend was rebuilt to stop.
    out.append(("Assets", "road / access",
                _hex((0.82, 0.78, 0.66)), "sq"))
    asset_labels = {"building": "building",
                    "critical": "critical facility (power, water, transport, "
                                "telecom, hospital, ...)",
                    "population": "population",
                    "evac_route": "evacuation exit (where E sends people)"}
    for kind, lab in asset_labels.items():
        c = _ASSET_STYLE[kind]["color"]
        # the glyph the MAP draws for this kind, not an approximation of it
        _gl = _ASSET_STYLE[kind].get("glyph", "sq")
        out.append(("Assets", lab,
                    f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}", _gl))
    out.append(("Markers", "ignition point (ring + cross)", "#a200de",
                "ignite"))
    # The wind is shown by the compass rose in the corner, which is a
    # widget of its own and needs no key. There is no per-cell wind arrow on
    # the map, so a legend line for one described something that is not
    # drawn.
    # ---- DSS orders: the intervention icon vocabulary (all six base
    # interventions S D C P E W, in order) ----
    out.append(("DSS orders (base)", "S — suppression effort "
                "(water on engaged cells)", "#2878ff", "supp"))
    out.append(("DSS orders (base)", "D — resource deployment "
                "(staged capacity, violet glow)", "#d83cff", "deploy"))
    # C (containment) and its physical footprint (the dug fuel break) are the
    # SAME operation, so they share ONE entry and ONE map glyph: brown diagonal
    # dozer strokes. The order and the dug result are drawn identically.
    out.append(("DSS orders (base)",
                "C — containment line / dug fuel break (diagonal strokes)",
                "#6e461e", "cont"))
    out.append(("DSS orders (base)", "P — asset protection (shield rings)",
                "#78ff96", "prot"))
    out.append(("DSS orders (base)", "E — evacuation (arrow + EVAC at people)",
                "#ff8c00", "evac"))
    out.append(("DSS orders (base)", "W — public warning (region corner)",
                "#ffdc00", "warn"))
    # ---- actuator library: physics-backed channels beyond the six
    # base orders. The map draws them on the worked cells with the SAME
    # chip macro_style(name) yields, so this entry cannot drift from
    # the map.
    try:
        from dss.rules import ACTUATOR_LIBRARY as _ALIB
    except Exception:
        _ALIB = {"tactical_burn": "", "water_drafting": "",
                 "retardant_drop": ""}
    for _an, _ad in _ALIB.items():
        _c, _shape = macro_style(_an)
        out.append((
            "DSS orders (actuator library)",
            f"{BASE_IV_LABEL.get(_an, _an.replace('_', ' '))} "
            f"[{macro_tag(_an)}]" + (f" — {_ad}" if _ad else ""),
            f"#{_c[0]:02x}{_c[1]:02x}{_c[2]:02x}", _shape))
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
            # THE LABEL ALREADY CARRIES THE RANGE, in the parentheses the
            # catalog puts there. Repeating it after a dash said the same
            # thing twice and pushed every sensor line off the panel.
            out.append(("Sensors (+ coverage fill)",
                        str(_sp.get("label", _k)),
                        f"#{_c[0]:02x}{_c[1]:02x}{_c[2]:02x}", "dot"))
    except Exception:
        pass
    # ---- resources (kinds match dss.RESOURCE_KINDS) ----
    # two PLACEABLE units (Add dropdown) + two derived elements:
    # PLAIN NAMES. A legend says what a symbol IS; how a unit is added and
    # whether it is placeable belong in the panel that places it, and the
    # explanations made every line wrap.
    out.append(("Resources", "ground depot", "#3cc878", "sq"))
    out.append(("Resources", "helibase / aerial", "#00bee6", "dot"))
    out.append(("Resources", "service radius", "#ff6464", "ring"))
    out.append(("Resources", "road corridor",
                _hex((0.82, 0.78, 0.66)), "sq"))
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
    # plotly's 3D marker set has no house and no exclamation plate, so the
    # shapes here are the closest it offers; the COLOURS at least come from
    # the one asset table the 2D map and the legend read, so a facility is
    # the same red and a settlement the same brown in every view instead of
    # turning white in this one.
    # plotly's 3D marker set has no exclamation plate; "x" was read as a
    # deletion mark rather than as a facility, so the upright cross is used
    # - the shape a hospital or a utility is drawn with everywhere else.
    sym = {"building": "square", "critical": "cross",
           "population": "circle", "evac_route": "diamond"}
    col = {k: "rgb({}, {}, {})".format(*v["color"])
           for k, v in _ASSET_STYLE.items()}
    if world.assets:
        for kind in sym:
            pts = [a for a in world.assets if a.kind == kind]
            if not pts:
                continue
            # ONLY THE SETTLEMENTS ARE LABELLED. Plotly writes 3D text
            # wherever the point lands and has no de-cluttering, so a town
            # holding six facilities came out as six names stacked on top
            # of each other and on the town's own. The names that are not
            # drawn are still one hover away, and the settlement label
            # carries the head count, as it does on the 2D map.
            _lab = []
            for a in pts:
                _g = str(getattr(a, "group", "") or "")
                if kind == "building" and _g:
                    _p = sum(float(getattr(b, "population", 0.0) or 0.0)
                             for b in world.assets
                             if getattr(b, "kind", "") == "population"
                             and str(getattr(b, "group", "")) == _g)
                    _lab.append(f"{_g}  {_p / 1000:.0f}K" if _p >= 1000
                                else (f"{_g}  {_p:.0f}" if _p >= 1 else _g))
                elif kind == "evac_route":
                    _lab.append(str(a.name))
                else:
                    _lab.append("")
            data.append(go.Scatter3d(
                x=[a.x for a in pts], y=[a.y for a in pts],
                z=[zfull[int(np.clip(a.y, 0, ny - 1)), int(np.clip(a.x, 0, nx - 1))]
                   + 0.6 for a in pts],
                mode="markers+text", text=_lab,
                hovertext=[str(a.name) for a in pts],
                textposition="top center", textfont=dict(size=11,
                                                         color="white"),
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
    # SAME BLOCK AS THE 2D MAP. This was 460 px tall on an opaque white
    # sheet while the 2D view is 560 on the page's own background, so
    # switching between them resized the map area and the 3D terrain sat in
    # a white box of its own instead of in the map card.
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                      showlegend=False, uirevision="keep",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      scene=dict(aspectmode="data", uirevision="keep",
                                 bgcolor="rgba(0,0,0,0)",
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


def cell_hover_text(world, sim=None, actions=None, alloc=None,
                    regions=None, network=None, only=None, engine=None):
    """One tooltip per grid cell: where it started, where it is, what moved.

    The map answers "what is happening" at a glance and "what happened HERE"
    not at all. The burn state, the fuel that was there before the fire, the
    assets at risk, the orders that landed and the rules behind them were
    spread across the legend, the step table and the decision log, and none
    of them was addressable by pointing at the place.

    `ordered` and `applied` are deliberately separate. `ordered` is what the
    DSS DECIDED for this cell; `applied` is what actually reached the physics
    after the coordination share, the budget cut and the fail-safe had their
    say. A cell that is ordered but not applied is one the plan reached and
    the funding did not.

    Every quantity is given as start -> now with the change, because a lone
    current value cannot say whether a cell was saved or was never at risk.
    The fire line carries both instants: the step it was lit and the step it
    went out, which is what makes a burn scar readable as a history rather
    than a colour.

    `only=(x, y)` answers for ONE cell. The animated view cannot afford the
    hover layer (its payload is a hundred times the compressed frame), so a
    pinned readout is how a place is inspected while the fire runs, and it
    must not build seven thousand strings to show one.

    Returns an (ny, nx) object array, or a single string when `only` is set.
    """
    ny, nx = np.asarray(world.fuel.ftype).shape
    ft = np.asarray(world.fuel.ftype)
    fl = np.asarray(world.fuel.fload)
    fl0 = np.asarray(getattr(world.fuel, "fload0", fl))
    fm = np.asarray(world.fuel.fmoist)
    fm0 = np.asarray(getattr(sim, "_fmoist0", fm)) if sim is not None else fm
    names = {i: m.name for i, m in FUEL_MODELS.items()}

    burning = ever = fis = bout = None
    step_now = None
    if sim is not None:
        burning = np.asarray(sim.state.burning) > 0.5
        ever = np.asarray(sim.ever_burned)
        fis = np.asarray(getattr(sim, "first_ignition_step",
                                 np.full((ny, nx), -1)))
        bout = np.asarray(getattr(sim, "burnout_step",
                                  np.full((ny, nx), -1)))
        step_now = int(getattr(sim.state, "step", 0))
        inten = np.asarray(getattr(sim.state, "intensity", np.zeros((ny, nx))))
    else:
        inten = np.zeros((ny, nx))

    val = getattr(world, "value", None)
    vbld = np.asarray(val.vbld) if val is not None else None
    vcrit = np.asarray(val.vcrit) if val is not None else None
    vpop = np.asarray(val.vpop) if val is not None else None
    vpop0 = np.asarray(getattr(sim, "_vpop0", vpop)) \
        if (sim is not None and vpop is not None) else vpop

    supp = cont = prot = None
    if actions:
        supp, cont, prot = (actions.get("supp"), actions.get("cont"),
                            actions.get("prot"))
    mac = (actions or {}).get("macro_cells") or {}

    res = getattr(sim, "last_applied_resource", None) if sim is not None \
        else None

    def _arr(name):
        a = getattr(res, name, None) if res is not None else None
        return np.asarray(a) if a is not None else None

    rcap, rcut = _arr("rcap"), _arr("rcut")
    revac, rwarn = _arr("revac"), _arr("rwarn")

    # the caller may hand regions as Region objects or as the
    # (x0, y0, x1, y1, label) tuples the renderer takes; accept both rather
    # than making one of the two callers convert
    reg_lab = np.empty((ny, nx), dtype=object)
    for r in (regions or []):
        try:
            if isinstance(r, dict):
                x0, y0, x1, y1 = r.get("box", (0, 0, 0, 0))
                lab = r.get("name", "?")
            elif hasattr(r, "x0"):
                x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
                lab = getattr(r, "name", "?")
            else:
                x0, y0, x1, y1, lab = tuple(r)[:5]
        except Exception:
            continue
        reg_lab[max(0, int(y0)):min(ny, int(y1)),
                max(0, int(x0)):min(nx, int(x1))] = str(lab)

    # WHICH RULES PUT THE ORDERS THERE. The orders on a cell are the
    # output of the rules that fired for the region that owns it, so the
    # rules and the adaptation verdict of the last cycle are named per
    # region and attached to every cell inside it.
    by_region = {}
    cyc = None
    if engine is not None:
        cycles = getattr(engine, "cycles", None) or []
        cyc = cycles[-1] if cycles else None
    if cyc:
        for _rn, _rd in (cyc.get("regions") or {}).items():
            _fired = ", ".join(
                f"{n} {float(w):.2f}"
                for n, w in (_rd.get("fired") or [])[:4] if float(w) > 0.05)
            by_region[_rn] = _fired or "none above 0.05"

    def _d(now, start, fmt="{:.2f}"):
        """start -> now with the signed change, or just the value if equal."""
        if abs(float(now) - float(start)) < 5e-3:
            return fmt.format(float(now))
        return (fmt.format(float(start)) + " → " + fmt.format(float(now))
                + f" ({float(now) - float(start):+.2f})")

    out = np.empty((ny, nx), dtype=object)
    if only is not None:
        _ox, _oy = int(only[0]), int(only[1])
        if not (0 <= _ox < nx and 0 <= _oy < ny):
            return None
        cells = [(_oy, _ox)]
    else:
        cells = None

    for y, x in (cells if cells is not None
                 else ((yy, xx) for yy in range(ny) for xx in range(nx))):
        L = [f"<b>cell ({x}, {y})</b> · {names.get(int(ft[y, x]), '?')}"
             + (f" · {reg_lab[y, x]}" if reg_lab[y, x] else "")]
        # ---- state_0 -> state_k, with the deltas
        L.append("fuel load " + _d(fl[y, x], fl0[y, x])
                 + " · moisture " + _d(fm[y, x], fm0[y, x]))
        # ---- the fire's own history: lit at k, out at k'
        if burning is not None:
            _k0 = int(fis[y, x])
            if burning[y, x]:
                L.append(f"<b>BURNING</b> since k={_k0}"
                         + (f", {step_now - _k0} steps"
                            if _k0 >= 0 and step_now is not None else "")
                         + f" · intensity {float(inten[y, x]):.2f}")
            elif ever is not None and ever[y, x]:
                _k1 = int(bout[y, x]) if bout is not None else -1
                if _k0 >= 0 and _k1 >= 0:
                    L.append(f"burned: lit k={_k0}, out k={_k1} "
                             f"({_k1 - _k0} steps)")
                elif _k0 >= 0:
                    L.append(f"burned: lit k={_k0}, still going out")
                else:
                    L.append("burned")
        # ---- what is at stake here
        _a = []
        if vbld is not None and float(vbld[y, x]) > 1e-6:
            _a.append(f"building {float(vbld[y, x]):.2f}")
        if vcrit is not None and float(vcrit[y, x]) > 1e-6:
            _a.append(f"critical {float(vcrit[y, x]):.2f}")
        if vpop is not None and (float(vpop[y, x]) > 1e-6
                                 or (vpop0 is not None
                                     and float(vpop0[y, x]) > 1e-6)):
            _a.append("population "
                      + _d(vpop[y, x], vpop0[y, x], "{:.0f}") + "/km²")
        if _a:
            L.append("at stake: " + ", ".join(_a))
        # WHO GOT OUT AND WHO DID NOT. Evacuation is the only thing that
        # removes people from a cell, so the drop since the start IS the
        # number who left; whoever is still there when the cell burns is
        # the exposure the population cost integrates. Reading a headcount
        # beside a burn scar could not tell those two apart.
        if (vpop is not None and vpop0 is not None
                and float(vpop0[y, x]) > 1e-6):
            _left = float(vpop0[y, x]) - float(vpop[y, x])
            _stay = float(vpop[y, x])
            _km2 = " /km²"
            if _left > 1e-6:
                L.append(f"people: {_left:.0f} evacuated, "
                         f"{_stay:.0f} still here{_km2}")
            elif burning is not None and ever is not None and ever[y, x]:
                L.append(f"people: NONE evacuated, {_stay:.0f} were here "
                         f"when it burned{_km2}")
        # ---- the orders that landed on this cell
        _o = []
        if supp is not None and supp[y, x]:
            _o.append("S suppression")
        if cont is not None and cont[y, x]:
            _o.append("C containment line")
        if prot is not None and prot[y, x]:
            _o.append("P asset protection")
        for _mn, _cells_m in mac.items():
            try:
                if _cells_m[y, x]:
                    _o.append(f"{macro_tag(_mn)} {_mn}")
            except Exception:
                pass
        if _o:
            L.append("ordered: " + ", ".join(_o))
        _ap = []
        if rcap is not None and float(rcap[y, x]) > 1e-6:
            _ap.append(f"capacity {float(rcap[y, x]):.2f}")
        if rcut is not None and float(rcut[y, x]) > 1e-6:
            _ap.append("fuel break dug")
        if revac is not None and float(revac[y, x]) > 1e-6:
            _ap.append(f"evacuation {float(revac[y, x]):.2f}")
        if rwarn is not None and float(rwarn[y, x]) > 1e-6:
            _ap.append(f"warning {float(rwarn[y, x]):.2f}")
        if alloc is not None:
            try:
                _v = float(np.asarray(alloc)[y, x])
                if _v > 1e-6:
                    _ap.append(f"staged {_v:.2f}")
            except Exception:
                pass
        if _ap:
            L.append("applied: " + ", ".join(_ap))
        # ---- and the decision those orders came out of
        _rl = by_region.get(reg_lab[y, x])
        if _rl:
            L.append("rules fired: " + _rl)
        out[y, x] = "<br>".join(L)

    if cells is not None:
        return out[cells[0][0], cells[0][1]]
    return out


def map_figure_2d(world, sim=None, scale: int = 6, hover: bool = True,
                  max_hover_cells: int = 60000, engine=None, **flags):
    """2D map as a plotly image so it supports scroll zoom and pan like the 3D
    view. Draws the same content as render_pil (land cover, roads, assets,
    flame fire), plus a per-cell hover.

    `max_hover_cells` bounds the tooltip payload: one string per cell is a
    fine price on a 100x70 map and an unreasonable one on a 400x400, so
    beyond the bound the hover layer is simply left off.
    """
    import plotly.graph_objects as go
    # NO supersampling: the app pairs this figure with an
    # image-rendering: pixelated stylesheet, and pixelated wants the
    # raster NEAR 1:1 (oversized input + nearest decimation reads as
    # soft mush; near-1:1 + pixelated reads like the st.image path).
    uirev = str(flags.pop("uirevision", "keep"))
    # read by the hover layer as well as the renderer
    actions = flags.get("actions")
    alloc = flags.get("alloc")
    regions = flags.get("regions")
    # THE LETTERING IS TEXT, NOT PIXELS, and it is placed by the RENDERER.
    # This view hands the raster to the browser, which scales it: lettering
    # baked into the image is resampled with it and turns to mush. But the
    # placement has to happen where the markers, the plates and the chrome
    # are known, or the labels come out clear and on top of the symbols.
    # render_pil places every label - settlements, facilities, sensors,
    # depots, order badges - and reports where each one goes; they are
    # drawn here as annotations, razor sharp at any zoom.
    pil = render_pil(world, sim=sim, scale=scale, show_labels=True,
                     defer_text=True, **flags)
    _labels = list((pil.info or {}).get("label_boxes") or [])
    arr = np.asarray(pil)
    # map the raster onto GRID CELL coordinates (dx = dy = 1/scale) so the
    # axes and the hover read the same x,y the user types when placing a
    # sensor/asset (cells), NOT pixels (pixel = cell * scale, which made a
    # sensor at cell 400 hover as 1600).
    _sc = max(float(scale), 1e-6)
    fig = go.Figure(go.Image(z=arr, x0=0.5 / _sc, dx=1.0 / _sc,
                             y0=0.5 / _sc, dy=1.0 / _sc,
                             hoverinfo="skip"))
    # A TRANSPARENT CELL GRID ON TOP, purely to be hovered. The image trace
    # could only ever report the pixel it was under; what the reader wants
    # when pointing at a place is what is happening THERE, which lives in
    # the world, the simulator and the orders rather than in the picture.
    # The overlay carries one preformatted line per cell and draws nothing.
    if hover:
        _ny, _nx = np.asarray(world.fuel.ftype).shape
        if _ny * _nx <= max_hover_cells:
            _txt = cell_hover_text(world, sim=sim, actions=actions,
                                   alloc=alloc, regions=regions,
                                   engine=engine)
            fig.add_trace(go.Heatmap(
                z=np.zeros((_ny, _nx)), x0=0.5, dx=1.0, y0=0.5, dy=1.0,
                customdata=_txt,
                hovertemplate="%{customdata}<extra></extra>",
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False, hoverongaps=False))
            fig.update_layout(hoverlabel=dict(align="left",
                                              bgcolor="rgba(255,255,255,.96)",
                                              font=dict(size=12)))
    # IMPORTANT: no explicit axis ranges. With a constant uirevision the
    # user's zoom/pan survives every step; re-setting ranges on every
    # rebuild would fight the preserved UI state and snap the view back.
    # VECTOR LABELS, at the positions the renderer worked out. Deriving
    # them a second time here would mean a second de-clutter that knows
    # about the assets only: sensor names, depot names and order badges
    # live in the raster pass, and a label placed here in ignorance of
    # them lands on top of them.
    for _l in _labels:
        _b = _l.get("box") or [0, 0, 0, 0]
        _fx = tuple(int(v) for v in (_l.get("fill") or (255, 255, 255, 255)))
        _pl = _l.get("plate")
        fig.add_annotation(
            x=(_b[0] + 2) / _sc, y=(_b[1] + _b[3]) / (2.0 * _sc),
            text=str(_l.get("text", "")), showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=13, family="Helvetica, Arial, sans-serif",
                      color=f"rgb({_fx[0]},{_fx[1]},{_fx[2]})"),
            bgcolor=(f"rgba({_pl[0]},{_pl[1]},{_pl[2]},"
                     f"{(_pl[3] if len(_pl) > 3 else 255) / 255:.2f})"
                     if _pl else "rgba(15,15,15,0.72)"),
            borderpad=3)

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
