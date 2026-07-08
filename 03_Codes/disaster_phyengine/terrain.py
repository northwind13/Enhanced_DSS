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


def _flow_accumulation(elev):
    """D8 flow accumulation. Every cell drains to its lowest neighbor and
    cells are processed from highest to lowest, so the result traces the
    dendritic drainage network of the terrain (large values = stream lines).
    The receiver of each cell is computed vectorized; only the accumulation
    pass itself is a linear loop."""
    ny, nx = elev.shape
    e = np.pad(elev, 1, mode="edge")
    stacks = []
    offs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            stacks.append(e[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx])
            offs.append(dy * nx + dx)
    nb = np.stack(stacks)                       # (8, ny, nx)
    kmin = np.argmin(nb, axis=0)
    emin = np.take_along_axis(nb, kmin[None], axis=0)[0]
    offs = np.asarray(offs)
    flat = np.arange(ny * nx)
    recv = flat + offs[kmin.ravel()]
    recv = np.where(emin.ravel() < elev.ravel(), recv, flat)
    # border receivers that would wrap are treated as outlets
    yy, xx = np.divmod(flat, nx)
    ry, rx = np.divmod(np.clip(recv, 0, ny * nx - 1), nx)
    recv = np.where((np.abs(ry - yy) <= 1) & (np.abs(rx - xx) <= 1),
                    recv, flat)
    acc = np.ones(ny * nx, dtype=float)
    order = np.argsort(elev.ravel())[::-1]
    rv = recv
    for i in order:
        r = rv[i]
        if r != i:
            acc[r] += acc[i]
    return acc.reshape(ny, nx)


def _erode(elev, rng, strength=0.30):
    """Hydraulic-style erosion pass. The terrain is carved along the flow
    accumulation network (a stream-power shortcut), which converts the
    isolated pits of raw fractal noise into CONNECTED, dendritic valleys the
    way water shapes real relief; a light smoothing afterwards plays the
    role of thermal erosion (talus creep). Returns the carved elevation and
    the normalized drainage map (0..1, large where streams run) for use as
    a topographic wetness index."""
    acc = _flow_accumulation(_gaussian_smooth(elev, 1.0))
    a = np.log1p(acc)
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-9)
    relief = float(elev.max() - elev.min())
    carved = elev - strength * relief * (a ** 1.6)
    carved = _gaussian_smooth(carved, 0.8)
    carved -= carved.min()
    return carved, a


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


def _carve_downhill_river(elev, ftype, water_id, rng):
    """Carve one continuous river that flows strictly downhill from the highest
    headwater to the sea, a lake or the map edge (never uphill). Returns the
    per-column river row, NaN where the river does not pass."""
    ny, nx = elev.shape
    water = (ftype == water_id)
    col_y = np.full(nx, np.nan)
    land = ~water
    if not land.any():
        return col_y
    e = _gaussian_smooth(elev.astype(float), 1.2)          # route on a mild DEM
    erange = float(e.max() - e.min()) + 1e-9
    y, x = map(int, np.unravel_index(
        int(np.argmax(np.where(land, e, -1e18))), e.shape))
    visited = np.zeros((ny, nx), dtype=bool)
    for _ in range(6 * (nx + ny)):
        ftype[y, x] = water_id
        col_y[x] = y
        visited[y, x] = True
        if water[y, x]:
            break                                          # reached sea / lake
        best = None
        bestc = 1e18
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if not (0 <= yy < ny and 0 <= xx < nx) or visited[yy, xx]:
                    continue
                c = e[yy, xx] + rng.uniform(0.0, 0.01) * erange   # tie-break
                if c < bestc:
                    bestc = c
                    best = (yy, xx)
        if best is None:
            break
        by, bx = best
        if e[by, bx] > e[y, x] + 1e-9:                     # only a pit remains
            break                                          # no uphill flow
        y, x = by, bx
        if x <= 0 or x >= nx - 1 or y <= 0 or y >= ny - 1:
            ftype[y, x] = water_id
            col_y[x] = y                                   # reached the mouth
            break
    return col_y


def _road_cost_grid(elev, slope, ftype, water_id, bridge_cost=18.0):
    """Per-cell road traversal cost: steep ground is expensive so roads keep
    to gentle low terrain and switchback up slopes. Water is crossable but
    VERY expensive (bridge_cost per cell), so a road bridges a river with the
    shortest possible perpendicular crossing and never runs along the water;
    long crossings (lakes, sea) stay effectively prohibitive."""
    en = (elev - elev.min()) / (float(elev.max() - elev.min()) + 1e-9)
    cost = 1.0 + 8.0 * np.tan(np.clip(slope, 0.0, 1.3)) + 0.4 * en
    return np.where(ftype == water_id, float(bridge_cost), cost)


def _least_cost_path(cost, sources_mask, goal):
    """Dijkstra shortest path from any True cell in sources_mask to goal
    (gx, gy) over the finite-cost cells. Returns the list of (x, y) or None."""
    import heapq
    ny, nx = cost.shape
    gx, gy = int(goal[0]), int(goal[1])
    if not np.isfinite(cost[gy, gx]):
        return None
    dist = np.full((ny, nx), np.inf)
    prevx = np.full((ny, nx), -1, dtype=np.int32)
    prevy = np.full((ny, nx), -1, dtype=np.int32)
    pq = []
    ys, xs = np.where(sources_mask)
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        dist[yy, xx] = 0.0
        heapq.heappush(pq, (0.0, yy, xx))
    SQ2 = 1.4142135623730951
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        if x == gx and y == gy:
            break
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                    continue
                c = cost[yy, xx]
                if not np.isfinite(c):
                    continue
                nd = d + c * (SQ2 if (dy and dx) else 1.0)
                if nd < dist[yy, xx]:
                    dist[yy, xx] = nd
                    prevy[yy, xx] = y
                    prevx[yy, xx] = x
                    heapq.heappush(pq, (nd, yy, xx))
    if not np.isfinite(dist[gy, gx]):
        return None
    path = []
    cx, cy = gx, gy
    while cx >= 0 and cy >= 0:
        path.append((cx, cy))
        px = int(prevx[cy, cx])
        py = int(prevy[cy, cx])
        if px < 0:
            break
        cx, cy = px, py
    return path


def _flood_sea(below):
    """Sea = the below-sea-level cells connected to a map border, so the coast
    is naturally indented (bays, peninsulas) and interior lows stay as lakes."""
    from collections import deque
    ny, nx = below.shape
    sea = np.zeros_like(below)
    dq = deque()
    for x in range(nx):
        for y in (0, ny - 1):
            if below[y, x] and not sea[y, x]:
                sea[y, x] = True
                dq.append((y, x))
    for y in range(ny):
        for x in (0, nx - 1):
            if below[y, x] and not sea[y, x]:
                sea[y, x] = True
                dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and below[yy, xx] and not sea[yy, xx]:
                sea[yy, xx] = True
                dq.append((yy, xx))
    return sea


def _dijkstra_field(cost, source):
    """Single-source least-cost field over finite-cost cells. Returns distance
    and predecessor arrays so any target can be traced back to the source."""
    import heapq
    ny, nx = cost.shape
    dist = np.full((ny, nx), np.inf)
    prevx = np.full((ny, nx), -1, dtype=np.int32)
    prevy = np.full((ny, nx), -1, dtype=np.int32)
    sx, sy = int(source[0]), int(source[1])
    dist[sy, sx] = 0.0
    pq = [(0.0, sy, sx)]
    SQ2 = 1.4142135623730951
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                    continue
                c = cost[yy, xx]
                if not np.isfinite(c):
                    continue
                nd = d + c * (SQ2 if (dy and dx) else 1.0)
                if nd < dist[yy, xx]:
                    dist[yy, xx] = nd
                    prevy[yy, xx] = y
                    prevx[yy, xx] = x
                    heapq.heappush(pq, (nd, yy, xx))
    return dist, prevx, prevy


def _prune_puddles(ftype, water_id, grass_id, min_cells=4):
    """Remove 1-3 cell water specks (quantile + erosion artifacts) so open
    water means the sea, lakes and the river, not scattered puddles (8-connected, so a diagonal river
    stays one body). The sea, the lakes
    and the river are far larger than the threshold, so only specks go; pruned cells
    revert to grass and get their fuel with the normal fload pass."""
    from collections import deque
    wm = (ftype == water_id)
    ny, nx = wm.shape
    seen = np.zeros_like(wm)
    for y0 in range(ny):
        for x0 in range(nx):
            if not wm[y0, x0] or seen[y0, x0]:
                continue
            comp = [(y0, x0)]
            seen[y0, x0] = True
            dq = deque(comp)
            while dq:
                y, x = dq.popleft()
                for dy in (-1, 0, 1):
                  for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < ny and 0 <= xx < nx and wm[yy, xx]
                            and not seen[yy, xx]):
                        seen[yy, xx] = True
                        comp.append((yy, xx))
                        dq.append((yy, xx))
            if len(comp) < int(min_cells):
                for y, x in comp:
                    ftype[y, x] = grass_id


def _dilate(mask, r):
    """Grow a boolean mask by r cells (4-neighbour), for water buffers."""
    out = mask.copy()
    for _ in range(int(max(0, r))):
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        out = d
    return out


def _place_settlements(elev, slope, ftype, water_id, n, rng):
    """Scatter settlements across the map with blue-noise spacing on SOLID land
    that is set back from water by a buffer, so a town never sits on or
    straddles the shoreline. Low, gentle ground is preferred."""
    ny, nx = elev.shape
    land = (ftype >= 1) & (ftype <= 4)
    if n <= 0 or not land.any():
        return []
    water = (ftype == water_id)
    # require a buffer of dry land around each site (shrink it only if the map
    # is too watery to fit any settlement otherwise)
    buf = 4
    safe = land & ~_dilate(water, buf)
    while not safe.any() and buf > 1:
        buf -= 1
        safe = land & ~_dilate(water, buf)
    if not safe.any():
        safe = land
    en = (elev - elev.min()) / (float(elev.max() - elev.min()) + 1e-9)
    score = (1.0 - en) * (1.0 - np.clip(slope / 1.3, 0.0, 1.0))
    # historical siting: settlements grow near (but not on) water
    if water.any():
        nw = _gaussian_smooth(water.astype(float), max(2.0, max(nx, ny) / 60.0))
        nw = nw / max(float(nw.max()), 1e-9)
        score = score * (0.7 + 0.6 * np.clip(nw * 2.0, 0.0, 1.0))
    score = score + rng.uniform(0.0, 0.15, size=score.shape)
    score[~safe] = -1.0
    ys, xs = np.where(safe)
    sc = score[ys, xs]
    keep = int(min(sc.size, max(2000, n * 60)))
    top = np.argpartition(-sc, keep - 1)[:keep]
    top = top[np.argsort(-sc[top])]
    min_d = max(3.0, 0.62 * np.sqrt(float(safe.sum()) / float(n)))
    md2 = min_d * min_d
    picks = []
    for idx in top:
        x, y = int(xs[idx]), int(ys[idx])
        if all((x - px) ** 2 + (y - py) ** 2 >= md2 for px, py in picks):
            picks.append((x, y))
            if len(picks) >= n:
                break
    if len(picks) < n:
        have = set(picks)
        for idx in top:
            xy = (int(xs[idx]), int(ys[idx]))
            if xy not in have:
                picks.append(xy)
                have.add(xy)
                if len(picks) >= n:
                    break
    return picks


def _nearest_land(land, x, y):
    """Nearest cell that is not water. Guarantees structures stay off water."""
    ny, nx = land.shape
    x = int(np.clip(x, 0, nx - 1))
    y = int(np.clip(y, 0, ny - 1))
    if land[y, x]:
        return x, y
    for r in range(1, max(ny, nx)):
        x0 = max(0, x - r); x1 = min(nx - 1, x + r)
        y0 = max(0, y - r); y1 = min(ny - 1, y + r)
        sub = land[y0:y1 + 1, x0:x1 + 1]
        if sub.any():
            ys, xs = np.where(sub)
            d = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
            k = int(np.argmin(d))
            return int(xs[k] + x0), int(ys[k] + y0)
    return x, y


def _urban_blob(ny, nx, cx, cy, rad, rng):
    """Organic built-up footprint: a blob with an irregular (non-rectangular)
    boundary, so a settlement does not look like a rigid SimCity grid."""
    yy, xx = np.ogrid[:ny, :nx]
    dx = xx - cx
    dy = yy - cy
    ang = np.arctan2(dy, dx)
    ph = rng.uniform(0.0, 6.283, size=3)
    rmod = rad * (0.74 + 0.16 * np.sin(2 * ang + ph[0])
                  + 0.10 * np.sin(3 * ang + ph[1])
                  + 0.07 * np.sin(5 * ang + ph[2]))
    dist = np.sqrt(dx * dx + dy * dy)
    return dist <= np.maximum(rmod, 2.0)


def _paint_urban(world, x0, y0, x1, y1, urban_id, water_id,
                 load=0.6, moisture=0.06):
    """Paint a built-up block, but ONLY on land cells (never on water)."""
    ft = world.fuel.ftype
    ny, nx = ft.shape
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(nx - 1, int(x1)); y1 = min(ny - 1, int(y1))
    if x1 < x0 or y1 < y0:
        return
    sub = ft[y0:y1 + 1, x0:x1 + 1]
    landm = sub != water_id
    sub[landm] = urban_id
    world.fuel.fload[y0:y1 + 1, x0:x1 + 1][landm] = load
    world.fuel.fmoist[y0:y1 + 1, x0:x1 + 1][landm] = moisture
    if getattr(world.fuel, "fload0", None) is not None:
        world.fuel.fload0[y0:y1 + 1, x0:x1 + 1][landm] = load


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
                       precipitation: float = 0.0,
                       with_assets: bool = True,
                       with_roads: bool = True,
                       settlement_density: float = 0.5,
                       n_settlements: int | None = None,
                       population_per_settlement: int | None = None,
                       building_scale: float = 1.0,
                       accessibility: float | None = None,
                       preset: str | None = None,
                       coast: bool = False,
                       river: bool = False) -> World:
    """Generate a realistic World.

    preset          optional named landscape type (see PRESETS); overrides the
                    relief / forest / water / coast / river arguments
    relief_m        peak to valley elevation difference in meters
    forest_density  fraction of land covered by forest clusters (0 to 1)
    water_level     fraction of lowest cells turned into water (non fuel)
    population_per_settlement  TOTAL population of the map, split across
                    the settlements with a skewed share (town largest)
    settlement_density  0..1: how built-up the landscape is. Scales the
                    number of settlements (1..4), their footprint, their
                    population and the road connections, so studies can
                    vary human exposure (and therefore the cost terms)
                    while keeping the same terrain
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

    # relief: rolling fBm base + ridged mountains + gentle power-law valley
    # flattening, which reads like real topography (plains, foothills, ridges)
    base = fractal_noise(ny, nx, rng, octaves=6, persistence=0.55)
    ridge = 1.0 - np.abs(2.0 * fractal_noise(ny, nx, rng, octaves=5,
                                             persistence=0.5) - 1.0)
    field = 0.55 * base + 0.45 * ridge ** 1.6
    field = (field - field.min()) / max(field.max() - field.min(), 1e-9)
    elev = (field ** 1.25) * relief_m
    # erosion pass: carve connected, dendritic valleys along the drainage
    # network (raw fBm has isolated pits; real valleys are cut by water)
    elev, drain = _erode(elev, rng)
    gy, gx = np.gradient(elev, cfg.cell_size_m)
    slope = np.clip(np.arctan(np.hypot(gx, gy)), 0.0, 1.3)
    aspect = np.arctan2(-gy, -gx)
    elev_norm = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-9)

    grass = FUEL_NAME_TO_ID["grass"]
    shrub = FUEL_NAME_TO_ID["shrub"]
    pine = FUEL_NAME_TO_ID["pine_litter"]
    hard = FUEL_NAME_TO_ID["hardwood"]
    WATER = FUEL_NAME_TO_ID["water"]

    ftype = np.full((ny, nx), grass, dtype=int)
    # larger, smoother vegetation clusters look like real forest stands
    veg = fractal_noise(ny, nx, rng, octaves=3, persistence=0.55)
    veg = _gaussian_smooth(veg, max(nx, ny) / 60.0)
    veg = (veg - veg.min()) / max(veg.max() - veg.min(), 1e-9)
    # sun exposure by aspect: slopes whose downslope direction points to the
    # map south (+y) receive more sun, so they are drier and carry lighter
    # fuels; pole-facing slopes hold denser, moister forest (N hemisphere)
    _gmag = np.hypot(gx, gy) + 1e-9
    sun = 0.5 - 0.5 * (gy / _gmag) * np.clip(slope / 0.5, 0.0, 1.0)
    veg_eff = np.clip(veg + 0.35 * (0.5 - sun), 0.0, 1.2)
    ftype[(veg_eff > 0.45) & (elev_norm > 0.25)] = shrub
    forest_threshold = np.quantile(veg_eff,
                                   1.0 - np.clip(forest_density, 0.0, 0.95))
    forest = veg_eff >= forest_threshold
    # elevation zonation: moist broadleaf woods on the valley floors, a
    # conifer belt at mid elevations, subalpine scrub near the ridgelines
    ftype[forest & (elev_norm <= 0.30)] = hard
    ftype[forest & (elev_norm > 0.30) & (elev_norm <= 0.82)] = pine
    ftype[forest & (elev_norm > 0.82)] = shrub
    # sunny mid-slopes stay open (grass / shrub) even inside the forest
    # envelope, the classic south-slope garrigue pattern
    sunny = (sun > 0.58) & (slope > 0.10)
    ftype[sunny & forest & (elev_norm > 0.30) & (veg < 0.7)] = shrub
    ftype[sunny & (ftype == shrub) & (veg < 0.35)] = grass
    # shrub ecotone around forest edges (transition belt, as in nature)
    edge = _gaussian_smooth(forest.astype(float), 1.5)
    ftype[(edge > 0.15) & (edge < 0.5) & ~forest & (elev_norm > 0.2)] = shrub

    if water_level > 0:
        thr = np.quantile(elev_norm, np.clip(water_level, 0.0, 0.3))
        ftype[elev_norm <= thr] = WATER

    # coastline: sea along the eastern edge with a wavy boundary
    if coast:
        # descend the land toward the sea on the east/southeast, then flood the
        # sea inland below a sea level. The shoreline then follows the terrain,
        # producing an indented coast with bays, peninsulas and offshore
        # islands (an Aegean / Marmara-like shore) rather than a straight cliff.
        yy = np.arange(ny)[:, None]
        xx = np.arange(nx)[None, :]
        tilt = (0.7 * (xx / max(nx - 1, 1)) + 0.3 * (yy / max(ny - 1, 1))) ** 1.2
        elev = elev * (1.0 - 0.85 * tilt)
        elev_norm = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-9)
        below = elev_norm < 0.12
        sea = _flood_sea(below)
        ftype[sea] = WATER
        elev[sea] = 0.0
        elev_norm = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-9)

    # river: a single continuous watercourse that flows strictly downhill from
    # its headwater to the sea, a lake or the map edge (never uphill). Because
    # it is traced cell by cell it is always connected (no dashes).
    river_y = None
    if river:
        river_y = _carve_downhill_river(elev, ftype, WATER, rng)

    # water bodies must read as sea / lake / river: drop 1-3 cell puddles
    _prune_puddles(ftype, WATER, grass, min_cells=4)

    # refresh derived terrain fields (coast / river may have changed elev)
    elev_norm = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-9)
    _gy2, _gx2 = np.gradient(elev, cfg.cell_size_m)
    _slope2 = np.clip(np.arctan(np.hypot(_gx2, _gy2)), 0.0, 1.3)
    # riparian belt: land within a couple of cells of water carries moist
    # broadleaf vegetation, the green corridor seen along real streams
    _wm = (ftype == WATER)
    if _wm.any():
        rip = _dilate(_wm, max(1, min(nx, ny) // 100 + 1)) & ~_wm
        ftype[rip & (ftype >= 1) & (ftype <= 4) & (elev_norm < 0.6)] = hard
    # rock outcrops: on steep ground near the crests the soil is thin and
    # bare rock breaks the fuel continuity (natural firebreaks)
    rock = (elev_norm > 0.80) & (_slope2 > 0.70) & (veg < 0.55)
    ftype[rock & (ftype != WATER)] = 0

    # only burnable land carries fuel: water (and bare ground) have zero fuel
    # so a river / lake / the sea can never catch fire
    fload = np.where((ftype > 0) & (ftype != WATER),
                     0.55 + 0.45 * fractal_noise(ny, nx, rng, octaves=4), 0.0)
    fload = np.clip(fload, 0.0, 1.0)
    # shallow soils near the ridgelines carry thinner fuels; rock carries none
    fload *= 1.0 - 0.5 * np.clip((elev_norm - 0.75) / 0.25, 0.0, 1.0)
    fload[ftype == 0] = 0.0
    # old burn scars: a few irregular patches of early-succession fuel
    # (grass with reduced load), the way past fires leave their mark
    for _ in range(int(rng.integers(0, 3))):
        _bx = int(rng.uniform(0.1, 0.9) * nx)
        _by = int(rng.uniform(0.1, 0.9) * ny)
        _br = int(max(3, min(nx, ny) * rng.uniform(0.04, 0.10)))
        scar = _urban_blob(ny, nx, _bx, _by, _br, rng)
        scar &= (ftype >= 1) & (ftype <= 4)
        ftype[scar] = grass
        fload[scar] *= 0.35
    # valleys hold more moisture; riparian belts near water are greener/wetter
    # precipitation: scattered showers / a wet band across the map (mm/h).
    # Rain wets the fuel, so cells under rain start with higher moisture and
    # therefore resist ignition and spread.
    p_noise = fractal_noise(ny, nx, rng, octaves=3, persistence=0.5)
    p_noise = (p_noise - p_noise.min()) / max(p_noise.max() - p_noise.min(), 1e-9)
    prec_field = (np.clip((p_noise - 0.55) / 0.45, 0.0, 1.0)
                  * float(max(0.0, precipitation)))
    m_noise = fractal_noise(ny, nx, rng, octaves=4, persistence=0.5)
    m_noise = (m_noise - m_noise.min()) / max(m_noise.max() - m_noise.min(), 1e-9)
    moisture = (base_moisture + 0.10 * (1.0 - elev_norm)
                + 0.06 * (m_noise - 0.5)
                + 0.30 * np.clip(prec_field / 10.0, 0.0, 1.0)
                + 0.08 * (drain ** 0.7))   # damp valley floors (wetness index)
    wmask = (ftype == WATER).astype(float)
    if wmask.any():
        near_water = _gaussian_smooth(wmask, max(2.0, max(nx, ny) / 80.0))
        moisture = moisture + 0.10 * np.clip(near_water * 3.0, 0.0, 1.0)
    moisture = np.clip(moisture, 0.02, 0.5)

    world = World.blank(cfg)
    gy, gx = np.gradient(elev, cfg.cell_size_m)
    slope = np.clip(np.arctan(np.hypot(gx, gy)), 0.0, 1.3)
    aspect = np.arctan2(-gy, -gx)
    world.topo = TopoLayer(elev=elev, slope=slope, aspect=aspect,
                           access=np.clip((1.0 - slope / 1.3) * (1.0 if accessibility is None else float(np.clip(accessibility, 0.0, 1.0))), 0.05, 1.0))
    world.fuel = FuelLayer(ftype=ftype, fload=fload, fmoist=moisture)
    # wind is not global: a base flow plus fractal gustiness, stronger on
    # exposed high ground and eased in sheltered valleys, with the direction
    # wandering locally. Temperature and humidity also vary across the map.
    g_s = fractal_noise(ny, nx, rng, octaves=4, persistence=0.5)
    g_s = (g_s - g_s.min()) / max(g_s.max() - g_s.min(), 1e-9)
    g_d = fractal_noise(ny, nx, rng, octaves=4, persistence=0.5)
    g_d = (g_d - g_d.min()) / max(g_d.max() - g_d.min(), 1e-9)
    # the STORED field is synoptic wind + fractal gustiness only. The
    # terrain response (ridge speed-up, valley shelter, direction
    # channeling toward the slope axis) is applied by the ENGINE at every
    # step, so baking it into the stored field would double-count it.
    wws_field = np.clip(wind_speed * (0.7 + 0.6 * g_s), 0.0, None)
    world.meteo.wws[:] = wws_field
    world.meteo.wwd[:] = wind_dir_rad + 0.6 * (g_d - 0.5)
    world.meteo.gust[:] = wws_field * 1.4
    world.meteo.rh[:] = np.clip(world.meteo.rh + 18.0 * (g_s - 0.5), 5.0, 95.0)
    world.meteo.temp[:] = np.clip(world.meteo.temp + 6.0 * (0.5 - g_s)
                                  + 5.0 * (1.0 - elev_norm), 0.0, 55.0)
    world.meteo.prec[:] = prec_field

    # land-constrained road path: lowest LAND cell per column, so the road never
    # runs through the sea or the river
    land = (ftype >= 1) & (ftype <= 4)
    big = float(elev.max()) + 1e6
    elev_land = np.where(land, elev, big)
    road_y = _smooth1d(np.argmin(elev_land, axis=0).astype(float),
                       max(5, nx // 10))

    def _nearest_land_row(x, y):
        yi = int(np.clip(round(y), 0, ny - 1))
        if land[yi, x]:
            return yi
        for r in range(1, ny):
            for dy in (-r, r):
                yy = yi + dy
                if 0 <= yy < ny and land[yy, x]:
                    return yy
        return yi

    # a road runs BESIDE the river, not on top of it: hold a fixed buffer to
    # one bank so the two never share the same strip and never cross back and
    # forth. Snap the offset line to the nearest dry land.
    if river_y is not None:
        buf = max(2.0, ny / 16.0)
        for _x in range(nx):
            ry = river_y[_x]
            if not np.isnan(ry) and abs(road_y[_x] - float(ry)) < buf:
                cand = float(ry) - buf
                if not (0 <= int(round(cand)) < ny
                        and land[int(round(cand)), _x]):
                    cand = float(ry) + buf
                road_y[_x] = cand
        for _x in range(nx):
            road_y[_x] = _nearest_land_row(_x, road_y[_x])

    # limit the vertical step per column so the road bends like a real road
    for _x in range(1, nx):
        road_y[_x] = np.clip(road_y[_x], road_y[_x - 1] - 1.5,
                             road_y[_x - 1] + 1.5)
    road_y = _smooth1d(road_y, 7).round().astype(int)
    valid = land.any(axis=0)

    def _to_land(x, y):
        x = int(np.clip(x, 0, nx - 1)); y = int(np.clip(y, 0, ny - 1))
        if land[y, x]:
            return x, y
        for r in range(1, max(ny, nx)):
            for dy in (-r, r):
                yy = y + dy
                if 0 <= yy < ny and land[yy, x]:
                    return x, yy
        return x, y

    # settlements on accessible land near the road, kept clear of water.
    # settlement_density scales how many, how large and how populated.
    dens = float(np.clip(settlement_density, 0.0, 1.0))
    if n_settlements is not None:
        n_set = 0 if not with_assets else int(max(0, n_settlements))
    else:
        n_set = 0 if (not with_assets or dens <= 0.0) else max(
            1, int(round(dens * 4)))
    sites = _place_settlements(elev, slope, ftype, WATER, n_set, rng)
    if not sites:
        sites = [_to_land(int(nx * 0.6), ny // 2)]
    tx, ty = sites[0]

    # agricultural quilt: rectangular parcels of cultivated (cured-grass,
    # low-load) fuel on flat, low ground around the settlements, like the
    # field mosaics of real valley floors. Unselected parcels keep their
    # natural cover, so the quilt is broken rather than wall-to-wall.
    if with_assets and n_set > 0:
        par_h = max(3, ny // 40)
        par_w = max(4, nx // 30)
        flat_low = (slope < 0.15) & (elev_norm < 0.45) \
            & (ftype >= 1) & (ftype <= 4)
        near_set = np.zeros((ny, nx), dtype=bool)
        _oy, _ox = np.ogrid[:ny, :nx]
        _r2 = (max(nx, ny) * 0.18) ** 2
        for _sx, _sy in sites[:n_set]:
            near_set |= ((_ox - _sx) ** 2 + (_oy - _sy) ** 2) <= _r2
        for _py in range(0, ny - par_h, par_h + 1):
            for _px in range(0, nx - par_w, par_w + 1):
                _sl = (slice(_py, _py + par_h), slice(_px, _px + par_w))
                ok = flat_low[_sl] & near_set[_sl]
                if ok.mean() > 0.7 and rng.random() < 0.55:
                    m = (ftype[_sl] >= 1) & (ftype[_sl] <= 4)
                    ftype[_sl][m] = grass
                    world.fuel.fload[_sl][m] = 0.30 + 0.15 * float(rng.random())
                    world.fuel.fmoist[_sl][m] = np.clip(
                        world.fuel.fmoist[_sl][m] + 0.03, 0.02, 0.5)
                    if getattr(world.fuel, "fload0", None) is not None:
                        world.fuel.fload0[_sl][m] = world.fuel.fload[_sl][m]

    evac_xy = None
    if with_roads and n_set > 0:
        # one least-cost shortest-path tree rooted at the town. Water is
        # impassable (roads go AROUND lakes and the sea) and steep ground is
        # costly (roads follow valleys / low ground and switchback up slopes).
        cost = _road_cost_grid(elev, slope, ftype, WATER)
        dist, pvx, pvy = _dijkstra_field(cost, sites[0])
        net = np.zeros((ny, nx), dtype=bool)

        def _trace(gx, gy):
            gx, gy = int(gx), int(gy)
            if not np.isfinite(dist[gy, gx]):
                return False
            cx, cy = gx, gy
            while True:
                net[cy, cx] = True
                nxp = int(pvx[cy, cx])
                nyp = int(pvy[cy, cx])
                if nxp < 0:
                    break
                cx, cy = nxp, nyp
            return True

        for gxp, gyp in sites[1:n_set]:
            _trace(gxp, gyp)
        # a road always leaves the map: connect to the reachable border land
        # cell with the lowest routing cost and place the evacuation route
        # there, on the road that exits the map (never in the sea).
        border = np.zeros((ny, nx), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        border_land = border & land & np.isfinite(dist)
        if border_land.any():
            bd = np.where(border_land, dist, np.inf)
            by, bx = np.unravel_index(int(np.argmin(bd)), bd.shape)
            _trace(bx, by)
            evac_xy = (int(bx), int(by))
        # rasterize the tree as roads (with a little width), never on water
        ys, xs = np.where(net)
        for yy, xx in zip(ys.tolist(), xs.tolist()):
            world.add_road_disk(int(xx), int(yy), 1)
        # bridges are allowed: road cells over water stay roads (the cell
        # itself remains water / non-burnable, only crossing is possible)

    # ---- settlements as proper, land-only built-up areas --------------------
    _CIVIC = [("Hospital", 1.00), ("Power plant", 0.95),
              ("Water treatment", 0.88), ("Government office", 0.82),
              ("School", 0.78), ("Fire station", 0.72),
              ("Telecom tower", 0.66), ("Fuel depot", 0.70)]
    urban_id = FUEL_NAME_TO_ID["urban"]
    if with_assets and n_set > 0:
        land_nw = ftype != WATER
        # the population input is the TOTAL population of the whole map: it
        # is split across the settlements with a skewed share (the first
        # settlement is the town and takes the largest part, the rest are
        # villages of varying size) and the shares sum EXACTLY to the total
        if population_per_settlement is not None:
            _total = int(max(0, population_per_settlement))
            _w = np.sort(rng.random(n_set) ** 2 + 1e-3)[::-1]
            _w[0] = max(_w[0], 0.35 * _w.sum())   # the town is clearly largest
            _w = _w / _w.sum()
            _pops = [int(_total * float(v)) for v in _w]
            _pops[0] += _total - sum(_pops)          # exact sum
        else:
            _pops = [int(round((12000 if i == 0 else 3000)
                               * (0.6 + 0.9 * dens))) for i in range(n_set)]
        for i, (sx_, sy_) in enumerate(sites[:n_set]):
            main = i == 0
            pop = int(_pops[i])
            rad = int(np.clip(2 + (pop ** 0.5) / 14.0
                              * float(max(0.3, building_scale)),
                              2, max(3, min(nx, ny) // 5)))
            name = ("City" if (main and pop >= 20000)
                    else "Town" if main else f"Village {i}")
            x0, y0, x1, y1 = sx_ - rad, sy_ - rad, sx_ + rad, sy_ + rad
            if rad >= 4:
                blk = 5
                for by in range(y0, y1 + 1, blk):
                    for bx in range(x0, x1 + 1, blk):
                        _paint_urban(world, bx, by, bx + blk - 2,
                                     by + blk - 2, urban_id, WATER)
                for gx in range(x0 + blk, x1, 2 * blk):
                    world.add_road_rect(int(np.clip(gx, 0, nx - 1)), max(0, y0),
                                        int(np.clip(gx, 0, nx - 1)),
                                        min(ny - 1, y1))
                for gy in range(y0 + blk, y1, 2 * blk):
                    world.add_road_rect(max(0, x0), int(np.clip(gy, 0, ny - 1)),
                                        min(nx - 1, x1),
                                        int(np.clip(gy, 0, ny - 1)))
            else:
                _paint_urban(world, x0, y0, x1, y1, urban_id, WATER)
            cxl, cyl = _nearest_land(land_nw, sx_, sy_)
            world.add_asset(Asset(f"{name} residents", "population", cxl, cyl,
                                  radius=max(2, rad), value=1.0, population=pop))
            world.add_asset(Asset(f"{name} centre", "building", cxl, cyl,
                                  radius=max(1, rad // 2),
                                  value=1.0 if main else 0.7))
            if main:
                k = int(min(len(_CIVIC), rng.integers(5, 8)))
                idx = list(range(k))
            else:
                # at least one facility per village (school / fire station
                # scale); more with population
                hi = int(min(len(_CIVIC), 2 + pop // 4000))
                k = int(rng.integers(1, max(2, hi)))
                idx = list(rng.permutation(len(_CIVIC))[:k])
            ang0 = float(rng.uniform(0.0, 6.283))
            for j, t in enumerate(idx):
                fname, fval = _CIVIC[t]
                a = ang0 + j * (6.283 / max(len(idx), 1))
                r = rad * (0.35 + 0.55 * ((j % 3) / 2.0))
                fx, fy = _nearest_land(land_nw,
                                       sx_ + int(round(r * np.cos(a))),
                                       sy_ + int(round(r * np.sin(a))))
                world.add_asset(Asset(fname if main else f"{fname} ({name})",
                                      "critical", fx, fy, radius=1,
                                      value=float(fval)))
        ex, ey = (evac_xy if evac_xy is not None
                  else _nearest_land(land_nw, nx - 2, ty))
        world.add_asset(Asset("Evacuation route", "evac_route", ex, ey,
                              radius=0))
        # bridges are allowed: road cells over water stay roads (the cell
        # itself remains water / non-burnable, only crossing is possible)
    return world
