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

from .config import SimConfig, FUEL_NAME_TO_ID, CROP_FUEL_LOADS
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


def _grade_shores(elev, ftype, water_id):
    """Bring the land down to the water instead of dropping it off a cliff.

    Only the water cells were being levelled. The land beside them kept
    whatever height the noise had given it, so a coastline came out as a
    plateau standing three hundred metres above the sea with a vertical
    wall between them, and the ground behind a lake could sit below the
    lake's own surface. On the 3D view it read exactly as what it was: a
    blue sheet pasted against the side of a hill.

    Every water body is taken separately, because a sea and a lake do not
    share a surface. Within a shore band whose width scales with the body,
    the land is drawn toward that body's level, reaching it at the water's
    edge; further inland the terrain is untouched. Nothing on land is left
    below the level of the water it drains into.
    """
    from collections import deque
    ny, nx = elev.shape
    wm = (ftype == water_id)
    if not wm.any():
        return elev
    seen = np.zeros_like(wm)
    for y0, x0 in zip(*np.where(wm)):
        if seen[y0, x0]:
            continue
        # ---- one water body
        seen[y0, x0] = True
        dq = deque([(y0, x0)])
        cells = []
        while dq:
            y, x = dq.popleft()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < ny and 0 <= xx < nx and wm[yy, xx]
                            and not seen[yy, xx]):
                        seen[yy, xx] = True
                        dq.append((yy, xx))
        if len(cells) < 6:
            continue                       # a river cell, not a shore
        level = float(np.median([elev[y, x] for y, x in cells]))
        width = float(np.clip(0.6 * np.sqrt(len(cells)), 3.0,
                              0.20 * min(nx, ny)))
        # ---- distance from this body, out to the shore band
        dist = np.full((ny, nx), np.inf)
        dq = deque()
        for y, x in cells:
            dist[y, x] = 0.0
            dq.append((y, x))
        while dq:
            y, x = dq.popleft()
            if dist[y, x] >= width:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < ny and 0 <= xx < nx
                        and dist[yy, xx] > dist[y, x] + 1.0):
                    dist[yy, xx] = dist[y, x] + 1.0
                    dq.append((yy, xx))
        band = np.isfinite(dist) & ~wm
        if not band.any():
            continue
        # the beach is at the water's level and the relief comes back with
        # distance; the exponent keeps the first cells gentle rather than
        # starting the climb at full slope
        w = np.clip(dist[band] / width, 0.0, 1.0) ** 0.75
        elev[band] = level + (elev[band] - level) * w
        # AND NOTHING DRAINS UPHILL. Land inside the band that was below
        # the water it sits beside is brought up to it: a coast cannot be
        # under sea level, which is what sea level means.
        elev[band] = np.maximum(elev[band], level)
    return elev


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


def _one_lake(elev, ftype, water_id, target_frac):
    """Fill ONE basin to the requested area instead of flooding every pit.

    Taking a global elevation quantile made water wherever the ground dipped,
    so a map came out as a lake district: eleven separate ponds on a 200x200
    grid at water_level 0.06, none of them big enough to matter and all of
    them in the way. A lake is one basin filling upward, so the deepest
    basin is seeded and grown over its lowest rim cell at a time until it
    holds the area asked for. What comes out is a single connected body of
    the size the slider requested.

    Returns (lake mask, water surface elevation). The SURFACE matters as
    much as the mask: water has no slope. Painting the cells blue and
    leaving the ground under them alone put a lake on the side of a hill,
    which is what it looked like in the 3D view: a blue stripe running
    downhill. The caller flattens the bed to the returned level, which is
    also what a real lake does to the land it covers.
    """
    import heapq
    ny, nx = elev.shape
    land = (ftype != water_id)
    target = int(round(float(np.clip(target_frac, 0.0, 0.3)) * ny * nx))
    lake = np.zeros((ny, nx), dtype=bool)
    if target <= 0 or not land.any():
        return lake, 0.0
    # the deepest piece of dry ground is where the water would collect
    _e = np.where(land, elev, np.inf)
    sy, sx = np.unravel_index(int(np.argmin(_e)), _e.shape)
    lake[sy, sx] = True
    seen = lake.copy()
    frontier = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = sy + dy, sx + dx
        if 0 <= yy < ny and 0 <= xx < nx and not seen[yy, xx]:
            seen[yy, xx] = True
            heapq.heappush(frontier, (float(elev[yy, xx]), yy, xx))
    n = 1
    level = float(elev[sy, sx])
    while frontier and n < target:
        _lv, y, x = heapq.heappop(frontier)
        lake[y, x] = True
        level = max(level, float(_lv))     # the rim it has risen to
        n += 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and not seen[yy, xx]:
                seen[yy, xx] = True
                heapq.heappush(frontier, (float(elev[yy, xx]), yy, xx))
    return lake, level


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


def _place_settlements(elev, slope, ftype, water_id, n, rng,
                       town_radius=6, water_pull=0.0):
    """Scatter settlements across the map with blue-noise spacing on SOLID land
    that is set back from water by a buffer, so a town never sits on or
    straddles the shoreline. Low, gentle ground is preferred.

    `town_radius` is how far the built block will reach from the site: the
    water buffer has to clear THAT, not just the centre cell, or a town
    placed four cells from a lake still paints itself into the water.

    `water_pull` is how much waterside ground is preferred, and it is OFF
    by default. Real towns do grow by water, but as a generator rule it
    made every town on every map sit on the same shoreline: measured at the
    original strength they were 6.1x closer to water than average ground
    (155 m against 950 m). A map is more useful when the towns are spread
    over it, so the term is kept for anyone who wants it and not applied.
    """
    ny, nx = elev.shape
    land = (ftype >= 1) & (ftype <= 4)
    if n <= 0 or not land.any():
        return []
    water = (ftype == water_id)
    # require a buffer of dry land around each site (shrink it only if the map
    # is too watery to fit any settlement otherwise)
    # HALF THE BLOCK, NOT ALL OF IT. Clearing the full town radius removed
    # the whole waterside band from the running, and with it every site the
    # historical preference is about: measured, towns then sat at exactly
    # the same distance from water as average ground and no amount of pull
    # changed it, because there was nothing near the water left to pick.
    # _paint_urban already refuses to paint a water cell, so the block can
    # meet the shore without standing in it.
    buf = max(3, int(town_radius) // 2)
    safe = land & ~_dilate(water, buf)
    while not safe.any() and buf > 1:
        buf -= 1
        safe = land & ~_dilate(water, buf)
    if not safe.any():
        safe = land
    # AND NOT AGAINST THE FRAME. A settlement on the border is drawn
    # half-off the map, its labels are clipped, half its protective ring
    # falls outside the domain and a fire reaching it leaves the world
    # instead of burning it. The margin scales with the map so it means the
    # same thing on any grid.
    _mrg = max(3, int(round(0.06 * min(nx, ny))))
    _inside = np.zeros_like(safe)
    _inside[_mrg:ny - _mrg, _mrg:nx - _mrg] = True
    if (safe & _inside).any():
        safe = safe & _inside
    en = (elev - elev.min()) / (float(elev.max() - elev.min()) + 1e-9)
    score = (1.0 - en) * (1.0 - np.clip(slope / 1.3, 0.0, 1.0))
    # historical siting: settlements grow near (but not on) water. A gentle
    # preference, so SOME towns are waterside and others are not.
    if water.any() and water_pull > 0.0:
        nw = _gaussian_smooth(water.astype(float), max(2.0, max(nx, ny) / 60.0))
        nw = nw / max(float(nw.max()), 1e-9)
        score = score * (1.0 + water_pull * np.clip(nw * 2.0, 0.0, 1.0))
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


#: the civic facilities a settlement may hold, most important first. One
#: definition, used by the generator and by the editor's settlement tool.
#: (name, protection value, scope). SCOPE is what keeps the map honest: a
#: power plant serves a region, not a hamlet, and two sparse villages on one
#: map were each being given their own. "regional" facilities are built at
#: most ONCE on a map and only in its main settlement; "local" ones are the
#: things a village really does have.
_CIVIC_FACILITIES = [("Hospital", 1.00, "regional"),
                     ("Power station", 0.95, "regional"),
                     ("Water works", 0.88, "regional"),
                     ("Town hall", 0.82, "regional"),
                     ("School", 0.78, "local"),
                     ("Fire station", 0.72, "local"),
                     ("Telecom tower", 0.66, "local"),
                     ("Fuel depot", 0.70, "local")]


def _facility_base(name: str) -> str:
    """The catalogue name behind a placed facility ("Hospital (Village 1)")."""
    return str(name or "").split(" (")[0].strip()


def place_settlement(world, cx, cy, pop, building_scale=1.0, rng=None,
                     main=False, name=None, label_index=1):
    """Build ONE settlement: the block, its streets, its people, its civic
    facilities. Shared by the generator and the map editor.

    The editor had no way to add a town: the Asset tool drops a single
    marker, but a settlement is a painted block of built-up ground with a
    street grid, a population spread over it and civic facilities around
    its centre. Writing that a second time in the editor would give the two
    paths different towns, which is the drift this file has repeatedly been
    corrected for, so it is written once and called from both.

    Returns the number of assets added.
    """
    rng = np.random.default_rng() if rng is None else rng
    ny, nx = np.asarray(world.fuel.ftype).shape
    urban_id = FUEL_NAME_TO_ID["urban"]
    WATER = FUEL_NAME_TO_ID["water"]
    land_nw = np.asarray(world.fuel.ftype) != WATER
    _CIVIC = _CIVIC_FACILITIES
    _n0 = len(world.assets)
    # the slider is a DENSITY: it scales the footprint AND how many
    # civic facilities a settlement carries. It used to touch only
    # the footprint, and it was clamped at 0.3, so asking for 0.2
    # gave 0.3 and every village still got its hospital.
    _dsc = float(max(0.0, building_scale))
    rad = int(np.clip(2 + (pop ** 0.5) / 14.0 * max(_dsc, 0.15),
                      2, max(3, min(nx, ny) // 5)))
    name = name or ("City" if (main and pop >= 20000)
                    else "Town" if main else f"Village {label_index}")
    x0, y0, x1, y1 = cx - rad, cy - rad, cx + rad, cy + rad
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
    cxl, cyl = _nearest_land(land_nw, cx, cy)
    world.add_asset(Asset(f"{name} residents", "population", cxl, cyl,
                          radius=max(2, rad), value=1.0, population=pop,
                          group=name))
    world.add_asset(Asset(f"{name} centre", "building", cxl, cyl,
                          radius=max(1, rad // 2),
                          value=1.0 if main else 0.7, group=name))
    # WHAT IS ALREADY ON THE MAP. A regional facility is built once: the
    # generator placed each settlement in isolation, so two villages a
    # kilometre apart each got a power plant and the map read as nonsense.
    _built = {_facility_base(getattr(_a, "name", ""))
              for _a in (getattr(world, "assets", []) or [])
              if getattr(_a, "kind", "") == "critical"}
    _avail = [i for i, (fn, _v, _sc) in enumerate(_CIVIC)
              if not (_sc == "regional"
                      and (not main or fn in _built))]
    if main:
        k = int(round(min(len(_avail), rng.integers(5, 8) * _dsc)))
        k = max(1, k)          # a town always has something civic
        idx = _avail[:k]
    else:
        # A HAMLET NEED NOT HAVE A HOSPITAL. Every village used to
        # be given at least one civic facility whatever its size or
        # the density asked for, which is how a map of twelve small
        # settlements ended up covered in critical-facility markers
        # even at the lowest density setting.
        hi = int(min(len(_avail), 2 + pop // 4000))
        k = int(round(rng.integers(0, max(2, hi)) * _dsc))
        idx = ([_avail[j] for j in rng.permutation(len(_avail))[:k]]
               if (k > 0 and _avail) else [])
    ang0 = float(rng.uniform(0.0, 6.283))
    for j, t in enumerate(idx):
        fname, fval, _scope = _CIVIC[t]
        a = ang0 + j * (6.283 / max(len(idx), 1))
        r = rad * (0.35 + 0.55 * ((j % 3) / 2.0))
        fx, fy = _nearest_land(land_nw,
                               cx + int(round(r * np.cos(a))),
                               cy + int(round(r * np.sin(a))))
        world.add_asset(Asset(fname if main else f"{fname} ({name})",
                              "critical", fx, fy, radius=1,
                              value=float(fval), group=name))
    return len(world.assets) - _n0


def settlements(world):
    """Every settlement on the map, as {group: summary}.

    The editor needs to talk about a TOWN, and the world only stores a flat
    list of assets. The group tag put on each part by place_settlement is
    what makes them one thing again.
    """
    out = {}
    for a in getattr(world, "assets", []) or []:
        g = str(getattr(a, "group", "") or "")
        if not g:
            continue
        d = out.setdefault(g, dict(name=g, x=int(a.x), y=int(a.y),
                                   radius=0, population=0.0, parts=0,
                                   facilities=0))
        d["parts"] += 1
        if a.kind == "population":
            d["population"] += float(getattr(a, "population", 0.0) or 0.0)
            d["radius"] = max(d["radius"], int(getattr(a, "radius", 0)))
            d["x"], d["y"] = int(a.x), int(a.y)      # the centre of the block
        elif a.kind == "critical":
            d["facilities"] += 1
    return out


def remove_settlement(world, group: str) -> int:
    """Take a settlement off the map: its people, its facilities AND the
    built-up ground it stands on.

    Deleting the assets alone left the urban block painted, so the map kept
    a town-shaped patch of built-up fuel with nobody in it: it still burned
    like a town, still steered the fire, and no longer cost anything when
    it did. The block is returned to the cover around it, taking its fuel
    load and moisture from the natural ground next to it rather than from a
    constant, so the patch does not read as a scar.

    The roads are left where they are. A road that ran through the town is
    also the road that runs PAST it, and cutting a hole in the network
    would strand whatever is on the far side.

    Returns the number of assets removed.
    """
    ft = np.asarray(world.fuel.ftype)
    ny, nx = ft.shape
    urban = FUEL_NAME_TO_ID["urban"]
    WATER = FUEL_NAME_TO_ID["water"]
    parts = [a for a in (getattr(world, "assets", []) or [])
             if str(getattr(a, "group", "") or "") == str(group)]
    if not parts:
        return 0
    rad = max([int(getattr(a, "radius", 0)) for a in parts] + [2])
    cx = int(round(float(np.mean([a.x for a in parts]))))
    cy = int(round(float(np.mean([a.y for a in parts]))))
    pad = rad + 2
    x0, x1 = max(0, cx - pad), min(nx - 1, cx + pad)
    y0, y1 = max(0, cy - pad), min(ny - 1, cy + pad)

    sub = ft[y0:y1 + 1, x0:x1 + 1]
    built = sub == urban
    if built.any():
        # what the neighbourhood is made of, so the hole matches its edges
        ring = ft[max(0, y0 - 4):min(ny, y1 + 5),
                  max(0, x0 - 4):min(nx, x1 + 5)]
        nat = ring[(ring >= 1) & (ring <= 4)]
        fill = int(np.bincount(nat.astype(int)).argmax()) if nat.size \
            else FUEL_NAME_TO_ID["grass"]
        _fl = np.asarray(world.fuel.fload)
        _fm = np.asarray(world.fuel.fmoist)
        _nm = (ft >= 1) & (ft <= 4) & (ft != urban)
        load = float(np.median(_fl[_nm])) if _nm.any() else 0.7
        moist = float(np.median(_fm[_nm])) if _nm.any() else 0.08
        sub[built] = fill
        _fl[y0:y1 + 1, x0:x1 + 1][built] = load
        _fm[y0:y1 + 1, x0:x1 + 1][built] = moist
        if getattr(world.fuel, "fload0", None) is not None:
            np.asarray(world.fuel.fload0)[y0:y1 + 1, x0:x1 + 1][built] = load
        # a cell that was water stays water: _paint_urban never built on it
        sub[ft[y0:y1 + 1, x0:x1 + 1] == WATER] = WATER

    world.assets = [a for a in world.assets
                    if str(getattr(a, "group", "") or "") != str(group)]
    world.rebuild_value_layers()
    return len(parts)


def move_settlement(world, group: str, cx: int, cy: int,
                    name: str | None = None, rng=None) -> int:
    """Pick a settlement up and put it down somewhere else.

    It is a remove followed by a build, not a shift of coordinates: the
    block of built-up ground, the street grid and the facility ring are
    painted onto the terrain and have to be unpainted from the old place
    and painted at the new one.
    """
    info = settlements(world).get(str(group))
    if info is None:
        return 0
    pop = float(info["population"])
    rad = max(2, int(info["radius"]))
    # the density that reproduces this footprint (place_settlement's own
    # radius law, solved for the scale), so a moved town is the same size
    _dsc = float(np.clip((rad - 2) * 14.0 / max(pop ** 0.5, 1e-6),
                         0.15, 4.0)) if pop > 0 else 1.0
    _main = any(a.kind == "building" and float(getattr(a, "value", 0)) >= 1.0
                for a in world.assets
                if str(getattr(a, "group", "")) == str(group))
    remove_settlement(world, group)
    return place_settlement(world, int(cx), int(cy), pop,
                            building_scale=_dsc,
                            rng=rng or np.random.default_rng(0),
                            main=_main, name=str(name or group))


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
                       farmland: bool = True,
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
        # ONE LAKE, NOT A SCATTER OF PONDS. See _one_lake: the level is an
        # area, and the area goes into the deepest basin.
        _lake, _lvl = _one_lake(elev, ftype, WATER, water_level)
        ftype[_lake] = WATER
        # AND THE SURFACE IS FLAT. Only the cover was being repainted, so
        # the ground under the lake kept its slope and the 3D view showed
        # water lying along a hillside. A lake has one surface elevation,
        # and the land it covers is under that surface.
        elev[_lake] = _lvl
        elev_norm = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-9)

    # coastline: sea along the eastern edge with a wavy, indented boundary.
    if coast:
        # A "sea potential" that RISES toward the east/south edge decides the
        # shore, not the raw elevation: multiplying the terrain down (the old
        # way) left inland valleys lower than the coast, so the sea never
        # reached the east. A wavy noise term indents the shoreline and low
        # coastal land extends the sea inland as bays; a fixed quantile
        # guarantees a real sea forms whatever the relief. _flood_sea then
        # keeps only the body connected to the border (no inland puddles).
        yy = np.arange(ny)[:, None]
        xx = np.arange(nx)[None, :]
        ramp = 0.72 * (xx / max(nx - 1, 1)) + 0.28 * (yy / max(ny - 1, 1))
        wob = 0.12 * (fractal_noise(ny, nx, rng, octaves=4,
                                    persistence=0.5) - 0.5)
        sea_pot = ramp + wob - 0.25 * elev_norm
        sea_level = float(np.quantile(sea_pot, 0.70))    # ~30% of map is sea
        sea = _flood_sea(sea_pot > sea_level)
        ftype[sea] = WATER
        elev[sea] = 0.0
        elev_norm = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-9)

    # THE LAND MEETS THE WATER AT THE WATER'S LEVEL. Both the sea and the
    # lake were levelled without touching the ground beside them, which put
    # a cliff at every shoreline.
    _grade_shores(elev, ftype, WATER)
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
    # the largest block the generator is about to paint, so the water
    # buffer clears it: the settlement radius below is
    # 2 + sqrt(pop)/14 * density, capped by the grid
    _pop_hi = float(population_per_settlement or 0) or 12000.0
    _rad_hi = int(np.clip(2 + (_pop_hi ** 0.5) / 14.0
                          * max(float(building_scale), 0.15),
                          2, max(3, min(nx, ny) // 5)))
    sites = _place_settlements(elev, slope, ftype, WATER, n_set, rng,
                               town_radius=_rad_hi)
    if not sites:
        sites = [_to_land(int(nx * 0.6), ny // 2)]
    tx, ty = sites[0]

    # AGRICULTURAL QUILT: rectangular parcels of cultivated (cured-grass,
    # low-load) fuel on flat, low ground around the settlements, like the
    # field mosaics of real valley floors. Unselected parcels keep their
    # natural cover, so the quilt is broken rather than wall-to-wall.
    #
    # It is a real land-cover class, not decoration: a worked field carries
    # roughly half the continuous fine fuel of natural grass (measured on a
    # generated map, 0.30-0.45 against 0.80) and a little more moisture, so
    # it slows a front the way farmland does. The GIS import says the same
    # thing about the WorldCover cropland class. It is switchable because
    # the hard-edged blocks read as an artefact on a wildland scenario
    # where no one expects fields.
    if with_assets and n_set > 0 and farmland:
        # ITS OWN RANDOM STREAM. The parcel loop used to draw from the main
        # rng, so switching farmland off did not merely remove the fields:
        # every draw after it shifted and the settlements came out with
        # different facilities on a map that was supposed to differ in one
        # respect only. A separate stream keeps the switch to one effect.
        # DERIVED FROM THE SEED, NOT FROM hash(). Python randomises the hash
        # of a string per process, so hashing a tag here made the same seed
        # produce a different map on every run: the one thing a seed exists
        # to prevent.
        _frng = np.random.default_rng((int(seed) * 2654435761 + 12345)
                                      % (2 ** 32))
        par_h = max(3, ny // 40)
        par_w = max(4, nx // 30)
        # AN IRREGULAR EDGE. A parcel painted as a clean rectangle reads as
        # something stamped on the map rather than grown on it, and on a
        # pale grass background the blocks stood out as squares the eye
        # could not explain. Real field boundaries follow ditches, tracks
        # and the lie of the land, so the edge is broken with a noise field
        # of its own: the same seed, so a map is still reproducible.
        _fedge = fractal_noise(ny, nx, _frng, octaves=4, persistence=0.55)
        _fedge = (_fedge - _fedge.min()) / max(float(np.ptp(_fedge)), 1e-9)
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
                # THE FIELD FOLLOWS THE GROUND, NOT THE GRID. The parcel was
                # painted wall to wall once the block was mostly flat, so the
                # steep corner of it was cultivated too; and demanding that
                # 70% of a block qualify meant that on a map with 450 m of
                # relief, where barely a tenth of the ground is flat enough
                # to work, no parcel qualified at all and the class silently
                # disappeared. Half a block is enough to call it farmland,
                # and only the workable cells inside it are sown.
                if ok.mean() > 0.5 and _frng.random() < 0.55:
                    # A FIELD IS A FIELD, WITH AN EDGE. Cutting 30% of the
                    # block away along the noise broke the parcels up until
                    # they read as mottling rather than as farmland; the
                    # variety belongs in the COLOUR of each parcel, which
                    # is what a field mosaic actually looks like from the
                    # air. A tenth is enough to stop the sides being drawn
                    # with a ruler.
                    _cut = float(np.quantile(_fedge[_sl], 0.10))
                    m = ((ftype[_sl] >= 1) & (ftype[_sl] <= 4) & ok
                         & (_fedge[_sl] > _cut))
                    # A FIELD IS WORKED IN ONE PIECE. Where the ground
                    # inside a block is broken up, keeping only the
                    # qualifying cells left one- and two-cell scraps, and
                    # since each parcel is drawn in its own colour, those
                    # scraps rendered as confetti. A block that cannot hold
                    # a field is left as it is.
                    if int(m.sum()) < int(0.45 * m.size):
                        continue
                    ftype[_sl][m] = grass
                    # ONE value off the crop ladder per parcel: the
                    # renderer derives the field colour from this number,
                    # so a per-cell draw would speckle the parcel in five
                    # colours at once, and a value off the ladder would let
                    # wild grass be taken for a field.
                    world.fuel.fload[_sl][m] = float(
                        CROP_FUEL_LOADS[int(_frng.integers(
                            0, len(CROP_FUEL_LOADS)))])
                    world.fuel.fmoist[_sl][m] = np.clip(
                        world.fuel.fmoist[_sl][m] + 0.03, 0.02, 0.5)
                    if getattr(world.fuel, "fload0", None) is not None:
                        world.fuel.fload0[_sl][m] = world.fuel.fload[_sl][m]

    evac_xy = None
    _road_exits = []
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
        # SEVERAL ROADS LEAVE THE MAP, NOT ONE. A single exit made every
        # network a dead-end tree hanging off one corner: traffic could only
        # ever go one way out, and a fire across that road cut the domain
        # off entirely. Real road networks pass THROUGH a place and carry on.
        # One exit is taken per map side where the cost allows it, so the
        # network crosses the frame instead of terminating at it.
        _exits = []
        if border_land.any():
            _sides = [(border_land & (np.arange(ny)[:, None] == 0)),
                      (border_land & (np.arange(ny)[:, None] == ny - 1)),
                      (border_land & (np.arange(nx)[None, :] == 0)),
                      (border_land & (np.arange(nx)[None, :] == nx - 1))]
            for _sm in _sides:
                if not _sm.any():
                    continue
                _bd = np.where(_sm, dist, np.inf)
                if not np.isfinite(_bd).any():
                    continue
                _by, _bx = np.unravel_index(int(np.argmin(_bd)), _bd.shape)
                if _trace(_bx, _by):
                    _exits.append((int(_bx), int(_by)))
        if _exits:
            # the cheapest exit is the one the town actually uses, so the
            # primary evacuation route is placed there
            _exits.sort(key=lambda p: float(dist[p[1], p[0]]))
            evac_xy = _exits[0]
            _road_exits = list(_exits)
        # rasterize the tree as roads (with a little width), never on water
        ys, xs = np.where(net)
        for yy, xx in zip(ys.tolist(), xs.tolist()):
            world.add_road_disk(int(xx), int(yy), 1)
        # bridges are allowed: road cells over water stay roads (the cell
        # itself remains water / non-burnable, only crossing is possible)

    # ---- settlements as proper, land-only built-up areas --------------------

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
            place_settlement(world, sx_, sy_, int(_pops[i]),
                             building_scale=building_scale, rng=rng,
                             main=(i == 0), label_index=i)
        ex, ey = (evac_xy if evac_xy is not None
                  else _nearest_land(land_nw, nx - 2, ty))
        world.add_asset(Asset("Evacuation route", "evac_route", ex, ey,
                              radius=0))
        # MORE THAN ONE WAY OUT. Every road that leaves the map is a way
        # out of it, and a town with a single exit is one road-cut away
        # from having none: the alternates are marked so the evacuation
        # has somewhere to send people when the primary is cut off.
        for _i2, (_ex2, _ey2) in enumerate(
                [p for p in (_road_exits or []) if p != (ex, ey)][:2],
                start=2):
            world.add_asset(Asset(f"Evacuation route {_i2}", "evac_route",
                                  int(_ex2), int(_ey2), radius=0))
        # bridges are allowed: road cells over water stay roads (the cell
        # itself remains water / non-burnable, only crossing is possible)
    # WHAT THE MAP DRAWS AS A TOWN MUST BE WORTH SOMETHING. The built-up
    # footprint is painted far wider than the discs of the placed assets, so
    # without this the loss model saw 90% of the settlement as empty ground.
    world.seed_builtup_value()
    # and the people live across the town, not in a circle around its label
    world.spread_population_over_builtup()
    return world
