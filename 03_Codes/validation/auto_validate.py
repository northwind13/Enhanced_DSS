"""Fully automatic hindcast validation against a real historical fire.

WHAT THIS VALIDATES
-------------------
The simulator is given ONLY the real inputs of a documented fire:
    terrain   Copernicus GLO-30 DEM (AWS open data, no account)
    fuel      ESA WorldCover 10 m land cover -> internal fuel classes
    weather   real hourly wind + temperature + humidity (open-meteo archive;
              humidity drives dead-fuel moisture through the EMC model)
    ignition  the first satellite fire detection (NASA FIRMS, free MAP_KEY)
It then simulates the documented burn duration BLIND (no tuning on this
fire) and scores the simulated burned area against the observed fire
footprint (FIRMS detections, or an official EFFIS/EMS perimeter raster if
you pass one). Metrics: Sorensen-Dice, Jaccard/IoU, hit rate, false alarm,
area bias, front position error. Outputs a JSON report and an agreement map
(green = correctly predicted burn, red = overprediction, blue = missed).

USAGE
-----
    python examples/auto_validate.py --case manavgat2021 --firms-key XXXX
    python examples/auto_validate.py --case manavgat2021 --firms-key XXXX \
        --burned my_effis_perimeter.tif        # stronger ground truth
    python examples/auto_validate.py --offline-demo   # no internet self-test

Get a free FIRMS MAP_KEY (1 minute): https://firms.modaps.eosdis.nasa.gov/api/map_key/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disaster_phyengine import Simulator, World, SimConfig
from disaster_phyengine.gis import slope_aspect_from_dem
from disaster_phyengine.layers import TopoLayer, FuelLayer
from disaster_phyengine.fuel_moisture import equilibrium_moisture
from disaster_phyengine.validation import compare_masks, front_distance_errors

# ---------------------------------------------------------------- case book
CASES = {
    # Antalya-Manavgat, started 2021-07-28, strong wind-driven runs.
    "manavgat2021": dict(west=31.30, south=36.72, east=31.80, north=37.08,
                         start="2021-07-28", hours=48.0, tree_fuel=3,
                         label="Manavgat (Antalya) 2021"),
    # Mugla-Marmaris, started 2021-07-29.
    "marmaris2021": dict(west=28.10, south=36.75, east=28.55, north=37.05,
                         start="2021-07-29", hours=48.0, tree_fuel=3,
                         label="Marmaris (Mugla) 2021"),
    # Mugla-Milas (Kemerkoy/Oren): the fire reached this coastal zone in
    # its Kemerkoy phase (the power-plant threat), 1-5 August 2021; the
    # 29 July starts were the Kavaklidere complex further northeast.
    "milas2021": dict(west=27.70, south=36.95, east=28.35, north=37.35,
                      start="2021-08-01", hours=72.0, tree_fuel=3,
                      label="Milas-Kemerkoy (Mugla) 2021"),
    # Canakkale (Kayadere), started 2023-08-22; strong NE wind case that
    # closed the Dardanelles strait.
    "canakkale2023": dict(west=26.50, south=39.90, east=27.00, north=40.20,
                          start="2023-08-22", hours=48.0, tree_fuel=3,
                          label="Canakkale-Kayadere 2023"),
}

# ESA WorldCover class -> (internal fuel id, fuel load). Irrigated cropland
# and built-up areas carry far less continuous fine fuel than wildland, which
# matters a lot on Mediterranean coastal plains.
WORLDCOVER_TO_FUEL = {10: (3, 1.0),   # tree cover -> pine litter (P. brutia)
                      20: (2, 1.0),   # shrubland (maquis)
                      30: (1, 0.9),   # grassland
                      40: (1, 0.35),  # cropland (largely irrigated)
                      50: (6, 0.5),   # built-up
                      60: (0, 0.0),   # bare / sparse
                      70: (0, 0.0),   # snow & ice
                      80: (5, 0.0),   # water
                      90: (0, 0.0),   # herbaceous wetland
                      95: (0, 0.0),   # mangroves
                      100: (1, 0.5)}  # moss & lichen


def _grid(case, cell_m):
    lat0 = 0.5 * (case["south"] + case["north"])
    mx = 111320.0 * math.cos(math.radians(lat0))   # m per deg lon
    my = 110540.0                                   # m per deg lat
    nx = int((case["east"] - case["west"]) * mx / cell_m)
    ny = int((case["north"] - case["south"]) * my / cell_m)
    lons = case["west"] + (np.arange(nx) + 0.5) * cell_m / mx
    lats = case["north"] - (np.arange(ny) + 0.5) * cell_m / my
    return nx, ny, lons, lats


def _sample(arr, a_lons, a_lats, lons, lats):
    """Nearest-neighbour sample of a lon/lat raster onto the grid."""
    xi = np.clip(np.searchsorted(a_lons, lons) - 0, 0, len(a_lons) - 1)
    # a_lats descending
    yi = np.clip(np.searchsorted(-a_lats, -lats), 0, len(a_lats) - 1)
    return arr[np.ix_(yi, xi)]


def build_real_world(bbox, cell_m, cache_dir, moisture=0.08, tree_fuel=3):
    """Build a ready-to-simulate World from a real-world bounding box using
    the same open data as the validation pipeline: Copernicus GLO-30 DEM +
    ESA WorldCover fuel. `bbox` needs keys west/south/east/north (deg).
    Downloads are cached in cache_dir; a cached area builds offline."""
    os.makedirs(cache_dir, exist_ok=True)
    dem, dlons, dlats = _download_dem(bbox, cache_dir)
    wc, wlons, wlats = _download_worldcover(bbox, cache_dir)
    nx, ny, lons, lats = _grid(bbox, cell_m)
    cfg = SimConfig(nx=nx, ny=ny, cell_size_m=cell_m)
    w = World.blank(cfg)
    demg = _sample(dem, dlons, dlats, lons, lats).astype(float)
    slope, aspect = slope_aspect_from_dem(demg, cell_m)
    w.topo = TopoLayer(elev=demg, slope=slope, aspect=aspect,
                       access=np.ones((ny, nx)))
    wcg = _sample(wc, wlons, wlats, lons, lats).astype(int)
    ftype = np.zeros((ny, nx), dtype=int)
    fload = np.zeros((ny, nx), dtype=float)
    for code, (fid, load) in WORLDCOVER_TO_FUEL.items():
        fid = tree_fuel if code == 10 else fid
        sel = wcg == code
        ftype[sel] = fid
        fload[sel] = load
    w.fuel = FuelLayer(ftype=ftype, fload=fload,
                       fmoist=np.full((ny, nx), float(moisture)))
    return w


# ------------------------------------------------------------- downloaders
def _download_dem(case, cache):
    """Copernicus GLO-30 DEM tiles from the AWS open-data bucket."""
    import rasterio
    path = os.path.join(cache, "dem_win.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["arr"], z["lons"], z["lats"]
    tiles = []
    for lat in range(int(math.floor(case["south"])), int(math.floor(case["north"])) + 1):
        for lon in range(int(math.floor(case["west"])), int(math.floor(case["east"])) + 1):
            name = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
            url = (f"https://copernicus-dem-30m.s3.amazonaws.com/"
                   f"{name}/{name}.tif")
            tiles.append((lat, lon, url))
    print(f"[dem] reading {len(tiles)} Copernicus GLO-30 tile(s) via range "
          "requests ...")
    arrs = {}
    for lat, lon, url in tiles:
        with rasterio.open("/vsicurl/" + url) as src:
            arrs[(lat, lon)] = src.read(1)
    n = next(iter(arrs.values())).shape[0]
    lat_rows = sorted({t[0] for t in arrs}, reverse=True)
    lon_cols = sorted({t[1] for t in arrs})
    mosaic = np.block([[arrs[(la, lo)] for lo in lon_cols] for la in lat_rows])
    lons = lon_cols[0] + (np.arange(mosaic.shape[1]) + 0.5) / n
    lats = lat_rows[0] + 1.0 - (np.arange(mosaic.shape[0]) + 0.5) / n
    np.savez_compressed(path, arr=mosaic.astype(np.float32),
                        lons=lons, lats=lats)
    return mosaic, lons, lats


def _download_worldcover(case, cache):
    """ESA WorldCover 2021 v200 window from the AWS open-data bucket."""
    import rasterio
    from rasterio.windows import from_bounds
    path = os.path.join(cache, "wc_win.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["arr"], z["lons"], z["lats"]
    la3 = int(math.floor(case["south"] / 3.0)) * 3
    lo3 = int(math.floor(case["west"] / 3.0)) * 3
    name = f"ESA_WorldCover_10m_2021_v200_N{la3:02d}E{lo3:03d}_Map"
    url = (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/"
           f"map/{name}.tif")
    print("[fuel] reading ESA WorldCover 2021 window via range requests ...")
    with rasterio.open("/vsicurl/" + url) as src:
        win = from_bounds(case["west"], case["south"], case["east"],
                          case["north"], src.transform)
        arr = src.read(1, window=win)
        tr = src.window_transform(win)
    ny, nx = arr.shape
    lons = tr.c + (np.arange(nx) + 0.5) * tr.a
    lats = tr.f + (np.arange(ny) + 0.5) * tr.e
    np.savez_compressed(path, arr=arr.astype(np.uint8), lons=lons, lats=lats)
    return arr, lons, lats


def _download_weather(case, cache, lat=None, lon=None):
    """Real hourly weather from the open-meteo ERA5 archive (no key).

    The weather is taken AT THE FIRE (ignition point) when known: winds at
    the bbox centre can sit in a different terrain regime."""
    import requests
    if lat is None:
        lat = 0.5 * (case["south"] + case["north"])
    if lon is None:
        lon = 0.5 * (case["west"] + case["east"])
    path = os.path.join(cache, f"weather_{lat:.2f}_{lon:.2f}.json")
    legacy = os.path.join(cache, "weather.json")
    if os.path.exists(path):
        return json.load(open(path))
    start = case["start"]
    end_days = int(math.ceil(case["hours"] / 24.0)) + 1
    import datetime as dt
    d0 = dt.date.fromisoformat(start)
    d1 = d0 + dt.timedelta(days=end_days)
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat:.4f}&longitude={lon:.4f}"
           f"&start_date={d0}&end_date={d1}"
           "&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,"
           "relative_humidity_2m&windspeed_unit=ms&timezone=UTC")
    print(f"[weather] fetching hourly ERA5 at {lat:.3f}, {lon:.3f} "
          "(open-meteo) ...")
    try:
        js = requests.get(url, timeout=60).json()["hourly"]
    except Exception as exc:
        if os.path.exists(legacy):
            print(f"[weather] fetch failed ({exc}); using the cached "
                  "bbox-centre series")
            return json.load(open(legacy))
        raise
    json.dump(js, open(path, "w"))
    return js


def _download_firms(case, key, cache):
    """VIIRS S-NPP detections (standard processing) for the fire window."""
    import requests, datetime as dt
    path = os.path.join(cache, f"firms_{case['start']}.csv")
    legacy = os.path.join(cache, "firms.csv")
    if (not os.path.exists(path) and os.path.exists(legacy)
            and sum(1 for _ in open(legacy)) > 1):
        os.replace(legacy, path)   # keep a previously downloaded archive
    if not os.path.exists(path):
        bbox = f"{case['west']},{case['south']},{case['east']},{case['north']}"
        d0 = dt.date.fromisoformat(case["start"])
        days = min(10, int(math.ceil(case["hours"] / 24.0)) + 2)
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
               f"VIIRS_SNPP_SP/{bbox}/{days}/{d0}")
        print("[truth] fetching NASA FIRMS VIIRS detections ...")
        txt = requests.get(url, timeout=120).text
        if txt.lstrip().lower().startswith("<") or "invalid" in txt[:200].lower():
            raise RuntimeError(
                "FIRMS returned an error - check your MAP_KEY "
                "(https://firms.modaps.eosdis.nasa.gov/api/map_key/)\n"
                + txt[:300])
        if len(txt.strip().splitlines()) <= 1:
            raise RuntimeError(
                "FIRMS returned ZERO detections for this case window "
                f"(bbox {bbox}, {d0} +{days} d). The fire was not burning "
                "there in that window - check the case definition, or the "
                "key hit its 10-minute quota (wait and retry). Query: "
                + url.replace(key, "<KEY>"))
        open(path, "w").write(txt)
    rows = list(__import__("csv").DictReader(open(path)))
    if not rows:
        raise RuntimeError("cached FIRMS file has no detections; delete "
                           f"{path} and rerun")
    pts = [(float(r["longitude"]), float(r["latitude"]),
            r["acq_date"], r["acq_time"]) for r in rows]
    print(f"[truth] {len(pts)} satellite fire detections")
    return pts


def _binary_close(mask, r):
    """Morphological closing (dilate then erode) with a (2r+1) square."""
    def _dil(m):
        out = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
        return out

    def _ero(m):
        out = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out &= np.roll(np.roll(m, dy, 0), dx, 1)
        return out

    out = mask.copy()
    for _ in range(int(r)):
        out = _dil(out)
    for _ in range(int(r)):
        out = _ero(out)
    return out


def _firms_mask_and_ignition(case, pts, nx, ny, lons, lats, cell_m):
    """Matched-window burned footprint + first-detection ignition.

    The ground truth mask keeps ONLY the detections inside the simulated
    time window [t0, t0 + hours], where t0 is the first detection: scoring
    a 48 h simulation against a 4 day footprint would be meaningless.
    Returns (mask, first) with first = (gx, gy, date, time, t0_datetime).
    """
    import datetime as dt

    def _ts(p):
        hhmm = int(p[3])
        return dt.datetime.fromisoformat(p[2]) + dt.timedelta(
            hours=hhmm // 100, minutes=hhmm % 100)

    def _largest_component(m):
        from collections import deque
        lab = np.zeros(m.shape, dtype=int)
        best, bestn = 0, 0
        cur = 0
        for sy, sx in zip(*np.where(m)):
            if lab[sy, sx]:
                continue
            cur += 1
            q = deque([(sy, sx)]); lab[sy, sx] = cur; n = 0
            while q:
                y, x = q.popleft(); n += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if (0 <= yy < m.shape[0] and 0 <= xx < m.shape[1]
                                and m[yy, xx] and not lab[yy, xx]):
                            lab[yy, xx] = cur; q.append((yy, xx))
            if n > bestn:
                best, bestn = cur, n
        return lab == best

    pts = sorted(pts, key=_ts)
    mask = np.zeros((ny, nx), dtype=bool)
    mx = 111320.0 * math.cos(math.radians(float(np.mean(lats))))
    my = 110540.0
    r = max(1, int(round(187.5 / cell_m)))   # half a VIIRS 375 m pixel
    first = None
    t0 = None
    tend = None
    for p in pts:
        lon, lat = p[0], p[1]
        gx = int((lon - case["west"]) * mx / cell_m)
        gy = int((case["north"] - lat) * my / cell_m)
        if not (0 <= gx < nx and 0 <= gy < ny):
            continue
        if first is None:
            first = (gx, gy, p[2], p[3], _ts(p))
            t0 = _ts(p)
            tend = t0 + dt.timedelta(hours=float(case["hours"]))
        if _ts(p) > tend:
            continue
        y0, y1 = max(0, gy - r), min(ny, gy + r + 1)
        x0, x1 = max(0, gx - r), min(nx, gx + r + 1)
        mask[y0:y1, x0:x1] = True
    # several separate fires can burn in the same bbox on the same days;
    # validate against the MAIN cluster only, and take the ignition as the
    # earliest detection inside that cluster
    main = _largest_component(mask)
    # detection swaths are sparse samples of a continuous burn: consolidate
    # the footprint with a morphological closing (~2 cells = one VIIRS px)
    main = _binary_close(main, max(1, int(round(375.0 / cell_m))))
    first = None
    ign_cells = []
    import datetime as _dt
    for p in pts:
        lon, lat = p[0], p[1]
        gx = int((lon - case["west"]) * mx / cell_m)
        gy = int((case["north"] - lat) * my / cell_m)
        if not (0 <= gx < nx and 0 <= gy < ny) or not main[gy, gx]:
            continue
        if first is None:
            first = (gx, gy, p[2], p[3], _ts(p))
        # initialize from the FULL first observation: every detection of the
        # first overpass (t0 + 20 min) is part of the initial fire front -
        # the fire had already grown before the satellite first saw it
        if _ts(p) <= first[4] + _dt.timedelta(minutes=20):
            ign_cells.append((gx, gy))
    return main, first, ign_cells


# ------------------------------------------------------------------ hindcast
def _met_to_math_toward(met_from_deg):
    return math.radians((270.0 - met_from_deg) % 360.0)


def run_case(case, args, dem, dem_ll, wc, wc_ll, weather, obs_mask,
             ignition, progress_cb=None, frame_cb=None, frame_every=8):
    cell = args.cell
    nx, ny, lons, lats = _grid(case, cell)
    cfg = SimConfig(nx=nx, ny=ny, cell_size_m=cell,
                    step_minutes=args.step_minutes, max_steps=10 ** 6)
    w = World.blank(cfg)
    demg = _sample(dem, dem_ll[0], dem_ll[1], lons, lats).astype(float)
    slope, aspect = slope_aspect_from_dem(demg, cell)
    w.topo = TopoLayer(elev=demg, slope=slope, aspect=aspect,
                       access=np.ones((ny, nx)))
    wcg = _sample(wc, wc_ll[0], wc_ll[1], lons, lats).astype(int)
    ftype = np.zeros((ny, nx), dtype=int)
    fload = np.zeros((ny, nx), dtype=float)
    for code, (fid, load) in WORLDCOVER_TO_FUEL.items():
        fid = case.get("tree_fuel", 3) if code == 10 else fid
        sel = wcg == code
        ftype[sel] = fid
        fload[sel] = load
    w.fuel = FuelLayer(ftype=ftype, fload=fload,
                       fmoist=np.full((ny, nx), 0.10))
    cells = ignition if isinstance(ignition, list) else [ignition]
    for gx, gy in cells:
        w.add_ignition(int(gx), int(gy), step=0, radius=0)

    n_steps = int(round(case["hours"] * 60.0 / args.step_minutes))
    ws = weather["wind_speed_10m"]; wd = weather["wind_direction_10m"]
    tt = weather["temperature_2m"]; rh = weather["relative_humidity_2m"]
    # the weather series starts at 00:00 UTC of the start date; the fire
    # starts at the first-detection hour, so offset the series accordingly
    h0 = int(case.get("t0_hour", 0))

    runs = []
    for seed in range(args.seeds):
        wr = World.from_dict(w.to_dict())
        wr.config.rng_seed = seed
        sim = Simulator(wr)
        sim.record_states = False
        for k in range(n_steps):
            h = min(h0 + int(k * args.step_minutes // 60), len(ws) - 1)
            wr.meteo.wws[:] = float(ws[h])
            wr.meteo.wwd[:] = _met_to_math_toward(float(wd[h]))
            wr.fuel.fmoist[:] = float(np.clip(
                equilibrium_moisture(np.array([tt[h]]), np.array([rh[h]]))[0],
                0.02, 0.35))
            sim.step()
            if progress_cb is not None:
                progress_cb(seed, k + 1, n_steps)
            if (frame_cb is not None and seed == 0
                    and ((k + 1) % max(1, int(frame_every)) == 0
                         or k + 1 == n_steps)):
                frame_cb(k + 1, n_steps, sim.ever_burned,
                         float(ws[min(h0 + int(k * args.step_minutes // 60),
                                      len(ws) - 1)]))
        rep = compare_masks(sim.ever_burned, obs_mask)
        rep.update(front_distance_errors(sim.ever_burned, obs_mask, cell))
        runs.append((rep, sim.ever_burned.copy()))
        print(f"  seed {seed}: dice={rep['dice']:.3f} "
              f"jaccard={rep['jaccard']:.3f} hit={rep['hit_rate']:.3f} "
              f"FAR={rep['false_alarm']:.3f} bias={rep['area_bias']:.2f} "
              f"front={rep['mean_m']:.0f} m")
    return runs, (nx, ny)


def _basemap(ftype, dem):
    """Simple fuel-coloured, hillshaded basemap so results sit on a map."""
    cols = {0: (120, 116, 108), 1: (196, 192, 120), 2: (150, 160, 92),
            3: (58, 104, 66), 4: (88, 128, 72), 5: (70, 130, 185),
            6: (168, 160, 152)}
    ny, nx = ftype.shape
    img = np.zeros((ny, nx, 3), dtype=float)
    for fid, c in cols.items():
        img[ftype == fid] = c
    gy, gx = np.gradient(dem.astype(float))
    sh = np.clip(1.0 - (gx + gy) / (np.abs(gx) + np.abs(gy) + 30.0), 0.55,
                 1.25)
    return np.clip(img * sh[..., None] * 0.55, 0, 255).astype(np.uint8)


def run_wind_ensemble(case, args, dem, dem_ll, wc, wc_ll, weather,
                      obs_mask, ignition, offsets, progress_cb=None,
                      frame_cb=None):
    """Input-uncertainty ensemble: rerun the hindcast with the wind
    direction rotated by fixed offsets (deg). Gridded reanalysis winds are
    the dominant input uncertainty for fire hindcasts (local channeling and
    the fire's own convection are unresolved), so the ensemble reports how
    the score responds to direction and which member matches best."""
    import types as _t
    members = []
    for i, off in enumerate(offsets):
        w2 = dict(weather)
        w2["wind_direction_10m"] = [(d + off) % 360.0
                                    for d in weather["wind_direction_10m"]]
        a2 = _t.SimpleNamespace(**vars(args))
        a2.seeds = 1

        def _pcb(seed, k, n, _i=i):
            if progress_cb is not None:
                progress_cb(_i, len(offsets), k, n)

        runs, shape = run_case(case, a2, dem, dem_ll, wc, wc_ll, w2,
                               obs_mask, ignition, progress_cb=_pcb,
                               frame_cb=None)
        rep, mask = runs[0]
        members.append({"offset_deg": off, "rep": rep, "mask": mask})
        print(f"  offset {off:+4.0f} deg: dice={rep['dice']:.3f} "
              f"hit={rep['hit_rate']:.3f} front={rep['mean_m']:.0f} m")
    members.sort(key=lambda m: -m["rep"]["dice"])
    return members, shape


def _report(runs, shape, obs_mask, out, base=None, ignition=None):
    nx, ny = shape
    keys = ["jaccard", "dice", "hit_rate", "false_alarm", "area_bias",
            "mean_m", "p90_m"]
    summary = {k: {"mean": float(np.mean([r[0][k] for r in runs])),
                   "sd": float(np.std([r[0][k] for r in runs]))}
               for k in keys}
    print("\n=== VALIDATION SUMMARY (mean +/- sd over seeds) ===")
    for k, v in summary.items():
        print(f"  {k:12s} {v['mean']:8.3f} +/- {v['sd']:.3f}")
    json.dump({"summary": summary, "runs": [r[0] for r in runs]},
              open(out + ".json", "w"), indent=2)
    best = max(runs, key=lambda r: r[0]["dice"])[1]
    img = (base.copy() if base is not None
           else np.zeros((ny, nx, 3), dtype=np.uint8) + 24)
    img[best & obs_mask] = (46, 160, 67)
    img[best & ~obs_mask] = (200, 55, 44)
    img[~best & obs_mask] = (58, 110, 220)
    if ignition is not None:
        gx, gy = int(ignition[0]), int(ignition[1])
        rr = max(2, nx // 150)
        img[max(0, gy - rr):gy + rr + 1, max(0, gx - rr):gx + rr + 1] = \
            (255, 235, 60)
    try:
        from PIL import Image
        sc = max(1, 1000 // nx)
        Image.fromarray(img).resize((nx * sc, ny * sc),
                                    Image.NEAREST).save(out + ".png")
        print(f"\nwrote {out}.json and {out}.png "
              "(green=hit, red=false alarm, blue=missed)")
    except ImportError:
        print(f"\nwrote {out}.json")


def _offline_demo(args):
    """Prove the whole chain without internet: synthetic 'real' case."""
    from disaster_phyengine import terrain
    print("[offline-demo] synthetic landscape stands in for the downloads")
    w = terrain.generate_landscape(SimConfig(nx=160, ny=100,
                                             step_minutes=30.0),
                                   seed=5, preset="Mediterranean coast",
                                   with_assets=False, with_roads=False)
    w.fuel.fmoist[:] = 0.06
    w.set_uniform_wind(9.0, np.radians(30))
    w.add_ignition(40, 60, step=0, radius=1)
    s = Simulator(w); s.record_states = False
    for _ in range(24):
        s.step()
    obs = s.ever_burned.copy()
    runs = []
    for seed in range(args.seeds):
        w2 = terrain.generate_landscape(SimConfig(nx=160, ny=100,
                                                  step_minutes=30.0),
                                        seed=5, preset="Mediterranean coast",
                                        with_assets=False, with_roads=False)
        w2.fuel.fmoist[:] = 0.06
        w2.set_uniform_wind(9.0, np.radians(30))
        w2.add_ignition(40, 60, step=0, radius=1)
        w2.config.rng_seed = seed
        s2 = Simulator(w2); s2.record_states = False
        for _ in range(24):
            s2.step()
        rep = compare_masks(s2.ever_burned, obs)
        rep.update(front_distance_errors(s2.ever_burned, obs, 30.0))
        runs.append((rep, s2.ever_burned.copy()))
        print(f"  seed {seed}: dice={rep['dice']:.3f}")
    _report(runs, (160, 100), obs, args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=sorted(CASES), default="manavgat2021")
    ap.add_argument("--firms-key", default=os.environ.get("FIRMS_MAP_KEY", ""))
    ap.add_argument("--burned", default=None,
                    help="optional official perimeter raster (tif/npy) as a "
                         "stronger ground truth than FIRMS detections")
    ap.add_argument("--cell", type=float, default=90.0,
                    help="grid cell size in m (90 m matches the satellite "
                         "truth resolution and keeps the run fast)")
    ap.add_argument("--step-minutes", type=float, default=30.0)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--wind-ensemble", action="store_true",
                    help="also rerun with the wind direction rotated over "
                         "the full circle (8 members) to quantify the "
                         "input-wind uncertainty")
    ap.add_argument("--cache", default=os.path.join("validation", "cache"))
    ap.add_argument("--out", default=os.path.join("validation", "runs",
                                                  "cli_report"))
    ap.add_argument("--offline-demo", action="store_true")
    args = ap.parse_args()

    if args.offline_demo:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        _offline_demo(args)
        return

    case = dict(CASES[args.case])
    if args.hours:
        case["hours"] = args.hours
    # per-case cache (switching cases must never reuse another fire's data);
    # migrate a pre-existing flat cache into the case folder once
    root_cache = args.cache
    args.cache = os.path.join(args.cache, args.case)
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for fn in ("dem_win.npz", "wc_win.npz", "weather.json", "firms.csv"):
        oldp = os.path.join(root_cache, fn)
        newp = os.path.join(args.cache, fn)
        if os.path.exists(oldp) and not os.path.exists(newp):
            os.replace(oldp, newp)
    print(f"=== Hindcast validation: {case['label']} ===")
    print(__doc__.split("USAGE")[0])

    dem, dlons, dlats = _download_dem(case, args.cache)
    wc, wlons, wlats = _download_worldcover(case, args.cache)

    nx, ny, lons, lats = _grid(case, args.cell)
    if not args.firms_key and not args.burned:
        raise SystemExit("Ground truth needed: pass --firms-key (free, "
                         "https://firms.modaps.eosdis.nasa.gov/api/map_key/) "
                         "or --burned perimeter.tif")
    pts = _download_firms(case, args.firms_key, args.cache) \
        if args.firms_key else []
    fmask, first, ign_cells = (None, None, [])
    if pts:
        fmask, first, ign_cells = _firms_mask_and_ignition(
            case, pts, nx, ny, lons, lats, args.cell)
        case["t0_hour"] = first[4].hour
        print(f"[ignition] first overpass {first[2]} {first[3]} UTC: "
              f"{len(ign_cells)} front cells initialize the fire; truth "
              f"window = first {case['hours']:g} h of detections")
    if args.burned:
        from disaster_phyengine.gis import _read_resampled
        obs = (np.load(args.burned) if args.burned.endswith(".npy")
               else _read_resampled(args.burned, ny, nx, nearest=True)) > 0.5
    else:
        obs = fmask
    ignition = ign_cells if ign_cells else (nx // 2, ny // 2)
    _igx, _igy = (first[0], first[1]) if first else (nx // 2, ny // 2)
    ig_lat = case["north"] - (_igy + 0.5) * args.cell / 110540.0
    ig_lon = case["west"] + (_igx + 0.5) * args.cell / (
        111320.0 * math.cos(math.radians(ig_lat)))
    weather = _download_weather(case, args.cache, lat=ig_lat, lon=ig_lon)

    runs, shape = run_case(case, args, dem, (dlons, dlats),
                           wc, (wlons, wlats), weather, obs, ignition)
    if args.wind_ensemble:
        print("\nwind-direction uncertainty ensemble:")
        members, _ = run_wind_ensemble(
            case, args, dem, (dlons, dlats), wc, (wlons, wlats), weather,
            obs, ignition, offsets=[-135, -90, -45, 0, 45, 90, 135, 180])
        best = members[0]
        print(f"best member: {best['offset_deg']:+.0f} deg  "
              f"dice={best['rep']['dice']:.3f}  "
              f"hit={best['rep']['hit_rate']:.3f}")
        json.dump([{"offset_deg": m["offset_deg"], **m["rep"]}
                   for m in members],
                  open(args.out + "_wind_ensemble.json", "w"), indent=2)
    demg = _sample(dem, dlons, dlats, lons, lats)
    wcg = _sample(wc, wlons, wlats, lons, lats).astype(int)
    ftype = np.zeros_like(wcg)
    for code, (fid, _ld) in WORLDCOVER_TO_FUEL.items():
        ftype[wcg == code] = case.get("tree_fuel", 3) if code == 10 else fid
    _report(runs, shape, obs, args.out, base=_basemap(ftype, demg),
            ignition=(first[0], first[1]) if first else None)


if __name__ == "__main__":
    main()
