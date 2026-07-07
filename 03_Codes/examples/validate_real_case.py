"""Validate the simulator against a real historical fire.

Feeds a real landscape (DEM + fuel map), the observed weather and the real
ignition into the engine, runs it for the documented fire duration over
several random seeds, and scores the simulated burned area against the
observed final perimeter (Jaccard, Sorensen-Dice, hit rate, false alarm,
front position error). See 03_Codes/VALIDATION.md for where to download the
input data (EFFIS, SRTM/Copernicus DEM, CORINE, ERA5).

Inputs are GeoTIFF (needs rasterio) or plain .npy arrays that are already on
the simulation grid. All rasters must cover the same bounding box.

Example:
    python examples/validate_real_case.py \
        --dem dem.tif --fuel clc.tif --corine --burned observed.tif \
        --nx 300 --ny 200 --cell 30 --step-minutes 30 --hours 36 \
        --ignite 142,88 --wind-csv wind.csv --moisture 0.06 --seeds 5

wind.csv columns: minute,speed_ms,dir_deg  (direction the wind blows TOWARD,
math convention: 0 = east, 90 = north; leave out to use --wind-speed/dir).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disasteraware import Simulator, World, SimConfig
from disasteraware.gis import slope_aspect_from_dem, _read_resampled
from disasteraware.layers import TopoLayer, FuelLayer
from disasteraware.validation import (compare_masks, front_distance_errors,
                                      CORINE_TO_FUEL)


def _load(path: str, ny: int, nx: int, nearest: bool) -> np.ndarray:
    if path.endswith(".npy"):
        arr = np.load(path)
        if arr.shape != (ny, nx):
            raise SystemExit(f"{path}: shape {arr.shape} != grid ({ny},{nx}); "
                             "resample .npy inputs to the grid first")
        return arr.astype(float)
    return _read_resampled(path, ny, nx, nearest=nearest)


def _build_world(a) -> World:
    cfg = SimConfig(nx=a.nx, ny=a.ny, cell_size_m=a.cell,
                    step_minutes=a.step_minutes, max_steps=100000,
                    rng_seed=0)
    w = World.blank(cfg)
    dem = _load(a.dem, a.ny, a.nx, nearest=False)
    slope, aspect = slope_aspect_from_dem(dem, a.cell)
    w.topo = TopoLayer(elev=dem, slope=slope, aspect=aspect,
                       access=np.ones((a.ny, a.nx)))
    raw = _load(a.fuel, a.ny, a.nx, nearest=True).astype(int)
    if a.corine:
        ftype = np.zeros_like(raw)
        for code, fid in CORINE_TO_FUEL.items():
            ftype[raw == code] = fid
    else:
        ftype = np.clip(raw, 0, 6)
    w.fuel = FuelLayer(ftype=ftype,
                       fload=np.where(ftype > 0, 1.0, 0.0).astype(float),
                       fmoist=np.full((a.ny, a.nx), a.moisture))
    for pair in a.ignite:
        x, y = (int(v) for v in pair.split(","))
        w.add_ignition(x, y, step=0, radius=a.ignite_radius)
    return w


def _wind_series(a, n_steps):
    if a.wind_csv:
        rows = list(csv.DictReader(open(a.wind_csv)))
        pts = [(float(r["minute"]), float(r["speed_ms"]),
                np.radians(float(r["dir_deg"]))) for r in rows]
        out = []
        for k in range(n_steps):
            t = k * a.step_minutes
            last = pts[0]
            for p in pts:
                if p[0] <= t:
                    last = p
            out.append((last[1], last[2]))
        return out
    return [(a.wind_speed, np.radians(a.wind_dir))] * n_steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dem", required=True)
    ap.add_argument("--fuel", required=True)
    ap.add_argument("--corine", action="store_true",
                    help="fuel raster holds CORINE level-3 codes")
    ap.add_argument("--burned", required=True,
                    help="observed burned area raster (>0.5 = burned)")
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--cell", type=float, default=30.0)
    ap.add_argument("--step-minutes", type=float, default=30.0)
    ap.add_argument("--hours", type=float, required=True,
                    help="documented fire duration to simulate")
    ap.add_argument("--ignite", nargs="+", required=True,
                    help="ignition cells as x,y (grid indices)")
    ap.add_argument("--ignite-radius", type=int, default=1)
    ap.add_argument("--wind-speed", type=float, default=6.0)
    ap.add_argument("--wind-dir", type=float, default=0.0,
                    help="deg, math convention (0=east, 90=north), blows toward")
    ap.add_argument("--wind-csv", default=None)
    ap.add_argument("--moisture", type=float, default=0.06,
                    help="dead fuel moisture (mass fraction) during the event")
    ap.add_argument("--seeds", type=int, default=5,
                    help="stochastic spotting -> run several seeds")
    ap.add_argument("--out", default="validation_report")
    a = ap.parse_args()

    n_steps = int(round(a.hours * 60.0 / a.step_minutes))
    obs = _load(a.burned, a.ny, a.nx, nearest=True) > 0.5
    runs = []
    for seed in range(a.seeds):
        w = _build_world(a)
        w.config.rng_seed = seed
        sim = Simulator(w)
        sim.record_states = False
        winds = _wind_series(a, n_steps)
        for k in range(n_steps):
            ws, wd = winds[k]
            w.meteo.wws[:] = ws
            w.meteo.wwd[:] = wd
            sim.step()
        rep = compare_masks(sim.ever_burned, obs)
        rep.update(front_distance_errors(sim.ever_burned, obs, a.cell))
        runs.append((rep, sim.ever_burned.copy()))
        print(f"seed {seed}: dice={rep['dice']:.3f} jaccard={rep['jaccard']:.3f} "
              f"hit={rep['hit_rate']:.3f} FAR={rep['false_alarm']:.3f} "
              f"bias={rep['area_bias']:.2f} front_err={rep['mean_m']:.0f} m")

    keys = ["jaccard", "dice", "hit_rate", "false_alarm", "area_bias",
            "mean_m", "p90_m"]
    summary = {k: {"mean": float(np.mean([r[0][k] for r in runs])),
                   "sd": float(np.std([r[0][k] for r in runs]))} for k in keys}
    print("\nsummary over seeds:")
    for k, v in summary.items():
        print(f"  {k:12s} {v['mean']:8.3f} +/- {v['sd']:.3f}")

    with open(a.out + ".json", "w") as fh:
        json.dump({"summary": summary,
                   "runs": [r[0] for r in runs],
                   "n_steps": n_steps, "args": vars(a)}, fh, indent=2)

    # agreement image of the best-dice run: green hit, red false alarm,
    # blue miss (observed only)
    best = max(runs, key=lambda r: r[0]["dice"])[1]
    img = np.zeros((a.ny, a.nx, 3), dtype=np.uint8) + 24
    img[best & obs] = (46, 160, 67)
    img[best & ~obs] = (200, 55, 44)
    img[~best & obs] = (58, 110, 220)
    try:
        from PIL import Image
        sc = max(1, 900 // a.nx)
        Image.fromarray(img).resize((a.nx * sc, a.ny * sc),
                                    Image.NEAREST).save(a.out + ".png")
        print(f"\nwrote {a.out}.json and {a.out}.png "
              "(green=hit, red=false alarm, blue=missed)")
    except ImportError:
        print(f"\nwrote {a.out}.json (install Pillow for the image)")


if __name__ == "__main__":
    main()
