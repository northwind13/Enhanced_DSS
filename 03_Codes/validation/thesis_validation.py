"""Batch hindcast that produces the validation deliverables.

Runs the four documented Turkish fires (Manavgat 2021, Marmaris 2021,
Milas-Kemerkoy 2021, Canakkale-Kayadere 2023) through the same blind
hindcast pipeline as ``auto_validate.py`` and assembles, in one pass, the
exact objects the thesis reports:

    Table 5.4 case configuration and the simulated DURATION per case
    Table 5.5 agreement metrics per case (mean +/- sd over seeds)
    Figure 5.2 agreement maps (green correct burn, red overprediction,
               blue missed burn)
    Figure 5.3 simulated versus observed burned-area growth over time

The metric definitions are the standard wildfire-simulation overlap scores
(Jaccard/IoU, Sorensen-Dice, hit rate, false-alarm ratio, area bias, front
position error); the implementations live in ``disaster_phyengine.validation``
and are reused unchanged. Everything is written to
``validation/runs/thesis_<timestamp>/`` as CSV, JSON and PNG so the run is
reproducible and the numbers can be pasted straight into the document.

An ``offline`` self-test mode replaces the real downloads with a synthetic
reference so the whole chain (tables, figures, export, archive) can be
exercised without a NASA FIRMS key or internet. Offline numbers are a
DEMONSTRATION only and are labelled as such; they are not a real hindcast.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import numpy as np

from . import auto_validate as av
from disaster_phyengine.validation import (compare_masks,
                                           front_distance_errors,
                                           arrival_agreement)

# fixed reporting order and the Table 5.4 notes column (kept identical to the
# thesis so the produced table drops straight in)
THESIS_CASE_ORDER = ["manavgat2021", "marmaris2021", "milas2021",
                     "canakkale2023"]
CASE_NOTES = {
    "manavgat2021": "strong wind-driven runs",
    "marmaris2021": "coastal pine, complex terrain",
    "milas2021": "power-plant threat phase",
    "canakkale2023": "NE-wind case; strait closure",
}
OFFLINE_PRESET = {
    "manavgat2021": ("Mediterranean coast", 9.0, 30.0),
    "marmaris2021": ("Mediterranean coast", 7.0, 55.0),
    "milas2021": ("Mediterranean coast", 8.0, 20.0),
    "canakkale2023": ("Mediterranean coast", 10.0, 45.0),
}

# spread-validation criteria for a suppression-free model against satellite
# truth: coverage of the observed burn, front position error, and
# arrival-time (rate-of-spread) agreement. The area-overlap scores
# (Dice/Jaccard/false-alarm/area-bias) are NOT used here: a free-running
# model compared to a SUPPRESSED real fire overpredicts the final area, so
# those scores conflate the spread error with the missing suppression and
# the satellite undersampling.
METRIC_KEYS = ["hit_rate", "mean_m", "p90_m", "arrival_mae_h", "arrival_rho"]

# wind-direction rotation offsets for the input-uncertainty ensemble
# (Figure 5.4). Gridded reanalysis winds miss local channeling and the
# fire's own convection, the dominant input uncertainty of any hindcast.
WIND_OFFSETS = [-135, -90, -45, 0, 45, 90, 135, 180]

# the "how each number is computed" text shown on the page and archived with
# every run (A = simulated burned set, B = observed burned set)
METHODS_MD = (
    "**How the validation works.** Each case is a blind hindcast: the "
    "simulator is given only the real inputs of the documented fire "
    "(Copernicus GLO-30 terrain, ESA WorldCover fuel, hourly ERA5 weather "
    "at the ignition point, and the first NASA FIRMS overpass as the initial "
    "front) and is run for the recorded duration with no tuning on that "
    "fire.\n\n"
    "The simulation core models **free fire spread; it does not model "
    "suppression**, while the real fire was actively fought. A free run "
    "therefore overpredicts the FINAL burned area, so the classic "
    "area-overlap scores (Sorensen-Dice, Jaccard, false-alarm ratio, area "
    "bias) are NOT used: they would conflate the spread error with the "
    "missing suppression and with the satellite undersampling. The model is "
    "validated on what it actually claims, the **propagation of the fire "
    "front**, with three criteria ($A$ = simulated burn, $B$ = observed "
    "burn):\n\n"
    "| Criterion | Definition | What it checks |\n"
    "|---|---|---|\n"
    "| Coverage (POD) | $|A\\cap B|/|B|$ | fraction of the observed burn the "
    "model reproduced. Robust to the free-run overshoot. Target $>0.7$. |\n"
    "| Front position error | mean and 90th-percentile edge-to-edge "
    "distance between $\\partial A$ and $\\partial B$ (m) | how far the "
    "simulated front sits from the observed front |\n"
    "| Arrival-time agreement | over cells both burned, mean absolute "
    "difference between the simulated arrival time and the FIRMS "
    "first-detection time (h), and the Spearman rank correlation $\\rho$ of "
    "the arrival ORDER | the rate-of-spread test: does the front reach each "
    "place at about the right time, in the right order |\n\n"
    "Ember spotting is stochastic, so every case is run over several seeds "
    "and each criterion is reported as mean +/- standard deviation. "
    "Area-overlap scores are reported only when an official EFFIS/EMS "
    "perimeter supplies referee-grade area truth."
)


def _cell_km2(cell_m: float) -> float:
    return (cell_m * cell_m) / 1.0e6


def _aggregate(reps):
    """Mean +/- sd over seeds for each kept criterion, NaN-safe (the
    arrival scores are NaN when too few cells match)."""
    out = {}
    for k in METRIC_KEYS:
        vals = np.array([r.get(k, np.nan) for r in reps], dtype=float)
        if np.isfinite(vals).any():
            with np.errstate(all="ignore"):
                out[k] = {"mean": float(np.nanmean(vals)),
                          "sd": float(np.nanstd(vals))}
        else:
            out[k] = {"mean": float("nan"), "sd": float("nan")}
    return out


def _agreement_rgb(base, best_mask, obs_mask, ign=None, nx=None):
    """green = correct burn, red = overprediction, blue = missed burn."""
    img = base.copy()
    img[obs_mask] = (58, 110, 220)
    img[best_mask & obs_mask] = (46, 160, 67)
    img[best_mask & ~obs_mask] = (200, 55, 44)
    if ign is not None and nx:
        _i0 = ign[0] if isinstance(ign, list) else ign
        gx, gy = int(_i0[0]), int(_i0[1])
        rr = max(2, nx // 150)
        img[max(0, gy - rr):gy + rr + 1,
            max(0, gx - rr):gx + rr + 1] = (255, 235, 60)
    return img


def _observed_growth_km2(pts, case, nx, ny, cell, main_mask, t0, hours,
                         n_points=24):
    """Cumulative observed burned area (km^2) versus hours since ignition.

    Built from the FIRMS detection timestamps inside the main cluster; the
    curve keeps the real temporal shape and is scaled so its endpoint equals
    the final observed area |B|."""
    import datetime as dt
    ckm = _cell_km2(cell)
    mx = 111320.0 * math.cos(math.radians(0.5 * (case["south"]
                                                 + case["north"])))
    my = 110540.0

    def _ts(p):
        hhmm = int(p[3])
        return dt.datetime.fromisoformat(p[2]) + dt.timedelta(
            hours=hhmm // 100, minutes=hhmm % 100)

    seen = {}
    for p in pts:
        gx = int((p[0] - case["west"]) * mx / cell)
        gy = int((case["north"] - p[1]) * my / cell)
        if not (0 <= gx < nx and 0 <= gy < ny) or not main_mask[gy, gx]:
            continue
        th = (_ts(p) - t0).total_seconds() / 3600.0
        if th < 0 or th > hours:
            continue
        key = (gy, gx)
        if key not in seen or th < seen[key]:
            seen[key] = th
    ts = np.linspace(0.0, hours, n_points)
    if not seen:
        obs_area = float(main_mask.sum()) * ckm
        return {"t_h": ts.tolist(),
                "obs_km2": [obs_area * (t / hours) for t in ts]}
    arr = np.array(sorted(seen.values()))
    cum_cells = np.array([float((arr <= t).sum()) for t in ts])
    final_det = cum_cells[-1] if cum_cells[-1] > 0 else 1.0
    scale = (float(main_mask.sum()) / final_det) * ckm
    return {"t_h": ts.tolist(), "obs_km2": (cum_cells * scale).tolist()}


# --------------------------------------------------------------- online case
def _run_case_online(case_id, key, cell, stepm, seeds, cache_dir,
                     wind_ensemble=False, progress=None):
    case = dict(av.CASES[case_id])
    args = SimpleNamespace(cell=cell, step_minutes=stepm, seeds=seeds,
                           cache=cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    def _p(frac, msg):
        if progress:
            progress(frac, msg)

    _p(0.05, "terrain (Copernicus GLO-30 DEM)")
    dem, dlons, dlats = av._download_dem(case, cache_dir)
    _p(0.15, "fuel (ESA WorldCover)")
    wc, wlons, wlats = av._download_worldcover(case, cache_dir)
    nx, ny, lons, lats = av._grid(case, cell)
    _p(0.25, "fire truth (NASA FIRMS)")
    pts = av._download_firms(case, key, cache_dir) if key else []
    fmask, first, ign_cells = av._firms_mask_and_ignition(
        case, pts, nx, ny, lons, lats, cell) if pts else (None, None, [])
    if fmask is None or first is None:
        raise RuntimeError(f"{case_id}: no FIRMS ground truth (key needed "
                           "the first time a case is run)")
    case["t0_hour"] = first[4].hour
    t0 = first[4]
    iglat = case["north"] - (first[1] + 0.5) * cell / 110540.0
    iglon = case["west"] + (first[0] + 0.5) * cell / (
        111320.0 * math.cos(math.radians(iglat)))
    _p(0.35, "weather (hourly ERA5 at the fire)")
    weather = av._download_weather(case, cache_dir, lat=iglat, lon=iglon)
    ign = ign_cells if ign_cells else (nx // 2, ny // 2)

    demg = av._sample(dem, dlons, dlats, lons, lats)
    wcg = av._sample(wc, wlons, wlats, lons, lats).astype(int)
    ftype = np.zeros_like(wcg)
    for code, (fid, _ld) in av.WORLDCOVER_TO_FUEL.items():
        ftype[wcg == code] = case.get("tree_fuel", 3) if code == 10 else fid
    base = av._basemap(ftype, demg)

    ckm = _cell_km2(cell)
    frames = {"t_h": [], "sim_km2": []}

    def _frame(k, n, burned, ws_now):
        frames["t_h"].append(k * stepm / 60.0)
        frames["sim_km2"].append(float(burned.sum()) * ckm)

    def _cb(seed, k, n):
        if progress:
            progress(0.4 + 0.55 * (seed + k / n) / seeds,
                     f"simulating seed {seed + 1}/{seeds}")

    n_total = int(round(case["hours"] * 60.0 / stepm))
    obs_arr = av.firms_arrival_hours(case, pts, nx, ny, cell, fmask, t0)
    runs, shape = av.run_case(case, args, dem, (dlons, dlats),
                              wc, (wlons, wlats), weather, fmask, ign,
                              progress_cb=_cb, frame_cb=_frame,
                              frame_every=max(1, n_total // 24),
                              obs_arrival=obs_arr)
    best = max(runs, key=lambda r: r[0]["hit_rate"])[1]
    obs_growth = _observed_growth_km2(pts, case, nx, ny, cell, fmask, t0,
                                      case["hours"])
    metrics = _aggregate([r[0] for r in runs])

    ens = []
    if wind_ensemble:
        def _ecb(i, nmem, k, n):
            if progress:
                progress(0.95 + 0.05 * (i + k / n) / nmem,
                         f"wind ensemble member {i + 1}/{nmem}")
        members, _sh = av.run_wind_ensemble(
            case, args, dem, (dlons, dlats), wc, (wlons, wlats), weather,
            fmask, ign, offsets=WIND_OFFSETS, progress_cb=_ecb)
        ens = _ensemble_curve(members)

    return dict(
        case_id=case_id, label=av.CASES[case_id]["label"],
        bbox=_bbox_str(case), start=case["start"], hours=case["hours"],
        notes=CASE_NOTES.get(case_id, ""), n_seed=seeds, truth="firms",
        metrics=metrics, agreement=_agreement_rgb(base, best, fmask, ign, nx),
        growth={"t_h": frames["t_h"], "sim_km2": frames["sim_km2"],
                "obs_t_h": obs_growth["t_h"], "obs_km2": obs_growth["obs_km2"]},
        wind_ensemble=ens, obs_km2_final=float(fmask.sum()) * ckm)


def _ensemble_curve(members):
    """Sort wind-ensemble members by rotation offset and keep the scores
    used in Figure 5.4."""
    rows = [{"offset": float(m["offset_deg"]),
             "coverage": float(m["rep"]["hit_rate"]),
             "mean_m": float(m["rep"]["mean_m"])}
            for m in members]
    rows.sort(key=lambda r: r["offset"])
    return rows


# -------------------------------------------------------------- offline case
def _run_case_offline(case_id, cell, stepm, seeds, wind_ensemble=False,
                      progress=None):
    """Synthetic stand-in: a reference run is the 'observed' fire and the
    seeds are the same landscape under small wind perturbations. Exercises
    the full pipeline without any download. NOT a real hindcast."""
    from disaster_phyengine import terrain
    from disaster_phyengine import Simulator, SimConfig
    preset, wind, wdir = OFFLINE_PRESET.get(
        case_id, ("Mediterranean coast", 9.0, 30.0))
    nx, ny = 160, 100
    hours = float(av.CASES[case_id]["hours"])
    n_steps = max(6, int(round(hours * 60.0 / stepm)))
    ckm = _cell_km2(cell)

    def _mk(seed, wobble, moist=0.06, dxy=(0, 0)):
        cfg = SimConfig(nx=nx, ny=ny, cell_size_m=cell, step_minutes=stepm)
        w = terrain.generate_landscape(cfg, seed=7, preset=preset,
                                       with_assets=False, with_roads=False)
        w.fuel.fmoist[:] = moist
        w.set_uniform_wind(wind, np.radians(wdir + wobble))
        ix = int(np.clip(nx // 3 + dxy[0], 1, nx - 2))
        iy = int(np.clip(2 * ny // 3 + dxy[1], 1, ny - 2))
        w.add_ignition(ix, iy, step=0, radius=1)
        w.config.rng_seed = seed
        return Simulator(w)

    if progress:
        progress(0.1, "offline reference run")
    ref = _mk(0, 0.0)
    ref.record_states = False
    ref_growth = {"t_h": [], "km2": []}
    for k in range(n_steps):
        ref.step()
        ref_growth["t_h"].append((k + 1) * stepm / 60.0)
        ref_growth["km2"].append(float(ref.ever_burned.sum()) * ckm)
    obs = ref.ever_burned.copy()
    # observed arrival time (h) from the reference run, for the arrival-time
    # criterion; NaN outside the observed burn
    obs_arr = np.where(obs & (ref.first_ignition_step >= 0),
                       ref.first_ignition_step.astype(float) * (stepm / 60.0),
                       np.nan)

    base = np.zeros((ny, nx, 3), dtype=np.uint8) + 24
    runs = []
    frames = {"t_h": [], "sim_km2": []}
    for seed in range(seeds):
        if progress:
            progress(0.2 + 0.7 * seed / max(1, seeds),
                     f"offline seed {seed + 1}/{seeds}")
        # a real reanalysis wind is a few degrees off, the fuel moisture is
        # uncertain and the reported ignition is imprecise: perturb all three
        # so the demo shows realistic <1 overlap scores
        sgn = 1 if seed % 2 else -1
        s = _mk(seed + 1, (seed + 1) * 12.0 * sgn,
                moist=0.06 + 0.012 * (seed + 1),
                dxy=((seed + 1) * 4 * sgn, (seed + 1) * 3))
        s.record_states = False
        for k in range(n_steps):
            s.step()
            if seed == 0:
                frames["t_h"].append((k + 1) * stepm / 60.0)
                frames["sim_km2"].append(float(s.ever_burned.sum()) * ckm)
        rep = compare_masks(s.ever_burned, obs)
        rep.update(front_distance_errors(s.ever_burned, obs, cell))
        rep.update(arrival_agreement(s.first_ignition_step, obs_arr, stepm))
        runs.append((rep, s.ever_burned.copy()))
    best = max(runs, key=lambda r: r[0]["hit_rate"])[1]
    metrics = _aggregate([r[0] for r in runs])

    ens = []
    if wind_ensemble:
        for j, off in enumerate(WIND_OFFSETS):
            if progress:
                progress(0.9 + 0.1 * j / len(WIND_OFFSETS),
                         f"offline wind ensemble {j + 1}/{len(WIND_OFFSETS)}")
            se = _mk(100 + j, float(off))
            se.record_states = False
            for _k in range(n_steps):
                se.step()
            rep = compare_masks(se.ever_burned, obs)
            rep.update(front_distance_errors(se.ever_burned, obs, cell))
            ens.append({"offset_deg": off, "rep": rep,
                        "mask": se.ever_burned.copy()})
        ens = _ensemble_curve(ens)

    case = dict(av.CASES[case_id])
    return dict(
        case_id=case_id, label=av.CASES[case_id]["label"] + " (offline demo)",
        bbox=_bbox_str(case), start=case["start"], hours=hours,
        notes=CASE_NOTES.get(case_id, ""), n_seed=seeds, truth="demo",
        metrics=metrics, agreement=_agreement_rgb(base, best, obs, None, nx),
        growth={"t_h": frames["t_h"], "sim_km2": frames["sim_km2"],
                "obs_t_h": ref_growth["t_h"], "obs_km2": ref_growth["km2"]},
        wind_ensemble=ens, obs_km2_final=float(obs.sum()) * ckm)


def _bbox_str(case) -> str:
    return (f"{case['west']:.2f}-{case['east']:.2f} E, "
            f"{case['south']:.2f}-{case['north']:.2f} N")


# ------------------------------------------------------------------- figures
def _build_fig_5_2(cases, path):
    """2x2 agreement maps (Figure 5.2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    for ax, c in zip(axes.ravel(), cases):
        ax.imshow(c["agreement"])
        ax.set_title(c["label"], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    legend = [Patch(color=(46 / 255, 160 / 255, 67 / 255), label="correct burn"),
              Patch(color=(200 / 255, 55 / 255, 44 / 255), label="overprediction"),
              Patch(color=(58 / 255, 110 / 255, 220 / 255), label="missed burn")]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _build_fig_5_3(cases, path):
    """2x2 simulated vs observed burned-area growth (Figure 5.3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    for ax, c in zip(axes.ravel(), cases):
        g = c["growth"]
        if g["t_h"]:
            ax.plot(g["t_h"], g["sim_km2"], color="#c8372c", lw=1.8,
                    label="simulated")
        if g.get("obs_t_h"):
            ax.plot(g["obs_t_h"], g["obs_km2"], color="#3a6edc", lw=1.8,
                    ls="--", label="observed")
        ax.set_title(c["label"], fontsize=9)
        ax.set_xlabel("hours since ignition", fontsize=8)
        ax.set_ylabel("burned area (km$^2$)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _build_fig_5_4(cases, path):
    """Wind-direction ensemble: coverage (POD) per rotation offset, per case
    (Figure 5.4). One curve per case; the raw-reanalysis member is the
    0-degree point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have = [c for c in cases if c.get("wind_ensemble")]
    if not have:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for c in have:
        xs = [m["offset"] for m in c["wind_ensemble"]]
        ys = [m["coverage"] for m in c["wind_ensemble"]]
        ax.plot(xs, ys, marker="o", lw=1.6,
                label=c["label"].replace(" (offline demo)", ""))
    ax.axvline(0.0, color="0.6", lw=0.9, ls=":")
    ax.set_xlabel("wind-direction rotation offset (deg)")
    ax.set_ylabel("coverage (POD)")
    ax.set_title("Wind-direction ensemble (0 deg = raw reanalysis wind)",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


# -------------------------------------------------------------- table builders
def build_table_5_4(cases) -> List[Dict[str, str]]:
    rows = []
    for c in cases:
        rows.append({
            "Case": c["label"].replace(" (offline demo)", ""),
            "Bounding box (E, N)": c["bbox"],
            "Start date": c["start"],
            "Duration (h)": f"{c['hours']:g}",
            "Notes": c["notes"],
        })
    return rows


def _fmt(m, p=3):
    if m["mean"] != m["mean"]:            # NaN
        return "n/a"
    return f"{m['mean']:.{p}f} ± {m['sd']:.{p}f}"


def build_table_5_5(cases) -> List[Dict[str, str]]:
    rows = []
    for c in cases:
        m = c["metrics"]
        _rho = m["arrival_rho"]
        rows.append({
            "Case": c["label"].replace(" (offline demo)", ""),
            "Coverage (POD)": _fmt(m["hit_rate"]),
            "Front mean / p90 (m)": (f"{m['mean_m']['mean']:.0f} / "
                                     f"{m['p90_m']['mean']:.0f}"),
            "Arrival MAE (h)": _fmt(m["arrival_mae_h"], 2),
            "Arrival ρ": ("n/a" if _rho["mean"] != _rho["mean"]
                          else f"{_rho['mean']:.2f}"),
        })
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


# ------------------------------------------------------------------- driver
def run_thesis_validation(firms_key: str = "", cell: float = 90.0,
                          step_minutes: float = 30.0, seeds: int = 3,
                          cases: Optional[List[str]] = None,
                          offline: bool = False,
                          wind_ensemble: bool = True,
                          out_root: Optional[str] = None,
                          progress: Optional[Callable] = None) -> Dict:
    """Run the four cases and assemble the deliverables.

    ``progress(case_idx, n_cases, frac, message)`` is called throughout.
    Returns a dict with the two tables, the two figure paths, the per-case
    detail and the archive directory."""
    order = cases or THESIS_CASE_ORDER
    root = out_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "validation", "runs")
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = "offline" if offline else "firms"
    out_dir = os.path.join(root, f"thesis_{tag}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, cid in enumerate(order):
        def _pp(frac, msg, _i=i):
            if progress:
                progress(_i, len(order), frac, f"{av.CASES[cid]['label']}: {msg}")
        if offline:
            res = _run_case_offline(cid, cell, step_minutes, seeds,
                                    wind_ensemble=wind_ensemble, progress=_pp)
        else:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "validation", "cache", cid)
            res = _run_case_online(cid, firms_key, cell, step_minutes, seeds,
                                   cache_dir, wind_ensemble=wind_ensemble,
                                   progress=_pp)
        # per-case agreement PNG
        try:
            from PIL import Image
            Image.fromarray(res["agreement"]).save(
                os.path.join(out_dir, f"agreement_{cid}.png"))
        except Exception:
            pass
        # per-case wind-ensemble JSON (matches the CLI *_wind_ensemble.json)
        if res.get("wind_ensemble"):
            json.dump(res["wind_ensemble"],
                      open(os.path.join(out_dir, f"{cid}_wind_ensemble.json"),
                           "w"), indent=2)
        results.append(res)

    t54 = build_table_5_4(results)
    t55 = build_table_5_5(results)
    _write_csv(os.path.join(out_dir, "table_5_4_cases.csv"), t54)
    _write_csv(os.path.join(out_dir, "table_5_5_metrics.csv"), t55)

    fig52 = os.path.join(out_dir, "figure_5_2_agreement_maps.png")
    fig53 = os.path.join(out_dir, "figure_5_3_growth.png")
    try:
        _build_fig_5_2(results, fig52)
        _build_fig_5_3(results, fig53)
    except Exception as exc:
        fig52 = fig53 = None
        _write_note(out_dir, f"figure build failed: {exc}")
    fig54 = os.path.join(out_dir, "figure_5_4_wind_ensemble.png")
    try:
        if not _build_fig_5_4(results, fig54):
            fig54 = None
    except Exception as exc:
        fig54 = None
        _write_note(out_dir, f"figure 5.4 build failed: {exc}")

    report = {
        "stamp": stamp, "offline": offline,
        "settings": {"cell_m": cell, "step_minutes": step_minutes,
                     "seeds": seeds, "wind_ensemble": wind_ensemble},
        "table_5_4": t54, "table_5_5": t55,
        "cases": [{k: v for k, v in r.items() if k != "agreement"}
                  for r in results],
    }
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"),
              indent=2, default=float)
    open(os.path.join(out_dir, "methods.md"), "w", encoding="utf-8").write(
        METHODS_MD)

    return {
        "stamp": stamp, "offline": offline, "out_dir": out_dir,
        "table_5_4": t54, "table_5_5": t55,
        "fig_5_2": fig52, "fig_5_3": fig53, "fig_5_4": fig54,
        "methods": METHODS_MD, "cases": results,
    }


def _write_note(out_dir, text):
    try:
        open(os.path.join(out_dir, "NOTE.txt"), "a", encoding="utf-8").write(
            text + "\n")
    except Exception:
        pass
