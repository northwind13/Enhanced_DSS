# -*- coding: utf-8 -*-
r"""K4: hindcast of the four documented fires, 50 replicates, errors in
metres and hours, area ratio, under a STATED stopping rule.

Referee question: the 86 to 94% agreement hides how large the errors are and
how they were computed. The values of the article were produced by the
validation page of the app with its default stopping rule, "stop when the
simulated burn reaches the observed area" (one seed per fire). This run
repeats the hindcast from the same cached inputs with 50 seeds and reports
both rules side by side:

  --stop area   area-matched: the free burn stops when it reaches the
                observed footprint (the rule of the thesis and the article);
                the matched window in hours is reported with every score
  --stop none   full documented duration (48 or 72 h) with no suppression;
                the honest free-burn overprediction, reported as the ratio
                of simulated to observed area

Protocols:
  --protocol window12  (default) the observed footprint is the VIIRS detections
                     of the FIRST 12 HOURS of every fire, the first operational
                     period and the episode length of Section V; the free burn
                     stops when it reaches that area; no wind correction. Time
                     is compared by ARRIVAL ORDER (Spearman rho), which
                     suppression and detection latency stretch but do not
                     reorder; the hours the free burn needs to reach the
                     footprint are reported as the time compression.
  --protocol clean   documented duration (48, 48, 72, 48 h), ignition from the
                     first overpass, no wind correction
  --protocol thesis  the settings of the thesis runs: observed footprint and run
                     window 12 / 24 / 12 / 72 h, ERA5 wind direction rotated by
                     0 / +50 / 0 / -50 degrees (Manavgat, Milas)

Per fire and rule: probability of detection, front mean and p90 distance
(m), arrival-time MAE (h) with the window it is measured over, Spearman rho
of arrival order, simulated/observed area ratio. Nothing is tuned.

  python experiments\benchmark\revision\k4_hindcast.py --stop area --seeds 50
  python experiments\benchmark\revision\k4_hindcast.py --stop none --seeds 50

Resumable: every finished seed is saved under <out>/stop_<rule>/ckpt_<case>/,
an interrupted run continues from the first unfinished seed.
"""
import argparse, csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
import thesis_validation as TV                # noqa: E402
import auto_validate as AV                    # noqa: E402

OUT = os.path.join(HERE, "k4_hindcast")
LABEL = {"canakkale2023": "Sc1", "manavgat2021": "Sc2",
         "marmaris2021": "Sc3", "milas2021": "Sc4"}
ORDER = ["canakkale2023", "manavgat2021", "marmaris2021", "milas2021"]
# the settings of the thesis runs (validation/runs/*_20260721-23*): observed
# footprint and run window in hours, ERA5 wind-direction offset in degrees
THESIS = {"canakkale2023": dict(hours=12.0, offset=0.0),
          "manavgat2021": dict(hours=24.0, offset=50.0),
          "marmaris2021": dict(hours=12.0, offset=0.0),
          "milas2021": dict(hours=72.0, offset=-50.0)}
_orig_weather = AV._download_weather
_OFFSET = {"deg": 0.0}


def _weather_rotated(case, cache, lat=None, lon=None):
    wx = _orig_weather(case, cache, lat=lat, lon=lon)
    off = float(_OFFSET["deg"])
    if abs(off) > 1e-6:
        wx = dict(wx)
        wx["wind_direction_10m"] = [(float(d) + off) % 360.0
                                    for d in wx["wind_direction_10m"]]
    return wx

_orig_run_case = AV.run_case
_CKPT = {"dir": None, "stop": "none"}


def _run_case_checkpointed(case, args, dem, dem_ll, wc, wc_ll, weather,
                           obs_mask, ignition, **kw):
    """One seed at a time, each seed saved to disk as soon as it ends, so an
    interrupted run resumes without repeating finished seeds. Under the
    area-matched rule the stop area is the observed footprint."""
    import numpy as np
    if _CKPT["stop"] == "area" and kw.get("stop_area") is None:
        kw["stop_area"] = float(obs_mask.sum())
    cdir = os.path.join(_CKPT["dir"], f"ckpt_{case.get('label', 'case')}"
                        .replace(" ", "_").replace("/", "_"))
    os.makedirs(cdir, exist_ok=True)
    runs = []
    for seed in range(int(args.seeds)):
        jp = os.path.join(cdir, f"seed_{seed:03d}.json")
        mp = os.path.join(cdir, f"seed_{seed:03d}.npy")
        if os.path.exists(jp) and os.path.exists(mp):
            rep = json.load(open(jp, encoding="utf-8"))
            runs.append((rep, np.load(mp)))
            continue
        kw2 = dict(kw)
        if seed > 0:
            kw2["frame_cb"] = None
        one, _shape = _orig_run_case(case, args, dem, dem_ll, wc, wc_ll,
                                     weather, obs_mask, ignition,
                                     seed_list=[seed], **kw2)
        rep, mask = one[0]
        np.save(mp, mask)
        json.dump(rep, open(jp, "w", encoding="utf-8"), default=float)
        runs.append((rep, mask))
    nx, ny, _lons, _lats = AV._grid(case, args.cell)
    return runs, (nx, ny)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", default="area", choices=("area", "none"))
    ap.add_argument("--protocol", default="window12",
                    choices=("window12", "clean", "thesis"),
                    help="window12: observed footprint = detections of the "
                         "first 12 h of every fire (first operational period), "
                         "no wind offset, area-matched stop; clean: documented "
                         "duration (48/48/72/48 h), no offset; thesis: the "
                         "window and wind offset of the thesis runs "
                         "(12/24/12/72 h, 0/+50/0/-50 deg)")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--cell", type=float, default=90.0)
    ap.add_argument("--step", type=float, default=30.0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cases", default=",".join(ORDER))
    ap.add_argument("--firms-key", default=os.environ.get("FIRMS_MAP_KEY", ""),
                    help="NASA FIRMS MAP_KEY, needed once per case until the "
                         "detections are cached under validation/cache")
    args = ap.parse_args()
    out = os.path.join(args.out, f"{args.protocol}_stop_{args.stop}")
    cases = [c for c in args.cases.split(",") if c]
    if args.protocol == "thesis":
        if len(cases) != 1:
            ap.error("--protocol thesis takes one case per call (--cases X)")
        AV.CASES[cases[0]]["hours"] = THESIS[cases[0]]["hours"]
        _OFFSET["deg"] = THESIS[cases[0]]["offset"]
        AV._download_weather = _weather_rotated
    elif args.protocol == "window12":
        for c in cases:
            AV.CASES[c]["hours"] = 12.0
    os.makedirs(out, exist_ok=True)
    TV.METRIC_KEYS = ["hit_rate", "mean_m", "p90_m", "arrival_mae_h",
                      "arrival_rho", "area_bias", "stop_hours"]
    _CKPT["dir"] = out
    _CKPT["stop"] = args.stop
    AV.run_case = _run_case_checkpointed

    def prog(i, n, frac, msg):
        print(f"[{i + 1}/{n}] {frac * 100:5.1f}% {msg}", flush=True)
    # the FIRMS truth of every case is cached under validation/cache; the
    # driver only needs a non-empty key to read it
    res = TV.run_thesis_validation(
        firms_key=(args.firms_key or "cached"), cell=args.cell, step_minutes=args.step,
        seeds=args.seeds, cases=cases,
        offline=False, wind_ensemble=False, out_root=out, progress=prog)
    rows = []
    for d in res["cases"]:
        m = d["metrics"]

        def g(k, f="mean"):
            return float(m.get(k, {}).get(f, float("nan")))
        rows.append(dict(
            fire=LABEL.get(d["case_id"], d["case_id"]), case=d["case_id"],
            protocol=args.protocol, stop_rule=args.stop,
            wind_offset_deg=_OFFSET["deg"],
            n_seed=d.get("n_seed", args.seeds),
            duration_h=float(d.get("hours", float("nan"))),
            window_h=round(g("stop_hours"), 1),
            pod=round(g("hit_rate"), 3), pod_sd=round(g("hit_rate", "sd"), 3),
            front_mean_m=round(g("mean_m")),
            front_mean_sd=round(g("mean_m", "sd")),
            front_p90_m=round(g("p90_m")),
            arrival_mae_h=round(g("arrival_mae_h"), 2),
            arrival_mae_sd=round(g("arrival_mae_h", "sd"), 2),
            arrival_rho=round(g("arrival_rho"), 2),
            area_ratio_sim_obs=round(g("area_bias"), 2),
            area_ratio_sd=round(g("area_bias", "sd"), 2),
            obs_km2=round(float(d.get("obs_km2_final", float("nan"))), 1)))
    rows.sort(key=lambda r: r["fire"])
    tag = "_".join(LABEL.get(c, c) for c in args.cases.split(",") if c)
    path = os.path.join(out, f"hindcast_errors_{tag}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, f"result_{tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(rows=rows, out_dir=res.get("out_dir")), f, indent=1)
    print(f"\nprotocol {args.protocol}, stop rule {args.stop}, {args.seeds} seeds")
    print("fire  POD   front mean/p90 (m)  arrival MAE (h) over window (h)"
          "  area sim/obs")
    for r in rows:
        print(f"{r['fire']}  {r['pod']:.2f}  {r['front_mean_m']:>5} / "
              f"{r['front_p90_m']:<5}  {r['arrival_mae_h']:.2f} over "
              f"{r['window_h']:.1f} of {r['duration_h']:.0f}  "
              f"{r['area_ratio_sim_obs']:.2f}")
    print("written", path)


if __name__ == "__main__":
    main()
