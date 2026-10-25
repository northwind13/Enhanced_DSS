r"""K7: spread model different in the evaluation world, planning on the nominal one.

Referee question: T3k to T3p perturb the suppression and the coating; the
spread model itself (wind speed, wind direction) is never wrong in the
world the planner forecasts. Here the live world carries a different wind
while every shadow forecast (pre-selection, gates, evFIS trials, no-harm
guard) plans on the NOMINAL wind of the scenario.

Perturbations of the live world (world.meteo):
  wind_x1.5    wind speed 1.5 times the nominal (8 -> 12 m/s)
  wind_rot_p30 wind direction rotated +30 degrees (counter-clockwise)
  wind_rot_m30 wind direction rotated -30 degrees

Arms per perturbation on the 150 worlds, all planning on the nominal wind:
  closed   F5Ev, the closed layer alone
  replay   F5Ev with the objects T6 admitted re-installed at their recorded
           minutes (the T4c replay reference)
  live     VODA, exp E9 arm LLMScreen, the whole loop run anew (model calls)

Resumable. Pairs with final/t1_reference and final/t6_voda.

  python experiments\benchmark\revision\k7_spread.py --arms closed,replay --workers 16
  python experiments\benchmark\revision\k7_spread.py --arms live --workers 6
"""
import argparse, csv, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, BENCH)
from open_space import campaign as C          # noqa: E402
import loo as L                               # noqa: E402
from dss import evaluate as EV                # noqa: E402

SRC = os.path.join(BENCH, "final", "t6_voda")
OUT = os.path.join(HERE, "k7_spread")
SCEN = ("S1", "S2", "S3", "S4", "S5")
import numpy as np
PERTURB = {   # (speed factor, direction offset in degrees) of the live world
    "wind_x1.5":    (1.5, 0.0),
    "wind_rot_p30": (1.0, 30.0),
    "wind_rot_m30": (1.0, -30.0),
}
DEFAULT = "wind_x1.5,wind_rot_p30,wind_rot_m30"

_NOMINAL = {}          # nominal wws/wwd arrays of the current world
_FACTORS = (1.0, 0.0)  # (speed factor, direction offset deg) of the live world
_orig_build = C.build_world
_orig_clone = EV.clone_sim


def _build_world_perturbed(seed, scenario):
    w = _orig_build(seed, scenario)
    _NOMINAL["wws"] = np.array(w.meteo.wws, dtype=float, copy=True)
    _NOMINAL["wwd"] = np.array(w.meteo.wwd, dtype=float, copy=True)
    f, off = _FACTORS
    w.meteo.wws[:] = _NOMINAL["wws"] * float(f)
    w.meteo.wwd[:] = _NOMINAL["wwd"] + np.deg2rad(float(off))
    return w


def _clone_nominal(sim, keep_snapshots=False):
    """Every shadow copy plans on the NOMINAL wind."""
    s2 = _orig_clone(sim, keep_snapshots=keep_snapshots)
    if _NOMINAL:
        s2.world.meteo.wws[:] = _NOMINAL["wws"]
        s2.world.meteo.wwd[:] = _NOMINAL["wwd"]
    return s2


def _install(name):
    global _FACTORS
    _FACTORS = PERTURB[name]
    C.build_world = _build_world_perturbed
    EV.clone_sim = _clone_nominal


def _job(args):
    name, arm, sc, seed, script = args
    try:
        _install(name)
        if arm == "live":
            row, _c, _a, _m = C.run_once("E9", sc, "LLMScreen", seed)
        else:
            row, _c, _a, _m = C.run_once("LOO", sc, "F5Ev", seed,
                                         script=script)
        return dict(perturb=name, arm=arm, scenario=sc, seed=seed,
                    n_installed=(len(_m) if arm == "live" else len(script)),
                    burned_ha=row["burned_ha"],
                    coated_ha=row.get("coated_ha", ""),
                    success=row["success"], pop_affected=row["pop_affected"],
                    j_phys=row["j_phys"], j_total=row["j_total"],
                    j_resp=row["j_resp"], rules_final=row["rules_final"],
                    wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, (f"K7/{name}/{arm}/{sc}/{seed}: {type(exc).__name__}: "
                      f"{exc}\n" + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturb", default=DEFAULT,
                    help="comma list of " + ",".join(PERTURB))
    ap.add_argument("--arms", default="closed,replay")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", default=",".join(SCEN))
    ap.add_argument("--seeds", default="201-230")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = L.load_admitted(args.src, exp="E9", arm="LLMScreen")
    names = [p for p in args.perturb.split(",") if p]
    arms = [a for a in args.arms.split(",") if a]
    scs = [s for s in args.scenarios.split(",") if s]
    seeds = C._parse_seeds(args.seeds)
    path = os.path.join(args.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["perturb"], r["arm"], r["scenario"], int(r["seed"]))
                    for r in csv.DictReader(f)}
    jobs = []
    for name in names:
        for arm in arms:
            for sc in scs:
                for sd in seeds:
                    if (name, arm, sc, sd) in done:
                        continue
                    script = list(recs.get((sc, sd), [])) if arm == "replay" \
                        else []
                    jobs.append((name, arm, sc, sd, script))
    print(f"K7: {len(jobs)} runs to do ({len(done)} done)", flush=True)
    fields = ["perturb", "arm", "scenario", "seed", "n_installed",
              "burned_ha", "coated_ha", "success", "pop_affected", "j_phys", "j_total",
              "j_resp", "rules_final", "wall_s"]
    t0 = time.time()
    n = 0

    def sink(res):
        nonlocal n
        row, err = res
        n += 1
        if err:
            print("ERR", err, flush=True)
            with open(os.path.join(args.out, "errors.log"), "a") as f:
                f.write(err + "\n")
            return
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[{n}/{len(jobs)}] {row['perturb']} {row['arm']} "
              f"{row['scenario']} s{row['seed']}: burned {row['burned_ha']} "
              f"ha, success {row['success']}, {row['wall_s']}s "
              f"(elapsed {time.time() - t0:.0f}s)", flush=True)

    if args.workers <= 1:
        for j in jobs:
            sink(_job(j))
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for res in pool.imap_unordered(_job, jobs):
                sink(res)
    print("done", flush=True)


if __name__ == "__main__":
    main()
