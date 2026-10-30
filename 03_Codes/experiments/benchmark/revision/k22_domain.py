# -*- coding: utf-8 -*-
r"""K22: larger domain. The same S1-S5 worlds generated on a 120 x 90 grid
(3.6 x 2.7 km, 972 ha, 3.2 times the article domain), so that the burned
area of a free-running fire is no longer bounded by the edge of the map.
Everything else as in the main campaign: same presets, settlement counts,
fuel moisture, wind, ignition rule, pool rule, decision interval, horizons,
gates and budget.

Referee question: the 2.16 x 1.50 km worlds truncate growth and inflate the
extinguished fraction. Arms on 30 worlds (S1-S5, seeds 201-206):
  closed   F5Ev, the closed layer alone
  live     GODA, exp E9 arm LLMScreen (model calls)

Pairs within itself (live against closed). Resumable. DSS_K22_NX and
DSS_K22_NY override the grid.

  python experiments\benchmark\revision\k22_domain.py --arms closed --workers 16
  python experiments\benchmark\revision\k22_domain.py --arms live --workers 6
"""
import argparse, csv, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, BENCH)
from open_space import campaign as C          # noqa: E402
import campaign5 as C5                        # noqa: E402
from disaster_phyengine.config import SimConfig   # noqa: E402
from disaster_phyengine import terrain        # noqa: E402
import numpy as np

OUT = os.path.join(HERE, "k22_domain")
SCEN = ("S1", "S2", "S3", "S4", "S5")
NX = int(os.environ.get("DSS_K22_NX", "120"))
NY = int(os.environ.get("DSS_K22_NY", "90"))


def _build_world_large(seed, scenario):
    """campaign5.build_world with the larger grid, otherwise identical."""
    cfg = SimConfig(nx=NX, ny=NY, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    if scenario in C5.WUI:
        w = terrain.generate_landscape(
            cfg, seed=seed, preset="Rolling hills", n_settlements=12,
            population_per_settlement=8000)
    else:
        w = terrain.generate_landscape(
            cfg, seed=seed, preset="Mountain forest", n_settlements=2,
            population_per_settlement=4000)
    w.fuel.fmoist[:] = 0.08
    w.meteo.wws[:] = 8.0
    return w


def _install():
    C5.build_world = _build_world_large     # campaign.build_world delegates here for S1-S5


def _job(args):
    arm, sc, seed = args
    try:
        _install()
        if arm == "live":
            row, _c, _a, _m = C.run_once("E9", sc, "LLMScreen", seed)
        else:
            row, _c, _a, _m = C.run_once("E2P", sc, "F5Ev", seed)
        return dict(perturb=f"domain_{NX}x{NY}", arm=arm,
                    scenario=sc, seed=seed,
                    n_installed=(len(_m) if arm == "live" else 0),
                    burned_ha=row["burned_ha"],
                    coated_ha=row.get("coated_ha", ""),
                    success=row["success"], pop_affected=row["pop_affected"],
                    j_phys=row["j_phys"], j_total=row["j_total"],
                    j_resp=row["j_resp"], rules_final=row["rules_final"],
                    wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, (f"K22/{arm}/{sc}/{seed}: {type(exc).__name__}: {exc}\n"
                      + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="closed")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", default=",".join(SCEN))
    ap.add_argument("--seeds", default="201-206")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    if "live" in arms and os.environ.get("DSS_NO_CONCEPT", "") != "1":
        sys.exit("K22: set DSS_NO_CONCEPT=1, as in the model-error campaign")
    scs = [s for s in args.scenarios.split(",") if s]
    seeds = C._parse_seeds(args.seeds)
    path = os.path.join(args.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["arm"], r["scenario"], int(r["seed"]))
                    for r in csv.DictReader(f)}
    jobs = [(arm, sc, sd) for arm in arms for sc in scs for sd in seeds
            if (arm, sc, sd) not in done]
    print(f"K22: {len(jobs)} runs to do ({len(done)} done)", flush=True)
    fields = ["perturb", "arm", "scenario", "seed", "n_installed",
              "burned_ha", "coated_ha", "success", "pop_affected", "j_phys",
              "j_total", "j_resp", "rules_final", "wall_s"]
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
        print(f"[{n}/{len(jobs)}] {row['arm']} {row['scenario']} "
              f"s{row['seed']}: burned {row['burned_ha']} ha, success "
              f"{row['success']}, {row['wall_s']}s "
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
