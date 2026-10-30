# -*- coding: utf-8 -*-
r"""K20: sensitivity to the cost function.

Referee question: the results depend on the evacuee weight (0.05), the
capacity charges for coating, ignition and warning, the reference burned
fraction (0.5) and the response and delay weights (0.2). The gates and the
acceptance test judge the physical cost, burned area, asset loss and
population exposure, so the response and delay weights and the capacity
charges enter the reported total only and are reweighted offline; the
evacuee weight and the reference fraction enter the physical cost and can
change decisions, so they are replayed here. Each setting changes the cost
function of both the live world and every shadow forecast (it is the
objective, not a model error), and both arms see the same setting.

Settings (disaster_phyengine.config cost parameters):
  evac_x5     evacuation_weight 0.05 -> 0.25
  bref_x0.5   burn_reference_fraction 0.50 -> 0.25 (area charged twice as hard)
  charges_x2  coat, ignite capacity steps 6 -> 12, warning 0.1 -> 0.2 per step
  resp_1      w_resp and w_delay 0.2 -> 1.0

Two arms per setting on the 150 worlds:
  closed   F5Ev, the closed layer alone
  voda     F5Ev with the objects admitted in that world re-installed at
           their recorded minutes (the replay reference)

Resumable, CPU only.

  python experiments\benchmark\revision\k20_cost_weights.py --workers 16
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
OUT = os.path.join(HERE, "k20_cost_weights")
SCEN = ("S1", "S2", "S3", "S4", "S5")
PERTURB = {
    "evac_x5":    dict(evacuation_weight=0.25),
    "bref_x0.5":  dict(burn_reference_fraction=0.25),
    "charges_x2": dict(coat_capacity_steps=12.0, ignite_capacity_steps=12.0,
                       warn_capacity_per_step=0.2),
    "resp_1":     dict(w_resp=1.0, w_delay=1.0),
}
DEFAULT = ",".join(PERTURB)

_SET = {}
_orig_build = C.build_world


def _build_world_costed(seed, scenario):
    w = _orig_build(seed, scenario)
    for k, v in _SET.items():
        setattr(w.config.cost, k, float(v))
    return w


def _install(name):
    _SET.clear()
    _SET.update(PERTURB[name])
    C.build_world = _build_world_costed


def _job(args):
    name, arm, sc, seed, script = args
    try:
        _install(name)
        row, _c, _a, _m = C.run_once("LOO", sc, "F5Ev", seed, script=script)
        return dict(perturb=name, arm=arm, scenario=sc, seed=seed,
                    n_installed=len(script), burned_ha=row["burned_ha"],
                    success=row["success"], pop_affected=row["pop_affected"],
                    j_phys=row["j_phys"], j_total=row["j_total"],
                    j_resp=row["j_resp"], rules_final=row["rules_final"],
                    wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, (f"K20/{name}/{arm}/{sc}/{seed}: {type(exc).__name__}: "
                      f"{exc}\n" + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturb", default=DEFAULT,
                    help="comma list of " + ",".join(PERTURB))
    ap.add_argument("--arms", default="closed,voda")
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
                    script = list(recs.get((sc, sd), [])) if arm == "voda" \
                        else []
                    jobs.append((name, arm, sc, sd, script))
    print(f"K20: {len(jobs)} runs to do ({len(done)} done)", flush=True)
    fields = ["perturb", "arm", "scenario", "seed", "n_installed",
              "burned_ha", "success", "pop_affected", "j_phys", "j_total",
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
