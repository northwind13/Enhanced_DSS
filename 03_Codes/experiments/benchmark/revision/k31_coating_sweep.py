# -*- coding: utf-8 -*-
r"""K31: coating effectiveness sweep.

Referee question (round 4, M4): the coating operator stops spread in a fully
coated cell (moisture floor above extinction) and about two thirds of the
gain travels with objects that coat; three points (nominal, weakened, zero)
do not show how the gain depends on the assumed effectiveness. This sweep
runs the coating at five effectiveness levels, with the burn reduction and
the moisture floor scaled together:

  level   burn reduction   moisture floor   coated cell burns at
  r100    0.80 (nominal)   0.45             0.20 of its rate, spread stops
  r75     0.60             0.34             0.40
  r50     0.40             0.23             0.60
  r25     0.20             0.11             0.80
  r0      0.00             0.00             1.00, coating does nothing

Two modes:
  world  the LIVE world carries the weaker coating and every shadow forecast
         plans on the nominal one (model error, as K6; K6's coat_half and
         coat_off are the 0.5/0.30 and zero points of this family)
  both   planner and world carry the same weaker coating (a calibrated
         operator, not an error)

Arms: closed (F5Ev) and replay (F5Ev with the objects GODA admitted in that
world re-installed at their minutes). CPU only, no model call. Resumable.
Pairs with final/t1_reference and final/t10_benefit (removed=none).

  python experiments\benchmark\revision\k31_coating_sweep.py --mode world --workers 16
  python experiments\benchmark\revision\k31_coating_sweep.py --mode both --workers 16
"""
import argparse, csv, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)
from open_space import campaign as C          # noqa: E402
import loo as L                               # noqa: E402
from dss import evaluate as EV                # noqa: E402
import k6_coating as K6                       # noqa: E402

SRC = os.path.join(BENCH, "final", "t6_voda")
OUT = os.path.join(HERE, "k31_coating_sweep")
SCEN = ("S1", "S2", "S3", "S4", "S5")
LEVELS = {
    "r100": dict(coat_burn_reduction=0.80, coat_moisture_floor=0.45),
    "r75": dict(coat_burn_reduction=0.60, coat_moisture_floor=0.34),
    "r50": dict(coat_burn_reduction=0.40, coat_moisture_floor=0.23),
    "r25": dict(coat_burn_reduction=0.20, coat_moisture_floor=0.11),
    "r0": dict(coat_burn_reduction=0.00, coat_moisture_floor=0.00),
}
DEFAULT = "r75,r50,r25,r0"      # r100 is final/t1_reference and t10_benefit


def _install(level, mode):
    K6._FACTORS.clear()
    K6._FACTORS.update(LEVELS[level])
    K6._PRICE.clear()
    K6._RULE.clear()
    C.build_world = K6._build_world_perturbed
    # 'world': shadow copies restored to the nominal coating (K6 machinery);
    # 'both': the copies inherit the weaker coating of the world
    EV.clone_sim = K6._clone_nominal if mode == "world" else K6._orig_clone


def _job(args):
    level, mode, arm, sc, seed, script = args
    try:
        _install(level, mode)
        row, _c, _a, _m = C.run_once("LOO", sc, "F5Ev", seed, script=script)
        return dict(level=level, mode=mode, arm=arm, scenario=sc, seed=seed,
                    n_installed=len(script), burned_ha=row["burned_ha"],
                    coated_ha=row.get("coated_ha", ""),
                    success=row["success"], pop_affected=row["pop_affected"],
                    j_phys=row["j_phys"], j_total=row["j_total"],
                    j_resp=row["j_resp"], rules_final=row["rules_final"],
                    wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, (f"K31/{level}/{mode}/{arm}/{sc}/{seed}: "
                      f"{type(exc).__name__}: {exc}\n"
                      + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default=DEFAULT,
                    help="comma list of " + ",".join(LEVELS))
    ap.add_argument("--mode", default="world", choices=("world", "both"))
    ap.add_argument("--arms", default="closed,replay")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", default=",".join(SCEN))
    ap.add_argument("--seeds", default="201-230")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = L.load_admitted(args.src, exp="E9", arm="LLMScreen")
    levels = [p for p in args.levels.split(",") if p]
    arms = [a for a in args.arms.split(",") if a]
    scs = [s for s in args.scenarios.split(",") if s]
    seeds = C._parse_seeds(args.seeds)
    path = os.path.join(args.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["level"], r["mode"], r["arm"], r["scenario"],
                     int(r["seed"])) for r in csv.DictReader(f)}
    jobs = []
    for lv in levels:
        for arm in arms:
            for sc in scs:
                for sd in seeds:
                    if (lv, args.mode, arm, sc, sd) in done:
                        continue
                    script = list(recs.get((sc, sd), [])) \
                        if arm == "replay" else []
                    jobs.append((lv, args.mode, arm, sc, sd, script))
    print(f"K31 {args.mode}: {len(jobs)} runs to do ({len(done)} done)",
          flush=True)
    fields = ["level", "mode", "arm", "scenario", "seed", "n_installed",
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
        print(f"[{n}/{len(jobs)}] {row['level']} {row['mode']} {row['arm']} "
              f"{row['scenario']} s{row['seed']}: burned {row['burned_ha']} "
              f"ha, coated {row['coated_ha']}, success {row['success']}, "
              f"{row['wall_s']}s (elapsed {time.time() - t0:.0f}s)",
              flush=True)

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
