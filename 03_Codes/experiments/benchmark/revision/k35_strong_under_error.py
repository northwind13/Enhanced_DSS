# -*- coding: utf-8 -*-
r"""K35: the equal-budget comparisons under model error and with independent
gate futures.

Referee question (three reports): the margin of GODA over the strongest
searches (32% over the evolutionary search on the six families, 24% over the
evolutionary search on the same grammar) was measured with a planner that
runs the live model from the exact state. This run repeats the two
evolutionary searches (no model call) in two settings whose GODA runs exist:

  err   the combined live-world error of K18 (cosine kernel and short
        embers, wind x1.5, weakened coating), every forecast on the nominal
        model; pairs with k18_combined (arm live = GODA, arm closed)
  ind   the nominal world with independent gate futures (DSS_GATE_SEEDS
        303,404); pairs with k8_gate_seeds (GODA with the same futures)

Arms: evo_err, evofam_err, evo_ind, evofam_ind. CPU only. Resumable.

  python experiments\benchmark\revision\k35_strong_under_error.py --workers 8
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
from dss import evaluate as EV                # noqa: E402
import k18_combined as K18                    # noqa: E402

OUT = os.path.join(HERE, "k35_strong_under_error")
SCEN = ("S1", "S2", "S3", "S4", "S5")
ARMS = {"evo_err": ("Evo", True, False), "evofam_err": ("Evo-fam", True, False),
        "evo_ind": ("Evo", False, True), "evofam_ind": ("Evo-fam", False, True)}


def _job(args):
    arm, sc, seed = args
    base, err, ind = ARMS[arm]
    try:
        C.build_world = K18._orig_build
        EV.clone_sim = K18._orig_clone
        if err:
            K18._install()
        os.environ["DSS_GATE_SEEDS"] = "303,404" if ind else ""
        row, _c, _a, _m = C.run_once("E15", sc, base, seed)
        return dict(arm=arm, scenario=sc, seed=seed, n_installed=len(_m),
                    burned_ha=row["burned_ha"], coated_ha=row.get("coated_ha", ""),
                    success=row["success"], pop_affected=row["pop_affected"],
                    j_phys=row["j_phys"], j_total=row["j_total"],
                    j_resp=row["j_resp"], rules_final=row["rules_final"],
                    wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, (f"K35/{arm}/{sc}/{seed}: {type(exc).__name__}: {exc}\n"
                      + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", default=",".join(SCEN))
    ap.add_argument("--seeds", default="201-230")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    scs = [s for s in args.scenarios.split(",") if s]
    seeds = C._parse_seeds(args.seeds)
    path = os.path.join(args.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["arm"], r["scenario"], int(r["seed"])) for r in csv.DictReader(f)}
    jobs = [(a, sc, sd) for a in arms for sc in scs for sd in seeds
            if (a, sc, sd) not in done]
    print(f"K35: {len(jobs)} runs to do ({len(done)} done)", flush=True)
    fields = ["arm", "scenario", "seed", "n_installed", "burned_ha", "coated_ha",
              "success", "pop_affected", "j_phys", "j_total", "j_resp",
              "rules_final", "wall_s"]
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
        print(f"[{n}/{len(jobs)}] {row['arm']} {row['scenario']} s{row['seed']}: "
              f"burned {row['burned_ha']} ha, success {row['success']}, "
              f"{row['wall_s']}s (elapsed {time.time() - t0:.0f}s)", flush=True)

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
