# -*- coding: utf-8 -*-
r"""K9: recorded tactics against continued generation on unseen conditions.

Question: once the tactics found by the open layer are recorded as rules,
does keeping the open layer active add anything on incidents outside the
conditions the tactics were derived from?

Tactics: the four canonical tactics of K2b, derived from the admissions in
S1-S5, seeds 201-215 (revision/k2b_generalization/k2_doctrine.json).

Arms, paired on the same worlds:
  A  closed layer, five seed rules                      (exp E2P, F5Ev)
  B  closed layer + the four tactics at minute 0         (exp LOO, F5Ev)
  C  B + open layer active (LLM pre-selection and gates) (exp E9, LLMScreen)
  D  VODA from the five seed rules                       (exp E9, LLMScreen)

World sets:
  novel    S6, S7, S8 (landscapes, wind, moisture, ignitions, pool never
           seen in S1-S5), seeds 201-230
  control  S1-S5, seeds 216-230 (the 75 unseen worlds of T3i); A, B, D of
           the control set exist in final/t1_reference, k2b_generalization,
           final/t6_voda

The planner models the world correctly in every arm, so the test measures
novelty, not model error. Resumable.

  python experiments\benchmark\revision\k9_novelty.py --arm B --set novel --workers 16
"""
import argparse, csv, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, BENCH)
from open_space import campaign as C          # noqa: E402

OUT = os.path.join(HERE, "k9_novelty")
TACTICS = os.path.join(HERE, "k2b_generalization", "k2_doctrine.json")
SETS = {"novel": (("S6", "S7", "S8"), "201-230"),
        "control": (("S1", "S2", "S3", "S4", "S5"), "216-230")}
EXP = {"A": ("E2P", "F5Ev", False), "B": ("LOO", "F5Ev", True),
       "C": ("E9", "LLMScreen", True), "D": ("E9", "LLMScreen", False)}
KEEP = ["burned_ha", "success", "pop_affected", "j_phys", "j_total",
        "j_resp", "j_delay", "rules_final", "coated_ha", "ignited_ha",
        "prop_calls", "prop_candidates", "prop_forecasts", "acc_1", "acc_2",
        "acc_3", "n_macros", "n_actuators", "n_concepts", "wall_s"]


def _job(args):
    arm, wset, sc, seed, tactics = args
    exp, carm, use_t = EXP[arm]
    try:
        script = [dict(p, t_min=0.0) for p in tactics] if use_t else None
        row, _c, _att, adm = C.run_once(exp, sc, carm, seed, script=script)
        out = dict(arm=arm, set=wset, scenario=sc, seed=seed,
                   n_installed=len(script or []))
        for k in KEEP:
            out[k] = row.get(k, "")
        out["n_admitted"] = len(adm or [])
        return out, adm, None
    except Exception as exc:
        import traceback
        return None, None, (f"K9/{arm}/{wset}/{sc}/{seed}: "
                            f"{type(exc).__name__}: {exc}\n"
                            + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=tuple(EXP))
    ap.add_argument("--set", default="novel", choices=tuple(SETS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tactics", default=TACTICS)
    args = ap.parse_args()
    if EXP[args.arm][1] == "LLMScreen" and \
            os.environ.get("DSS_NO_CONCEPT", "") != "1":
        sys.exit("K9: set DSS_NO_CONCEPT=1, as in the reported campaign")
    os.makedirs(args.out, exist_ok=True)
    tactics = json.load(open(args.tactics, encoding="utf-8"))
    scs, seeds_s = SETS[args.set]
    seeds = C._parse_seeds(seeds_s)
    path = os.path.join(args.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["arm"], r["set"], r["scenario"], int(r["seed"]))
                    for r in csv.DictReader(f)}
    jobs = [(args.arm, args.set, sc, sd, tactics) for sc in scs for sd in seeds
            if (args.arm, args.set, sc, sd) not in done]
    print(f"K9 arm {args.arm}, set {args.set}: {len(jobs)} runs to do "
          f"({len(done)} rows on disk)", flush=True)
    fields = ["arm", "set", "scenario", "seed", "n_installed", "n_admitted"] + KEEP
    t0 = time.time(); n = 0

    def sink(res):
        nonlocal n
        row, adm, err = res
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
        if adm:
            with open(os.path.join(args.out, "admitted.jsonl"), "a",
                      encoding="utf-8") as f:
                for a in adm:
                    f.write(json.dumps(dict(a, k9_arm=row["arm"],
                                            k9_set=row["set"]),
                                       default=str) + "\n")
        print(f"[{n}/{len(jobs)}] {row['arm']} {row['set']} {row['scenario']} "
              f"s{row['seed']}: burned {row['burned_ha']} ha, success "
              f"{row['success']}, calls {row['prop_calls']}, admitted "
              f"{row['n_admitted']} (elapsed {time.time() - t0:.0f}s)",
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
