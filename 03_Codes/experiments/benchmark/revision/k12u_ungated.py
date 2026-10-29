# -*- coding: utf-8 -*-
r"""K12u: the ungated control.

Referee question: the gates are said to keep the proposer from deciding.
The direct test removes them. This run uses the single-shot model proposer
of the LLM arm (one candidate per call) and admits every candidate that
clears G1 to G4; the forecast gates G5 to G8 are switched off
(DSS_UNGATED=1, see dss.adapt). It pairs with final/t7_llm_single (the same
proposer under the full chain) and with final/t1_reference (Closed) on the
same 150 worlds, S1-S5, seeds 201-230.

The model is called. Resumable: finished rows are skipped.

  python experiments\benchmark\revision\k12u_ungated.py --workers 6
"""
import argparse, csv, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, BENCH)
from open_space import campaign as C          # noqa: E402

OUT = os.path.join(HERE, "k12u_ungated")
SCEN = ("S1", "S2", "S3", "S4", "S5")
ARM = "LLM-ungated"


def _job(args):
    mode, sc, seed = args
    try:
        row, _c, _a, _m = C.run_once("E12U", sc, ARM, seed)
        return dict(mode=mode, scenario=sc, seed=seed, arm=ARM,
                    burned_ha=row["burned_ha"], success=row["success"],
                    pop_affected=row["pop_affected"], j_phys=row["j_phys"],
                    j_total=row["j_total"], j_resp=row["j_resp"],
                    j_delay=row.get("j_delay", ""),
                    rules_final=row["rules_final"],
                    n_macros=row.get("n_macros", ""),
                    n_actuators=row.get("n_actuators", ""),
                    prop_calls=row.get("prop_calls", ""),
                    coated_ha=row.get("coated_ha", ""),
                    wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, (f"K12u/{mode}/{sc}/{seed}: {type(exc).__name__}: {exc}\n"
                      + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", default=",".join(SCEN))
    ap.add_argument("--seeds", default="201-230")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scs = [s for s in args.scenarios.split(",") if s]
    seeds = C._parse_seeds(args.seeds)
    path = os.path.join(args.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["mode"], r["scenario"], int(r["seed"]))
                    for r in csv.DictReader(f)}
    jobs = [("ungated", sc, sd)
            for sc in scs for sd in seeds if ("ungated", sc, sd) not in done]
    print(f"K12u ungated: {len(jobs)} runs to do ({len(done)} done)", flush=True)
    fields = ["mode", "scenario", "seed", "arm", "burned_ha", "success",
              "pop_affected", "j_phys", "j_total", "j_resp", "j_delay",
              "rules_final", "n_macros", "n_actuators", "prop_calls",
              "coated_ha", "wall_s"]
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
        print(f"[{n}/{len(jobs)}] {row['scenario']} s{row['seed']}: burned "
              f"{row['burned_ha']} ha, success {row['success']}, rules "
              f"{row['rules_final']}, {row['wall_s']}s "
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
