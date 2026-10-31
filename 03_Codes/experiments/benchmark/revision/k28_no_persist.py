# -*- coding: utf-8 -*-
r"""K28: the same search without a persistent doctrine.

Referee question: how much of the gain comes from recording the admitted
objects as doctrine, against finding them again every cycle? Here an object
admitted by the open layer is used in the next cycle only and withdrawn
after it (DSS_NO_PERSIST=1); the proposer, the search budget, the gates and
the no-harm guard are those of GODA, so the open layer runs as an online
optimizer that keeps nothing. Pairs with the closed layer of
final/t1_reference and with GODA (final/t6_voda).

  python experiments\benchmark\revision\k28_no_persist.py --arms live --workers 6"""
import argparse, csv, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, BENCH)
from open_space import campaign as C          # noqa: E402
from dss import evaluate as EV                # noqa: E402
from dss import adapt as A                    # noqa: E402
import numpy as np

OUT = os.path.join(HERE, "k28_no_persist")
SCEN = ("S1", "S2", "S3", "S4", "S5")

def _install():
    os.environ["DSS_NO_PERSIST"] = "1"


def _job(args):
    arm, sc, seed = args
    try:
        _install()
        if arm == "live":
            row, _c, _a, _m = C.run_once("E9", sc, "LLMScreen", seed)
        else:
            row, _c, _a, _m = C.run_once("E2P", sc, "F5Ev", seed)
        _att = list(_a)
        for a in _att:
            a.setdefault("margin", row.get("margin", ""))
        return dict(_attempts=_att, perturb="no_persist", arm=arm,
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
        return None, (f"K28/{arm}/{sc}/{seed}: {type(exc).__name__}: {exc}\n"
                      + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="live")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", default="S1,S2,S3,S4,S5")
    ap.add_argument("--seeds", default="201-230")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    if "live" in arms and os.environ.get("DSS_NO_CONCEPT", "") != "1":
        sys.exit("K28: set DSS_NO_CONCEPT=1, as in the model-error campaign")
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
    print(f"K28: {len(jobs)} runs to do ({len(done)} done)", flush=True)
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
        import json
        with open(os.path.join(args.out, "proposals.jsonl"), "a", encoding="utf-8") as f:
            for a in row.pop("_attempts", []):
                f.write(json.dumps(a, default=str) + "\n")
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
