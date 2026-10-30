# -*- coding: utf-8 -*-
r"""K19: situation-picture error. The shadow core starts from a perturbed
fire state.

Referee question: the model-error tests perturb the dynamics, never the
fire state the planner starts from. Here every shadow forecast (acceptance
test, closed moves, G5 to G8) starts from a copy of the live simulator in
which a fraction of the burning cells is dropped, the S5 outage rate of
0.25 (DSS_K19_DROP): the planner does not see those cells burning, their
fuel is intact in the copy, so they may reignite from neighbours. The
drop is drawn once per live step from a seed-and-step-dependent stream,
so the two futures of a gate see the same perturbed state. The live world
is the nominal one.

Arms on the 150 worlds of S1-S5, seeds 201-230:
  closed   F5Ev, the closed layer with perturbed shadow starts
  live     GODA, exp E9 arm LLMScreen, the whole loop run anew (model calls)

Pairs within itself (live against closed). Resumable.

  python experiments\benchmark\revision\k19_perimeter.py --arms closed --workers 16
  python experiments\benchmark\revision\k19_perimeter.py --arms live --workers 6
"""
import argparse, csv, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, BENCH)
from open_space import campaign as C          # noqa: E402
from dss import evaluate as EV                # noqa: E402
import numpy as np

OUT = os.path.join(HERE, "k19_perimeter")
SCEN = ("S1", "S2", "S3", "S4", "S5")
DROP = float(os.environ.get("DSS_K19_DROP", "0.25"))
_orig_clone = EV.clone_sim


def _clone_perturbed(sim, keep_snapshots=False):
    """Shadow copy with a fraction DROP of the burning cells unseen."""
    s2 = _orig_clone(sim, keep_snapshots=keep_snapshots)
    if keep_snapshots:                 # counterfactual replay, not a forecast
        return s2
    st = s2.state
    on = np.asarray(st.burning) > 0.5
    if not on.any() or DROP <= 0.0:
        return s2
    seed = int(getattr(sim.cfg, "rng_seed", 0) or 0)
    rng = np.random.default_rng(1_000_003 * seed + 7 * int(s2.state.step) + 19)
    drop = on & (rng.random(on.shape) < DROP)
    st.burning[drop] = 0.0
    st.intensity[drop] = 0.0
    st.tau[drop] = 0.0
    return s2


def _install():
    EV.clone_sim = _clone_perturbed


def _job(args):
    arm, sc, seed = args
    try:
        _install()
        if arm == "live":
            row, _c, _a, _m = C.run_once("E9", sc, "LLMScreen", seed)
        else:
            row, _c, _a, _m = C.run_once("E2P", sc, "F5Ev", seed)
        return dict(perturb=f"shadow_drop_{DROP:g}", arm=arm,
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
        return None, (f"K19/{arm}/{sc}/{seed}: {type(exc).__name__}: {exc}\n"
                      + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="closed")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", default=",".join(SCEN))
    ap.add_argument("--seeds", default="201-230")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    if "live" in arms and os.environ.get("DSS_NO_CONCEPT", "") != "1":
        sys.exit("K19: set DSS_NO_CONCEPT=1, as in the model-error campaign")
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
    print(f"K19: {len(jobs)} runs to do ({len(done)} done)", flush=True)
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
