# -*- coding: utf-8 -*-
r"""K3: planning model different from the evaluation world.

Referee question: the open layer selects proposals on the same simulator
that scores them, so a gain could be an adaptation to the simulator's own
assumptions. Here the EVALUATION world is perturbed while every shadow
forecast (pre-selection, gates, evFIS trials) still runs on the NOMINAL
suppression model: the decision cores plan on a model that is wrong about
the crews.

Perturbations of the live world (disaster_phyengine.config.SuppressionParams):
  gammaI_x1.5   crews lose more effectiveness against intensity (eta_eff)
  betat_x1.5    effectiveness decays faster with travel time (eta_reach)
  knock_x1.5    knockdown needs 50% more pressure (harder to put out)
  gammaI_x0.67  crews more effective than planned (favourable error)

Two arms per perturbation on the 150 worlds, both planning on the nominal
model:
  closed   F5Ev, the closed layer alone
  voda     F5Ev with the objects T6 admitted in that world re-installed at
           their recorded minutes (the T10 replay reference), so the open
           layer's output is scored under model error without a new
           proposer run; worlds with no admission are the closed layer

Resumable. Pairs with final/t1_reference and final/t10_benefit (removed=none).

  python experiments\benchmark\revision\k3_model_error.py --workers 8
  python experiments\benchmark\revision\k3_model_error.py --perturb gammaI_x1.5 --arms closed
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
OUT = os.path.join(HERE, "k3_model_error")
SCEN = ("S1", "S2", "S3", "S4", "S5")
PERTURB = {
    "gammaI_x1.5":  dict(gamma_I=1.5),
    "betat_x1.5":   dict(beta_t=1.5),
    "knock_x1.5":   dict(knockdown_ratio=1.5),
    "gammaI_x0.67": dict(gamma_I=1.0 / 1.5),
}
DEFAULT = "gammaI_x1.5,betat_x1.5,knock_x1.5"

_NOMINAL = {}          # field -> nominal value of the current world
_FACTORS = {}          # field -> multiplier applied to the live world
_orig_build = C.build_world
_orig_clone = EV.clone_sim


def _build_world_perturbed(seed, scenario):
    w = _orig_build(seed, scenario)
    sp = w.config.suppression
    _NOMINAL.clear()
    for k, f in _FACTORS.items():
        _NOMINAL[k] = float(getattr(sp, k))
        setattr(sp, k, _NOMINAL[k] * float(f))
    return w


def _clone_nominal(sim, keep_snapshots=False):
    """Every shadow copy plans on the NOMINAL suppression model."""
    s2 = _orig_clone(sim, keep_snapshots=keep_snapshots)
    for k, v in _NOMINAL.items():
        setattr(s2.cfg.suppression, k, v)
        try:
            setattr(s2.world.config.suppression, k, v)
        except AttributeError:
            pass
    return s2


def _install(name):
    _FACTORS.clear()
    _FACTORS.update(PERTURB[name])
    C.build_world = _build_world_perturbed
    EV.clone_sim = _clone_nominal


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
        return None, (f"K3/{name}/{arm}/{sc}/{seed}: {type(exc).__name__}: "
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
    print(f"K3: {len(jobs)} runs to do ({len(done)} done)", flush=True)
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
