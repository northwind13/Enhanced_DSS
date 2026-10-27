r"""K6: coating effect weaker in the evaluation world than in the planning model.

Referee question: the retardant coating (burn rate x 0.2, moisture floor
0.45, decay 0.3 %/min) is a strong modelling assumption, and a method that
coats more could gain a systematic advantage from it. Here the live world
carries a weaker coating while every shadow forecast (pre-selection, gates,
evFIS trials, no-harm guard) plans on the NOMINAL coating.

Perturbations of the live world (disaster_phyengine.config.SuppressionParams):
  coat_half   burn reduction 0.8 -> 0.5 (coated cell burns at half rate, not a
              fifth), moisture floor 0.45 -> 0.30 (at the extinction moisture
              of the wettest fuel, so a coated cell can still carry fire),
              decay 0.3 %/min -> 1 %/min (about 1.5 h of protection)
  coat_off    the coating does nothing (reduction 0, floor 0); it is still
              priced and still passes the gates on the nominal model
  coat_price_x2  the coating costs twice as much capacity (12 instead of 6
              capacity-steps per cell) in the planner AND the world: a
              budget rule, not a model error, so G4 refuses it sooner
  coat_stock_20ha  at most 20 ha can be coated in one incident (nominal VODA
              coats about 43 ha), planner and world alike
  coat_delay_12min a coating order takes effect one decision cycle (12 min)
              after it is given, planner and world alike

Arms per perturbation on the 150 worlds, all planning on the nominal model:
  closed   F5Ev, the closed layer alone (coats only through its seed rules)
  replay   F5Ev with the objects T6 admitted in that world re-installed at
           their recorded minutes (the T4c replay reference)
  live     VODA, exp E9 arm LLMScreen, the whole propose, pre-select, admit
           loop run anew on the perturbed world (model calls)

Resumable. Pairs with final/t1_reference, final/t10_benefit (removed=none)
and final/t6_voda.

  python experiments\benchmark\revision\k6_coating.py --arms closed,replay --workers 16
  python experiments\benchmark\revision\k6_coating.py --arms live --workers 4
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
OUT = os.path.join(HERE, "k6_coating")
SCEN = ("S1", "S2", "S3", "S4", "S5")
PERTURB = {   # absolute values set on the live world
    "coat_half": dict(coat_burn_reduction=0.5, coat_moisture_floor=0.30,
                      coat_decay_per_min=0.01),
    "coat_off":  dict(coat_burn_reduction=0.0, coat_moisture_floor=0.0,
                      coat_decay_per_min=0.003),
    "coat_price_x2": dict(),
    "coat_stock_20ha": dict(),
    "coat_delay_12min": dict(),
}
# application RULES, set on planner and world alike (not a model error)
PRICE = {"coat_price_x2": dict(coat_capacity_steps=12.0)}
RULE = {"coat_stock_20ha": dict(coat_stock_ha=20.0),
        "coat_delay_12min": dict(coat_delay_min=12.0)}
DEFAULT = "coat_half,coat_off,coat_price_x2,coat_stock_20ha,coat_delay_12min"

_NOMINAL = {}          # field -> nominal value of the current world
_FACTORS = {}          # field -> value set on the live world
_orig_build = C.build_world
_orig_clone = EV.clone_sim


_PRICE = {}            # cost fields set on planner and world alike
_RULE = {}             # suppression rule fields set on planner and world alike


def _build_world_perturbed(seed, scenario):
    w = _orig_build(seed, scenario)
    sp = w.config.suppression
    _NOMINAL.clear()
    for k, v in _FACTORS.items():
        _NOMINAL[k] = float(getattr(sp, k))
        setattr(sp, k, float(v))
    for k, v in _PRICE.items():
        setattr(w.config.cost, k, float(v))
    for k, v in _RULE.items():
        setattr(sp, k, float(v))
    return w


def _clone_nominal(sim, keep_snapshots=False):
    """Every shadow copy plans on the NOMINAL coating."""
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
    _PRICE.clear()
    _PRICE.update(PRICE.get(name, {}))
    _RULE.clear()
    _RULE.update(RULE.get(name, {}))
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
                    n_concepts=row.get("n_concepts", ""),
                    wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, (f"K6/{name}/{arm}/{sc}/{seed}: {type(exc).__name__}: "
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
    if "live" in arms and os.environ.get("DSS_NO_CONCEPT", "") != "1":
        sys.exit("K6: set DSS_NO_CONCEPT=1, as in the reported campaign")
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
    print(f"K6: {len(jobs)} runs to do ({len(done)} done)", flush=True)
    fields = ["perturb", "arm", "scenario", "seed", "n_installed",
              "burned_ha", "coated_ha", "success", "pop_affected", "j_phys", "j_total",
              "j_resp", "rules_final", "n_concepts", "wall_s"]
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
