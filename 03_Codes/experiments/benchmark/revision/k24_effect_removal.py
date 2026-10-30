# -*- coding: utf-8 -*-
r"""K24: marginal contribution of each physical effect.

Referee question: how much of the gain rests on each effect operator, coating
in particular. For each of the seven effects, every admitted object whose
clauses use that effect is withheld from the replay (together with the later
objects that build on it), and the closed layer runs with the remaining
objects installed at their recorded minutes, on the 150 worlds of the main
campaign. Pairs with the full replay (final/t10_benefit, removed=none) and
with the closed layer. CPU only, resumable.

  python experiments\benchmark\revision\k24_effect_removal.py --workers 12
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

SRC = os.path.join(BENCH, "final", "t6_voda")
OUT = os.path.join(HERE, "k24_effect_removal")
EFFECTS = ("wet", "clear", "ignite", "coat", "evacuate", "prime", "draft")


def _uses_effect(rec, eff):
    cl = rec.get("clauses") or (rec.get("new_intervention") or {}).get("clauses") or []
    return any(str(c.get("effect", "")) == eff for c in cl)


def script_without_effect(recs, eff):
    """Drop every record using eff and every later record that depends on a
    dropped object (same dependency rule as loo.script_without)."""
    gone = set()
    out = []
    for r in recs:
        ants, cons = C._parse_rule_text(r["rule"])
        uses = {v for v, _t in ants} | {c for c, _x in cons}
        if _uses_effect(r, eff) or (uses & gone):
            nm = L.new_object_name(r)
            if nm:
                gone.add(nm)
            continue
        out.append(r)
    return out


def _job(args):
    eff, sc, seed, script, n_full = args
    try:
        row, _c, _a, _m = C.run_once("LOO", sc, "F5Ev", seed, script=script)
        return dict(removed_effect=eff, scenario=sc, seed=seed,
                    n_installed=len(script), n_full=n_full,
                    burned_ha=row["burned_ha"], success=row["success"],
                    pop_affected=row["pop_affected"], j_phys=row["j_phys"],
                    j_total=row["j_total"], j_resp=row["j_resp"],
                    rules_final=row["rules_final"], wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, (f"K24/{eff}/{sc}/{seed}: {type(exc).__name__}: {exc}\n"
                      + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effects", default=",".join(EFFECTS))
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", default="S1,S2,S3,S4,S5")
    ap.add_argument("--seeds", default="201-230")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = L.load_admitted(args.src, exp="E9", arm="LLMScreen")
    effs = [e for e in args.effects.split(",") if e]
    scs = [s for s in args.scenarios.split(",") if s]
    seeds = C._parse_seeds(args.seeds)
    path = os.path.join(args.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["removed_effect"], r["scenario"], int(r["seed"]))
                    for r in csv.DictReader(f)}
    jobs = []
    for eff in effs:
        for sc in scs:
            for sd in seeds:
                if (eff, sc, sd) in done:
                    continue
                rl = list(recs.get((sc, sd), []))
                jobs.append((eff, sc, sd, script_without_effect(rl, eff), len(rl)))
    print(f"K24: {len(jobs)} runs to do ({len(done)} done)", flush=True)
    fields = ["removed_effect", "scenario", "seed", "n_installed", "n_full",
              "burned_ha", "success", "pop_affected", "j_phys", "j_total",
              "j_resp", "rules_final", "wall_s"]
    t0 = time.time(); n = 0

    def sink(res):
        nonlocal n
        row, err = res; n += 1
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
        print(f"[{n}/{len(jobs)}] -{row['removed_effect']} {row['scenario']} "
              f"s{row['seed']}: {row['n_installed']}/{row['n_full']} objects, "
              f"burned {row['burned_ha']} ha (elapsed {time.time() - t0:.0f}s)", flush=True)

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
