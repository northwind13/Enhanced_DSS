# -*- coding: utf-8 -*-
r"""K11: a hand-written doctrine over the SAME seven physical effects.

Referee question (four reports): the closed layer orders six families, the
open layer composes interventions from wet, clear, ignite, coat, evacuate,
prime and draft. Does the gain come from the open decision space or from
access to those effects? K2b answered half of it with the four DISCOVERED
tactics; this test gives the closed layer tactics an officer would write by
hand, without looking at what VODA found.

Arm 'expert': the five seed rules plus six expert-written composed
interventions installed at minute 0 in every world (arm F5Ev of the LOO
machinery, no proposer, no gates, no model call). Same 150 worlds as the
main campaign (S1-S5, seeds 201-230). Pairs with final/t1_reference (closed)
and final/t6_voda (VODA) on the same worlds.

The six tactics are written from standard initial-attack practice and use
only the vocabulary of (19): an aerial retardant line ahead of the head,
structure protection with water and retardant, a flank line, a rear anchor,
an evacuation and warning belt, and a burnout behind a cleared line. The
antecedents use the five concepts of the article at medium and high terms
so that the tactics fire in ordinary situations, not only at the extremes.

No language model is called. Resumable: finished rows are skipped.

  python experiments\benchmark\revision\k11_expert_doctrine.py --workers 16
"""
import argparse, csv, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, BENCH)
from open_space import campaign as C          # noqa: E402

OUT = os.path.join(HERE, "k11_expert_doctrine")
SCEN = ("S1", "S2", "S3", "S4", "S5")


def _tactic(rule, clauses):
    """A recorded-admission record as _install_scripted expects it: the
    rule text carries the antecedents and the object name, the clauses
    define the composed intervention. Installed at minute 0."""
    return dict(rule=rule, kind="actuator", clauses=clauses,
                composition=None, concept=None, t_min=0.0,
                source="expert", accepted=True)


EXPERT = [
    # 1 aerial retardant line ahead of the head
    _tactic("IF fire threat level is H THEN head retardant line 1.0",
            [dict(effect="coat", sector="head", range=[2, 6], amount=1.0)]),
    # 2 structure protection: wet the settlement, retardant on its edge
    _tactic("IF asset exposure risk is H THEN structure protection 1.0",
            [dict(effect="wet", sector="populated", range=[0, 2], amount=1.0),
             dict(effect="coat", sector="assets", range=[0, 3], amount=0.8)]),
    # 3 flank line where crews can work
    _tactic("IF suppression feasibility is H AND fire threat level is M "
            "THEN flank line 0.9",
            [dict(effect="clear", sector="flank", range=[0, 2], amount=1.0)]),
    # 4 rear anchor point
    _tactic("IF suppression feasibility is M THEN rear anchor 0.8",
            [dict(effect="wet", sector="rear", range=[0, 2], amount=1.0)]),
    # 5 evacuation and warning belt around the settlement
    _tactic("IF evacuation pressure is H THEN evacuation belt 1.0",
            [dict(effect="prime", sector="populated", range=[0, 4], amount=1.0),
             dict(effect="evacuate", sector="populated", range=[0, 2],
                  amount=1.0)]),
    # 6 burnout between the head and a cleared line, feasibility permitting
    _tactic("IF suppression feasibility is VH AND fire threat level is H "
            "THEN burnout behind line 0.8",
            [dict(effect="clear", sector="head", range=[5, 6], amount=1.0),
             dict(effect="ignite", sector="head", range=[3, 4], amount=0.6)]),
]


def _job(args):
    mode, sc, seed, script = args
    try:
        row, _c, _a, _m = C.run_once("LOO", sc, "F5Ev", seed, script=script)
        return dict(mode=mode, scenario=sc, seed=seed, n_installed=len(script),
                    burned_ha=row["burned_ha"], success=row["success"],
                    pop_affected=row["pop_affected"], j_phys=row["j_phys"],
                    j_total=row["j_total"], j_resp=row["j_resp"],
                    j_delay=row.get("j_delay", ""),
                    rules_final=row["rules_final"],
                    coated_ha=row.get("coated_ha", ""),
                    wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, (f"K11/{mode}/{sc}/{seed}: {type(exc).__name__}: {exc}\n"
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
    jobs = [("expert", sc, sd, [dict(p) for p in EXPERT])
            for sc in scs for sd in seeds if ("expert", sc, sd) not in done]
    print(f"K11 expert: {len(jobs)} runs to do ({len(done)} done)", flush=True)
    fields = ["mode", "scenario", "seed", "n_installed", "burned_ha",
              "success", "pop_affected", "j_phys", "j_total", "j_resp",
              "j_delay", "rules_final", "coated_ha", "wall_s"]
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
        print(f"[{n}/{len(jobs)}] {row['mode']} {row['scenario']} "
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
