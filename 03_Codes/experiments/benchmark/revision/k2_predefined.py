# -*- coding: utf-8 -*-
r"""K2: the discovered tactics given to the CLOSED layer in advance.

Referee question: does VODA win because it discovers alternatives during the
incident, or would a closed doctrine that already contained those tactics do
as well? Two closed arms answer it, both without a proposer (arm F5Ev of the
LOO replay machinery, objects installed at minute 0 without the gates):

  mode canon  one canonical exemplar of each recurring tactic of T6
              (asset collar, head coating, interface hardening macro,
              line-and-attack macro, Section V-D of the article) installed
              at minute 0 in EVERY world; the same four objects everywhere,
              the "extended doctrine"
  mode own    every object the T6 run admitted in THAT world, installed at
              minute 0 (perfect-foresight bound of the same objects)

  K2b, generalization: --derive-seeds 201-215 --seeds 216-230 derives the
  four tactics from 75 worlds and evaluates the frozen library on the 75
  worlds it never saw (closed layer, no proposer); pairs with T6 there.

  python experiments\benchmark\revision\k2_predefined.py --mode canon --derive-seeds 201-215 --seeds 216-230 --workers 16 --out experiments\benchmark\revision\k2b_generalization

Pairs with final/t1_reference (F5Ev) and final/t6_voda (LLMScreen) on the
same worlds. Resumable.

  python experiments\benchmark\revision\k2_predefined.py --mode canon --workers 8
  python experiments\benchmark\revision\k2_predefined.py --mode own   --workers 8
"""
import argparse, collections, csv, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, BENCH)
from open_space import campaign as C          # noqa: E402
import loo as L                               # noqa: E402

SRC = os.path.join(BENCH, "final", "t6_voda")
OUT = os.path.join(HERE, "k2_predefined")
SCEN = ("S1", "S2", "S3", "S4", "S5")
BASE_CONCEPTS = {"fire_threat_level", "asset_exposure_risk",
                 "suppression_feasibility", "intervention_urgency",
                 "evacuation_pressure"}


def _has(p, eff, sec):
    return any(c["effect"] == eff and c["sector"] == sec
               for c in (p.get("clauses") or []))


def archetype(p):
    """The four tactics of the article (Section V-D): asset collar = coating
    of populated cells or assets, head coating = coating of the head,
    interface hardening = asset protection blended with a containment line,
    line-and-attack = containment line blended with suppression effort."""
    k = p.get("kind")
    if k == "actuator":
        if _has(p, "coat", "populated") or _has(p, "coat", "assets"):
            return "asset_collar"
        if _has(p, "coat", "head"):
            return "head_coating"
    if k == "macro":
        fs = {f for f, _w in p["composition"]}
        if {"asset_protection", "containment_line"} <= fs:
            return "interface_hardening"
        if {"containment_line", "suppression_effort"} <= fs:
            return "line_and_attack"
    return None


def base_only(p):
    """True when the rule reads base concepts only (no world-born concept)."""
    ants, _c = C._parse_rule_text(p["rule"])
    return all(v in BASE_CONCEPTS for v, _t in ants)


def canonical_set(recs):
    """One exemplar per archetype: the modal GEOMETRY (effect@sector set of
    an actuator, family set of a macro) over distinct worlds, and within it
    the modal antecedent, base concepts only."""
    def geom(p):
        if p.get("kind") == "actuator":
            return " + ".join(sorted({f"{c['effect']}@{c['sector']}"
                                      for c in p["clauses"]}))
        return " + ".join(sorted(f for f, _w in p["composition"]))

    def ants(p):
        a, _c = C._parse_rule_text(p["rule"])
        return " AND ".join(f"{v} is {t}" for v, t in a)
    gw = collections.defaultdict(set)       # (arch, geom) -> worlds
    aw = collections.defaultdict(set)       # (arch, geom, ants) -> worlds
    ex = {}
    for (sc, sd), rl in recs.items():
        for p in rl:
            a = archetype(p)
            if a is None or not base_only(p):
                continue
            g = geom(p)
            gw[(a, g)].add((sc, sd))
            aw[(a, g, ants(p))].add((sc, sd))
            ex.setdefault((a, g, ants(p)), p)
    out = []
    for a in ("asset_collar", "head_coating", "interface_hardening",
              "line_and_attack"):
        cands = [(len(w), k) for k, w in gw.items() if k[0] == a]
        if not cands:
            print("WARNING: no exemplar for", a)
            continue
        ng, (_a, g) = max(cands)
        na, k3 = max((len(w), k) for k, w in aw.items()
                     if k[0] == a and k[1] == g)
        p = dict(ex[k3])
        p["t_min"] = 0.0
        p["archetype"] = a
        p["geometry"] = g
        p["n_worlds_geometry"] = ng
        p["n_worlds_exact_antecedent"] = na
        out.append(p)
    return out


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
                    wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, (f"K2/{mode}/{sc}/{seed}: {type(exc).__name__}: {exc}\n"
                      + traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="canon", choices=("canon", "own"))
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenarios", default=",".join(SCEN))
    ap.add_argument("--seeds", default="201-230",
                    help="worlds to EVALUATE on")
    ap.add_argument("--derive-seeds", default="",
                    help="canon mode: derive the canonical tactics from "
                         "these seeds only (e.g. 201-215), evaluate on "
                         "--seeds; empty = derive from all worlds")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = L.load_admitted(args.src, exp="E9", arm="LLMScreen")
    scs = [s for s in args.scenarios.split(",") if s]
    seeds = C._parse_seeds(args.seeds)
    canon = []
    if args.mode == "canon":
        src_recs = recs
        if args.derive_seeds:
            ds = set(C._parse_seeds(args.derive_seeds))
            src_recs = {k: v for k, v in recs.items() if k[1] in ds}
            print(f"deriving the tactics from {len(src_recs)} worlds "
                  f"(seeds {args.derive_seeds}), evaluating on seeds "
                  f"{args.seeds}")
        canon = canonical_set(src_recs)
        with open(os.path.join(args.out, "k2_doctrine.json"), "w",
                  encoding="utf-8") as f:
            json.dump(canon, f, indent=1, default=str)
        print("canonical extension of the doctrine:")
        for p in canon:
            print(f"  {p['archetype']:20s} ({p['kind']}, geometry in "
                  f"{p['n_worlds_geometry']} worlds, antecedent in "
                  f"{p['n_worlds_exact_antecedent']}): {p['rule'][:110]}")
    path = os.path.join(args.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["mode"], r["scenario"], int(r["seed"]))
                    for r in csv.DictReader(f)}
    jobs = []
    for sc in scs:
        for sd in seeds:
            if (args.mode, sc, sd) in done:
                continue
            if args.mode == "canon":
                script = [dict(p) for p in canon]
            else:
                script = [dict(p, t_min=0.0) for p in recs.get((sc, sd), [])]
            jobs.append((args.mode, sc, sd, script))
    print(f"K2 {args.mode}: {len(jobs)} runs to do ({len(done)} done)",
          flush=True)
    fields = ["mode", "scenario", "seed", "n_installed", "burned_ha",
              "success", "pop_affected", "j_phys", "j_total", "j_resp",
              "j_delay", "rules_final", "wall_s"]
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
              f"s{row['seed']} ({row['n_installed']} objects): burned "
              f"{row['burned_ha']} ha, success {row['success']}, "
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
