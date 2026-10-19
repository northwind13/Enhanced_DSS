# -*- coding: utf-8 -*-
"""Leave-one-out replay of the admitted objects of a VODA campaign.

For every world of the source campaign (default experiments/benchmark/out_nolib,
experiment E5) that admitted at least one object, the world is replayed with the
closed layer alone (arm F5Ev, no proposer) while the recorded admitted objects
are re-installed at their recorded minutes:

  removed = none   every admitted object re-installed (replay reference)
  removed = all    nothing re-installed (closed-layer replay)
  removed = k      object k left out, together with later objects that order it

The difference between 'none' and 'k' attributes the realized outcome to object k
without the proposer's run-to-run variance. Resumable: finished rows are skipped.

  python experiments\benchmark\loo.py --workers 2
  python experiments\benchmark\loo.py --scenarios S4 --seeds 213 --workers 1
"""
import argparse, csv, json, os, sys, time, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from open_space import campaign as C          # noqa: E402

SRC = os.path.join(HERE, "out_nolib")
OUT = os.path.join(HERE, "out_loo")


def load_admitted(src, exp="E5", arm="LLM"):
    recs = collections.defaultdict(list)
    with open(os.path.join(src, "proposals.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("exp") == exp and d.get("arm") == arm and d.get("accepted"):
                recs[(d["scenario"], int(d["seed"]))].append(d)
    for k in recs:
        recs[k].sort(key=lambda r: float(r["t_min"]))
    return recs


def new_object_name(rec):
    kind = rec.get("kind")
    if kind in ("actuator", "macro"):
        _a, cons = C._parse_rule_text(rec["rule"])
        for i, _x in cons:
            if i not in C.dss_rules.INTERVENTIONS:
                return i
    if kind == "concept" and rec.get("concept"):
        return rec["concept"]["name"]
    return None


def script_without(recs, k):
    """All records except k and the later ones that depend on k's object."""
    if k == "none":
        return list(recs)
    if k == "all":
        return []
    k = int(k)
    gone = {new_object_name(recs[k])} - {None}
    out = []
    for i, r in enumerate(recs):
        if i == k:
            continue
        ants, cons = C._parse_rule_text(r["rule"])
        uses = {v for v, _t in ants} | {c for c, _x in cons}
        if uses & gone:
            nm = new_object_name(r)
            if nm:
                gone.add(nm)
            continue
        out.append(r)
    return out


def _job(args):
    sc, seed, removed, script, kind, name = args
    try:
        row, _c, _a, _m = C.run_once("LOO", sc, "F5Ev", seed, script=script)
        return dict(scenario=sc, seed=seed, removed=removed, removed_kind=kind,
                    removed_name=name, n_installed=len(script),
                    burned_ha=row["burned_ha"], success=row["success"],
                    pop_affected=row["pop_affected"], j_phys=row["j_phys"],
                    j_total=row["j_total"], j_resp=row["j_resp"],
                    rules_final=row["rules_final"], wall_s=row["wall_s"]), None
    except Exception as exc:
        import traceback
        return None, f"LOO/{sc}/{seed}/{removed}: {type(exc).__name__}: {exc}\n" + traceback.format_exc()[-800:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--controls-only", action="store_true",
                    help="only the 'none' and 'all' replays")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = load_admitted(args.src)
    scs = set(s for s in args.scenarios.split(",") if s)
    seeds = set(int(s) for s in args.seeds.split(",") if s)
    path = os.path.join(args.out, "loo.csv")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["scenario"], int(r["seed"]), r["removed"]) for r in csv.DictReader(f)}
    jobs = []
    for (sc, seed), rl in sorted(recs.items()):
        if scs and sc not in scs:
            continue
        if seeds and seed not in seeds:
            continue
        keys = ["none", "all"] + ([] if args.controls_only else [str(i) for i in range(len(rl))])
        for k in keys:
            if (sc, seed, k) in done:
                continue
            kind = rl[int(k)]["kind"] if k not in ("none", "all") else ""
            name = (new_object_name(rl[int(k)]) or "rule") if k not in ("none", "all") else ""
            jobs.append((sc, seed, k, script_without(rl, k), kind, name))
    print(f"LOO: {len(jobs)} replays to do ({len(done)} done) over {len(recs)} worlds", flush=True)
    import multiprocessing as mp
    t0 = time.time(); n = 0
    fields = ["scenario", "seed", "removed", "removed_kind", "removed_name", "n_installed",
              "burned_ha", "success", "pop_affected", "j_phys", "j_total", "j_resp", "rules_final", "wall_s"]

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
            wri = csv.DictWriter(f, fieldnames=fields)
            if new:
                wri.writeheader()
            wri.writerow(row)
        print(f"[{n}/{len(jobs)}] {row['scenario']} s{row['seed']} removed={row['removed']} "
              f"({row['removed_kind']} {row['removed_name']}): burned {row['burned_ha']} ha, "
              f"success {row['success']}, {row['wall_s']}s (elapsed {time.time() - t0:.0f}s)", flush=True)

    if args.workers <= 1:
        for j in jobs:
            sink(_job(j))
    else:
        with mp.Pool(args.workers) as pool:
            for res in pool.imap_unordered(_job, jobs):
                sink(res)
    print("done", flush=True)


if __name__ == "__main__":
    main()
