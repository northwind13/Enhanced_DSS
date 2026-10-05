"""Tables for the open intervention space campaign.

  python experiments/open_space/report.py --out experiments/open_space/out
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x, d=np.nan):
    try:
        return float(x)
    except Exception:
        return d


def _ci(xs):
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return np.nan, np.nan, 0
    a = np.asarray(xs, dtype=float)
    hw = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return float(a.mean()), float(hw), len(a)


def table_outcomes(rows, exp):
    g = defaultdict(list)
    for r in rows:
        if r["exp"] != exp:
            continue
        g[(r["scenario"], r["arm"])].append(r)
    lines = [f"\n== {exp}: outcome per scenario and arm "
             f"(mean, 95% half-width, n) ==",
             f"{'sc':<4}{'arm':<16}{'burned ha':>18}{'J_phys':>18}"
             f"{'success%':>10}{'pop aff.':>12}{'evac ord.':>10}"
             f"{'sited ord.':>11}{'st3 acc/try':>12}{'actuat.':>8}"]
    out = []
    for (sc, arm) in sorted(g):
        rr = g[(sc, arm)]
        b, bh, n = _ci([_f(r["burned_ha"]) for r in rr])
        j, jh, _ = _ci([_f(r["j_phys"]) for r in rr])
        s = 100 * np.mean([_f(r["success"]) for r in rr])
        pa, _, _ = _ci([_f(r["pop_affected"]) for r in rr])
        ev = np.mean([_f(r["evac_orders"]) for r in rr])
        so = np.mean([_f(r["sited_orders"]) for r in rr])
        a3 = sum(_f(r["acc_3"]) for r in rr)
        t3 = sum(_f(r["tried_3"]) for r in rr)
        na = np.mean([_f(r["n_actuators"]) for r in rr])
        lines.append(f"{sc:<4}{arm:<16}{b:>10.1f} ±{bh:>5.1f} ({n:>2})"
                     f"{j:>10.3f} ±{jh:>5.3f}{s:>9.0f}%{pa:>12.0f}"
                     f"{ev:>10.1f}{so:>11.1f}{a3:>6.0f}/{t3:<5.0f}"
                     f"{na:>8.2f}")
        out.append(dict(exp=exp, scenario=sc, arm=arm, n=n,
                        burned_ha=round(b, 2), burned_hw=round(bh, 2),
                        j_phys=round(j, 4), j_phys_hw=round(jh, 4),
                        success_pct=round(s, 1), pop_affected=round(pa, 0),
                        evac_orders=round(ev, 2), sited_orders=round(so, 2),
                        acc_3=int(a3), tried_3=int(t3),
                        n_actuators=round(na, 2)))
    return "\n".join(lines), out


def table_paired(rows, exp, ref="F5Ev"):
    """Paired deltas against the closed arm on the same (scenario, seed)."""
    by = {}
    for r in rows:
        if r["exp"] == exp:
            by[(r["scenario"], r["arm"], int(r["seed"]))] = r
    arms = sorted({k[1] for k in by} - {ref})
    scs = sorted({k[0] for k in by})
    lines = [f"\n== {exp}: paired difference in burned area against "
             f"{ref} (negative = open arm burned less) =="]
    out = []
    for sc in scs:
        for arm in arms:
            ds = []
            for (s_, a_, sd), r in by.items():
                if s_ == sc and a_ == arm and (sc, ref, sd) in by:
                    ds.append(_f(r["burned_ha"])
                              - _f(by[(sc, ref, sd)]["burned_ha"]))
            if ds:
                m, hw, n = _ci(ds)
                wins = sum(1 for d in ds if d < 0)
                lines.append(f"{sc:<4}{arm:<16}{m:>9.1f} ±{hw:>5.1f} ha "
                             f"(n={n}, open better in {wins}/{n})")
                out.append(dict(exp=exp, scenario=sc, arm=arm, n=n,
                                delta_ha=round(m, 2), hw=round(hw, 2),
                                wins=wins))
    return "\n".join(lines), out


def table_rediscovery(rows, exp):
    g = defaultdict(list)
    for r in rows:
        if r["exp"] == exp and _f(r["withhold"]) == 1:
            g[(r["scenario"], r["arm"])].append(r)
    lines = [f"\n== {exp}: rediscovery of the withheld doctrine "
             f"(share of runs; median minute of first admission) =="]
    out = []
    for (sc, arm) in sorted(g):
        rr = g[(sc, arm)]
        n = len(rr)
        cells = []
        rec = dict(exp=exp, scenario=sc, arm=arm, n=n)
        for tgt, col, tcol in (("draft", "redis_draft", "t_draft"),
                               ("coat", "redis_coat", "t_coat"),
                               ("ignite", "redis_ignite", "t_ignite")):
            k = sum(1 for r in rr if _f(r[col]) == 1)
            ts = [_f(r[tcol]) for r in rr if r[tcol] not in ("", None)]
            med = float(np.median(ts)) if ts else np.nan
            cells.append(f"{tgt}: {k}/{n}"
                         + (f" @{med:.0f} min" if ts else ""))
            rec[f"{tgt}_share"] = round(k / n, 3) if n else np.nan
            rec[f"{tgt}_median_min"] = round(med, 1) if ts else ""
        any_k = sum(1 for r in rr if max(_f(r["redis_draft"]),
                                         _f(r["redis_coat"]),
                                         _f(r["redis_ignite"])) == 1)
        rec["any_share"] = round(any_k / n, 3) if n else np.nan
        lines.append(f"{sc:<4}{arm:<16}" + "   ".join(cells)
                     + f"   any: {any_k}/{n}")
        out.append(rec)
    return "\n".join(lines), out


def table_proposers(rows, props, exp):
    g = defaultdict(list)
    for r in rows:
        if r["exp"] == exp and r["proposer"]:
            g[r["arm"]].append(r)
    pa = defaultdict(lambda: defaultdict(int))
    for p in props:
        if p["exp"] != exp:
            continue
        a = p["arm"]
        pa[a]["attempts"] += 1
        pa[a]["accepted"] += 1 if p["accepted"] else 0
        pa[a][f"kind_{p['kind']}"] += 1
        if p["accepted"]:
            pa[a][f"acc_{p['kind']}"] += 1
        gate = str(p.get("gate") or "")
        if not p["accepted"]:
            key = ("S0" if "S0" in gate else
                   "G5b" if "G5b" in gate else
                   "G5" if "G5" in gate else
                   "G4" if "G4" in gate else
                   "G3" if "G3" in gate else "cheap")
            pa[a][f"rej_{key}"] += 1
    lines = [f"\n== {exp}: proposer accounting =="]
    out = []
    for arm in sorted(g):
        rr = g[arm]
        fc = sum(_f(r["prop_forecasts"]) for r in rr)
        calls = sum(_f(r["prop_calls"]) for r in rr)
        cand = sum(_f(r["prop_candidates"]) for r in rr)
        acc = pa[arm]["accepted"]
        att = pa[arm]["attempts"]
        lines.append(
            f"{arm:<16} runs {len(rr):>3}  calls {calls:>5.0f}  "
            f"candidates priced {cand:>6.0f}  forecasts {fc:>7.0f}  "
            f"attempts {att:>4}  admitted {acc:>3} "
            f"({100 * acc / max(att, 1):.0f}%)  forecasts/admission "
            f"{fc / max(acc, 1):>7.1f}  kinds "
            + ", ".join(f"{k[4:]} {pa[arm][k]}/{pa[arm]['kind_' + k[4:]]}"
                        for k in sorted(pa[arm]) if k.startswith("acc_"))
            + "  rejections "
            + ", ".join(f"{k[4:]} {pa[arm][k]}" for k in sorted(pa[arm])
                        if k.startswith("rej_")))
        out.append(dict(exp=exp, arm=arm, runs=len(rr), calls=calls,
                        candidates=cand, forecasts=fc, attempts=att,
                        admitted=acc,
                        forecasts_per_admission=round(fc / max(acc, 1), 1),
                        **{k: v for k, v in pa[arm].items()
                           if k not in ('attempts', 'accepted')}))
    return "\n".join(lines), out


def _split_margins(rows, props):
    """Experiments with several gate margins: label the arm by margin.

    Older proposal records carry no margin field; they were appended in the
    same order as the run rows, one block of ``tried_3`` records per run, so
    the margin is recovered by walking the two files together.
    """
    for exp in sorted({r["exp"] for r in rows}):
        er = [r for r in rows if r["exp"] == exp]
        if len({r.get("margin", "") for r in er}) <= 1:
            continue
        ep = [p for p in props if p["exp"] == exp]
        if ep and "margin" not in ep[0]:
            i = 0
            for r in er:
                n = int(float(r.get("tried_3", 0) or 0))
                for p in ep[i:i + n]:
                    p["margin"] = r["margin"]
                i += n
        for r in er:
            r["arm"] = f"{r['arm']}@{float(r['margin']):g}"
        for p in ep:
            if p.get("margin") not in (None, ""):
                p["arm"] = f"{p['arm']}@{float(p['margin']):g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()
    rows = _load(os.path.join(args.out, "runs.csv"))
    props = []
    pp = os.path.join(args.out, "proposals.jsonl")
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            props = [json.loads(l) for l in f if l.strip()]
    _split_margins(rows, props)
    exps = sorted({r["exp"] for r in rows})
    tdir = os.path.join(args.out, "tables")
    os.makedirs(tdir, exist_ok=True)
    text = []
    for exp in exps:
        for fn in (table_outcomes, table_paired, table_rediscovery):
            t, data = fn(rows, exp)
            text.append(t)
            if data:
                with open(os.path.join(tdir, f"{exp}_{fn.__name__[6:]}.csv"),
                          "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                    w.writeheader()
                    w.writerows(data)
        t, data = table_proposers(rows, props, exp)
        text.append(t)
        if data:
            keys = sorted({k for d in data for k in d})
            with open(os.path.join(tdir, f"{exp}_proposers.csv"), "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(data)
    report = "\n".join(text)
    print(report)
    with open(os.path.join(args.out, "summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
