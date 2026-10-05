"""Results for the IEEE Systems Journal article, v4 (benefits of DisasterAware).

Four configurations on the same worlds: T0 (no decision support), F5 (five
seed rules, static), F5+Ev (stages 1 and 2) and DisasterAware (all three
stages). Two input formats are read:

  thesis   experiments/out_n50/ladder_runs.csv (+ ladder_curves.csv), arms
           Test0 / F5 / F5Ev / F5EvAI, 50 worlds per scenario, episodes to
           12 h; the DisasterAware column there was fed by the stand-in
           proposer.
  open     experiments/open_space/<out>/runs.csv (+ curves.csv), arms
           T0 / F5 / F5Ev / <llm arm>, as written by campaign.py; the
           DisasterAware column is the LLM arm when --da-arm LLM.

Usage
  python experiments/article_v4/results_v4.py --source thesis \
      --runs experiments/out_n50/ladder_runs.csv --out experiments/article_v4/out_thesis
  python experiments/article_v4/results_v4.py --source open --da-arm LLM \
      --runs experiments/open_space/out_llm/runs.csv \
      --extra experiments/open_space/out/runs.csv --out experiments/article_v4/out_llm

Writes tables/*.csv, figures/*.png and fill.json (placeholder values for
write_article_v4.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "legend.frameon": False})

ARMS = ["T0", "F5", "F5Ev", "DA"]
LABEL = {"T0": "T0 (no DSS)", "F5": "F5", "F5Ev": "F5+Ev", "DA": "DisasterAware"}
COLOR = {"T0": "#9a9a9a", "F5": "#6f6f6f", "F5Ev": "#3C6E9F", "DA": "#8B2635"}
SCEN = ["S1", "S2", "S3", "S4", "S5"]
POP_PERSONS = 9e-4   # population density sum -> persons (thesis ladder_report)


def _f(x, d=np.nan):
    try:
        return float(x)
    except Exception:
        return d


def _ci(xs):
    a = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return np.nan, np.nan, 0
    hw = 1.96 * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0
    return float(a.mean()), float(hw), int(a.size)


# ------------------------------------------------------------------ load
def load_thesis(runs_path, curves_path, da_arm="F5EvAI"):
    amap = {"Test0": "T0", "F5": "F5", "F5Ev": "F5Ev", da_arm: "DA"}
    runs = []
    with open(runs_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["arm"] not in amap or r["scenario"] not in SCEN:
                continue
            runs.append(dict(
                scenario=r["scenario"], arm=amap[r["arm"]], seed=int(r["seed"]),
                burned=_f(r["burned_ha"]), pop=_f(r["pop_affected"]) * POP_PERSONS,
                success=_f(r["success"]), out_min=_f(r["out_min"]),
                j_burn=_f(r["end_j_burn"]), j_asset=_f(r["end_j_asset"]),
                j_pop=_f(r["end_j_pop"]), j_resp=_f(r["end_j_resp"]),
                j_delay=_f(r["end_j_delay"]), j_total=_f(r["end_j_total"]),
                j_phys=_f(r["end_j_phys"]), rules_final=_f(r.get("rules_final")),
                adapt_share=_f(r.get("adapt_share")),
                tried_3=_f(r.get("tried_3")), acc_3=_f(r.get("acc_3")),
                wall_s=_f(r.get("wall_s"))))
    curves = defaultdict(list)   # (scenario, arm, seed) -> [(t, burned_cells)]
    if curves_path and os.path.exists(curves_path):
        with open(curves_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["arm"] in amap and r["scenario"] in SCEN:
                    curves[(r["scenario"], amap[r["arm"]], int(r["seed"]))].append(
                        (float(r["t_min"]), float(r["burned_cells"])))
    return runs, curves


def load_open(runs_paths, curves_paths, da_arm="LLM"):
    amap = {"T0": "T0", "F5": "F5", "F5Ev": "F5Ev", da_arm: "DA"}
    runs, seen = [], set()
    for p in runs_paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("exp") not in ("E2", None, "") and r.get("exp") != "E2":
                    continue
                if r["arm"] not in amap or r["scenario"] not in SCEN:
                    continue
                key = (r["scenario"], amap[r["arm"]], int(r["seed"]))
                if key in seen:
                    continue
                seen.add(key)
                runs.append(dict(
                    scenario=r["scenario"], arm=amap[r["arm"]], seed=int(r["seed"]),
                    burned=_f(r["burned_ha"]), pop=_f(r["pop_affected"]) * POP_PERSONS,
                    success=_f(r["success"]), out_min=_f(r["out_min"]),
                    j_burn=_f(r["j_burn"]), j_asset=_f(r["j_asset"]),
                    j_pop=_f(r["j_pop"]), j_resp=_f(r["j_resp"]),
                    j_delay=_f(r["j_delay"]), j_total=_f(r["j_total"]),
                    j_phys=_f(r["j_phys"]), rules_final=_f(r.get("rules_final")),
                    adapt_share=np.nan,
                    tried_3=_f(r.get("tried_3")), acc_3=_f(r.get("acc_3")),
                    wall_s=_f(r.get("wall_s"))))
    curves = defaultdict(list)
    for p in curves_paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            rd = csv.reader(f)
            for row in rd:
                # campaign.py writes: exp, scenario, arm, seed, t_min, burned_cells, j_phys
                if len(row) < 6:
                    continue
                try:
                    exp, sc, arm, sd, t, b = row[0], row[1], row[2], int(row[3]), float(row[4]), float(row[5])
                except Exception:
                    continue
                if exp != "E2" or arm not in amap or sc not in SCEN:
                    continue
                curves[(sc, amap[arm], sd)].append((t, b))
    return runs, curves


# ---------------------------------------------------------------- tables
def balanced(runs):
    """Keep only seeds present for all four arms in a scenario."""
    have = defaultdict(set)
    for r in runs:
        have[(r["scenario"], r["arm"])].add(r["seed"])
    out = []
    for r in runs:
        sc = r["scenario"]
        common = set.intersection(*(have[(sc, a)] for a in ARMS if (sc, a) in have)) \
            if all((sc, a) in have for a in ARMS) else set()
        if r["seed"] in common:
            out.append(r)
    return out


def outcome_table(runs):
    g = defaultdict(list)
    for r in runs:
        g[(r["scenario"], r["arm"])].append(r)
    rows = []
    for arm in ARMS:
        for sc in SCEN + ["Avg"]:
            rr = [x for x in runs if x["arm"] == arm] if sc == "Avg" \
                else g.get((sc, arm), [])
            if not rr:
                continue
            b, bh, n = _ci([x["burned"] for x in rr])
            jp, jph, _ = _ci([x["j_phys"] for x in rr])
            jt, jth, _ = _ci([x["j_total"] for x in rr])
            rows.append(dict(
                scenario=sc, arm=arm, n=n, burned=b, burned_hw=bh,
                pop=float(np.mean([x["pop"] for x in rr])),
                success_pct=100 * float(np.mean([x["success"] for x in rr])),
                j_burn=float(np.mean([x["j_burn"] for x in rr])),
                j_asset=float(np.mean([x["j_asset"] for x in rr])),
                j_pop=float(np.mean([x["j_pop"] for x in rr])),
                j_resp=float(np.mean([x["j_resp"] for x in rr])),
                j_delay=float(np.mean([x["j_delay"] for x in rr])),
                j_phys=jp, j_phys_hw=jph, j_total=jt, j_total_hw=jth,
                rules_final=float(np.nanmean([x["rules_final"] for x in rr])),
                adapt_share=(float(np.nanmean([x["adapt_share"] for x in rr])) if any(np.isfinite(x["adapt_share"]) for x in rr) else float("nan")),
                tried_3=float(np.nansum([x["tried_3"] for x in rr])),
                acc_3=float(np.nansum([x["acc_3"] for x in rr]))))
    return rows


def paired_table(runs):
    by = {(r["scenario"], r["arm"], r["seed"]): r for r in runs}
    pairs = [("DA", "T0"), ("F5", "T0"), ("F5Ev", "F5"), ("DA", "F5"), ("DA", "F5Ev")]
    rows = []
    for a, b in pairs:
        for sc in SCEN + ["Avg"]:
            scs = SCEN if sc == "Avg" else [sc]
            ds, dp, dj, dt, succ = [], [], [], [], []
            for (s_, arm, sd), r in by.items():
                if arm != a or s_ not in scs or (s_, b, sd) not in by:
                    continue
                q = by[(s_, b, sd)]
                ds.append(r["burned"] - q["burned"])
                dp.append(r["pop"] - q["pop"])
                dj.append(r["j_phys"] - q["j_phys"])
                dt.append(r["j_total"] - q["j_total"])
                succ.append(r["success"] - q["success"])
            if not ds:
                continue
            m, hw, n = _ci(ds)
            mp, hwp, _ = _ci(dp)
            mj, hwj, _ = _ci(dj)
            mt, hwt, _ = _ci(dt)
            rows.append(dict(pair=f"{a}-{b}", scenario=sc, n=n,
                             d_burned=m, d_burned_hw=hw,
                             wins=int(sum(1 for d in ds if d < 0)),
                             ties=int(sum(1 for d in ds if d == 0)),
                             d_pop=mp, d_pop_hw=hwp, d_jphys=mj, d_jphys_hw=hwj,
                             d_jtotal=mt, d_jtotal_hw=hwt,
                             d_success_pts=100 * float(np.mean(succ))))
    return rows


# --------------------------------------------------------------- figures
def fig_ladder(out_rows, figdir):
    """Grouped bars: burned area per scenario and arm, with 95 % CI."""
    fig, ax = plt.subplots(figsize=(3.4, 2.48))
    w = 0.8 / len(ARMS)
    for j, arm in enumerate(ARMS):
        vals = [next((r["burned"] for r in out_rows if r["scenario"] == sc and r["arm"] == arm), np.nan) for sc in SCEN]
        hws = [next((r["burned_hw"] for r in out_rows if r["scenario"] == sc and r["arm"] == arm), 0) for sc in SCEN]
        x = np.arange(len(SCEN)) + (j - (len(ARMS) - 1) / 2) * w
        ax.bar(x, vals, w, yerr=hws, color=COLOR[arm], label=LABEL[arm],
               error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(np.arange(len(SCEN)))
    ax.set_xticklabels(SCEN)
    ax.set_ylabel("burned area at episode end (ha)")
    ax.legend(fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.8)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()
    p = os.path.join(figdir, "fig11_ladder_burned.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_curves(curves, figdir, cell_ha=0.09, t_end=720.0, scen=("S2", "S4")):
    """Mean burned area against incident time for the four arms; a run that
    ends early keeps its final value to the end of the window."""
    if not curves:
        return None
    grid = np.arange(0.0, t_end + 1e-9, 12.0)
    fig, axes = plt.subplots(1, len(scen), figsize=(3.4 * len(scen) / 1.0 if len(scen) == 1 else 7.0, 2.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scen):
        for arm in ARMS:
            series = []
            for (s_, a_, sd), pts in curves.items():
                if s_ != sc or a_ != arm or not pts:
                    continue
                pts = sorted(pts)
                ts = np.array([p[0] for p in pts])
                bs = np.array([p[1] for p in pts]) * cell_ha
                y = np.interp(grid, ts, bs, left=0.0, right=bs[-1])
                series.append(y)
            if series:
                m = np.mean(series, axis=0)
                ax.plot(grid / 60.0, m, color=COLOR[arm], lw=1.2, label=LABEL[arm],
                        ls="-" if arm == "DA" else ("--" if arm == "F5Ev" else (":" if arm == "F5" else "-.")))
        ax.set_title(sc, fontsize=8)
        ax.set_xlabel("incident time (h)")
        ax.set_xlim(0, t_end / 60.0)
        ax.grid(lw=0.4, alpha=0.4)
    axes[0].set_ylabel("mean burned area (ha)")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    p = os.path.join(figdir, "fig12_curves.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ fill
def make_fill(out_rows, pair_rows, source, n_seeds):
    F = {"SRC": source, "NSEEDS": str(n_seeds)}
    o = {(r["scenario"], r["arm"]): r for r in out_rows}
    p = {(r["pair"], r["scenario"]): r for r in pair_rows}

    def g(sc, arm, k, fmt):
        r = o.get((sc, arm))
        return (fmt % r[k]) if r else "n/a"
    for arm in ARMS:
        for sc in SCEN + ["Avg"]:
            F[f"burn.{sc}.{arm}"] = g(sc, arm, "burned", "%.1f")
            F[f"burn0.{sc}.{arm}"] = g(sc, arm, "burned", "%.0f")
            F[f"succ.{sc}.{arm}"] = g(sc, arm, "success_pct", "%.0f")
            F[f"pop.{sc}.{arm}"] = g(sc, arm, "pop", "%.0f")
            F[f"jphys.{sc}.{arm}"] = g(sc, arm, "j_phys", "%.3f")
            F[f"jtot.{sc}.{arm}"] = g(sc, arm, "j_total", "%.3f")
            F[f"jresp.{sc}.{arm}"] = g(sc, arm, "j_resp", "%.3f")
            F[f"jdelay.{sc}.{arm}"] = g(sc, arm, "j_delay", "%.3f")
        F[f"rules.{arm}"] = g("Avg", arm, "rules_final", "%.1f")
        r = o.get(("Avg", arm))
        if r and np.isfinite(r["adapt_share"]):
            v = r["adapt_share"] * (100.0 if r["adapt_share"] <= 1.0 else 1.0)
            F[f"adapt_share.{arm}"] = f"{v:.0f}"
        else:
            F[f"adapt_share.{arm}"] = "n/a"
        F[f"tried3.{arm}"] = g("Avg", arm, "tried_3", "%.0f")
        F[f"acc3.{arm}"] = g("Avg", arm, "acc_3", "%.0f")

    def pct(a, b):
        return f"{100 * (a - b) / a:.0f}"
    if ("Avg", "T0") in o and ("Avg", "DA") in o:
        F["red.DA.T0"] = pct(o[("Avg", "T0")]["burned"], o[("Avg", "DA")]["burned"])
        F["red.F5.T0"] = pct(o[("Avg", "T0")]["burned"], o[("Avg", "F5")]["burned"])
        F["red.DA.F5"] = pct(o[("Avg", "F5")]["burned"], o[("Avg", "DA")]["burned"])
        F["red.DA.F5Ev"] = pct(o[("Avg", "F5Ev")]["burned"], o[("Avg", "DA")]["burned"])
        F["popred.DA.F5"] = pct(o[("Avg", "F5")]["pop"], o[("Avg", "DA")]["pop"])
        F["popred.DA.F5Ev"] = pct(o[("Avg", "F5Ev")]["pop"], o[("Avg", "DA")]["pop"])
        F["popred.DA.T0"] = pct(o[("Avg", "T0")]["pop"], o[("Avg", "DA")]["pop"])
        s5, s4 = o.get(("S5", "DA")), o.get(("S4", "DA"))
        if s5 and s4:
            F["s5loss"] = f"{100 * (s5['burned'] - s4['burned']) / s4['burned']:.0f}"
        F["succ_ratio"] = f"{o[('Avg', 'DA')]['success_pct'] / max(o[('Avg', 'F5')]['success_pct'], 1e-9):.1f}"
    for pr in ("DA-T0", "F5-T0", "F5Ev-F5", "DA-F5", "DA-F5Ev"):
        for sc in SCEN + ["Avg"]:
            r = p.get((pr, sc))
            if r:
                F[f"d.{pr}.{sc}"] = f"{-r['d_burned']:.0f} ± {r['d_burned_hw']:.0f}"
                F[f"d1.{pr}.{sc}"] = f"{-r['d_burned']:.1f} ± {r['d_burned_hw']:.1f}"
                F[f"wins.{pr}.{sc}"] = f"{r['wins']}"
                F[f"n.{pr}.{sc}"] = f"{r['n']}"
                F[f"dpop.{pr}.{sc}"] = f"{-r['d_pop']:.0f} ± {r['d_pop_hw']:.0f}"
                F[f"djphys.{pr}.{sc}"] = f"{-r['d_jphys']:.3f} ± {r['d_jphys_hw']:.3f}"
                F[f"djtot.{pr}.{sc}"] = f"{-r['d_jtotal']:.3f} ± {r['d_jtotal_hw']:.3f}"
                F[f"dsucc.{pr}.{sc}"] = f"{r['d_success_pts']:.0f}"
    # families where DA has the lowest burned mean
    k = sum(1 for sc in SCEN if all(o.get((sc, "DA"), {}).get("burned", np.inf) <= o.get((sc, a), {}).get("burned", np.inf) for a in ("F5", "F5Ev")))
    F["da_lowest_families"] = str(k)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["thesis", "open"], default="thesis")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--curves", default=None)
    ap.add_argument("--extra", default=None, help="open: second runs.csv (T0/F5/F5Ev rows)")
    ap.add_argument("--extra-curves", default=None)
    ap.add_argument("--da-arm", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.source == "thesis":
        da = args.da_arm or "F5EvAI"
        curves = args.curves or os.path.join(os.path.dirname(args.runs), "ladder_curves.csv")
        runs, curves = load_thesis(args.runs, curves, da)
    else:
        da = args.da_arm or "LLM"
        paths = [args.runs] + ([args.extra] if args.extra else [])
        cps = [args.curves or os.path.join(os.path.dirname(args.runs), "curves.csv")]
        if args.extra:
            cps.append(args.extra_curves or os.path.join(os.path.dirname(args.extra), "curves.csv"))
        runs, curves = load_open(paths, cps, da)
    runs = balanced(runs)
    seeds = sorted({r["seed"] for r in runs})
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    out_rows = outcome_table(runs)
    pair_rows = paired_table(runs)
    for name, rows in (("outcomes", out_rows), ("paired", pair_rows)):
        with open(os.path.join(args.out, "tables", f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    fig_ladder(out_rows, os.path.join(args.out, "figures"))
    fig_curves(curves, os.path.join(args.out, "figures"))
    F = make_fill(out_rows, pair_rows, args.source, len(seeds))
    with open(os.path.join(args.out, "fill.json"), "w", encoding="utf-8") as f:
        json.dump(F, f, indent=1, ensure_ascii=False)
    print(f"{len(runs)} runs, {len(seeds)} seeds per scenario, source {args.source}, DA arm {da}")
    for arm in ARMS:
        r = next(x for x in out_rows if x["scenario"] == "Avg" and x["arm"] == arm)
        print(f"  {LABEL[arm]:<14} burned {r['burned']:6.1f}  success {r['success_pct']:3.0f}%  pop {r['pop']:6.0f}  Jphys {r['j_phys']:.3f}  Jtot {r['j_total']:.3f}")
    for pr in ("DA-F5", "DA-F5Ev"):
        r = next(x for x in pair_rows if x["pair"] == pr and x["scenario"] == "Avg")
        print(f"  {pr}: {-r['d_burned']:.1f} ± {r['d_burned_hw']:.1f} ha, wins {r['wins']}/{r['n']}")


if __name__ == "__main__":
    main()
