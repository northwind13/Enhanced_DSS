"""Benchmark tables: No DSS / F5 / F5+EvFIS / DisasterAware.

Reads the runs.csv written by experiments/open_space/campaign.py (arms T0,
F5, F5Ev, LLM) and writes, under <out>/report/:

  table1_outcome.csv / .md     burned area, affected population, extinguished
                               share per scenario and configuration (mean,
                               95 % half-width)
  table2_cost.csv / .md        cost components J_burn, J_asset, J_pop,
                               J_resp, J_delay, J_phys, J_total per scenario
                               and configuration
  table3_paired.csv / .md      paired differences on the same worlds:
                               DisasterAware vs each reference, F5+Ev vs F5,
                               F5 vs No DSS
  table4_adaptation.csv / .md  rules at episode end, stage-3 attempts and
                               admissions, LLM calls, wall time
  fig_burned.png, fig_curves.png
  summary.txt

  python experiments/benchmark/benchmark_tables.py --out experiments/benchmark/out
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "article_v4"))
import results_v4 as R   # noqa: E402

NAMES = {"T0": "No DSS", "F5": "F5", "F5Ev": "F5+EvFIS", "DA": "DisasterAware"}
R.LABEL.update({"T0": "No DSS", "F5": "F5", "F5Ev": "F5+EvFIS", "DA": "DisasterAware"})


def _md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def _write(path_base, header, rows):
    with open(path_base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(path_base + ".md", "w", encoding="utf-8") as f:
        f.write(_md(header, rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="campaign output dir (runs.csv, curves.csv)")
    ap.add_argument("--da-arm", default="LLM")
    ap.add_argument("--scenarios", default="S1,S2,S3,S4,S5")
    ap.add_argument("--arms", default="T0,F5,F5Ev,DA", help="columns, e.g. T0,F5,DA to leave evFIS out")
    ap.add_argument("--report", default="report", help="output subfolder under --out")
    args = ap.parse_args()
    R.SCEN[:] = [s for s in args.scenarios.split(",") if s]
    R.ARMS[:] = [a for a in args.arms.split(",") if a]
    runs_p = os.path.join(args.out, "runs.csv")
    curves_p = os.path.join(args.out, "curves.csv")
    runs, curves = R.load_open([runs_p], [curves_p], args.da_arm)
    runs = [r for r in runs if r["arm"] in R.ARMS]
    curves = {k: v for k, v in curves.items() if k[1] in R.ARMS} if isinstance(curves, dict) else curves
    runs = R.balanced(runs)
    if not runs:
        print("no balanced runs yet (every scenario needs all requested arms on the same seeds)")
        return
    rep = os.path.join(args.out, args.report)
    os.makedirs(rep, exist_ok=True)
    out_rows = R.outcome_table(runs)
    pair_rows = R.paired_table(runs)
    o = {(r["scenario"], r["arm"]): r for r in out_rows}
    arms = R.ARMS
    scen = R.SCEN + ["Avg"]
    seeds = sorted({r["seed"] for r in runs})

    # table 1: outcome
    h = ["Scenario", "Measure"] + [NAMES[a] for a in arms]
    rows = []
    for sc in scen:
        rows.append([sc, "burned area (ha)"] + [f"{o[(sc,a)]['burned']:.1f} ± {o[(sc,a)]['burned_hw']:.1f}" for a in arms])
        rows.append([sc, "affected population"] + [f"{o[(sc,a)]['pop']:.0f}" for a in arms])
        rows.append([sc, "extinguished within episode (%)"] + [f"{o[(sc,a)]['success_pct']:.0f}" for a in arms])
    _write(os.path.join(rep, "table1_outcome"), h, rows)

    # table 2: cost components
    h = ["Scenario", "Cost term"] + [NAMES[a] for a in arms]
    rows = []
    for sc in scen:
        for k, lab in (("j_burn", "J_burn"), ("j_asset", "J_asset"), ("j_pop", "J_pop"),
                       ("j_resp", "J_resp"), ("j_delay", "J_delay"), ("j_phys", "J_phys"), ("j_total", "J_total")):
            rows.append([sc, lab] + [f"{o[(sc,a)][k]:.3f}" for a in arms])
    _write(os.path.join(rep, "table2_cost"), h, rows)

    # table 3: paired
    h = ["Comparison", "Scenario", "n worlds", "burned area saved (ha)", "better in", "population saved", "J_phys saved", "J_total saved", "extinguished (points)"]
    lab = {"DA-T0": "DisasterAware vs No DSS", "F5-T0": "F5 vs No DSS", "F5Ev-F5": "F5+EvFIS vs F5",
           "DA-F5": "DisasterAware vs F5", "DA-F5Ev": "DisasterAware vs F5+EvFIS"}
    rows = []
    for r in pair_rows:
        rows.append([lab[r["pair"]], r["scenario"], r["n"],
                     f"{-r['d_burned']:.1f} ± {r['d_burned_hw']:.1f}", f"{r['wins']}/{r['n']}",
                     f"{-r['d_pop']:.0f} ± {r['d_pop_hw']:.0f}",
                     f"{-r['d_jphys']:.3f} ± {r['d_jphys_hw']:.3f}",
                     f"{-r['d_jtotal']:.3f} ± {r['d_jtotal_hw']:.3f}",
                     f"{r['d_success_pts']:+.0f}"])
    _write(os.path.join(rep, "table3_paired"), h, rows)

    # table 4: adaptation and cost of running
    h = ["Configuration", "rules at episode end", "stage-3 attempts", "stage-3 admitted", "admission rate (%)", "wall s / run"]
    rows = []
    for a in arms:
        rr = [x for x in runs if x["arm"] == a]
        t3 = float(np.nansum([x["tried_3"] for x in rr]))
        a3 = float(np.nansum([x["acc_3"] for x in rr]))
        rows.append([NAMES[a], f"{np.nanmean([x['rules_final'] for x in rr]):.1f}", f"{t3:.0f}", f"{a3:.0f}",
                     f"{100 * a3 / t3:.0f}" if t3 else "-", f"{np.nanmean([x['wall_s'] for x in rr]):.0f}"])
    _write(os.path.join(rep, "table4_adaptation"), h, rows)

    R.fig_ladder(out_rows, rep)
    os.replace(os.path.join(rep, "fig11_ladder_burned.png"), os.path.join(rep, "fig_burned.png"))
    if curves:
        R.fig_curves(curves, rep, scen=tuple(R.SCEN[:2]) if len(R.SCEN) < 4 else ("S2", "S4"))
        if os.path.exists(os.path.join(rep, "fig12_curves.png")):
            os.replace(os.path.join(rep, "fig12_curves.png"), os.path.join(rep, "fig_curves.png"))

    lines = [f"benchmark: {len(runs)} runs, {len(seeds)} worlds per scenario, scenarios {','.join(R.SCEN)}, DisasterAware arm = {args.da_arm}", ""]
    for a in arms:
        r = o[("Avg", a)]
        lines.append(f"{NAMES[a]:<14} burned {r['burned']:6.1f} ha  extinguished {r['success_pct']:3.0f} %  population {r['pop']:6.0f}  Jphys {r['j_phys']:.3f}  Jtot {r['j_total']:.3f}")
    lines.append("")
    for r in pair_rows:
        if r["scenario"] == "Avg":
            lines.append(f"{lab[r['pair']]:<28} burned {-r['d_burned']:6.1f} ± {r['d_burned_hw']:.1f} ha, better in {r['wins']}/{r['n']}, extinguished {r['d_success_pts']:+.0f} points")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(rep, "summary.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
