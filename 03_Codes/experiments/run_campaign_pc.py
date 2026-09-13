"""Run the full Chapter 5 ladder campaign on this PC and produce the
thesis tables and figures from it.

    python experiments\run_campaign_pc.py              # run + report
    python experiments\run_campaign_pc.py --workers 6  # choose workers
    python experiments\run_campaign_pc.py --report     # report only

Everything is written to experiments/out_n50/ (the original
experiments/out/ of the thesis is never touched). The campaign is
resumable: finished (scenario, arm, seed) rows are skipped, so the
script can be stopped and restarted at any time. If out_n50/ already
holds rows from an earlier (cloud) session they are kept and only the
missing runs are computed.

Outputs in experiments/out_n50/:
  ladder_runs.csv, ladder_curves.csv, ladder_funnel.csv,
  ladder_products.jsonl                      raw campaign outputs
  table5_9_physical.csv                      Table 5.9 (mean, 95% CI)
  table5_10_cost.csv                         Table 5.10 (end-of-run costs)
  table5_11_funnel.csv                       stage counts for Table 5.11
  paired_deltas.csv                          paired differences vs T0/F5
  fig5_16_outcome_average.png, fig5_17_Jphys_average.png,
  fig5_18_Jtotal_average.png                 Figures 5.16 to 5.18
  summary.txt                                the numbers used in the text
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_n50")
SEEDS, SEED0 = 50, 201
ARMS = ["Test0", "F5", "F5Ev", "F5EvAI"]
LABELS = {"Test0": "T_0", "F5": "T_F5", "F5Ev": "T_F5+Ev",
          "F5EvAI": "T_DisasterAware"}
SCEN = ["S1", "S2", "S3", "S4", "S5"]
W = [1, 1, 1, 0.2, 0.2]      # cost weights, divided by 3.4
sys.path.insert(0, HERE)


def run_campaign(workers: int) -> None:
    import campaign5
    campaign5.OUT = OUT              # redirect every output file
    os.makedirs(OUT, exist_ok=True)
    sys.argv = ["campaign5.py", "--seeds", str(SEEDS), "--seed0",
                str(SEED0), "--workers", str(workers)]
    t0 = time.time()
    campaign5.main()
    print(f"campaign finished in {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------- report
def _ci(x):
    import numpy as np
    x = np.asarray(x, float)
    n = len(x)
    return 1.96 * x.std(ddof=1) / n ** 0.5 if n > 1 else 0.0


def report() -> None:
    import numpy as np
    import pandas as pd
    p = os.path.join(OUT, "ladder_runs.csv")
    d = pd.read_csv(p)
    d = d.drop_duplicates(["scenario", "arm", "seed"])
    n_expected = SEEDS * len(ARMS) * len(SCEN)
    lines = [f"runs: {len(d)} of {n_expected} "
             f"(seeds {SEED0}..{SEED0 + SEEDS - 1})"]
    if len(d) < n_expected:
        lines.append("WARNING: campaign incomplete, tables are partial")
    d["popp"] = d.pop_affected * 9e-4            # persons (cell density)

    # ---- Table 5.9: physical outcomes per scenario and averaged
    rows = []
    for arm in ARMS:
        for sc in SCEN + ["Avg"]:
            g = d[(d.arm == arm) & ((d.scenario == sc) if sc != "Avg"
                                    else True)]
            if sc == "Avg":                 # mean of scenario means
                gm = d[d.arm == arm].groupby("scenario").mean(
                    numeric_only=True)
                burned, ci = gm.burned_ha.mean(), _ci(
                    d[d.arm == arm].groupby("scenario").burned_ha
                    .mean())
            else:
                burned, ci = g.burned_ha.mean(), _ci(g.burned_ha)
            ext = g[g.out_min > 0]
            rows.append(dict(
                scenario=sc, arm=LABELS[arm], n=len(g),
                burned_ha=round(burned, 1), burned_ci=round(ci, 1),
                forest_ha=round(g.forest_ha.mean(), 1),
                pop_affected=round(g.popp.mean(), 0),
                evacuated=round(g.evacuated.mean(), 0),
                out_min=(round(ext.out_min.mean(), 0) if len(ext)
                         else "-"),
                success_pct=round(100 * g.success.mean(), 0),
                not_extinguished=f"{int((g.out_min <= 0).sum())}/{len(g)}"))
    t9 = pd.DataFrame(rows)
    t9.to_csv(os.path.join(OUT, "table5_9_physical.csv"), index=False)

    # ---- Table 5.10: end-of-run cost terms
    cols = ["end_j_burn", "end_j_asset", "end_j_pop", "end_j_resp",
            "end_j_delay", "end_j_total", "end_j_phys"]
    rows = []
    for arm in ARMS:
        for sc in SCEN + ["Avg"]:
            if sc == "Avg":
                g = d[d.arm == arm].groupby("scenario").mean(
                    numeric_only=True)
            else:
                g = d[(d.arm == arm) & (d.scenario == sc)]
            r = dict(scenario=sc, arm=LABELS[arm], n=len(g))
            for c in cols:
                r[c.replace("end_", "")] = round(g[c].mean(), 3)
            r["j_total_ci"] = round(_ci(g.end_j_total), 3)
            r["j_phys_ci"] = round(_ci(g.end_j_phys), 3)
            rows.append(r)
    t10 = pd.DataFrame(rows)
    t10.to_csv(os.path.join(OUT, "table5_10_cost.csv"), index=False)

    # ---- paired deltas (same world, same seed) with 95% CI
    piv = d.pivot_table(index=["scenario", "seed"], columns="arm",
                        values=["burned_ha", "end_j_phys",
                                "end_j_total"])
    rows = []
    for a, b in [("F5EvAI", "Test0"), ("F5EvAI", "F5"),
                 ("F5EvAI", "F5Ev"), ("F5Ev", "F5"), ("F5", "Test0")]:
        for sc in SCEN + ["Avg"]:
            sub = piv if sc == "Avg" else piv.loc[sc]
            r = dict(pair=f"{LABELS[a]} - {LABELS[b]}", scenario=sc)
            for v in ["burned_ha", "end_j_phys", "end_j_total"]:
                dd = (sub[(v, a)] - sub[(v, b)]).dropna()
                r[v + "_delta"] = round(dd.mean(), 3)
                r[v + "_ci"] = round(_ci(dd), 3)
                r[v + "_wins_pct"] = round(100 * (dd < 0).mean(), 0)
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "paired_deltas.csv"),
                              index=False)

    # ---- Table 5.11 funnel: trials and acceptances per stage
    rows = []
    for arm in ["F5Ev", "F5EvAI"]:
        g = d[d.arm == arm]
        for k in (1, 2, 3):
            tried, acc = g[f"tried_{k}"].sum(), g[f"acc_{k}"].sum()
            rows.append(dict(arm=LABELS[arm], stage=k, tried=int(tried),
                             accepted=int(acc),
                             accept_pct=round(100 * acc / tried, 1)
                             if tried else 0.0))
    fp = os.path.join(OUT, "ladder_funnel.csv")
    if os.path.exists(fp):
        f = pd.read_csv(fp)
        for (arm, gate), g in f.groupby(["arm", "gate"]):
            rows.append(dict(arm=LABELS.get(arm, arm), stage=f"gate:{gate}",
                             tried=len(g), accepted=int(g.accepted.sum()),
                             accept_pct=round(100 * g.accepted.mean(), 1)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table5_11_funnel.csv"),
                              index=False)

    # ---- headline numbers for the text
    m = t9[t9.scenario == "Avg"].set_index("arm")
    c = t10[t10.scenario == "Avg"].set_index("arm")
    b0, bf, be, ba = [m.loc[LABELS[a], "burned_ha"] for a in ARMS]
    lines += [
        f"mean burned area (ha): T0 {b0}, F5 {bf}, F5+Ev {be}, "
        f"DisasterAware {ba}",
        f"reduction vs no action: {100 * (1 - ba / b0):.0f} %",
        f"reduction vs static F5: {100 * (1 - ba / bf):.0f} %",
        f"mean physical cost: T0 {c.loc['T_0', 'j_phys']}, "
        f"F5 {c.loc['T_F5', 'j_phys']}, "
        f"DisasterAware {c.loc['T_DisasterAware', 'j_phys']}",
        f"mean total cost: T0 {c.loc['T_0', 'j_total']}, "
        f"F5 {c.loc['T_F5', 'j_total']}, "
        f"DisasterAware {c.loc['T_DisasterAware', 'j_total']}",
    ]
    s5 = t9[(t9.scenario == "S5") & (t9.arm == "T_DisasterAware")]
    s4 = t9[(t9.scenario == "S4") & (t9.arm == "T_DisasterAware")]
    if len(s4) and len(s5):
        lines.append(f"S4 -> S5 burned area, DisasterAware: "
                     f"{s4.burned_ha.iloc[0]} -> {s5.burned_ha.iloc[0]} ha")
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        figures(d)
    except Exception as e:                    # figures are optional
        print("figures skipped:", e)


def figures(d) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r"$T_0$", r"$T_{F5}$", r"$T_{F5+Ev}$", r"$T_{DisasterAware}$"]
    sc = d.groupby(["arm", "scenario"]).mean(numeric_only=True)
    m = sc.groupby("arm").mean().loc[ARMS]
    Wn = np.array(W) / 3.4
    RED, GRN, BLU, PUR, GRY = ("#c0392b", "#27ae60", "#2980b9",
                               "#8e44ad", "#95a5a6")

    def bars(ax, series, colors, names, fmt, ylim, ylabel, title,
             share=True):
        n = len(series); w = 0.8 / n; x = np.arange(len(ARMS))
        for k, (s, col, nm) in enumerate(zip(series, colors, names)):
            v = np.array(s); h = 100 * v / v[0] if share else v
            ax.bar(x + (k - (n - 1) / 2) * w, h, w, color=col, label=nm)
            for xi, hi, vi in zip(x + (k - (n - 1) / 2) * w, h, v):
                ax.text(xi, hi + (1.5 if share else 0.003), fmt(vi),
                        rotation=90, ha="center", va="bottom", fontsize=8)
        if share:
            ax.axhline(100, color="0.3", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(*ylim); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10); ax.tick_params(axis="y",
                                                         labelsize=8)
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=n,
                  frameon=False, fontsize=8, handlelength=1.5,
                  columnspacing=1.2)

    fig, ax = plt.subplots(figsize=(6.06, 4.42), dpi=200)
    bars(ax, [m.burned_ha, m.forest_ha, m.popp], [RED, GRN, BLU],
         ["Burned (ha)", "Burned Forest (ha)", "AffectedPop."],
         lambda v: f"{v:,.0f}" if v > 1000 else f"{v:.1f}", (0, 128),
         "share of the no-DSS run $T_0$ (%)", "S1–S5")
    plt.setp(ax.get_xticklabels(), rotation=15); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_16_outcome_average.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(6.07, 3.67), dpi=200)
    bars(ax, [m.end_j_burn, m.end_j_asset, m.end_j_pop], [RED, GRN, BLU],
         ["burned area", "asset loss", "population"],
         lambda v: f"{v:.3f}", (0, 128),
         "physical decision cost, share of $T_0$ (%)", "Avg")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_17_Jphys_average.png")); plt.close()

    fig, ax = plt.subplots(figsize=(6.07, 3.67), dpi=200)
    terms = [m.end_j_burn * Wn[0], m.end_j_asset * Wn[1],
             m.end_j_pop * Wn[2], m.end_j_resp * Wn[3],
             m.end_j_delay * Wn[4]]
    top = max(float(sum(terms).max()) * 1.35, 0.33)
    bars(ax, terms, [RED, GRN, BLU, PUR, GRY],
         ["burned area", "asset loss", "population", "response", "delay"],
         lambda v: f"{v:.3f}", (0, top),
         "weighted cost term (sums to $J_{total}$)", "Avg", share=False)
    for i, a in enumerate(ARMS):
        ax.text(i, top * 0.91, f"$J_{{total}}$ = {m.end_j_total[a]:.3f}",
                ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_18_Jtotal_average.png")); plt.close()
    print("figures written to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--report", action="store_true",
                    help="skip the campaign, only build tables/figures")
    a = ap.parse_args()
    if not a.report:
        run_campaign(a.workers)
    report()
