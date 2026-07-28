"""Draw the two ladder figures FROM the filled table CSVs.

Reads  experiments/out/table58_phys.csv   (physical outcome table)
       experiments/out/table59_cost.csv   (decision-cost table)
Writes 01_Thesis/figures/fig_table58_burned.png
       01_Thesis/figures/fig_table59_cost.png

Everything visual sits in the STYLE block below: colors, labels,
order, figure size, dpi, fonts, output names. Edit and re-run; the
data never has to be regenerated.

Run:  python experiments/plot_tables.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGDIR = os.path.join(HERE, "..", "..", "01_Thesis", "figures")

# ============================ STYLE (edit freely) ============================
ARM_ORDER = ["Test0", "F5", "F5Ev", "F5EvAI"]
ARM_LABEL = {
    "Test0": "$T_0$ (no DSS)",
    "F5": "$T_{F5}$",
    "F40": "$T_{F40}$",
    "F5Ev": "$T_{F5+Ev}$",
    "F5EvAI": "$T_{DisasterAware}$",
}
ARM_COLOR = {
    "Test0": "#7f8c8d",
    "F5": "#2980b9",
    "F40": "#16a085",
    "F5Ev": "#27ae60",
    "F5EvAI": "#c0392b",
}
SCEN_ORDER = ["S1", "S2", "S3", "S4", "S5"]
SCEN_LABEL = {s: s for s in SCEN_ORDER}      # e.g. {"S1": "S1 remote"}

# cost terms: (csv column, legend label, color), stack order bottom-up
COST_TERMS = [
    ("j_burn", "burned area", "#e67e22"),
    ("j_asset", "asset loss", "#c0392b"),
    ("j_pop", "population", "#2980b9"),
    ("j_resp", "response", "#8e44ad"),
    ("j_delay", "delay", "#95a5a6"),
]
WEIGHTED = True        # True: stack weighted terms so bar height = J_total
FIGSIZE_58 = (10.5, 3.6)
FIGSIZE_59 = (10.5, 3.6)
DPI = 220
BAR_W = 0.26           # bar width inside a scenario group
GRID_ALPHA = 0.25
FONT_AXIS = 10
FONT_LEGEND = 9
OUT_58 = "fig_table58_burned.png"
OUT_59 = "fig_table59_cost.png"
# ============================================================================


def _weights():
    """Cost weights of the Balanced profile, normalized (single
    source: the simulator config), so the stacked bar height equals
    the decision cost J itself."""
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from disaster_phyengine.config import SimConfig
    c = SimConfig().cost
    w = {"j_burn": c.w_burn, "j_asset": c.w_asset, "j_pop": c.w_pop,
         "j_resp": c.w_resp, "j_delay": getattr(c, "w_delay", 0.2)}
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


def _load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return {(r["scenario"], r["arm"]): r for r in csv.DictReader(f)}


def fig_58():
    """Grouped bars: mean burned area per scenario per arm, with the
    95% CI whisker from the table's ± column."""
    t = _load("table58_phys.csv")
    fig, ax = plt.subplots(figsize=FIGSIZE_58)
    x = np.arange(len(SCEN_ORDER))
    n = len(ARM_ORDER)
    for i, arm in enumerate(ARM_ORDER):
        ys = [float(t[(s, arm)]["burned_ha"]) for s in SCEN_ORDER]
        es = [float(t[(s, arm)]["burned_ci"]) for s in SCEN_ORDER]
        ax.bar(x + (i - (n - 1) / 2) * BAR_W, ys, BAR_W,
               yerr=es, capsize=3, color=ARM_COLOR[arm],
               label=ARM_LABEL[arm],
               error_kw=dict(lw=1.0, alpha=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN_ORDER],
                       fontsize=FONT_AXIS)
    ax.set_ylabel("burned area at 6 h (ha)", fontsize=FONT_AXIS)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(fontsize=FONT_LEGEND, ncol=len(ARM_ORDER),
              frameon=False)
    fig.tight_layout()
    p = os.path.join(FIGDIR, OUT_58)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def fig_59():
    """Stacked bars: decision-cost terms per scenario per arm. With
    WEIGHTED = True the stack height equals J_total."""
    t = _load("table59_cost.csv")
    w = _weights() if WEIGHTED else {k: 1.0 for k, _l, _c in COST_TERMS}
    fig, ax = plt.subplots(figsize=FIGSIZE_59)
    x = np.arange(len(SCEN_ORDER))
    n = len(ARM_ORDER)
    for i, arm in enumerate(ARM_ORDER):
        bottoms = np.zeros(len(SCEN_ORDER))
        xs = x + (i - (n - 1) / 2) * BAR_W
        for key, lab, col in COST_TERMS:
            ys = []
            for s in SCEN_ORDER:
                v = t[(s, arm)].get(key, "")
                ys.append(w[key] * float(v) if v not in ("", "—")
                          else 0.0)
            ax.bar(xs, ys, BAR_W, bottom=bottoms, color=col,
                   label=lab if i == 0 else None)
            bottoms += np.array(ys)
        # arm tag above each stack
        for xi, bi in zip(xs, bottoms):
            ax.text(xi, bi + 0.008, ARM_LABEL[arm], rotation=90,
                    ha="center", va="bottom", fontsize=6.0,
                    color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN_ORDER],
                       fontsize=FONT_AXIS)
    ax.set_ylabel("decision cost J at 6 h"
                  + (" (weighted terms)" if WEIGHTED else ""),
                  fontsize=FONT_AXIS)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(fontsize=FONT_LEGEND, ncol=len(COST_TERMS),
              frameon=False, loc="upper right")
    ax.margins(y=0.18)
    fig.tight_layout()
    p = os.path.join(FIGDIR, OUT_59)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("written:", os.path.abspath(fig_58()))
    print("written:", os.path.abspath(fig_59()))


if __name__ == "__main__":
    main()
