"""Per-scenario decision-cost ladder figures FROM the filled table CSV.

Reads  experiments/out/table59_cost.csv
Writes 01_Thesis/figures/fig_Jphy_<SCEN>.png   physical decision cost
       01_Thesis/figures/fig_Jtot_<SCEN>.png   total decision cost

Two figure families, one pair per scenario plus an Avg panel:

  J_phy : grouped bars {J_burn, J_asset, J_pop}, each shown as a share of
          the no-DSS run T_0 (%). T_0 is the 100 % reference, so the bars
          read directly as "how much physical cost the configuration
          still carries relative to doing nothing". Every configuration is
          present, T_0 included.

  J_tot : stacked bars of the weighted cost terms, so the bar height IS the
          total decision cost J_total. T_0 is dropped here: it issues no
          orders, so its response and delay terms are zero and a share of
          T_0 is undefined for those terms. The panel therefore compares
          the four acting configurations on their true total cost.

Everything visual sits in the STYLE block. Edit and re-run; the data
never has to be regenerated.

Run:  python experiments/plot_J_ladder.py
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
SCEN_ORDER = ["S1", "S2", "S3", "S4", "S5"]
AVG_LABEL = "Avg"

# configuration order and x-axis labels (match the reference figure)
ARM_ORDER = ["Test0", "F5", "F5Ev", "F5EvAI"]
ARM_LABEL = {
    "Test0": r"$T_{0}$",
    "F5": r"$T_{F5}$",
    "F40": r"$T_{F40}$",
    "F5Ev": r"$T_{F5+Ev}$",
    "F5EvAI": r"$T_{DisasterAware}$",
}

# physical cost terms drawn in the J_phy figure (share of T_0)
PHY_TERMS = [
    ("j_burn", "burned area", "#c0392b"),
    ("j_asset", "asset loss", "#27ae60"),
    ("j_pop", "population", "#2980b9"),
]
# all cost terms stacked in the J_tot figure (weighted -> height = J_total)
TOT_TERMS = [
    ("j_burn", "burned area", "#c0392b"),
    ("j_asset", "asset loss", "#27ae60"),
    ("j_pop", "population", "#2980b9"),
    ("j_resp", "response", "#8e44ad"),
    ("j_delay", "delay", "#95a5a6"),
]

FIGSIZE = (7.6, 4.6)
DPI = 220
BAR_W = 0.26
GRID_ALPHA = 0.25
FONT_AXIS = 11
FONT_LEGEND = 9
FONT_VAL = 7.0
# ============================================================================


def _weights():
    """Normalized cost weights (single source: the simulator config), so
    the stacked term contributions add up to J_total exactly."""
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from disaster_phyengine.config import SimConfig
    c = SimConfig().cost
    w = {"j_burn": c.w_burn, "j_asset": c.w_asset, "j_pop": c.w_pop,
         "j_resp": c.w_resp, "j_delay": getattr(c, "w_delay", 0.2)}
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


def _load():
    with open(os.path.join(OUT, "table59_cost.csv"), encoding="utf-8") as f:
        return {(r["scenario"], r["arm"]): r for r in csv.DictReader(f)}


def _f(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _term(t, scen, arm, key):
    """Mean of a cost term for (scen, arm); Avg averages over scenarios."""
    if scen != AVG_LABEL:
        return _f(t[(scen, arm)], key)
    return float(np.mean([_f(t[(s, arm)], key) for s in SCEN_ORDER]))


def fig_jphy(t, scen):
    """Grouped bars: each physical cost term as a share of the no-DSS run
    T_0 (%). T_0 is the 100 % reference."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(ARM_ORDER))
    m = len(PHY_TERMS)
    base = {k: _term(t, scen, "Test0", k) for k, _l, _c in PHY_TERMS}
    for j, (key, lab, col) in enumerate(PHY_TERMS):
        ys, raw = [], []
        for arm in ARM_ORDER:
            v = _term(t, scen, arm, key)
            raw.append(v)
            ys.append(100.0 * v / base[key] if base[key] > 0 else 0.0)
        xs = x + (j - (m - 1) / 2) * BAR_W
        ax.bar(xs, ys, BAR_W, color=col, label=lab)
        for xi, yi, rv in zip(xs, ys, raw):
            ax.text(xi, yi + 1.5, f"{rv:.3f}", rotation=90, ha="center",
                    va="bottom", fontsize=FONT_VAL, color="#333333")
    ax.axhline(100.0, ls="--", lw=1.0, color="#888888")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], fontsize=FONT_AXIS)
    ax.set_ylabel(r"physical decision cost, share of $T_{0}$ (%)",
                  fontsize=FONT_AXIS)
    ax.set_title(scen, fontsize=FONT_AXIS + 1)
    ax.set_ylim(0, 128)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(fontsize=FONT_LEGEND, ncol=m, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    p = os.path.join(FIGDIR, f"fig_Jphy_{scen}.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def fig_jtot(t, scen):
    """Grouped bars: every weighted cost term, all four configurations.

    Built like the physical-cost figure so the two can be read one after
    the other without relearning the picture, with two differences that
    the data forces.

    The axis is ABSOLUTE, not a share of T_0. T_0 issues no orders, so
    its response and delay terms are exactly zero, and a percentage of
    zero is not a number: the two terms that the decision layer actually
    pays would be the two the figure could not draw. On an absolute axis
    they are simply there, and T_0 keeps its place as the first group.

    The bar heights are the WEIGHTED contributions, so the five bars of a
    group sum to that configuration's J_total, which is printed above the
    group. Stacking them made that sum the bar height and hid the terms
    inside it; grouped, the terms are comparable across configurations
    and the total is still stated.
    """
    w = _weights()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(ARM_ORDER))
    m = len(TOT_TERMS)
    totals = np.zeros(len(ARM_ORDER))
    top = 0.0
    for j, (key, lab, col) in enumerate(TOT_TERMS):
        ys = [w[key] * _term(t, scen, a, key) for a in ARM_ORDER]
        totals += np.array(ys)
        top = max(top, max(ys))
        xs = x + (j - (m - 1) / 2) * BAR_W * 0.75
        ax.bar(xs, ys, BAR_W * 0.75, color=col, label=lab)
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + top * 0.015, f"{yi:.3f}", rotation=90,
                    ha="center", va="bottom", fontsize=FONT_VAL,
                    color="#333333")
    for xi, ti in zip(x, totals):
        ax.text(xi, top * 1.30, f"$J_{{total}}$ = {ti:.3f}", ha="center",
                va="bottom", fontsize=FONT_VAL + 1.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER],
                       fontsize=FONT_AXIS)
    ax.set_ylabel(r"weighted cost term (sums to $J_{total}$)",
                  fontsize=FONT_AXIS)
    ax.set_title(scen, fontsize=FONT_AXIS + 1)
    ax.set_ylim(0, top * 1.45)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FONT_LEGEND, ncol=m, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    p = os.path.join(FIGDIR, f"fig_Jtot_{scen}.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only-average", action="store_true",
                    help="only the Avg panel, not the five "
                         "per-scenario pairs")
    a = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    t = _load()
    scens = [AVG_LABEL] if a.only_average else SCEN_ORDER + [AVG_LABEL]
    for scen in scens:
        print("written:", os.path.abspath(fig_jphy(t, scen)))
        print("written:", os.path.abspath(fig_jtot(t, scen)))


if __name__ == "__main__":
    main()
