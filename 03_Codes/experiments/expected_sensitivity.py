"""SCHEMATIC figure: what the literature expects each sensitivity
dial to do. These are REPRESENTATIVE curves, not data: they draw the
qualitative shapes the cited literature predicts, so the measured
sweeps of Figure 5.15/5.16 can be compared against an explicit
expectation instead of an implicit one.

Solid blue   = expected at the thesis operating point (large fire,
               scarce pool: guardrails not binding)
Dashed red   = expected in the MARGINAL regime (small fire, ample
               pool: the regime in which guardrail thresholds bind)
Dotted grey  = the engagement/secondary quantity where the outcome
               is expected flat but the mechanism is expected to move

Sources for the shapes (qualitative): initial-attack success vs
response delay and quasi-quadratic early fire growth (decision
cycle); detection-limited suppression and coverage saturation
(observation); hierarchical/distributed control near-centralized
performance (regions); Simon's satisficing aspiration level binding
only near the margin (J_TH); supervisory guardrails moving engagement
rather than outcome (quality gate, no-harm).

Run:  python experiments/expected_sensitivity.py
Out:  01_Thesis/figures/fig_expected_sensitivity.png
Edit the STYLE block and re-run; everything is synthetic.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "..", "01_Thesis", "figures")

# ============================ STYLE =============================
FIGSIZE = (16.5, 7.2)
DPI = 220
C_OP = "#2980b9"        # operating-point expectation (solid)
C_MARG = "#c0392b"      # marginal-regime expectation (dashed)
C_SEC = "#7f8c8d"       # secondary quantity (dotted)
BASE = 53.0             # base configuration level (%)
YLIM = (0, 100)
OUT = "fig_expected_sensitivity.png"
# ================================================================


def _ax(ax, title, xlabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylim(*YLIM)
    ax.grid(alpha=0.25)
    ax.axhline(BASE, color="#444444", ls="--", lw=0.9, alpha=0.6)


def main():
    fig, axes = plt.subplots(2, 4, figsize=FIGSIZE)

    # 1 decision cycle: early fire growth is quasi-quadratic, so the
    # prevented share falls steeply and convexly with the revisit
    # interval (initial-attack literature: containment probability
    # drops fast with response delay)
    ax = axes[0, 0]
    t = np.linspace(2, 20, 60)
    y = 30 + 40 * np.exp(-(t - 2) / 9.0)
    ax.plot(t, y, color=C_OP, lw=2)
    _ax(ax, "decision cycle: steep, convex decline",
        "revisit interval (min)")
    ax.annotate("fire caught while small", (3, 66), fontsize=8)

    # 2 observation: detection-limited suppression; a sharp gain from
    # first coverage, then saturation BELOW the full-observation
    # ceiling (adding sensors after coverage buys little)
    ax = axes[0, 1]
    n = np.array([1, 2, 3, 5, 9])
    y = 44 - 26 * np.exp(-(n - 1) / 0.8)
    ax.plot(n, y, color=C_OP, lw=2, marker="o", ms=4)
    ax.axhline(BASE, color=C_SEC, ls=":", lw=1.4)
    _ax(ax, "observation: jump, then saturation",
        "observation assets")
    ax.annotate("full-observation ceiling", (2.6, 56), fontsize=8,
                color=C_SEC)

    # 3 regions N: hierarchical decomposition with a working
    # coordinator stays near the centralized level; only extreme
    # fragmentation degrades mildly
    ax = axes[0, 2]
    n = np.array([1, 2, 4, 8, 16])
    y = BASE - 0.4 * np.log2(n) - 1.5 * (n >= 16)
    ax.plot(n, y, color=C_OP, lw=2, marker="o", ms=4)
    ax.set_xscale("log", base=2)
    _ax(ax, "regions: near-flat under coordination", "local regions N")

    # 4 quality gate: OUTCOME flat, ENGAGEMENT moving - a guardrail
    # governs who acts, not what it costs, until it binds everything
    ax = axes[0, 3]
    e = np.linspace(0.30, 0.90, 60)
    ax.plot(e, np.full_like(e, BASE), color=C_OP, lw=2,
            label="outcome")
    eng = 100 / (1 + np.exp(-(e - 0.68) * 14))
    ax.plot(e, eng, color=C_SEC, ls=":", lw=1.8,
            label="fail-safe engagement %")
    ax.legend(fontsize=7, loc="center left")
    _ax(ax, "quality gate: outcome flat,\nengagement moves",
        "quality gate $\\eta$")

    # 5 attention threshold: flat over the sensible middle; both
    # extremes hurt (no focus at all, or starving every non-focus
    # region)
    ax = axes[1, 0]
    tau = np.linspace(0.05, 0.95, 60)
    y = BASE - 6 * np.exp(-(tau - 0.05) / 0.06) \
        - 8 * np.exp((tau - 0.95) / 0.06)
    ax.plot(tau, y, color=C_OP, lw=2)
    _ax(ax, "attention: flat middle, weak extremes",
        "attention threshold $\\tau$")

    # 6 no-harm horizon: flat where offense clearly helps (solid);
    # in the MARGINAL regime an inverted U - too short misfires on
    # noise, too long withholds on stale forecasts
    ax = axes[1, 1]
    h = np.linspace(8, 48, 60)
    ax.plot(h, np.full_like(h, BASE), color=C_OP, lw=2,
            label="large fire (guard rarely binds)")
    ym = 40 - 0.012 * (h - 24) ** 2
    ax.plot(h, ym, color=C_MARG, ls="--", lw=1.8,
            label="marginal regime")
    ax.legend(fontsize=7, loc="lower center")
    _ax(ax, "no-harm horizon: flat unless marginal",
        "no-harm horizon (min)")

    # 7 satisficing bound: flat while the cost sits far above the
    # bound (aspiration not binding, Simon); in the marginal regime a
    # knee appears where the bound crosses the typical cost - a loose
    # bound starts accepting weak plans
    ax = axes[1, 2]
    j = np.linspace(0.15, 0.60, 60)
    ax.plot(j, np.full_like(j, BASE), color=C_OP, lw=2,
            label="large fire (bound never binds)")
    ym = BASE - 14 / (1 + np.exp(-(j - 0.38) * 25)) + 2
    ax.plot(j, ym, color=C_MARG, ls="--", lw=1.8,
            label="marginal regime (knee at the bind)")
    ax.legend(fontsize=7, loc="lower left")
    _ax(ax, "satisficing bound: flat until it binds",
        "satisficing bound $J_{TH}$")

    # 8 legend / reading panel
    ax = axes[1, 3]
    ax.axis("off")
    ax.text(0.02, 0.92, "READING", fontsize=10, weight="bold")
    ax.text(0.02, 0.02,
            "SCHEMATIC expectations, not data.\n\n"
            "solid blue: expected at the thesis operating point\n"
            "(large fire, scarce pool - guardrails not binding)\n\n"
            "dashed red: expected in the marginal regime\n"
            "(small fire, ample pool - thresholds bind);\n"
            "the cheap follow-up sweep runs THERE\n\n"
            "dotted grey: the mechanism that should move\n"
            "even where the outcome stays flat\n\n"
            "a measured flat line matches the literature\n"
            "only together with the dotted/dashed evidence",
            fontsize=8.5, va="bottom")

    for ax in axes[:, 0]:
        ax.set_ylabel("prevented share of the free-burn cost (%)",
                      fontsize=9)
    fig.suptitle("Literature-expected shapes for the sensitivity "
                 "dials (schematic)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, OUT)
    fig.savefig(p, dpi=DPI)
    print("written:", os.path.abspath(p))


if __name__ == "__main__":
    main()
