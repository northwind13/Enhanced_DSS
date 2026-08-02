"""Full-range sweep of the two acceptance thresholds, 0.05 to 0.95.

Corrected acceptance gate: the symptoms tighten the bound instead of
bypassing it, the ceiling is read on the total cost and the relative
margin on the physical cost, and the forecast horizon matches the
no-harm horizon at 45 minutes.

Top row  = mechanism, measured per decision.
Bottom row = outcome, paired by world against the setting at which the
             threshold is inactive, so the reading is "what the guardrail
             buys, or costs, against not having it".

Source: experiments/out/jth_sweep_v2.csv (200 runs, 10 worlds)
"""
import csv, json, math
import statistics as st
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# PATHS ARE RESOLVED FROM THIS FILE, never from the machine that first
# ran it. The script was written in a sandbox whose absolute paths do
# not exist anywhere else, so it failed on the first line the moment it
# was run from the repository. Everything is now relative to the script
# itself, which is the convention the rest of the experiments follow.
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGDIR = os.path.join(HERE, "..", "..", "01_Thesis", "figures")
ROOT = OUT
C1, C2, C3 = "#1f5f9e", "#b3492d", "#7a5aa8"
INK, MUTED, GRID = "#33322e", "#77746c", "#e3e1dc"
DEAD = "#f2efe9"

rows = list(csv.DictReader(open(f"{ROOT}/jth_sweep_v2.csv")))
fb = json.load(open(f"{ROOT}/jth_freeburn.json"))
SEEDS = sorted({r["seed"] for r in rows})
TT = 2.262

D = {}
for r in rows:
    D[(r["dial"], float(r["level"]), r["seed"])] = dict(
        prev=100.0 * (1.0 - float(r["j_phys"]) / fb[r["seed"]][0]),
        adapt=100.0 * float(r["adapt_frac"]),
        fs=100.0 * float(r["fs_frac"]))

L = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


def agg(dial, key):
    m, h = [], []
    for lv in L:
        v = [D[(dial, lv, s)][key] for s in SEEDS]
        m.append(st.mean(v))
        h.append(TT * st.stdev(v) / math.sqrt(len(v)) if st.stdev(v) > 0
                 else 0.0)
    return np.array(m), np.array(h)


def paired(dial, base):
    m, h = [], []
    for lv in L:
        v = [D[(dial, lv, s)]["prev"] - D[(dial, base, s)]["prev"]
             for s in SEEDS]
        m.append(st.mean(v))
        h.append(TT * st.stdev(v) / math.sqrt(len(v)) if st.stdev(v) > 0
                 else 0.0)
    return np.array(m), np.array(h)


def style(ax, xlab, ylab, title):
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=7)
    ax.set_xlabel(xlab, fontsize=9, color=MUTED)
    ax.set_ylabel(ylab, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, 1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


fig, ax = plt.subplots(2, 2, figsize=(12.4, 7.9))

# ------------------------------------------------------- (a) J_TH mechanism
a = ax[0, 0]
a.axvspan(0.45, 1.0, color=DEAD, zorder=0)
m, h = agg("j_threshold", "adapt")
a.errorbar(L, m, yerr=h, color=C1, lw=2, marker="o", ms=6, capsize=3,
           elinewidth=1, markeredgecolor="white", markeredgewidth=1.2,
           label="a decision cycle engages the adaptation ladder")
a.axvline(0.35, color=MUTED, lw=1, ls=(0, (2, 3)))
a.text(0.345, 104, "configured 0.35", fontsize=8, color=MUTED, ha="right")
a.text(0.72, 60, "inactive:\nthe relative margin\nis always the smaller",
       fontsize=8, color=MUTED, ha="center")
a.text(0.06, 47, "operable range\n0.05 to 0.45", fontsize=8, color=C1)
a.set_ylim(15, 112)
style(a, "satisficing bound  $J_{TH}$",
      "decision cycles that engage adaptation (%)",
      "(a) the bound sets how often the ladder is climbed")
a.legend(fontsize=8, frameon=False, loc="lower left", labelcolor=INK)

# ------------------------------------------------------- (b) eta mechanism
b = ax[0, 1]
b.axvspan(0.0, 0.55, color=DEAD, zorder=0)
b.axvspan(0.85, 1.0, color=DEAD, zorder=0)
m, h = agg("eta", "fs")
b.errorbar(L, m, yerr=h, color=C3, lw=2, marker="^", ms=7, capsize=3,
           elinewidth=1, markeredgecolor="white", markeredgewidth=1.2,
           label="a region decision is derated by the fail-safe")
b.axvline(0.60, color=MUTED, lw=1, ls=(0, (2, 3)))
b.text(0.585, -4.5, "configured 0.60", fontsize=8, color=MUTED, ha="right")
b.text(0.27, 55, "never engages:\nno protection", fontsize=8, color=MUTED,
       ha="center")
b.text(0.925, 42, "always\nengaged:\na permanent\nderating", fontsize=8,
       color=MUTED, ha="center")
b.text(0.70, 14, "operable range 0.55 to 0.85", fontsize=8, color=C3)
b.set_ylim(-8, 112)
style(b, "fail-safe quality gate  $\\eta$",
      "region decisions derated by the fail-safe (%)",
      "(b) the gate sets how often orders are derated")
b.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)

# ------------------------------------------------------- (c) J_TH outcome
c = ax[1, 0]
c.axvspan(0.45, 1.0, color=DEAD, zorder=0)
m, h = paired("j_threshold", 0.95)
c.errorbar(L, m, yerr=h, color=C1, lw=2, marker="o", ms=6, capsize=3,
           elinewidth=1, markeredgecolor="white", markeredgewidth=1.2)
c.axhline(0, color=GRID, lw=1.4)
c.axvline(0.35, color=MUTED, lw=1, ls=(0, (2, 3)))
c.text(0.045, -13.5, "reference: the ceiling never binds, which prevents "
       f"{st.mean([D[('j_threshold', 0.95, s)]['prev'] for s in SEEDS]):.0f} "
       "per cent of the free-burn cost", fontsize=8, color=MUTED)
style(c, "satisficing bound  $J_{TH}$",
      "change against an inactive ceiling (points)",
      "(c) a tighter bound trends better, but not separably")

# ------------------------------------------------------- (d) eta outcome
d = ax[1, 1]
d.axvspan(0.0, 0.55, color=DEAD, zorder=0)
d.axvspan(0.85, 1.0, color=DEAD, zorder=0)
m, h = paired("eta", 0.05)
d.errorbar(L, m, yerr=h, color=C3, lw=2, marker="^", ms=7, capsize=3,
           elinewidth=1, markeredgecolor="white", markeredgewidth=1.2)
d.axhline(0, color=GRID, lw=1.4)
d.axvline(0.60, color=MUTED, lw=1, ls=(0, (2, 3)))
d.text(0.045, -13.5, "reference: the fail-safe never engages, which prevents "
       f"{st.mean([D[('eta', 0.05, s)]['prev'] for s in SEEDS]):.0f} "
       "per cent of the free-burn cost", fontsize=8, color=MUTED)
style(d, "fail-safe quality gate  $\\eta$",
      "change against a fail-safe that never engages (points)",
      "(d) a stricter gate trends worse, but not separably")

lo = min(ax[1, 0].get_ylim()[0], ax[1, 1].get_ylim()[0])
hi = max(ax[1, 0].get_ylim()[1], ax[1, 1].get_ylim()[1])
for k in (0, 1):
    ax[1, k].set_ylim(lo, hi)

fig.suptitle("The two acceptance thresholds over their full range. "
             "Corrected gate, the relative margin read on the physical cost, "
             "forecast horizon 45 minutes.\n10 worlds, 4 simultaneous "
             "ignitions, resource pool 0.25, all three adaptation stages "
             "active. A decision cycle is one pass of the loop; a region "
             "decision is one\nregion within one cycle. Bars are 95 per cent "
             "intervals over the worlds; the outcome is paired by world.",
             fontsize=9.5, color=INK, x=0.008, ha="left", y=0.997)
fig.tight_layout(rect=(0, 0, 1, 0.885))
os.makedirs(FIGDIR, exist_ok=True)
fig.savefig(os.path.join(FIGDIR, "fig_thresholds_full.png"), dpi=220, facecolor="white")
print("written")
