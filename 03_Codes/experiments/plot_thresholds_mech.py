"""What the two acceptance thresholds govern.

Only the mechanism is plotted. The outcome is reported in the text as a
null result, since the world-to-world spread swamps any location shift
and the sign of the change is inconsistent across worlds.

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
C1, C3 = "#1f5f9e", "#7a5aa8"
INK, MUTED, GRID = "#33322e", "#77746c", "#e3e1dc"
DEAD = "#f2efe9"
TT = 2.262

rows = list(csv.DictReader(open(f"{ROOT}/jth_sweep_v2.csv")))
SEEDS = sorted({r["seed"] for r in rows})
D = {}
for r in rows:
    D[(r["dial"], float(r["level"]), r["seed"])] = dict(
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


def style(ax, xlab, ylab, title):
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel(xlab, fontsize=10, color=MUTED)
    ax.set_ylabel(ylab, fontsize=10, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-8, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)


fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6))

# ----------------------------------------------------------- (a) J_TH
a = ax[0]
a.axvspan(0.45, 1.0, color=DEAD, zorder=0)
m, h = agg("j_threshold", "adapt")
a.errorbar(L, m, yerr=h, color=C1, lw=2, marker="o", ms=6, capsize=3,
           elinewidth=1, markeredgecolor="white", markeredgewidth=1.2)
a.axvline(0.35, color=MUTED, lw=1, ls=(0, (2, 3)))
a.text(0.345, 104, "configured 0.35", fontsize=8.5, color=MUTED, ha="right")
a.text(0.725, 62, "inactive:\nthe relative margin\nis always the smaller",
       fontsize=8.5, color=MUTED, ha="center")
a.annotate("", xy=(0.05, 8), xytext=(0.45, 8),
           arrowprops=dict(arrowstyle="<->", color=C1, lw=1.2))
a.text(0.25, 12, "operable range", fontsize=8.5, color=C1, ha="center")
style(a, "satisficing bound  $J_{TH}$",
      "decision cycles that engage adaptation (%)",
      "(a) how often the adaptation ladder is climbed")

# ----------------------------------------------------------- (b) eta
b = ax[1]
b.axvspan(0.0, 0.55, color=DEAD, zorder=0)
b.axvspan(0.85, 1.0, color=DEAD, zorder=0)
m, h = agg("eta", "fs")
b.errorbar(L, m, yerr=h, color=C3, lw=2, marker="^", ms=7, capsize=3,
           elinewidth=1, markeredgecolor="white", markeredgewidth=1.2)
b.axvline(0.60, color=MUTED, lw=1, ls=(0, (2, 3)))
b.text(0.615, 104, "configured 0.60", fontsize=8.5, color=MUTED)
b.text(0.27, 62, "never engages:\nno protection", fontsize=8.5, color=MUTED,
       ha="center")
b.text(0.925, 62, "always engaged:\na permanent\nderating", fontsize=8.5,
       color=MUTED, ha="center")
b.annotate("", xy=(0.55, 8), xytext=(0.85, 8),
           arrowprops=dict(arrowstyle="<->", color=C3, lw=1.2))
b.text(0.70, 12, "operable range", fontsize=8.5, color=C3, ha="center")
style(b, "fail-safe quality gate  $\\eta$",
      "region decisions derated by the fail-safe (%)",
      "(b) how often offensive orders are derated")

fig.suptitle("What the two acceptance thresholds govern. 10 worlds, four "
             "simultaneous ignitions, resource pool 0.25, all three "
             "adaptation stages active.\nA decision cycle is one pass of the "
             "loop; a region decision is one region within one cycle. Bars "
             "are 95 per cent intervals over the worlds.",
             fontsize=9.5, color=INK, x=0.008, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.855))
os.makedirs(FIGDIR, exist_ok=True)
fig.savefig(os.path.join(FIGDIR, "fig_thresholds_mech.png"), dpi=220, facecolor="white")
print("written")
