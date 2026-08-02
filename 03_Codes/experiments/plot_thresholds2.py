"""Figure: what the two acceptance thresholds govern, and what it costs.

Corrected acceptance gate (the symptoms tighten the bound instead of
bypassing it), adaptation on with all three stages including the
generative one. 10 worlds, 4 simultaneous ignitions, resource pool 0.25.

Top row  = mechanism, measured on the running incident.
Bottom row = outcome, paired by world against the configured setting.

Source: experiments/out/jth_sweep_genai.csv (120 runs)
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

rows = list(csv.DictReader(open(f"{ROOT}/jth_sweep_genai.csv")))
fb = json.load(open(f"{ROOT}/jth_freeburn.json"))
SEEDS = sorted({r["seed"] for r in rows})
TT = 2.262 if len(SEEDS) == 10 else 2.776

D = {}
for r in rows:
    k = (r["dial"], float(r["level"]), r["seed"])
    D[k] = dict(
        prev=100.0 * (1.0 - float(r["j_phys"]) / fb[r["seed"]][0]),
        adapt=100.0 * float(r["adapt_frac"]),
        fs=100.0 * float(r["fs_frac"]),
        s3=int(r["tried3"]), acc3=int(r["acc3"]), rules=int(r["rules"]))


def agg(dial, levels, key):
    m, h = [], []
    for lv in levels:
        v = [D[(dial, lv, s)][key] for s in SEEDS]
        m.append(st.mean(v))
        h.append(TT * st.stdev(v) / math.sqrt(len(v)) if st.stdev(v) > 0
                 else 0.0)
    return np.array(m), np.array(h)


def paired(dial, levels, base):
    m, h = [], []
    for lv in levels:
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
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


JTH = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45]
ETA = [0.60, 0.70, 0.75, 0.80, 0.90]

fig, ax = plt.subplots(2, 2, figsize=(12.2, 7.8))

# ---------------------------------------------------- (a) J_TH mechanism
a = ax[0, 0]
a.axvspan(0.35, 0.47, color=DEAD, zorder=0)
a.text(0.412, 88, "saturated:\nthe relative\nmargin binds", fontsize=8,
       color=MUTED, ha="center")
m, h = agg("j_threshold", JTH, "adapt")
a.errorbar(JTH, m, yerr=h, color=C1, lw=2, marker="o", ms=6, capsize=3,
           elinewidth=1, markeredgecolor="white", markeredgewidth=1.2,
           label="a decision cycle engages the adaptation ladder")
a.axvline(0.35, color=MUTED, lw=1, ls=(0, (2, 3)))
a.text(0.345, 104, "configured 0.35", fontsize=8, color=MUTED, ha="right")
a.text(0.048, 26, "before the fix this curve was flat at 100 per cent at every\n"
       "level: the symptoms opened the gate on their own, so the\n"
       "bound reached no decision", fontsize=8, color=C2)
a.set_ylim(14, 110)
style(a, "satisficing bound  $J_{TH}$", "share of decision cycles (%)",
      "(a) the bound sets how often the ladder is climbed")
a.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK)

# ---------------------------------------------------- (b) eta mechanism
b = ax[0, 1]
m, h = agg("eta", ETA, "fs")
b.errorbar(ETA, m, yerr=h, color=C3, lw=2, marker="^", ms=7, capsize=3,
           elinewidth=1, markeredgecolor="white", markeredgewidth=1.2,
           label="a region decision is derated by the fail-safe")
b.axvline(0.60, color=MUTED, lw=1, ls=(0, (2, 3)))
b.text(0.605, -1, "configured 0.60", fontsize=8, color=MUTED)
b.set_ylim(-8, 112)
style(b, "fail-safe quality gate  $\\eta$",
      "share of region decisions (%)",
      "(b) the gate sets how often orders are derated")
b.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)

# ---------------------------------------------------- (c) J_TH outcome
c = ax[1, 0]
c.axvspan(0.35, 0.47, color=DEAD, zorder=0)
m, h = paired("j_threshold", JTH, 0.35)
c.errorbar(JTH, m, yerr=h, color=C1, lw=2, marker="o", ms=6, capsize=3,
           elinewidth=1, markeredgecolor="white", markeredgewidth=1.2)
c.axhline(0, color=GRID, lw=1.4)
c.axvline(0.35, color=MUTED, lw=1, ls=(0, (2, 3)))
b0 = st.mean([D[("j_threshold", 0.35, s)]["prev"] for s in SEEDS])
c.text(0.052, 11.5, f"configured setting prevents {b0:.0f} per cent of "
       "the free-burn cost", fontsize=8, color=MUTED)
style(c, "satisficing bound  $J_{TH}$",
      "change in prevented share (points)",
      "(c) no swept level separates from the configured one")

# ---------------------------------------------------- (d) eta outcome
d = ax[1, 1]
m, h = paired("eta", ETA, 0.60)
d.errorbar(ETA, m, yerr=h, color=C3, lw=2, marker="^", ms=7, capsize=3,
           elinewidth=1, markeredgecolor="white", markeredgewidth=1.2)
d.axhline(0, color=GRID, lw=1.4)
d.axvline(0.60, color=MUTED, lw=1, ls=(0, (2, 3)))
style(d, "fail-safe quality gate  $\\eta$",
      "change in prevented share (points)",
      "(d) the guardrail costs nothing it can be shown to cost")

lo = min(ax[1, 0].get_ylim()[0], ax[1, 1].get_ylim()[0])
hi = max(ax[1, 0].get_ylim()[1], ax[1, 1].get_ylim()[1])
for k in (0, 1):
    ax[1, k].set_ylim(lo, hi)

fig.suptitle("The two acceptance thresholds under the corrected gate. "
             "10 worlds, 4 simultaneous ignitions, resource pool 0.25,\n"
             "all three adaptation stages active. Bars are 95 per cent "
             "intervals over the worlds; the outcome is paired by world.",
             fontsize=10, color=INK, x=0.008, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.90))
os.makedirs(FIGDIR, exist_ok=True)
fig.savefig(os.path.join(FIGDIR, "fig_thresholds_genai.png"), dpi=220, facecolor="white")
print("written")
