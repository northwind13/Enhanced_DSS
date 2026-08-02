"""Figure: what the satisficing bound and the quality gate actually do.

Top row  = mechanism, the share of decision cycles the threshold governs.
Bottom row = outcome, the physical cost prevented against the free burn.

All from data already on disk:
  experiments/out/jth_probe_trace.csv   150 decision cycles, 5 worlds
  experiments/out/jth_sweep.csv         100 runs, 5 worlds
"""
import csv, json, math, collections
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
# validated categorical palette (scripts/validate_palette.js, light mode)
C1 = "#1f5f9e"      # first series
C2 = "#b3492d"      # second series
C3 = "#7a5aa8"      # third series
INK, MUTED, GRID = "#33322e", "#77746c", "#e3e1dc"
DEAD = "#f2efe9"    # inert-range shading
T = {5: 2.776}

# ------------------------------------------------------------------ data
tr = list(csv.DictReader(open(f"{ROOT}/jth_probe_trace.csv")))
jc = np.array([float(r["j_c"]) for r in tr])
j0 = np.array([float(r["j_0"]) for r in tr])
sd = np.array([int(r["seed"]) for r in tr])
SEEDS = sorted(set(sd.tolist()))

rows = list(csv.DictReader(open(f"{ROOT}/jth_sweep.csv")))
fb = json.load(open(f"{ROOT}/jth_freeburn.json"))
prev = {}
fsf = {}
for r in rows:
    k = (r["dial"], float(r["spread_tighten"]), float(r["level"]),
         int(r["seed"]))
    prev[k] = 100.0 * (1.0 - float(r["j_phys"]) / fb[str(r["seed"])][0])
    fsf[k] = 100.0 * float(r["fs_frac"])


def mech_jth(levels):
    """Share of cycles in which the cost deficit fires, and in which the
    ceiling rather than the relative margin is the binding constraint."""
    fires, binds = [], []
    for lv in levels:
        a, b = [], []
        for s in SEEDS:
            m = sd == s
            need = np.minimum(lv, 0.95 * j0[m])
            a.append(100.0 * np.mean(jc[m] > need))
            b.append(100.0 * np.mean(lv < 0.95 * j0[m]))
        fires.append(a)
        binds.append(b)
    return fires, binds


def band(series):
    m = [st.mean(v) for v in series]
    h = [(T[len(v)] * st.stdev(v) / math.sqrt(len(v))) if st.stdev(v) > 0
         else 0.0 for v in series]
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


JTH = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.60]
ETA = [0.30, 0.45, 0.60, 0.75, 0.90]
SWEPT = [0.15, 0.25, 0.35, 0.45, 0.60]

fig, ax = plt.subplots(2, 2, figsize=(12.2, 7.6))

# ---------------------------------------------- (a) J_TH mechanism
a = ax[0, 0]
a.axvspan(0.45, 0.62, color=DEAD, zorder=0)
a.text(0.525, 12, "inert:\nthe relative\nmargin binds", fontsize=8,
       color=MUTED, ha="center")
fires, binds = mech_jth(JTH)
m, h = band(fires)
a.plot(JTH, m, color=C1, lw=2, marker="o", ms=6, markeredgecolor="white",
       markeredgewidth=1.2, label="a cost deficit is declared")
a.fill_between(JTH, m - h, m + h, color=C1, alpha=0.13, lw=0)
m2, h2 = band(binds)
a.plot(JTH, m2, color=C2, lw=2, ls=(0, (5, 2)), marker="s", ms=6,
       markeredgecolor="white", markeredgewidth=1.2,
       label="the ceiling is the binding constraint")
a.fill_between(JTH, m2 - h2, m2 + h2, color=C2, alpha=0.13, lw=0)
a.axvline(0.35, color=MUTED, lw=1, ls=(0, (2, 3)))
a.text(0.355, 103, "configured 0.35", fontsize=8, color=MUTED)
a.set_ylim(-4, 112)
style(a, "satisficing bound  $J_{TH}$",
      "share of decision cycles (%)",
      "(a) what the bound governs")
a.legend(fontsize=8, frameon=False, loc="lower left", labelcolor=INK)

# ---------------------------------------------- (b) eta mechanism
b = ax[0, 1]
mm, hh = [], []
for lv in ETA:
    v = [fsf[("eta", 0.0, lv, s)] for s in SEEDS]
    mm.append(st.mean(v))
    hh.append(T[5] * st.stdev(v) / math.sqrt(5) if st.stdev(v) > 0 else 0.0)
mm, hh = np.array(mm), np.array(hh)
b.plot(ETA, mm, color=C3, lw=2, marker="^", ms=7, markeredgecolor="white",
       markeredgewidth=1.2, label="the graduated fail-safe engages")
b.fill_between(ETA, mm - hh, mm + hh, color=C3, alpha=0.13, lw=0)
b.axvspan(0.28, 0.60, color=DEAD, zorder=0)
b.text(0.44, 55, "inert:\nquality never\nfalls this low", fontsize=8,
       color=MUTED, ha="center")
b.axvline(0.60, color=MUTED, lw=1, ls=(0, (2, 3)))
b.text(0.607, 103, "configured 0.60", fontsize=8, color=MUTED)
b.set_ylim(-4, 112)
style(b, "fail-safe quality gate  $\\eta$",
      "share of region decisions (%)",
      "(b) what the gate governs")
b.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)

# ---------------------------------------------- (c) J_TH outcome
def paired(dial, tg, levels, base):
    m, h, b0 = [], [], st.mean([prev[(dial, tg, base, s)] for s in SEEDS])
    for lv in levels:
        d = [prev[(dial, tg, lv, s)] - prev[(dial, tg, base, s)]
             for s in SEEDS]
        m.append(st.mean(d))
        h.append(T[5] * st.stdev(d) / math.sqrt(5) if st.stdev(d) > 0 else 0.0)
    return np.array(m), np.array(h), b0


c = ax[1, 0]
c.axvspan(0.45, 0.62, color=DEAD, zorder=0)
for tg, col, mk, ls, lab in (
        (1.0, C2, "s", (0, (5, 2)), "as coded: symptoms bypass the bound"),
        (0.0, C1, "o", "solid", "corrected: symptoms tighten the bound")):
    m, h, b0 = paired("j_threshold", tg, SWEPT, 0.35)
    c.errorbar(SWEPT, m, yerr=h, color=col, lw=2, ls=ls, marker=mk, ms=6,
               capsize=3, elinewidth=1, markeredgecolor="white",
               markeredgewidth=1.2, label=f"{lab}  (base {b0:.0f}%)")
c.axhline(0, color=GRID, lw=1.4)
c.axvline(0.35, color=MUTED, lw=1, ls=(0, (2, 3)))
c.annotate("as coded: identically zero,\nthe bound reaches no decision",
           xy=(0.45, 0.0), xytext=(0.30, -6.5), fontsize=8, color=C2,
           arrowprops=dict(arrowstyle="-", color=C2, lw=0.9))
style(c, "satisficing bound  $J_{TH}$",
      "change in prevented share (points)",
      "(c) what the bound costs")
c.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)

# ---------------------------------------------- (d) eta outcome
d = ax[1, 1]
d.axvspan(0.28, 0.60, color=DEAD, zorder=0)
for tg, col, mk, ls, lab in (
        (1.0, C2, "s", (0, (5, 2)), "as coded"),
        (0.0, C3, "^", "solid", "corrected")):
    m, h, b0 = paired("eta", tg, ETA, 0.60)
    d.errorbar(ETA, m, yerr=h, color=col, lw=2, ls=ls, marker=mk, ms=6,
               capsize=3, elinewidth=1, markeredgecolor="white",
               markeredgewidth=1.2, label=f"{lab}  (base {b0:.0f}%)")
d.axhline(0, color=GRID, lw=1.4)
d.axvline(0.60, color=MUTED, lw=1, ls=(0, (2, 3)))
style(d, "fail-safe quality gate  $\\eta$",
      "change in prevented share (points)",
      "(d) what the gate costs")
d.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)

lo = min(ax[1, 0].get_ylim()[0], ax[1, 1].get_ylim()[0])
hi = max(ax[1, 0].get_ylim()[1], ax[1, 1].get_ylim()[1])
ax[1, 0].set_ylim(lo, hi)
ax[1, 1].set_ylim(lo, hi)

fig.suptitle("The two acceptance thresholds: what they govern (top) and "
             "what that costs (bottom).\n5 worlds, 4 simultaneous "
             "ignitions, resource pool 0.25, adaptation on. "
             "Shaded bands are 95 per cent intervals over the worlds.",
             fontsize=10, color=INK, x=0.008, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.90))
os.makedirs(FIGDIR, exist_ok=True)
fig.savefig(os.path.join(FIGDIR, "fig_thresholds.png"), dpi=220, facecolor="white")
print("written")
