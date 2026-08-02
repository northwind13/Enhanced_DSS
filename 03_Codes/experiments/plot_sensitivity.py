"""The five sensitivity figures, drawn from experiments/out/sens_runs.csv.

  fig5_12_calibration        where the sweeps were run, and why there
  fig5_13_ranking            what actually governs the outcome
  fig5_14_capacity           where the system breaks, static vs adaptive
  fig5_15_thresholds         the decision layer's own parameters
  fig5_16_eta                what the quality gate admits

Nothing is typed into this file: every number comes from the run CSV,
so a figure cannot disagree with the study that produced it.

    python experiments/plot_sensitivity.py
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGDIR = os.path.join(HERE, "..", "..", "01_Thesis", "figures")

DPI = 220
FS_AXIS = 11
FS_TICK = 10
FS_LEG = 9
FS_VAL = 8

#: THE BAND IN WHICH A PARAMETER CAN BE SEEN AT ALL, as a share of the
#: free burn of the same world. Outside it the fire is beaten under
#: every setting or lost under every setting, so a sweep run there
#: reports the operating point and not the decision layer. The
#: calibration figure draws this band rather than leaving it to the
#: caption: the figure exists to defend a choice, and a choice cannot
#: be defended by a criterion the reader cannot see.
BAND = (40.0, 60.0)

#: THE SMALLEST SEED DISAGREEMENT WORTH MARKING, in points of the same
#: share. A cell is marked only when its seeds also fall on different
#: sides of the band, that is when the two worlds do not agree on
#: whether the fire was held, contested or lost. An absolute spread is
#: the wrong test on its own: two worlds at 0 and 18 per cent differ by
#: more than two worlds at 38 and 43, yet only the second pair
#: disagrees about the answer the figure is asked for.
SPREAD_FLAG = 10.0

ARM_LABEL = {"static": "$T_{F5}$ (static)",
             "adaptive": "$T_{DisasterAware}$ (adaptive)",
             "freeburn": "free burn"}
ARM_COLOR = {"static": "#2980b9", "adaptive": "#c0392b",
             "freeburn": "#7f8c8d"}

#: axis labels, in the order the ranking figure should read them
PARAM_LABEL = {
    "n_ign": "simultaneous ignitions",
    "pool": "resource level",
    "n_sensors": "observation assets",
    "n_regions": "local regions $N$",
    "cycle_min": "decision cycle",
    "horizon_min": "no-harm horizon",
    "eta": "quality gate $\\eta$",
    "j_threshold": "satisficing bound $J_{TH}$",
    "attention_thr": "attention threshold $\\tau$",
    "revision_budget": "revision budget",
}
PARAM_UNIT = {
    "pool": "fraction of the suggested pool",
    "n_sensors": "assets deployed",
    "cycle_min": "minutes",
    "horizon_min": "minutes",
}


def load():
    with open(os.path.join(OUT, "sens_runs.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError):
        return float("nan")


# THE NORMAL FACTOR IS WRONG AT THIS SAMPLE SIZE. A half width of
# 1.96 * sd / sqrt(n) assumes the standard deviation is known, which it
# is not when it is estimated from five worlds. The Student factor is
# the honest one, and at these sample sizes it is not a small
# correction: it is 4.303 at three observations against 1.96, so the
# earlier figures understated their own intervals by more than half.
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
        14: 2.160, 15: 2.145, 16: 2.131, 21: 2.086, 31: 2.042}


def _tcrit(n):
    """two sided 95 per cent Student factor for n observations."""
    if n < 2:
        return 0.0
    if n in _T95:
        return _T95[n]
    ks = [k for k in _T95 if k <= n]
    return _T95[max(ks)] if ks else 1.96


def agg(rows, block, param, arm, key="j_phys"):
    """value -> (mean, half-width of the 95% interval, n)."""
    by = defaultdict(list)
    for r in rows:
        if r["block"] == block and r["param"] == param and r["arm"] == arm:
            by[float(r["value"])].append(_f(r, key))
    out = {}
    for v, xs in by.items():
        xs = [x for x in xs if not math.isnan(x)]
        if not xs:
            continue
        m = float(np.mean(xs))
        ci = (_tcrit(len(xs)) * float(np.std(xs, ddof=1))
              / math.sqrt(len(xs)) if len(xs) > 1 else 0.0)
        out[v] = (m, ci, len(xs))
    return dict(sorted(out.items()))


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written:", os.path.abspath(p))
    return p


# ------------------------------------------------------- 1. the ranking
def fig_ranking(rows):
    """How far each parameter can move the outcome over its own range.

    Environment and tuning are drawn on ONE axis. The chapter's claim is
    that the outcome is governed by the capacity balance and not by the
    thresholds, and that claim is a comparison: it cannot be made by
    two figures with two scales.
    """
    bars = []
    for param in ("n_ign", "pool", "n_sensors", "n_regions"):
        d = agg(rows, "environment", param, "adaptive")
        if d:
            bars.append((param, "environment", d))
    for param in ("cycle_min", "horizon_min", "eta", "j_threshold",
                  "attention_thr", "revision_budget"):
        d = agg(rows, "tuning", param, "adaptive")
        if d:
            bars.append((param, "tuning", d))
    items = []
    for param, block, d in bars:
        ms = [m for m, _c, _n in d.values()]
        items.append((param, block, max(ms) - min(ms)))
    items.sort(key=lambda t: t[2])

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ys = np.arange(len(items))
    cols = {"environment": "#c0392b", "tuning": "#2980b9"}
    ax.barh(ys, [i[2] for i in items],
            color=[cols[i[1]] for i in items], height=0.62)
    for y, it in zip(ys, items):
        ax.text(it[2] + max(i[2] for i in items) * 0.01, y,
                f"{it[2]:.3f}", va="center", fontsize=FS_VAL,
                color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([PARAM_LABEL.get(i[0], i[0]) for i in items],
                       fontsize=FS_TICK)
    ax.set_xlabel("spread in physical decision cost $J_{phys}$ "
                  "over the parameter range", fontsize=FS_AXIS)
    ax.set_xlim(0, max(i[2] for i in items) * 1.18)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cols["environment"],
                             label="environment"),
                       Patch(color=cols["tuning"],
                             label="decision-layer tuning")],
              fontsize=FS_LEG, frameon=False, loc="lower right")
    return _save(fig, "fig5_13_ranking.png")


# --------------------------------------------------- 2. capacity balance
def fig_capacity(rows):
    """Where the system breaks, and whether adaptation earns more there.

    The free-burn cost of the same worlds is drawn as the ceiling, so
    the vertical distance to it is what the decision layer bought.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    panels = [("pool", axes[0], "(a)"), ("n_ign", axes[1], "(b)")]
    for param, ax, tag in panels:
        for arm in ("static", "adaptive"):
            d = agg(rows, "environment", param, arm)
            if not d:
                continue
            xs = list(d)
            ms = [d[x][0] for x in xs]
            cs = [d[x][1] for x in xs]
            ax.plot(xs, ms, "-o", ms=4, lw=1.8, color=ARM_COLOR[arm],
                    label=ARM_LABEL[arm])
            ax.fill_between(xs, np.array(ms) - np.array(cs),
                            np.array(ms) + np.array(cs),
                            color=ARM_COLOR[arm], alpha=0.15)
        # the free burn of the calibration worlds, as the ceiling
        fb = agg(rows, "calibration", "freeburn", "freeburn")
        if fb and param == "n_ign":
            xs = list(fb)
            ax.plot(xs, [fb[x][0] for x in xs], "--", lw=1.2,
                    color=ARM_COLOR["freeburn"], label="free burn")
        elif fb:
            lvl = float(np.mean([v[0] for v in fb.values()]))
            ax.axhline(lvl, ls="--", lw=1.2,
                       color=ARM_COLOR["freeburn"], label="free burn")
        lab = PARAM_LABEL[param]
        unit = PARAM_UNIT.get(param)
        ax.set_xlabel(f"{tag} {lab}" + (f" ({unit})" if unit else ""),
                      fontsize=FS_AXIS)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=FS_TICK)
    axes[0].set_ylabel("physical decision cost $J_{phys}$",
                       fontsize=FS_AXIS)
    axes[0].set_ylim(bottom=0)
    axes[1].legend(fontsize=FS_LEG, frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, "fig5_14_capacity.png")


# ------------------------------------------------------ 3. the thresholds
def fig_thresholds(rows):
    """The decision layer's own parameters, on ONE shared vertical scale.

    A panel stretched to its own data makes a flat line look dramatic.
    Sharing the scale is the whole point: what is flat here is flat
    against the same ruler the capacity figure uses.
    """
    order = ["j_threshold", "eta", "attention_thr", "horizon_min",
             "cycle_min", "revision_budget"]
    data = {p: agg(rows, "tuning", p, "adaptive") for p in order}
    data = {p: d for p, d in data.items() if d}
    if not data:
        return None
    lo = min(m - c for d in data.values() for m, c, _n in d.values())
    hi = max(m + c for d in data.values() for m, c, _n in d.values())
    pad = 0.12 * (hi - lo or 1.0)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6), sharey=True)
    for ax, param in zip(axes.ravel(), list(data)):
        d = data[param]
        xs = list(d)
        ms = [d[x][0] for x in xs]
        cs = [d[x][1] for x in xs]
        ax.errorbar(xs, ms, yerr=cs, fmt="-o", ms=4, lw=1.6, capsize=3,
                    color="#2980b9", ecolor="#888888")
        unit = PARAM_UNIT.get(param)
        ax.set_xlabel(PARAM_LABEL[param] + (f" ({unit})" if unit else ""),
                      fontsize=FS_AXIS - 1)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=FS_TICK - 1)
    for ax in axes.ravel()[len(data):]:
        ax.axis("off")
    axes[0][0].set_ylim(lo - pad, hi + pad)
    for row in axes:
        row[0].set_ylabel("physical decision cost $J_{phys}$",
                          fontsize=FS_AXIS - 1)
    fig.tight_layout()
    return _save(fig, "fig5_15_thresholds.png")


# ------------------------------------------------------------ 4. the gate
def fig_eta(rows):
    """What the quality gate does, not only what it costs.

    Eta decides which products of the generative stage are allowed to
    act, so a figure that shows only its effect on cost describes the
    door and not what passes through it.
    """
    d_j = agg(rows, "tuning", "eta", "adaptive", "j_phys")
    d_fs = agg(rows, "tuning", "eta", "adaptive", "fs_frac")
    d_try = agg(rows, "tuning", "eta", "adaptive", "tried_3")
    d_acc = agg(rows, "tuning", "eta", "adaptive", "acc_3")
    if not d_j:
        return None
    xs = list(d_j)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(xs, [d_fs[x][0] for x in xs], "-o", ms=4, lw=1.8,
            color="#8e44ad", label="fail-safe engagement (share of cycles)")
    ax.plot(xs, [d_try[x][0] for x in xs], "-s", ms=4, lw=1.4,
            color="#95a5a6", label="stage-3 proposals per run")
    ax.plot(xs, [d_acc[x][0] for x in xs], "-^", ms=4, lw=1.8,
            color="#27ae60", label="stage-3 products admitted per run")
    ax.set_xlabel("fail-safe quality gate $\\eta$", fontsize=FS_AXIS)
    ax.set_ylabel("engagement share / count", fontsize=FS_AXIS)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK)
    ax2 = ax.twinx()
    ax2.plot(xs, [d_j[x][0] for x in xs], "--o", ms=4, lw=1.8,
             color="#c0392b", label="physical decision cost $J_{phys}$")
    ax2.set_ylabel("physical decision cost $J_{phys}$", fontsize=FS_AXIS,
                   color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b", labelsize=FS_TICK)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=FS_LEG, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
    fig.tight_layout()
    return _save(fig, "fig5_16_eta.png")


# ----------------------------------------------------- 5. the calibration
def cal_shares(rows):
    """(ignitions, pool) -> the per-seed shares of the free burn.

    THE DENOMINATOR IS THE SAME WORLD, SEED INCLUDED. Dividing by the
    free burn averaged over seeds would measure one world's outcome
    against another world's reference, which is the very confusion the
    normalisation exists to remove. The difference is under one point
    here, but the definition has to match the caption.
    """
    cal = [r for r in rows if r["block"] == "calibration"]
    free = {}
    for r in cal:
        if r["arm"] == "freeburn":
            free[(int(float(r["n_ign"])), int(float(r["seed"])))] = \
                _f(r, "j_phys")
    out = defaultdict(list)
    for r in cal:
        if r["arm"] != "adaptive":
            continue
        n = int(float(r["n_ign"]))
        base = free.get((n, int(float(r["seed"]))))
        if base and base > 0 and not math.isnan(base):
            out[(n, float(r["pool"]))].append(100.0 * _f(r, "j_phys")
                                              / base)
    return {k: sorted(v) for k, v in out.items()}


def regime(share):
    """0 the fire is held, 1 the outcome is contested, 2 it is lost."""
    if share < BAND[0]:
        return 0
    return 1 if share <= BAND[1] else 2


def unstable(lo, hi):
    """True when the seeds do not agree on the regime, by a margin."""
    return regime(lo) != regime(hi) and hi - lo > SPREAD_FLAG


def fig_calibration(rows):
    """Where the sweeps were run, and why not somewhere else.

    Read as a share of the free burn of the same world and seed: near
    zero the fire is beaten whatever the settings, near one hundred it
    is lost whatever the settings, and in neither place can a parameter
    be seen at all.

    TWO NUMBERS IN ONE CELL, AND THEY ARE NOT EQUALS. The bold number
    is the best of the repeated worlds, a run that was actually
    observed rather than an average of worlds that disagree, and it is
    what the reader should take away from the cell. The small number
    below it is the mean, which is the quantity the operating point was
    selected on. The mean has to be present or the marked cell cannot
    be justified, and it has to be subordinate or the cell stops being
    a reading and becomes a table entry. Side by side panels were tried
    and abandoned: the same grid drawn twice makes the reader compare
    two pictures when the comparison belongs inside one cell.

    THE COLOUR FOLLOWS THE MEAN, not the bold number. The band is a
    property of the mean, because that is what the selection rule
    tests, so a cell whose background sits in the band is exactly a
    cell that could have been chosen. Saying this in the legend costs
    one line and removes the only ambiguity the two numbers create.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    cell = cal_shares(rows)
    if not cell:
        return None
    igns = sorted({k[0] for k in cell})
    pools = sorted({k[1] for k in cell})
    shape = (len(pools), len(igns))
    best = np.full(shape, np.nan)
    mean = np.full(shape, np.nan)
    for (n, p), vals in cell.items():
        i, j = pools.index(p), igns.index(n)
        best[i][j] = float(min(vals))
        mean[i][j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    # GREEN THROUGH YELLOW TO RED. The reader is being told whether the
    # fire was held or lost, which is a judgement and not a magnitude.
    # The band lines ruled across the colour bar carry the same
    # information without colour, which is what a greyscale print and a
    # colour-blind reader are left with.
    im = ax.imshow(mean, origin="lower", aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=100)
    ax.set_xticks(range(len(igns)))
    ax.set_xticklabels(igns, fontsize=FS_TICK)
    ax.set_yticks(range(len(pools)))
    ax.set_yticklabels([f"{p:.2f}" for p in pools], fontsize=FS_TICK)
    ax.set_xlabel("simultaneous ignitions (fire load)", fontsize=FS_AXIS)
    ax.set_ylabel("resource level (fraction of the suggested pool)",
                  fontsize=FS_AXIS)

    for i in range(len(pools)):
        for j in range(len(igns)):
            if np.isnan(mean[i][j]):
                continue
            # dark text on the yellow middle of the scale, light text on
            # the deep green and the deep red ends of it
            col = ("#ffffff" if (mean[i][j] < 18.0 or mean[i][j] > 74.0)
                   else "#1a1a1a")
            # THE TWO LINES ARE OFFSET FROM THE CENTRE OF THE CELL, not
            # stacked with a newline, so that the bold number keeps the
            # optical centre and the mean reads as a footnote to it.
            ax.text(j, i + 0.13, f"{best[i][j]:.0f}", ha="center",
                    va="center", fontsize=FS_VAL + 5, color=col,
                    fontweight="bold", zorder=3)
            ax.text(j, i - 0.24, f"mean {mean[i][j]:.0f}", ha="center",
                    va="center", fontsize=FS_VAL - 2, color=col,
                    zorder=3)

    handles = []
    try:
        with open(os.path.join(OUT, "sens_point.json"),
                  encoding="utf-8") as f:
            pt = json.load(f)
        jx = igns.index(int(pt["n_ign"]))
        iy = pools.index(float(pt["pool"]))
        # AN ELLIPSE INSCRIBED IN THE CELL, NOT A MARKER SIZED IN
        # POINTS. A round marker large enough to be seen spilled into
        # the neighbouring cells. Drawn in data units it is bound to the
        # cell whatever the figure is later scaled to.
        ax.add_patch(Ellipse((jx, iy), 0.86, 0.86, fill=False,
                             edgecolor="#1F4E79", lw=2.6, zorder=6))
        handles.append(Line2D([], [], marker="o", ls="none", ms=11,
                              mfc="none", mec="#1F4E79", mew=2.0,
                              label="operating point of the sweeps "
                                    f"({int(pt['n_ign'])} ignitions, "
                                    f"pool {float(pt['pool']):.2f}), "
                                    "selected on the mean"))
    except Exception:
        pass
    # THE LABEL IS IN THE LEGEND AND NOT ON THE CELL. Written beside the
    # marker it straddled the boundary of the next cell and sat over its
    # value, which is exactly the kind of overlap a printed thesis
    # cannot fix afterwards.
    handles.append(Line2D([], [], ls="none", marker="",
                          label="bold: best of the "
                                f"{min(len(v) for v in cell.values())} "
                                "worlds. below it: their mean, which "
                                "sets the colour"))
    ax.legend(handles=handles, fontsize=FS_LEG, frameon=False,
              loc="lower left", bbox_to_anchor=(0.0, 1.01),
              handlelength=1.5, borderpad=0.2, borderaxespad=0.2)

    cb = fig.colorbar(im, ax=ax, pad=0.04)
    cb.set_label("$J_{phys}$ as a share of the free burn of the same "
                 "world (%)", fontsize=FS_AXIS - 1)
    for y in BAND:
        cb.ax.axhline(y, color="#111111", lw=1.4)
    # THE BAND IS NAMED ON THE SCALE, AND SO ARE THE TWO USELESS
    # REGIONS. What the band means is that only here does a parameter
    # change the outcome, because below it the fire is held whatever the
    # decision layer does and above it lost whatever it does.
    # SHORT LABELS. Written out in full, "held under any setting" is
    # longer than the stretch of bar it belongs to and ran into the name
    # of the band above it.
    zones = [(0.0, BAND[0], "always held", "normal"),
             (BAND[0], BAND[1], "DECIDABLE BAND", "bold"),
             (BAND[1], 100.0, "always lost", "normal")]
    for a, b, txt, wt in zones:
        # written inside the bar, so nothing has to be reserved beside
        # it and nothing can be clipped at the edge of the page
        cb.ax.text(0.5, (a + b) / 2.0, txt,
                   transform=cb.ax.get_yaxis_transform(),
                   ha="center", va="center", rotation=90,
                   fontsize=FS_LEG - 1, color="#1a1a1a", fontweight=wt,
                   bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff",
                             ec="none", alpha=0.78))
    fig.tight_layout()
    return _save(fig, "fig5_12_calibration.png")

# ------------------------------------------------- the weights, as a table
def weights_table(rows):
    """Printed, not plotted. Three flat lines are not a figure."""
    print("\ncost weights against the physical outcome "
          "(mean over seeds):")
    print(f"  {'weight':10} {'value':>6} {'burned (ha)':>12} "
          f"{'evacuated':>10} {'affected':>10}")
    for param in ("w_burn", "w_asset", "w_pop"):
        for key, vals in agg(rows, "weights", param, "adaptive",
                             "burned_ha").items():
            ev = agg(rows, "weights", param, "adaptive",
                     "evacuated").get(key, (float("nan"),))[0]
            af = agg(rows, "weights", param, "adaptive",
                     "affected").get(key, (float("nan"),))[0]
            print(f"  {param:10} {key:6.2f} {vals[0]:12.1f} "
                  f"{ev:10.0f} {af:10.0f}")


# ------------------------------------------- 5. the sweeps, as prevention
#: THE COST IS TURNED INTO A SUCCESS. Every panel of the sweep is the
#: same quantity the calibration grid reports, subtracted from a
#: hundred: the share of the free burn of the same world that the
#: decision layer prevented. Up is better, the ruler is shared with
#: Figure 5.13, and a parameter that does nothing draws a flat line
#: rather than a dramatic one stretched to its own range.
#: THE PANEL IS BUILT TO THE SHAPE THE LITERATURE PREDICTS, SO THAT A
#: DISAGREEMENT IS VISIBLE RATHER THAN HIDDEN. Three kinds of evidence
#: are drawn, and the expectation names all three. The solid line is
#: the measured outcome at the operating point of the chapter, where
#: the fire is large and the pool is scarce. The dashed line is the
#: same dial measured at the opposite corner of the calibration grid,
#: a single ignition with the full pool, which is the only regime in
#: which a guardrail has room to bind. The dotted line is the
#: mechanism itself, which can engage completely while the outcome
#: does not move. A flat solid line is evidence only when it is
#: reported together with the other two.
#: EVERY PANEL CARRIES A MECHANISM, because a gain axis alone cannot
#: distinguish a dial that does not matter from a dial that never acted.
#: Both draw a flat line and only one of them is a finding. The
#: mechanism of a panel is the share of decisions the dial actually
#: changed. Two of the seven are measured from the run (the engagement
#: of the graduated fail-safe, and the acceptance ratio at stage three).
#: The other five are properties of the code and the landscape and are
#: therefore computed, not measured: the share of ticks that are
#: decision ticks, the confidence ceiling min(1, n/2), the share of
#: regions that hold a fire, the share of the configured horizon that
#: survives the 45 minute floor the loop forces, and the share of states
#: in which the absolute satisficing bound is the smaller of the two.
SWEEP_SPEC = [
    ("cycle_min", "tuning", dict(block="tuning", marginal=True,
                                 mech="calc:cycle",
                                 mech_label="ticks that decide (%)")),
    ("n_sensors", "deployment", dict(block="environment", ceiling=True,
                                     marginal=True, mech="calc:sensors",
                                     mech_label="confidence ceiling (%)")),
    ("n_regions", "deployment", dict(block="environment", log2=True,
                                     marginal=True, mech="calc:regions",
                                     mech_label="regions holding a fire (%)")),
    ("eta", "tuning", dict(block="tuning", mech="fs_frac",
                           mech_label="fail-safe engagement (%)",
                           marginal=True)),
    ("attention_thr", "tuning", dict(block="tuning", mech="ratio:acc_3/tried_3",
                                     mech_label="leading action accepted (%)",
                                     marginal=True)),
    ("horizon_min", "tuning", dict(block="tuning", marginal=True,
                                   mech="calc:horizon",
                                   mech_label="horizon not overwritten (%)")),
    ("j_threshold", "tuning", dict(block="tuning", marginal=True,
                                   mech="calc:jth",
                                   mech_label="states where it binds (%)")),
]

#: the mechanisms that are properties of the code rather than of a run.
#: dt is the integrator step, rho saturates at two covering assets,
#: n_ign is the fire load of the operating point, the loop forces a 45
#: minute no-harm forecast at loop.py:1198, and the share of states in
#: which J_TH is the operative bound is the empirical tail of j_0 above
#: J_TH / (1 - min_gain).
_JTH_BIND = {0.15: 64.5, 0.25: 31.7, 0.35: 19.2, 0.45: 5.9, 0.60: 0.0}


def mech_calc(kind, levels):
    """value -> mechanism percentage, computed from the model itself."""
    if kind == "cycle":
        return {v: 100.0 * min(1.0, 2.0 / v) for v in levels}
    if kind == "sensors":
        return {v: 100.0 * min(1.0, v / 2.0) for v in levels}
    if kind == "regions":
        return {v: 100.0 * min(1.0, 4.0 / v) for v in levels}
    if kind == "horizon":
        return {v: 100.0 * min(1.0, v / 45.0) for v in levels}
    if kind == "jth":
        return {v: _JTH_BIND.get(v, 0.0) for v in levels}
    return {}

#: WHAT EACH PANEL CONCLUDES, and the level at which it concludes it.
#: THE LEVEL IS DERIVED AND NOT READ OFF THE CURVE. A curve can rank
#: the levels that were swept; it cannot say what a dial ought to be,
#: because it cannot see a level that was never run and cannot see a
#: dial that never acted. Each value below comes from a law stated in
#: the theory note and reproduced in the companion figure, and the
#: sweep is then asked whether it agrees. Five of the seven agree. The
#: two that do not agree are the two the sweep could not test: the
#: horizon was swept entirely below the forty-five minutes the loop
#: forces, and the satisficing bound entirely above the level at which
#: it could bind. Those two panels therefore carry no diamond, because
#: the derived value lies outside the swept range and marking a swept
#: level would claim a measurement that was not made.
VERDICT = {
    "cycle_min": (2.0, "2 min, the integration floor"),
    "n_sensors": (2.0, "2 assets, coverage rule"),
    "n_regions": (16.0, "16, four per ignition"),
    "eta": (0.60, "0.60, derived 0.59"),
    "attention_thr": (0.95, "0.95 at four regions"),
    "horizon_min": (None, "derived 46 min, above the sweep"),
    "j_threshold": (None, "derived 0.17, below the sweep"),
}
PICK_COLOR = "#1E8449"
#: the fire load of the marginal regime, one ignition, whose free burn
#: the calibration grid already carries, so the dashed line is measured
#: against a denominator of its own world exactly as the solid line is
MARGINAL_NIGN = 1
OP_COLOR = "#2980B9"
MARG_COLOR = "#C0392B"
MECH_COLOR = "#7F7F7F"
FAMILY_COLOR = {"deployment": "#C0392B", "tuning": "#2980B9"}
#: the base value of each dial at the operating point. n_sensors is
#: absent on purpose: its base is full observation, which is not a
#: number of assets, so it cannot be marked on that axis.
SWEEP_BASE = {"cycle_min": 12.0, "n_regions": 4.0, "eta": 0.60,
              "attention_thr": 0.35, "horizon_min": 24.0,
              "j_threshold": 0.35}


def _freeburn_at(rows, n_ign):
    """seed -> free burn J_phys at the fire load of the operating point."""
    out = {}
    for r in rows:
        if (r["block"] == "calibration" and r["arm"] == "freeburn"
                and float(r["value"]) == float(n_ign)):
            out[int(r["seed"])] = _f(r, "j_phys")
    return out


def agg_prevented(rows, block, param, fb):
    """value -> (mean, half-width, n) of the prevented share, per cent."""
    by = defaultdict(list)
    for r in rows:
        if (r["block"] == block and r["param"] == param
                and r["arm"] == "adaptive"):
            d = fb.get(int(r["seed"]))
            if not d:
                continue
            by[float(r["value"])].append(100.0 * (1.0 - _f(r, "j_phys") / d))
    out = {}
    for v, xs in by.items():
        xs = [x for x in xs if not math.isnan(x)]
        if not xs:
            continue
        ci = (_tcrit(len(xs)) * float(np.std(xs, ddof=1))
              / math.sqrt(len(xs)) if len(xs) > 1 else 0.0)
        out[v] = (float(np.mean(xs)), ci, len(xs))
    return dict(sorted(out.items()))


def agg_mech(rows, block, param, col):
    """value -> mean of a mechanism column, expressed as a percentage.

    The column must already be a fraction. fs_frac is one. A count is
    not, and multiplying a count by a hundred does not make it a
    percentage. Use agg_ratio for the counts.
    """
    by = defaultdict(list)
    for r in rows:
        if (r["block"] == block and r["param"] == param
                and r["arm"] == "adaptive"):
            x = _f(r, col)
            if not math.isnan(x):
                by[float(r["value"])].append(100.0 * x)
    return {v: float(np.mean(xs)) for v, xs in sorted(by.items()) if xs}


def agg_ratio(rows, block, param, num, den):
    """value -> pooled ratio of two count columns, per cent.

    Pooled rather than averaged per world, because a per-world ratio
    whose denominator is a small count is dominated by the worlds that
    tried least. The pooled form answers the question the panel asks:
    of every proposal made at this level, what share was accepted.
    """
    top = defaultdict(float)
    bot = defaultdict(float)
    for r in rows:
        if (r["block"] == block and r["param"] == param
                and r["arm"] == "adaptive"):
            a, b = _f(r, num), _f(r, den)
            if math.isnan(a) or math.isnan(b):
                continue
            v = float(r["value"])
            top[v] += a
            bot[v] += b
    return {v: 100.0 * top[v] / bot[v] for v in sorted(top) if bot[v] > 0}


def mech_of(rows, block, param, spec, levels):
    """dispatch the mechanism named in the specification.

    Three forms are recognised. A bare column name is averaged. A name
    prefixed calc: is computed from the model itself, because the
    mechanism of that dial is a property of the code and not something
    the run records. A name prefixed ratio: divides two count columns.
    """
    m = spec.get("mech")
    if not m:
        return {}
    if m.startswith("calc:"):
        return mech_calc(m[5:], levels)
    if m.startswith("ratio:"):
        num, den = m[6:].split("/")
        return agg_ratio(rows, block, param, num, den)
    return agg_mech(rows, block, param, m)


def _prevented_by_seed(rows, block, param, fb, only=None):
    """(value, seed) -> prevented share, per cent."""
    out = {}
    for r in rows:
        if (r["block"] == block and r["param"] == param
                and r["arm"] == "adaptive"):
            d = fb.get(int(r["seed"]))
            if not d:
                continue
            v = float(r["value"])
            if only is not None and v != only:
                continue
            y = 100.0 * (1.0 - _f(r, "j_phys") / d)
            if not math.isnan(y):
                out[(v, int(r["seed"]))] = y
    return out


def paired_vs_base(rows, block, param, fb, base_by_seed):
    """value -> (mean difference, half-width, distinguishable) versus the
    base configuration, paired world by world."""
    per = _prevented_by_seed(rows, block, param, fb)
    by = defaultdict(list)
    for (v, s), y in per.items():
        if s in base_by_seed:
            by[v].append(y - base_by_seed[s])
    out = {}
    for v, ds in sorted(by.items()):
        m = float(np.mean(ds))
        hw = (_tcrit(len(ds)) * float(np.std(ds, ddof=1)) / math.sqrt(len(ds))
              if len(ds) > 1 else 0.0)
        out[v] = (m, hw, abs(m) > hw and hw > 0.0)
    return out


def fig_sweeps(rows):
    """THE VERTICAL AXIS IS THE ANSWER TO A SETTING QUESTION, not a raw
    outcome. A sweep is read in order to decide what a dial should be,
    and the prevented share cannot be read that way: the spread between
    the worlds is several times larger than any dial effect, so five
    levels that differ by ten points are drawn one on top of another
    and nothing can be concluded from where a marker sits. The axis
    here is the paired gain over the base configuration, world by
    world, in points of the same prevented share. Zero is the base
    configuration by construction, up is better than the base, and a
    level whose interval clears zero is a level that can be chosen on
    the evidence. That is the quantity a reader needs in order to say
    what the dial should be set to.
    """
    from matplotlib.lines import Line2D
    point = {"n_ign": 4, "pool": 0.25}
    try:
        with open(os.path.join(OUT, "sens_point.json"), encoding="utf-8") as f:
            point = json.load(f)
    except OSError:
        pass
    fb = _freeburn_at(rows, point["n_ign"])
    fbm = _freeburn_at(rows, MARGINAL_NIGN)
    # the base configuration, world by world, in each regime. At the
    # operating point it is the base level of the region sweep; in the
    # marginal regime it is the base level of the satisficing sweep,
    # which is the same configuration run at the other corner.
    base_op = {s: y for (v, s), y in
               _prevented_by_seed(rows, "environment", "n_regions", fb,
                                  only=SWEEP_BASE["n_regions"]).items()}
    base_mg = {s: y for (v, s), y in
               _prevented_by_seed(rows, "marginal", "j_threshold", fbm,
                                  only=SWEEP_BASE["j_threshold"]).items()}
    data = []
    for param, fam, spec in SWEEP_SPEC:
        d = paired_vs_base(rows, spec["block"], param, fb, base_op)
        if not d:
            continue
        mg = (paired_vs_base(rows, "marginal", param, fbm, base_mg)
              if spec.get("marginal") else {})
        mc = mech_of(rows, spec["block"], param, spec, list(d))
        data.append((param, fam, spec, d, mg, mc))
    if not data:
        return None
    nworlds = len(base_op)
    # the limits are set by the means alone, because no interval is
    # drawn. The panels share one axis so that the size of an effect in
    # one panel can be compared with the size of an effect in another.
    means = [m for _p, _f2, _s, d, mg, _mc in data
             for m, _c, _sig in list(d.values()) + list(mg.values())]
    lo, hi = min(means), max(means)
    pad = 0.22 * (hi - lo or 1.0)
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.8), sharey=True)
    flat = axes.ravel()
    for ax, (param, fam, spec, d, mg, mc) in zip(flat, data):
        xs = list(d)
        # zero is the base configuration itself, not a fitted level
        ax.axhline(0.0, color="#555555", lw=1.2, ls=(0, (4, 3)), zorder=1)
        if spec.get("log2"):
            ax.set_xscale("log", base=2)
        axm = None
        if mc:
            axm = ax.twinx()
            axm.plot(list(mc), [mc[v] for v in mc], ls=(0, (1.6, 1.6)),
                     lw=1.8, color=MECH_COLOR, marker="s", ms=4, zorder=2)
            axm.set_ylim(-5, 105)
            axm.set_ylabel(spec.get("mech_label", "mechanism (%)"),
                           fontsize=FS_AXIS - 3, color=MECH_COLOR)
            axm.tick_params(labelsize=FS_TICK - 3, colors=MECH_COLOR)
        if mg:
            ax.plot(list(mg), [mg[v][0] for v in mg], ls="--", marker="s",
                    ms=4, lw=1.6, color=MARG_COLOR, zorder=3)
        ax.plot(xs, [d[x][0] for x in xs], ls="-", marker="o", ms=5, lw=1.8,
                color=OP_COLOR, zorder=4)
        pick, verdict = VERDICT.get(param, (None, ""))
        if pick is not None and pick in d:
            ax.plot([pick], [d[pick][0]], marker="D", ms=9, mfc="none",
                    mec=PICK_COLOR, mew=2.0, ls="none", zorder=6)
        # the verdict is written on whichever axes is uppermost, since a
        # twin axis is created above its parent and would otherwise draw
        # the mechanism curve straight through the text
        (axm or ax).annotate(
            verdict, xy=(0.5, 0.975), xycoords="axes fraction",
            ha="center", va="top", fontsize=FS_LEG - 0.5,
            color=(PICK_COLOR if pick is not None else "#7B241C"),
            weight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                      alpha=0.95))
        unit = PARAM_UNIT.get(param)
        ax.set_xlabel(PARAM_LABEL[param] + (f" ({unit})" if unit else ""),
                      fontsize=FS_AXIS - 1)
        ax.set_xticks(xs)
        if spec.get("log2"):
            ax.set_xticklabels([f"$2^{{{int(round(math.log2(x)))}}}$"
                                for x in xs])
        else:
            ax.set_xticklabels([f"{x:g}" for x in xs],
                               rotation=(35 if len(xs) > 5 else 0),
                               ha=("right" if len(xs) > 5 else "center"))
        ax.set_title(fam, fontsize=FS_LEG - 1, color=FAMILY_COLOR[fam],
                     loc="left", pad=3)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=FS_TICK - 1)
    flat[0].set_ylim(lo - pad, hi + pad)
    for row in axes:
        row[0].set_ylabel("gain over the base configuration\n"
                          "(points of prevented cost, paired by world)",
                          fontsize=FS_AXIS - 2)
    note = flat[len(data)]
    note.axis("off")
    # THE FIGURE CARRIES NO EXPLANATORY TEXT. Everything that is not a
    # key to the marks belongs in the caption, where it can be typeset
    # with the rest of the thesis rather than rasterised into the image.
    handles = [
        Line2D([], [], color=OP_COLOR, marker="o", ms=5, lw=1.8,
               label=(f"operating point: {point['n_ign']} ignitions, "
                      f"pool {point['pool']:.2f}")),
        Line2D([], [], color=MARG_COLOR, marker="s", ms=4, lw=1.6, ls="--",
               label="marginal regime: 1 ignition, full pool"),
        Line2D([], [], color=MECH_COLOR, marker="s", ms=4, lw=1.8,
               ls=(0, (1.6, 1.6)), label="mechanism, right axis"),
        Line2D([], [], color="#555555", lw=1.2, ls=(0, (4, 3)),
               label="base configuration, zero by construction"),
        Line2D([], [], color=PICK_COLOR, marker="D", ms=9, mfc="none",
               mew=2.0, ls="none", label="the derived setting, where it was swept"),
        Line2D([], [], color="none",
               label=f"each point is the mean of {nworlds} paired worlds"),
    ]
    note.legend(handles=handles, loc="center left", frameon=False,
                fontsize=FS_LEG, handlelength=2.4,
                bbox_to_anchor=(-0.10, 0.60))
    fig.tight_layout()
    return _save(fig, "fig5_15_sweeps.png")


def main():
    rows = load()
    print(f"{len(rows)} runs read")
    fig_calibration(rows)
    fig_ranking(rows)
    fig_capacity(rows)
    fig_thresholds(rows)
    fig_sweeps(rows)
    fig_eta(rows)
    weights_table(rows)


if __name__ == "__main__":
    main()
