"""The five figures of the H1 and C2/H2 experiments.

  fig5_24_latency_scale     decision latency against the number of local
                            agents, distributed against centralized,
                            with the cycle budget drawn
  fig5_25_complexity        antecedent evaluations per cycle, the
                            implementation-independent form of the same
                            comparison
  fig5_26_outcome_scale     burned area and total cost against the
                            number of agents
  fig5_27_tornado           the factors ranked by how far they move the
                            decision cost, grouped by what they are
  fig5_28_flip_shift        decision flips and concept activation shift
                            per factor

Every number is read from the run CSVs. Nothing is typed in, so a figure
cannot disagree with the campaign that produced it.

    python experiments/plot_h1_h2.py
"""
from __future__ import annotations

import collections
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGDIR = os.path.join(HERE, "..", "..", "01_Thesis", "figures")

DPI = 220
FS_AXIS, FS_TICK, FS_LEG, FS_VAL = 11, 10, 9, 8

ARM = {"A_distributed_open": ("distributed, open", "#c0392b", "o", "-"),
       "B_centralized_closed": ("centralized, closed", "#2980b9", "s", "-"),
       "C_centralized_open": ("centralized, open", "#7f8c8d", "^", "--")}

KIND_LABEL = {"feature": "observation feature",
              "mf": "membership partition",
              "weight": "cost weight"}
KIND_COLOR = {"feature": "#c0392b", "mf": "#2980b9", "weight": "#27ae60"}

CONCEPT_SHORT = {"fire_threat_level": "fire threat",
                 "asset_exposure_risk": "asset exposure",
                 "suppression_feasibility": "feasibility",
                 "intervention_urgency": "urgency",
                 "evacuation_pressure": "evac. pressure"}


def _rows(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(OUT, pattern))):
        with open(f, encoding="utf-8") as fh:
            out += list(csv.DictReader(fh))
    return out


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written:", os.path.abspath(p))


def _mean_by(rows, key, value, where=None):
    d = collections.defaultdict(list)
    for r in rows:
        if where and not where(r):
            continue
        try:
            d[key(r)].append(float(r[value]))
        except (KeyError, ValueError):
            continue
    return {k: float(np.mean(v)) for k, v in d.items()}


# ------------------------------------------------------------ 1. latency
def fig_latency(lat):
    """Latency against scale, with the budget that decides pass or fail.

    The budget is drawn because the comparison alone does not answer the
    question the section asks. A configuration can be twice as slow as
    another and still decide in a thousandth of the time available, and
    a reader who sees only the ratio would not know that.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for cfg, (label, color, marker, ls) in ARM.items():
        d = _mean_by(lat, lambda r: int(r["n_regions"]), "lat_median_ms",
                     where=lambda r: r["config"] == cfg)
        if not d:
            continue
        xs = sorted(d)
        ax.plot(xs, [d[x] for x in xs], marker=marker, color=color,
                linestyle=ls, label=label, lw=1.8, ms=6)
    budget = 12 * 60 * 1000.0
    ax.axhline(budget, color="#444444", lw=1.4, ls=":")
    ax.text(1.02, budget * 0.72, "decision cycle, 12 min",
            fontsize=FS_VAL, color="#444444")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"], fontsize=FS_TICK)
    ax.set_xlabel("local agents $N$", fontsize=FS_AXIS)
    ax.set_ylabel("median decision latency (ms, log scale)",
                  fontsize=FS_AXIS)
    ax.grid(alpha=0.25, which="both")
    ax.set_axisbelow(True)
    ax.legend(fontsize=FS_LEG, frameon=False, loc="center left")
    _save(fig, "fig5_24_latency_scale.png")


# --------------------------------------------------------- 2. complexity
def fig_complexity(lat):
    """What the decision costs in rule evaluations, which no machine
    changes. This is the metric H1's complexity claim rests on."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for cfg, (label, color, marker, ls) in ARM.items():
        d = _mean_by(lat, lambda r: int(r["n_regions"]), "ante_median",
                     where=lambda r: r["config"] == cfg)
        if not d:
            continue
        xs = sorted(d)
        ax.plot(xs, [d[x] for x in xs], marker=marker, color=color,
                linestyle=ls, label=label, lw=1.8, ms=6)
        for x in xs:
            ax.annotate(f"{d[x]:,.0f}".replace(",", " "),
                        (x, d[x]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=FS_VAL,
                        color=color)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"], fontsize=FS_TICK)
    ax.set_xlabel("local agents $N$", fontsize=FS_AXIS)
    ax.set_ylabel("antecedent evaluations per decision\n"
                  "(log scale)", fontsize=FS_AXIS)
    ax.set_ylim(top=1.2e6)
    ax.grid(alpha=0.25, which="both")
    ax.set_axisbelow(True)
    ax.legend(fontsize=FS_LEG, frameon=False, loc="center left")
    _save(fig, "fig5_25_complexity.png")


# ------------------------------------------------------------ 3. outcome
def fig_outcome(outc):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, field, ylab in (
            (axes[0], "burned_ha", "burned area (ha)"),
            (axes[1], "j_total", "total decision cost $J$")):
        for cfg, (label, color, marker, ls) in ARM.items():
            d = _mean_by(outc, lambda r: int(r["n_regions"]), field,
                         where=lambda r: r["config"] == cfg)
            if not d:
                continue
            xs = sorted(d)
            ax.plot(xs, [d[x] for x in xs], marker=marker, color=color,
                    linestyle=ls, label=label, lw=1.8, ms=6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16])
        ax.set_xticklabels(["1", "4", "16"], fontsize=FS_TICK)
        ax.set_xlabel("local agents $N$", fontsize=FS_AXIS)
        ax.set_ylabel(ylab, fontsize=FS_AXIS)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=FS_LEG, frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig5_26_outcome_scale.png")


# ------------------------------------------------- 4. the factor ranking
def morris_mu(mor):
    """mu* and sigma per factor, from the trajectory rows.

    Each row carries the cost of a point relative to the unperturbed run,
    so the elementary effect of the factor that moved is the difference
    between consecutive points of the same trajectory, divided by the
    step. Only complete steps contribute: the first row of a trajectory
    has no predecessor.
    """
    by_traj = collections.defaultdict(list)
    for r in mor:
        by_traj[r["traj"]].append(r)
    eff = collections.defaultdict(list)
    for _t, rs in by_traj.items():
        prev = None
        for r in rs:
            if prev is not None:
                try:
                    step = float(r["delta"])
                    if abs(step) > 1e-9:
                        ee = (float(r["d_j_total"])
                              - float(prev["d_j_total"])) / step
                        eff[(r["kind"], r["factor"])].append(ee)
                except (KeyError, ValueError):
                    pass
            prev = r
    return {k: (float(np.mean(np.abs(v))), float(np.std(v)), len(v))
            for k, v in eff.items() if v}


def fig_tornado(oat, mor):
    """The factors ranked by the size of the effect they carry.

    One-at-a-time gives the effect of a factor on its own; Morris gives
    it averaged over the rest of the space. Both are drawn, because a
    factor that is large alone and small in company is a different thing
    from one that is large in both, and the ranking alone hides that.
    """
    d = collections.defaultdict(list)
    for r in oat:
        d[(r["kind"], r["factor"])].append(abs(float(r["d_j_total"])))
    items = sorted(((k, float(np.mean(v))) for k, v in d.items()),
                   key=lambda kv: kv[1])
    mu = morris_mu(mor)

    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    ys = np.arange(len(items))
    ax.barh(ys, [v for _k, v in items],
            color=[KIND_COLOR[k[0]] for k, _v in items], height=0.66)
    span = max(v for _k, v in items) or 1.0
    for y, (k, v) in zip(ys, items):
        txt = f"{v:.4f}"
        if k in mu:
            txt += f"   (Morris $\\mu^*$ {mu[k][0]:.3f})"
        ax.text(v + span * 0.015, y, txt, va="center", fontsize=FS_VAL,
                color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([k[1].replace("_", " ") for k, _v in items],
                       fontsize=FS_TICK)
    ax.set_xlim(0, span * 1.42)
    ax.set_xlabel("mean $|\\Delta J|$ over the perturbation range",
                  fontsize=FS_AXIS)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=KIND_COLOR[k], label=KIND_LABEL[k])
                       for k in ("feature", "mf", "weight")],
              fontsize=FS_LEG, frameon=False, loc="lower right")
    _save(fig, "fig5_27_tornado.png")


# --------------------------------------------- 5. flips and the concepts
def fig_flip_shift(oat):
    """What a perturbation does to the DECISION, not only to the cost.

    The left panel counts the region cycles whose ordered intervention
    family changed. The right panel shows where the movement landed in
    the concept space. A factor can be quiet on cost and loud here, and
    a framework that claims traceability owes the reader that view.
    """
    d = collections.defaultdict(list)
    for r in oat:
        d[(r["kind"], r["factor"])].append(100.0 * float(r["flip_rate"]))
    items = sorted(((k, float(np.mean(v))) for k, v in d.items()),
                   key=lambda kv: kv[1])

    concepts = list(CONCEPT_SHORT)
    shift = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in oat:
        for c in concepts:
            col = "shift_" + c
            if col in r and r[col] not in ("", None):
                shift[(r["kind"], r["factor"])][c].append(float(r[col]))

    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(items))
    ax.barh(ys, [v for _k, v in items],
            color=[KIND_COLOR[k[0]] for k, _v in items], height=0.66)
    span = max(v for _k, v in items) or 1.0
    for y, (_k, v) in zip(ys, items):
        ax.text(v + span * 0.015, y, f"{v:.1f}", va="center",
                fontsize=FS_VAL, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([k[1].replace("_", " ") for k, _v in items],
                       fontsize=FS_TICK)
    ax.set_xlim(0, span * 1.16)
    ax.set_xlabel("region cycles whose ordered intervention changed (%)",
                  fontsize=FS_AXIS)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    ax2 = fig.add_subplot(gs[0, 1])
    keys = [k for k, _v in items]
    mat = np.array([[float(np.mean(shift[k][c])) if shift[k][c] else 0.0
                     for c in concepts] for k in keys])
    im = ax2.imshow(mat, aspect="auto", cmap="magma_r",
                    interpolation="nearest")
    ax2.set_xticks(range(len(concepts)))
    ax2.set_xticklabels([CONCEPT_SHORT[c] for c in concepts],
                        rotation=35, ha="right", fontsize=FS_TICK)
    ax2.set_yticks(range(len(keys)))
    ax2.set_yticklabels([k[1].replace("_", " ") for k in keys],
                        fontsize=FS_TICK)
    ax2.set_xlabel("decision concept", fontsize=FS_AXIS)
    cb = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)
    cb.set_label("mean $|\\Delta a|$", fontsize=FS_LEG)
    cb.ax.tick_params(labelsize=FS_VAL)
    _save(fig, "fig5_28_flip_shift.png")


def main():
    lat = _rows("central_latency.csv")
    outc = _rows("central_outcome.csv")
    oat = _rows("prop_oat.csv")
    mor = _rows("prop_morris*.csv")
    if lat:
        fig_latency(lat)
        fig_complexity(lat)
    if outc:
        fig_outcome(outc)
    if oat:
        fig_tornado(oat, mor)
        fig_flip_shift(oat)


if __name__ == "__main__":
    main()
