"""Figures and tables for the DisasterAware benchmark (No DSS / F5 / F5+EvFIS /
DisasterAware with the language-model proposer).

    python experiments/benchmark/benchmark_figures.py --out experiments/benchmark/out

Reads runs.csv, curves.csv, proposals.jsonl; writes <out>/report/figures/*.png|pdf
and <out>/report/tables/*.csv|md. Figure widths follow IEEE column sizes
(3.5 in single, 7.16 in double).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

ARMS = ["T0", "F5", "F5Ev", "LLM"]
NAME = {"T0": "No DSS", "F5": "F5", "F5Ev": "F5+EvFIS", "LLM": "DisasterAware"}
COL = {"T0": "#8A8F96", "F5": "#3A6FC4", "F5Ev": "#1E9E82", "LLM": "#D9651E"}
SCEN = ["S1", "S2", "S3", "S4", "S5"]
SCEN_LONG = {"S1": "S1 forest, 1 ignition", "S2": "S2 forest, pool 0.6, 2 ignitions",
             "S3": "S3 interface, 1 ignition", "S4": "S4 interface, pool 0.6, 2 ignitions",
             "S5": "S5 as S4, outage 0.25, noise 0.10"}
POP = 9e-4
SCEN_SHORT = {"S1": "S1 forest", "S2": "S2 forest, pool 0.6, 2 ign.", "S3": "S3 interface",
              "S4": "S4 interface, pool 0.6, 2 ign.", "S5": "S5 as S4, outage, noise"}
INK = "#222"; INK2 = "#555"; GRID = "#E3E3E3"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def hw95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def load(out):
    runs = defaultdict(dict)   # (scenario, arm) -> seed -> row
    with open(os.path.join(out, "runs.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["exp"] != "E2" or r["arm"] not in ARMS or r["scenario"] not in SCEN:
                continue
            for k, v in list(r.items()):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
            runs[(r["scenario"], r["arm"])][int(r["seed"])] = r
    # balanced seeds per scenario
    seeds = {}
    for sc in SCEN:
        s = None
        for a in ARMS:
            ks = set(runs[(sc, a)])
            s = ks if s is None else s & ks
        seeds[sc] = sorted(s or [])
    per = defaultdict(dict)  # (scenario, arm, seed) -> t -> ha
    with open(os.path.join(out, "curves.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, fieldnames=["exp", "scenario", "arm", "seed", "t_min", "cells", "frac"]):
            if r["arm"] not in ARMS or r["scenario"] not in SCEN:
                continue
            try:
                sd = int(r["seed"])
                if sd not in seeds[r["scenario"]]:
                    continue
                per[(r["scenario"], r["arm"], sd)][float(r["t_min"])] = float(r["cells"]) * 0.09
            except ValueError:
                continue
    # forward-fill every world to the 12 h horizon (an extinguished fire keeps its final area)
    grid = np.arange(0, 721, 12.0)
    curves = defaultdict(lambda: defaultdict(list))  # (scenario, arm) -> t -> [ha]
    for (sc, arm, sd), ser in per.items():
        ts = np.array(sorted(ser)); vs = np.array([ser[t] for t in ts])
        idx = np.searchsorted(ts, grid, side="right") - 1
        filled = np.where(idx >= 0, vs[np.clip(idx, 0, len(vs) - 1)], 0.0)
        for t, v in zip(grid, filled):
            curves[(sc, arm)][t].append(v)
    props = []
    pp = os.path.join(out, "proposals.jsonl")
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("arm") == "LLM" and d.get("scenario") in SCEN and d.get("seed") in seeds.get(d["scenario"], []):
                    props.append(d)
    return runs, seeds, curves, props


def val(runs, seeds, sc, arm, key):
    return np.array([runs[(sc, arm)][s][key] for s in seeds[sc]], float)


def val_all(runs, seeds, arm, key):
    return np.concatenate([val(runs, seeds, sc, arm, key) for sc in SCEN])


def save(fig, path):
    fig.savefig(path + ".png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def md_table(header, rows):
    s = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    s += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(s) + "\n"


def write_table(base, header, rows):
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(md_table(header, rows))


# ---------------------------------------------------------------- figures
def fig_outcomes(runs, seeds, fd):
    """Fig. A: burned area, affected population, extinguished share per scenario (3 stacked panels)."""
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.6), sharex=True)
    x = np.arange(len(SCEN) + 1)
    w = 0.2
    labels = SCEN + ["Mean"]
    for i, arm in enumerate(ARMS):
        off = (i - 1.5) * w
        # burned
        m = [val(runs, seeds, sc, arm, "burned_ha").mean() for sc in SCEN]
        e = [hw95(val(runs, seeds, sc, arm, "burned_ha")) for sc in SCEN]
        allv = val_all(runs, seeds, arm, "burned_ha"); m.append(allv.mean()); e.append(hw95(allv))
        axes[0].bar(x + off, m, w * 0.92, color=COL[arm], label=NAME[arm], yerr=e,
                    error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor=INK2))
        # population
        m = [val(runs, seeds, sc, arm, "pop_affected").mean() * POP for sc in SCEN]
        e = [hw95(val(runs, seeds, sc, arm, "pop_affected") * POP) for sc in SCEN]
        allv = val_all(runs, seeds, arm, "pop_affected") * POP; m.append(allv.mean()); e.append(hw95(allv))
        m = np.array(m); e = np.array(e)
        lo = np.minimum(e, m - 6.0).clip(min=0)
        axes[1].bar(x + off, m, w * 0.92, color=COL[arm], yerr=[lo, e],
                    error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor=INK2))
        # success
        m = [100 * val(runs, seeds, sc, arm, "success").mean() for sc in SCEN]
        m.append(100 * val_all(runs, seeds, arm, "success").mean())
        axes[2].bar(x + off, m, w * 0.92, color=COL[arm])
    axes[0].set_ylabel("burned area (ha)")
    axes[1].set_ylabel("affected population\n(persons, log scale)")
    axes[1].set_yscale("log"); axes[1].set_ylim(5, 2e4)
    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    axes[2].set_ylabel("extinguished within\nepisode (%)")
    axes[2].set_ylim(0, 105)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    for ax, t in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(-0.02, 1.02, t, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    axes[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22), columnspacing=1.2, handlelength=1.2)
    fig.align_ylabels(axes)
    save(fig, os.path.join(fd, "figA_outcomes"))


def fig_paired(runs, seeds, fd):
    """Fig. B: paired differences in burned area on the same worlds, per scenario (forest plot)."""
    pairs = [("F5Ev", "F5", "F5+EvFIS vs F5"), ("LLM", "F5", "DisasterAware vs F5"), ("LLM", "F5Ev", "DisasterAware vs F5+EvFIS")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), sharey=True)
    rows = SCEN + ["Mean"]
    y = np.arange(len(rows))[::-1]
    for ax, (a, b, title) in zip(axes, pairs):
        ms, es, wins, ns = [], [], [], []
        for sc in SCEN:
            d = val(runs, seeds, sc, b, "burned_ha") - val(runs, seeds, sc, a, "burned_ha")
            ms.append(d.mean()); es.append(hw95(d)); wins.append(int((d > 1e-6).sum())); ns.append(len(d))
        d = val_all(runs, seeds, b, "burned_ha") - val_all(runs, seeds, a, "burned_ha")
        ms.append(d.mean()); es.append(hw95(d)); wins.append(int((d > 1e-6).sum())); ns.append(len(d))
        ax.axvline(0, color=INK2, lw=0.6)
        ax.errorbar(ms, y, xerr=es, fmt="o", color=COL[a], ecolor=COL[a], ms=4, lw=1, capsize=2)
        import matplotlib.transforms as mtr
        tr = mtr.blended_transform_factory(ax.transAxes, ax.transData)
        for yi, m, e, wn, n in zip(y, ms, es, wins, ns):
            ax.text(0.99, yi + 0.08, f"{m:+.1f} ± {e:.1f}", transform=tr, ha="right", va="bottom", fontsize=6.3, color=INK2)
            ax.text(0.99, yi - 0.08, f"{wn}/{n}", transform=tr, ha="right", va="top", fontsize=6.3, color=INK2)
        ax.set_title(title)
        ax.set_yticks(y); ax.set_yticklabels(rows)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.grid(axis="y", visible=False)
        ax.set_xlabel("burned area saved (ha)")
        lo = min(0, min(m - e for m, e in zip(ms, es))) - 5
        hi = max(m + e for m, e in zip(ms, es)) * 1.45 + 5
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, len(rows) - 0.4)
    axes[0].text(0.0, -0.32, "Point: mean paired difference with 95 % CI; right column: mean ± CI and worlds in which the first configuration burned less.",
                 transform=axes[0].transAxes, fontsize=6.3, color=INK2)
    save(fig, os.path.join(fd, "figB_paired"))


def fig_scatter(runs, seeds, fd):
    """Fig. C: per-world burned area, DisasterAware against F5+EvFIS."""
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    mk = {"S1": "o", "S2": "s", "S3": "^", "S4": "D", "S5": "v"}
    lim = 0
    for sc in SCEN:
        xv = val(runs, seeds, sc, "F5Ev", "burned_ha"); yv = val(runs, seeds, sc, "LLM", "burned_ha")
        lim = max(lim, xv.max(), yv.max())
        ax.scatter(xv, yv, s=16, marker=mk[sc], facecolor="none", edgecolor="#3A6FC4" if sc in ("S1", "S2") else "#1E9E82",
                   linewidths=0.8, label=SCEN_SHORT[sc], alpha=0.9)
    lim = lim * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.6, ls="--")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("F5+EvFIS, burned area (ha)"); ax.set_ylabel("DisasterAware, burned area (ha)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    d = val_all(runs, seeds, "F5Ev", "burned_ha") - val_all(runs, seeds, "LLM", "burned_ha")
    n = len(d); wins = int((d > 1e-6).sum()); ties = int((np.abs(d) <= 1e-6).sum())
    ax.text(0.03, 0.97, f"below the diagonal: DisasterAware burned less\n{wins} of {n} worlds ({ties} ties)",
            transform=ax.transAxes, va="top", fontsize=6.8, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=6.3, handletextpad=0.2, markerscale=1.1, columnspacing=0.9)
    save(fig, os.path.join(fd, "figC_scatter"))


def fig_curves(curves, fd, scen=("S1", "S2", "S3", "S4", "S5")):
    """Fig. D: mean burned area against time, five scenarios (extinguished worlds carried forward)."""
    fig, axes = plt.subplots(1, len(scen), figsize=(7.16, 1.9), sharey=True)
    for ax, sc in zip(axes, scen):
        for arm in ARMS:
            c = curves[(sc, arm)]
            if not c:
                continue
            t = np.array(sorted(c)); m = np.array([np.mean(c[k]) for k in t])
            ax.plot(t / 60, m, color=COL[arm], lw=1.4, label=NAME[arm])
        ax.set_title(sc); ax.set_xlabel("time (h)")
        ax.set_xlim(0, 12); ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    axes[0].set_ylabel("mean burned area (ha)")
    axes[0].legend(loc="upper left", fontsize=6.3, handlelength=1.4)
    save(fig, os.path.join(fd, "figD_curves"))


def fig_cost(runs, seeds, fd):
    """Fig. E: decision cost decomposition, mean over S1-S5 (stacked)."""
    terms = [("j_burn", "J_burn"), ("j_asset", "J_asset"), ("j_pop", "J_pop"), ("j_resp", "0.2 J_resp"), ("j_delay", "0.2 J_delay")]
    wts = {k: v / 3.4 for k, v in {"j_burn": 1, "j_asset": 1, "j_pop": 1, "j_resp": 0.2, "j_delay": 0.2}.items()}  # J_total = weighted sum / 3.4
    shade = ["#4a4a4a", "#7a7a7a", "#a8a8a8", "#cfcfcf", "#e8e8e8"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(ARMS))
    bottom = np.zeros(len(ARMS))
    for (k, lab), col in zip(terms, shade):
        v = np.array([wts[k] * val_all(runs, seeds, a, k).mean() for a in ARMS])
        ax.bar(x, v, 0.6, bottom=bottom, color=col, edgecolor="white", linewidth=1, label=lab)
        bottom += v
    for xi, b, a in zip(x, bottom, ARMS):
        ax.text(xi, b + 0.01, f"{b:.3f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x); ax.set_xticklabels([NAME[a] for a in ARMS])
    for lab, a in zip(ax.get_xticklabels(), ARMS):
        lab.set_color(COL[a]); lab.set_fontsize(6.8)
    ax.set_ylabel("decision cost J_total (mean, S1-S5)")
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=6.3, columnspacing=0.8, handlelength=1.0)
    save(fig, os.path.join(fd, "figE_cost"))


def fig_gates(props, fd):
    """Fig. F: fate of the language-model proposals per scenario."""
    cats = [("admitted", "admitted"), ("G5", "rejected at G5 (package margin)"), ("G4", "rejected at G4 (A/B forecast)"),
            ("G3", "rejected at G3 (no improvement)"), ("other", "rejected earlier (G1, G2)")]
    cols = {"admitted": COL["LLM"], "G5": "#b0b0b0", "G4": "#8c8c8c", "G3": "#666666", "other": "#3f3f3f"}
    cnt = {sc: defaultdict(int) for sc in SCEN}
    for d in props:
        g = d.get("gate") or ""
        if d.get("accepted"):
            k = "admitted"
        elif "G5" in g:
            k = "G5"
        elif "G4" in g:
            k = "G4"
        elif "G3" in g:
            k = "G3"
        else:
            k = "other"
        cnt[d["scenario"]][k] += 1
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(SCEN)); bottom = np.zeros(len(SCEN))
    for k, lab in cats:
        v = np.array([cnt[sc][k] for sc in SCEN], float)
        ax.bar(x, v, 0.6, bottom=bottom, color=cols[k], edgecolor="white", linewidth=1, label=lab)
        bottom += v
    tot = sum(sum(c.values()) for c in cnt.values()); adm = sum(c["admitted"] for c in cnt.values())
    for xi, b, sc in zip(x, bottom, SCEN):
        ax.text(xi, b + 3, f"{cnt[sc]['admitted']}/{int(b)}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x); ax.set_xticklabels(SCEN)
    ax.set_ylabel("stage-3 proposals (150 worlds)")
    ax.set_ylim(0, bottom.max() * 1.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), fontsize=6.0, handlelength=1.0, ncol=2, columnspacing=1.0)
    ax.text(0.01, 0.98, f"{adm} of {tot} proposals admitted ({100*adm/tot:.0f} %)", transform=ax.transAxes, ha="left", va="top", fontsize=6.8)
    save(fig, os.path.join(fd, "figF_gates"))
    return cnt


def fig_gain_vs_admit(runs, seeds, fd):
    """Fig. G: burned area saved against F5+EvFIS versus stage-3 admissions, per world."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    xs, ys = [], []
    for sc in SCEN:
        a = val(runs, seeds, sc, "LLM", "acc_3"); d = val(runs, seeds, sc, "F5Ev", "burned_ha") - val(runs, seeds, sc, "LLM", "burned_ha")
        xs.append(a); ys.append(d)
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    jit = (np.random.default_rng(1).random(len(xs)) - 0.5) * 0.3
    ax.axhline(0, color=INK2, lw=0.6)
    ax.scatter(xs + jit, ys, s=12, color=COL["LLM"], alpha=0.65, edgecolor="none")
    # medians per admission count
    for k in sorted(set(xs.astype(int))):
        sel = xs == k
        if sel.sum() >= 3:
            ax.plot([k - 0.3, k + 0.3], [np.median(ys[sel])] * 2, color=INK, lw=1.2)
    ax.set_xlabel("stage-3 admissions in the world"); ax.set_ylabel("burned area saved vs F5+EvFIS (ha)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.text(0.98, 0.04, "bars: median per admission count", transform=ax.transAxes, ha="right", fontsize=6.3, color=INK2)
    save(fig, os.path.join(fd, "figG_gain_vs_admissions"))


# ---------------------------------------------------------------- tables
def tables(runs, seeds, props, gate_cnt, td):
    arms_n = [NAME[a] for a in ARMS]
    # T1 outcome
    rows = []
    for sc in SCEN + ["Mean"]:
        g = (lambda a, k: val_all(runs, seeds, a, k)) if sc == "Mean" else (lambda a, k: val(runs, seeds, sc, a, k))
        n = len(g("T0", "burned_ha"))
        rows.append([sc, n, "burned area (ha)"] + [f"{g(a,'burned_ha').mean():.1f} ± {hw95(g(a,'burned_ha')):.1f}" for a in ARMS])
        rows.append([sc, n, "affected population"] + [f"{(g(a,'pop_affected')*POP).mean():.0f} ± {hw95(g(a,'pop_affected')*POP):.0f}" for a in ARMS])
        rows.append([sc, n, "extinguished within episode (%)"] + [f"{100*g(a,'success').mean():.0f}" for a in ARMS])
        rows.append([sc, n, "time to extinction, extinguished worlds (min)"] + [
            (f"{g(a,'out_min')[g(a,'success')>0].mean():.0f}" if (g(a, 'success') > 0).any() else "-") for a in ARMS])
    write_table(os.path.join(td, "T1_outcome"), ["Scenario", "n", "Measure"] + arms_n, rows)
    # T2 cost
    rows = []
    for sc in SCEN + ["Mean"]:
        g = (lambda a, k: val_all(runs, seeds, a, k)) if sc == "Mean" else (lambda a, k: val(runs, seeds, sc, a, k))
        for k, lab in (("j_burn", "J_burn"), ("j_asset", "J_asset"), ("j_pop", "J_pop"), ("j_resp", "J_resp"), ("j_delay", "J_delay"), ("j_phys", "J_phys"), ("j_total", "J_total")):
            rows.append([sc, lab] + [f"{g(a,k).mean():.3f}" for a in ARMS])
    write_table(os.path.join(td, "T2_cost"), ["Scenario", "Cost term"] + arms_n, rows)
    # T3 paired
    pairs = [("F5", "T0"), ("F5Ev", "F5"), ("LLM", "T0"), ("LLM", "F5"), ("LLM", "F5Ev")]
    rows = []
    for a, b in pairs:
        for sc in SCEN + ["Mean"]:
            g = (lambda x, k: val_all(runs, seeds, x, k)) if sc == "Mean" else (lambda x, k: val(runs, seeds, sc, x, k))
            db = g(b, "burned_ha") - g(a, "burned_ha"); dp = (g(b, "pop_affected") - g(a, "pop_affected")) * POP
            dj = g(b, "j_phys") - g(a, "j_phys"); dt = g(b, "j_total") - g(a, "j_total")
            ds = 100 * (g(a, "success").mean() - g(b, "success").mean())
            n = len(db); wins = int((db > 1e-6).sum()); ties = int((np.abs(db) <= 1e-6).sum())
            rows.append([f"{NAME[a]} vs {NAME[b]}", sc, n, f"{db.mean():.1f} ± {hw95(db):.1f}", f"{wins}/{n}", ties,
                         f"{dp.mean():.0f} ± {hw95(dp):.0f}", f"{dj.mean():.3f} ± {hw95(dj):.3f}", f"{dt.mean():.3f} ± {hw95(dt):.3f}", f"{ds:+.0f}"])
    write_table(os.path.join(td, "T3_paired"), ["Comparison", "Scenario", "n", "burned area saved (ha)", "better in", "ties",
                                                "population saved", "J_phys saved", "J_total saved", "extinguished (points)"], rows)
    # T4 adaptation and cost of running
    rows = []
    for a in ARMS:
        rf = val_all(runs, seeds, a, "rules_final"); t1 = val_all(runs, seeds, a, "tried_1"); a1 = val_all(runs, seeds, a, "acc_1")
        t2 = val_all(runs, seeds, a, "tried_2"); a2 = val_all(runs, seeds, a, "acc_2"); t3 = val_all(runs, seeds, a, "tried_3"); a3 = val_all(runs, seeds, a, "acc_3")
        calls = val_all(runs, seeds, a, "prop_calls"); wall = val_all(runs, seeds, a, "wall_s")
        so = val_all(runs, seeds, a, "sited_orders"); nm = val_all(runs, seeds, a, "n_macros"); na = val_all(runs, seeds, a, "n_actuators")
        rows.append([NAME[a], f"{rf.mean():.1f}", f"{a1.sum():.0f}/{t1.sum():.0f}", f"{a2.sum():.0f}/{t2.sum():.0f}", f"{a3.sum():.0f}/{t3.sum():.0f}",
                     (f"{100*a3.sum()/t3.sum():.0f}" if t3.sum() else "-"), f"{calls.mean():.1f}", f"{nm.mean():.1f}", f"{na.mean():.1f}", f"{so.mean():.1f}", f"{wall.mean():.0f}"])
    write_table(os.path.join(td, "T4_adaptation"), ["Configuration", "rules at end", "stage 1 adm/att", "stage 2 adm/att", "stage 3 adm/att",
                                                    "stage 3 admission (%)", "model calls / run", "macros / run", "actuators / run", "sited orders / run", "wall s / run"], rows)
    # T5 proposals by scenario and gate
    rows = []
    for sc in SCEN:
        c = gate_cnt[sc]; tot = sum(c.values())
        rows.append([sc, tot, c["admitted"], c["G5"], c["G4"], c["G3"], c["other"], f"{100*c['admitted']/tot:.0f}" if tot else "-"])
    tot = sum(sum(c.values()) for c in gate_cnt.values())
    rows.append(["All", tot] + [sum(gate_cnt[sc][k] for sc in SCEN) for k in ("admitted", "G5", "G4", "G3", "other")] +
                [f"{100*sum(gate_cnt[sc]['admitted'] for sc in SCEN)/tot:.0f}"])
    write_table(os.path.join(td, "T5_proposals_gates"), ["Scenario", "proposals", "admitted", "rejected G5 (package margin)", "rejected G4 (A/B)",
                                                         "rejected G3 (no improvement)", "rejected G1/G2", "admission (%)"], rows)
    # T6 admitted proposals by kind and package
    kinds = defaultdict(lambda: [0, 0])
    effects = defaultdict(int)
    for d in props:
        k = (d.get("kind") or "rule", "package" if d.get("package") else "single")
        kinds[k][1] += 1
        if d.get("accepted"):
            kinds[k][0] += 1
            for c in d.get("clauses") or []:
                effects[(c.get("effect"), c.get("sector"))] += 1
    rows = [[k[0], k[1], v[1], v[0], f"{100*v[0]/v[1]:.0f}"] for k, v in sorted(kinds.items())]
    write_table(os.path.join(td, "T6_proposals_kind"), ["kind", "form", "proposed", "admitted", "admission (%)"], rows)
    rows = [[e[0], e[1], n] for e, n in sorted(effects.items(), key=lambda x: -x[1])]
    write_table(os.path.join(td, "T7_admitted_clauses"), ["effect", "sector", "clauses in admitted packages"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs, seeds, curves, props = load(args.out)
    rep = os.path.join(args.out, "report")
    fd = os.path.join(rep, "figures"); td = os.path.join(rep, "tables")
    os.makedirs(fd, exist_ok=True); os.makedirs(td, exist_ok=True)
    print("worlds per scenario:", {sc: len(s) for sc, s in seeds.items()}, "proposals:", len(props))
    fig_outcomes(runs, seeds, fd)
    fig_paired(runs, seeds, fd)
    fig_scatter(runs, seeds, fd)
    fig_curves(curves, fd)
    fig_cost(runs, seeds, fd)
    cnt = fig_gates(props, fd)
    fig_gain_vs_admit(runs, seeds, fd)
    tables(runs, seeds, props, cnt, td)
    print("written to", rep)


if __name__ == "__main__":
    main()
