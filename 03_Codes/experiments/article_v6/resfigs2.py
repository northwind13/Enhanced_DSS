"""Sharper result figures: axis scaled to the two decision layers, relative savings, per-world differences."""
import csv, json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

OUT = "/home/claude/bench/out"; FD = "/home/claude/v6/fig"
ALL = ["T0", "F5", "F5Ev", "LLM"]; SCEN = ["S1", "S2", "S3", "S4", "S5"]; POP = 9e-4
NAME = {"T0": "No DSS", "F5Ev": "Layer 1 (closed)", "LLM": "DisasterAware"}
COL = {"T0": "#8A8F96", "F5": "#3A6FC4", "F5Ev": "#1E9E82", "LLM": "#D9651E"}
INK, INK2, GRID = "#222", "#555", "#E3E3E3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300})


def hw(x):
    x = np.asarray(x, float); return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


runs = collections.defaultdict(dict)
for r in csv.DictReader(open(f"{OUT}/runs.csv", encoding="utf-8")):
    if r["arm"] in ALL and r["scenario"] in SCEN:
        for k, v in list(r.items()):
            try: r[k] = float(v)
            except (TypeError, ValueError): pass
        runs[(r["scenario"], r["arm"])][int(r["seed"])] = r
seeds = {sc: sorted(set.intersection(*[set(runs[(sc, a)]) for a in ALL])) for sc in SCEN}


def V(arm, key, sc=None):
    scs = [sc] if sc else SCEN
    return np.array([runs[(s, arm)][x][key] for s in scs for x in seeds[s]], float)


def save(fig, name):
    fig.savefig(f"{FD}/{name}.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)


# ---------------- r1b: outcomes, axis scaled to the two layers, No DSS as a reference number
fig, axes = plt.subplots(3, 1, figsize=(3.45, 5.2), sharex=True)
x = np.arange(len(SCEN) + 1); w = 0.36; labels = SCEN + ["Mean"]
ARMS = ["F5Ev", "LLM"]
for i, a in enumerate(ARMS):
    off = (i - 0.5) * w
    m = [V(a, "burned_ha", sc).mean() for sc in SCEN] + [V(a, "burned_ha").mean()]; e = [hw(V(a, "burned_ha", sc)) for sc in SCEN] + [hw(V(a, "burned_ha"))]
    axes[0].bar(x + off, m, w * 0.9, color=COL[a], label=NAME[a], yerr=e, error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor=INK2))
    m2 = [V(a, "pop_affected", sc).mean() * POP for sc in SCEN] + [V(a, "pop_affected").mean() * POP]; e2 = [hw(V(a, "pop_affected", sc) * POP) for sc in SCEN] + [hw(V(a, "pop_affected") * POP)]
    axes[1].bar(x + off, m2, w * 0.9, color=COL[a], yerr=e2, error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor=INK2))
    m3 = [100 * V(a, "success", sc).mean() for sc in SCEN] + [100 * V(a, "success").mean()]
    axes[2].bar(x + off, m3, w * 0.9, color=COL[a])
# relative change labels
for j, sc in enumerate(SCEN + [None]):
    b1 = V("F5Ev", "burned_ha", sc).mean(); b2 = V("LLM", "burned_ha", sc).mean()
    p1 = V("F5Ev", "pop_affected", sc).mean(); p2 = V("LLM", "pop_affected", sc).mean()
    s1 = 100 * V("F5Ev", "success", sc).mean(); s2 = 100 * V("LLM", "success", sc).mean()
    axes[0].text(x[j], max(b1, b2) * 1.0 + hw(V("F5Ev", "burned_ha", sc)) + 8, f"−{100*(b1-b2)/b1:.0f} %", ha="center", fontsize=6.5, color=COL["LLM"], fontweight="bold")
    axes[1].text(x[j], max(p1, p2) * POP + hw(V("F5Ev", "pop_affected", sc) * POP) + 25, f"−{100*(p1-p2)/p1:.0f} %", ha="center", fontsize=6.5, color=COL["LLM"], fontweight="bold")
    axes[2].text(x[j], max(s1, s2) + 3, f"+{s2-s1:.0f} pts", ha="center", fontsize=6.5, color=COL["LLM"], fontweight="bold")
t0b = V("T0", "burned_ha").mean(); t0p = V("T0", "pop_affected").mean() * POP
axes[0].set_ylabel("burned area (ha)"); axes[0].set_ylim(0, 240)
axes[0].text(0.99, 0.60, f"No DSS: {t0b:.0f} ha (mean),\n0 % extinguished", transform=axes[0].transAxes, ha="right", va="top", fontsize=6.5, color=COL["T0"])
axes[1].set_ylabel("affected population (persons)"); axes[1].set_ylim(0, 1150)
axes[1].text(0.99, 0.60, f"No DSS: {t0p:.0f} persons (mean)", transform=axes[1].transAxes, ha="right", va="top", fontsize=6.5, color=COL["T0"])
axes[2].set_ylabel("extinguished within\nepisode (%)"); axes[2].set_ylim(0, 118); axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
for ax, t in zip(axes, ["(a)", "(b)", "(c)"]): ax.text(-0.02, 1.02, t, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
axes[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.26), columnspacing=1.2, handlelength=1.2)
fig.align_ylabels(axes); save(fig, "r1b_outcomes")

# ---------------- r2b: per-world paired differences sorted + relative savings panel
fig, axes = plt.subplots(2, 1, figsize=(3.45, 5.0), gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.45})
d = V("F5Ev", "burned_ha") - V("LLM", "burned_ha")
scs = np.array([s for s in SCEN for _ in seeds[s]])
order = np.argsort(-d); ds = d[order]; cs = scs[order]
colors = [COL["LLM"] if v > 1e-9 else ("#bbbbbb" if abs(v) <= 1e-9 else "#7a1f1f") for v in ds]
axes[0].bar(np.arange(len(ds)), ds, 1.0, color=colors, edgecolor="none"); axes[0].axhline(0, color=INK2, lw=0.6)
n_pos = int((d > 1e-9).sum()); n_tie = int((np.abs(d) <= 1e-9).sum()); n_neg = int((d < -1e-9).sum())
axes[0].text(0.02, 0.95, f"DisasterAware burned less in {n_pos} of 150 worlds\nties {n_tie} (small fires already held), worse in {n_neg}\nmean {d.mean():.1f} ± {hw(d):.1f} ha, median {np.median(d):.1f} ha", transform=axes[0].transAxes, va="top", fontsize=6.6)
axes[0].set_xlabel("worlds, sorted by saving"); axes[0].set_ylabel("burned area saved\nvs Layer 1 (ha)"); axes[0].set_xlim(-1, 150); axes[0].grid(axis="x", visible=False)
axes[0].text(-0.02, 1.02, "(a)", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8)
# relative savings vs layer 1 on five measures (mean over worlds, ratio of means)
meas = [("burned_ha", "burned area"), ("pop_affected", "affected population"), ("j_phys", "J_phys"), ("j_total", "J_total"), ("j_resp", "J_resp (resources)"), ("j_delay", "J_delay")]
rel = []; relf5 = []
for k, lab in meas:
    l1 = V("F5Ev", k).mean(); da = V("LLM", k).mean(); f5 = V("F5", k).mean()
    rel.append(100 * (l1 - da) / l1); relf5.append(100 * (f5 - l1) / f5)
y = np.arange(len(meas))
axes[1].barh(y + 0.2, relf5, 0.38, color=COL["F5Ev"], label="Layer 1 vs static seed")
axes[1].barh(y - 0.2, rel, 0.38, color=COL["LLM"], label="DisasterAware vs Layer 1")
for yi, v in zip(y, rel): axes[1].text(v + 1.5 if v >= 0 else v - 1.5, yi - 0.2, f"{v:+.0f} %", va="center", ha="left" if v >= 0 else "right", fontsize=6.3, color=COL["LLM"])
for yi, v in zip(y, relf5): axes[1].text(v + 1.5 if v >= 0 else v - 1.5, yi + 0.2, f"{v:+.0f} %", va="center", ha="left" if v >= 0 else "right", fontsize=6.3, color=COL["F5Ev"])
axes[1].axvline(0, color=INK2, lw=0.6); axes[1].set_yticks(y); axes[1].set_yticklabels([lab for k, lab in meas], fontsize=6.8); axes[1].invert_yaxis()
axes[1].set_xlabel("reduction of the mean (%)"); axes[1].set_xlim(-32, 100); axes[1].grid(axis="x", color=GRID, lw=0.5); axes[1].grid(axis="y", visible=False)
axes[1].legend(fontsize=6.2, loc="lower right"); axes[1].text(-0.02, 1.02, "(b)", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8)
save(fig, "r2b_paired")
print("ok", n_pos, n_tie, n_neg, [f"{v:.0f}" for v in rel], [f"{v:.0f}" for v in relf5])
