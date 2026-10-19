"""Result figures for the v6 article (three configurations: No DSS, Layer 1, DisasterAware)."""
import csv, json, os, re, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

OUT = "/home/claude/bench/out"; FD = "/home/claude/v6/fig"; os.makedirs(FD, exist_ok=True)
ARMS = ["T0", "F5Ev", "LLM"]; ALL = ["T0", "F5", "F5Ev", "LLM"]
NAME = {"T0": "No DSS", "F5": "static seed F5", "F5Ev": "Layer 1 (closed)", "LLM": "DisasterAware"}
COL = {"T0": "#8A8F96", "F5": "#3A6FC4", "F5Ev": "#1E9E82", "LLM": "#D9651E"}
SCEN = ["S1", "S2", "S3", "S4", "S5"]; POP = 9e-4
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
props = []
for line in open(f"{OUT}/proposals.jsonl", encoding="utf-8"):
    try: d = json.loads(line)
    except json.JSONDecodeError: continue
    if d.get("arm") == "LLM" and d.get("scenario") in SCEN and d.get("seed") in seeds[d["scenario"]]: props.append(d)


def V(arm, key, sc=None):
    scs = [sc] if sc else SCEN
    return np.array([runs[(s, arm)][x][key] for s in scs for x in seeds[s]], float)


def core(r):
    r = re.sub(r"^G\d+: ", "", r); return re.sub(r" \(adaptation-born.*$", "", r)


def save(fig, name):
    fig.savefig(f"{FD}/{name}.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)


# ---------------- Fig outcomes (single column, three stacked panels)
fig, axes = plt.subplots(3, 1, figsize=(3.45, 5.0), sharex=True)
x = np.arange(len(SCEN) + 1); w = 0.26; labels = SCEN + ["Mean"]
for i, a in enumerate(ARMS):
    off = (i - 1) * w
    m = [V(a, "burned_ha", sc).mean() for sc in SCEN] + [V(a, "burned_ha").mean()]; e = [hw(V(a, "burned_ha", sc)) for sc in SCEN] + [hw(V(a, "burned_ha"))]
    axes[0].bar(x + off, m, w * 0.92, color=COL[a], label=NAME[a], yerr=e, error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor=INK2))
    m = np.array([V(a, "pop_affected", sc).mean() * POP for sc in SCEN] + [V(a, "pop_affected").mean() * POP]); e = np.array([hw(V(a, "pop_affected", sc) * POP) for sc in SCEN] + [hw(V(a, "pop_affected") * POP)])
    axes[1].bar(x + off, m, w * 0.92, color=COL[a], yerr=[np.minimum(e, m - 6).clip(min=0), e], error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor=INK2))
    m = [100 * V(a, "success", sc).mean() for sc in SCEN] + [100 * V(a, "success").mean()]
    axes[2].bar(x + off, m, w * 0.92, color=COL[a])
axes[0].set_ylabel("burned area (ha)"); axes[1].set_ylabel("affected population\n(persons, log)"); axes[1].set_yscale("log"); axes[1].set_ylim(5, 2e4)
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{v:g}")); axes[2].set_ylabel("extinguished within\nepisode (%)"); axes[2].set_ylim(0, 105)
axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
for ax, t in zip(axes, ["(a)", "(b)", "(c)"]): ax.text(-0.02, 1.02, t, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.26), columnspacing=1.0, handlelength=1.2)
fig.align_ylabels(axes); save(fig, "r1_outcomes")

# ---------------- Fig paired: scatter DA vs L1 + forest of three steps (single column, two panels stacked)
fig, axes = plt.subplots(2, 1, figsize=(3.45, 5.6), gridspec_kw={"height_ratios": [1.5, 1]})
ax = axes[0]; mk = {"S1": "o", "S2": "s", "S3": "^", "S4": "D", "S5": "v"}; lim = 0
for sc in SCEN:
    xv = V("F5Ev", "burned_ha", sc); yv = V("LLM", "burned_ha", sc); lim = max(lim, xv.max(), yv.max())
    ax.scatter(xv, yv, s=16, marker=mk[sc], facecolor="none", edgecolor="#3A6FC4" if sc in ("S1", "S2") else "#1E9E82", linewidths=0.8, label=sc)
lim *= 1.05; ax.plot([0, lim], [0, lim], ls="--", color=INK2, lw=0.6); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
d = V("F5Ev", "burned_ha") - V("LLM", "burned_ha")
ax.text(0.03, 0.97, f"below the diagonal: DisasterAware burned less\n{int((d>1e-9).sum())} of 150 worlds, {int((np.abs(d)<=1e-9).sum())} ties", transform=ax.transAxes, va="top", fontsize=6.6)
ax.set_xlabel("Layer 1 alone, burned area (ha)"); ax.set_ylabel("DisasterAware, burned area (ha)"); ax.grid(axis="x", color=GRID, lw=0.5)
ax.legend(loc="lower right", fontsize=6.2, ncol=2, handletextpad=0.2, columnspacing=0.8); ax.text(-0.02, 1.02, "(a)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
ax = axes[1]
pairs = [("F5", "T0", "static seed vs No DSS"), ("F5Ev", "F5", "Layer 1 vs static seed"), ("LLM", "F5Ev", "DisasterAware vs Layer 1")]
y = np.arange(len(pairs))[::-1]; ms, es, txt = [], [], []
for a, b, lab in pairs:
    dd = V(b, "burned_ha") - V(a, "burned_ha"); ms.append(dd.mean()); es.append(hw(dd)); txt.append(f"{dd.mean():.1f} ± {hw(dd):.1f} ha, {int((dd>1e-9).sum())}/150")
ax.barh(y, ms, 0.5, color=[COL[a] for a, b, l in pairs], xerr=es, error_kw=dict(elinewidth=0.8, capsize=2.5, ecolor=INK)); ax.axvline(0, color=INK2, lw=0.6)
ax.set_xlim(0, 260)
for yi, m, e, t in zip(y, ms, es, txt): ax.text(m + e + 5, yi, t, va="center", fontsize=6.4, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([l for a, b, l in pairs], fontsize=6.8); ax.set_xlabel("burned area saved on the same worlds (ha)"); ax.grid(axis="x", color=GRID, lw=0.5); ax.grid(axis="y", visible=False)
ax.text(-0.02, 1.02, "(b)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
save(fig, "r2_paired")

# ---------------- Fig costs (single column): two cost types per config + decomposition
fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.2), gridspec_kw={"width_ratios": [1, 1.15]})
x = np.arange(len(ARMS)); w = 0.38
jp = [V(a, "j_phys").mean() for a in ARMS]; jt = [V(a, "j_total").mean() for a in ARMS]
axes[0].bar(x - w / 2, jp, w, color=[COL[a] for a in ARMS], label="J_phys")
axes[0].bar(x + w / 2, jt, w, color=[COL[a] for a in ARMS], alpha=0.45, hatch="////", edgecolor="white", label="J_total")
for xi, a, b in zip(x, jp, jt): axes[0].text(xi - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=5.6); axes[0].text(xi + w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=5.6)
axes[0].set_xticks(x); axes[0].set_xticklabels(["No\nDSS", "Layer\n1", "Disaster-\nAware"], fontsize=6); axes[0].set_ylabel("decision cost (mean)"); axes[0].legend(fontsize=6, loc="upper right"); axes[0].set_ylim(0, 0.68)
terms = [("j_burn", "J_burn", "#4a4a4a"), ("j_asset", "J_asset", "#7a7a7a"), ("j_pop", "J_pop", "#a8a8a8"), ("j_resp", "0.2 J_resp", "#cfcfcf"), ("j_delay", "0.2 J_delay", "#e8e8e8")]
wts = {"j_burn": 1, "j_asset": 1, "j_pop": 1, "j_resp": 0.2, "j_delay": 0.2}; bottom = np.zeros(len(ARMS))
for k, lab, c in terms:
    v = np.array([wts[k] * V(a, k).mean() / 3.4 for a in ARMS]); axes[1].bar(x, v, 0.6, bottom=bottom, color=c, edgecolor="white", label=lab); bottom += v
axes[1].set_xticks(x); axes[1].set_xticklabels(["No\nDSS", "Layer\n1", "Disaster-\nAware"], fontsize=6); axes[1].set_ylabel("J_total by term"); axes[1].legend(fontsize=5.4, ncol=1, loc="upper right", handlelength=1)
axes[1].set_ylim(0, 0.68)
for ax, t in zip(axes, ["(a)", "(b)"]): ax.text(-0.04, 1.02, t, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
save(fig, "r3_costs")

# ---------------- Fig products: kind x verdict + per-family heatmap (single column, stacked)
kinds = ["rule", "concept", "macro", "actuator"]; KN = {"rule": "rule", "concept": "concept", "macro": "macro", "actuator": "actuator"}
verd = lambda d: "admitted" if d.get("accepted") else ("G5 margin" if "G5" in (d.get("gate") or "") else "G4 second seed" if "G4" in (d.get("gate") or "") else "G3 no gain" if "G3" in (d.get("gate") or "") else "G1/G2")
VS = ["admitted", "G5 margin", "G4 second seed", "G3 no gain", "G1/G2"]; VC = {"admitted": COL["LLM"], "G5 margin": "#b5b5b5", "G4 second seed": "#8f8f8f", "G3 no gain": "#656565", "G1/G2": "#3a3a3a"}
cnt = {k: collections.Counter() for k in kinds}
for d in props: cnt[d.get("kind") or "rule"][verd(d)] += 1
fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.4), gridspec_kw={"height_ratios": [1, 1.1]})
x = np.arange(len(kinds)); bottom = np.zeros(len(kinds))
for v in VS:
    vals = np.array([cnt[k][v] for k in kinds], float); axes[0].bar(x, vals, 0.6, bottom=bottom, color=VC[v], edgecolor="white", label=v); bottom += vals
for xi, k, b in zip(x, kinds, bottom): axes[0].text(xi, b + 5, f"{cnt[k]['admitted']}/{int(b)} ({100*cnt[k]['admitted']/b:.0f} %)", ha="center", fontsize=6)
axes[0].set_xticks(x); axes[0].set_xticklabels(["new rule", "concept", "macro\n(blended)", "actuator\n(composed)"], fontsize=6.5); axes[0].set_ylabel("proposals"); axes[0].set_ylim(0, bottom.max() * 1.22)
axes[0].legend(fontsize=5.6, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1); axes[0].text(-0.02, 1.02, "(a)", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8)
adm = collections.defaultdict(collections.Counter); worlds = collections.defaultdict(set)
for d in props:
    if d.get("accepted"): k = d.get("kind") or "rule"; adm[d["scenario"]][k] += 1; worlds[(d["scenario"], k)].add(d["seed"])
M = np.array([[adm[sc][k] for k in kinds] for sc in SCEN], float)
axes[1].imshow(M, cmap="Oranges", aspect="auto")
for i, sc in enumerate(SCEN):
    for j, k in enumerate(kinds): axes[1].text(j, i, f"{int(M[i,j])}\n({len(worlds[(sc,k)])} w.)", ha="center", va="center", fontsize=6, color=INK if M[i, j] < M.max() * 0.6 else "white")
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(["new rule", "concept", "macro", "actuator"], fontsize=6.5); axes[1].set_yticks(range(5)); axes[1].set_yticklabels(SCEN); axes[1].grid(False)
axes[1].text(-0.02, 1.02, "(b)", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8)
save(fig, "r4_products")

# ---------------- Fig vocabulary (double column): consequents, concept weights, macro matrix
admp = [d for d in props if d.get("accepted")]
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9), gridspec_kw={"width_ratios": [0.85, 1.35, 1.0], "wspace": 0.75})
fams = ["suppression effort", "resource deployment", "containment line", "asset protection", "evacuation", "public warning", "retardant drop", "water drafting", "tactical burn"]
gen = collections.Counter()
for d in admp:
    if (d.get("kind") or "rule") == "rule":
        m = re.search(r"THEN (.*)$", core(d["rule"]))
        for part in (m.group(1) if m else "").split(","):
            nm = re.sub(r"\s[\d.]+$", "", part.strip())
            if nm in fams: gen[nm] += 1
seed = {f: 1 for f in fams[:6]}
y = np.arange(len(fams))
axes[0].barh(y + 0.2, [seed.get(f, 0) for f in fams], 0.4, color=COL["F5Ev"], label="seed base")
axes[0].barh(y - 0.2, [gen[f] for f in fams], 0.4, color=COL["LLM"], label="generated rules")
axes[0].set_yticks(y); axes[0].set_yticklabels(fams, fontsize=6); axes[0].invert_yaxis(); axes[0].set_xlabel("rules ordering the family"); axes[0].grid(axis="x", color=GRID, lw=0.5); axes[0].grid(axis="y", visible=False); axes[0].legend(fontsize=5.6, loc="lower right")
concepts = collections.OrderedDict(); cc = collections.Counter(); base = ["fire_threat_level", "suppression_feasibility", "asset_exposure_risk", "intervention_urgency", "evacuation_pressure"]
for d in admp:
    if d.get("kind") == "concept": k = d["concept"]; cc[k["name"]] += 1; concepts[k["name"]] = dict(k["inputs"])
names = [n for n, _ in cc.most_common()]; W = np.array([[concepts[n].get(b, 0) for b in base] for n in names])
axes[1].imshow(W, cmap="Greens", aspect="auto", vmin=0, vmax=0.5)
axes[1].set_yticks(range(len(names))); axes[1].set_yticklabels([f"{n.replace('_',' ')} (×{cc[n]})" for n in names], fontsize=5.2)
axes[1].set_xticks(range(5)); axes[1].set_xticklabels(["threat", "feasib.", "asset exp.", "urgency", "evac. pr."], fontsize=6, rotation=30); axes[1].grid(False)
for i in range(len(names)):
    for j in range(5):
        if W[i, j] > 0: axes[1].text(j, i, f"{W[i,j]:.2f}", ha="center", va="center", fontsize=4.8, color="white" if W[i, j] > 0.3 else INK)
fam6 = ["asset_protection", "containment_line", "suppression_effort", "resource_deployment", "retardant_drop", "water_drafting", "tactical_burn", "evacuation", "public_warning"]
C = np.zeros((9, 9))
for d in admp:
    if d.get("kind") == "macro":
        fs = [f for f, w_ in d["composition"] if f in fam6]
        for f1 in fs:
            for f2 in fs: C[fam6.index(f1), fam6.index(f2)] += 1
axes[2].imshow(C, cmap="Blues", aspect="auto")
for i in range(9):
    for j in range(9):
        if C[i, j] > 0: axes[2].text(j, i, int(C[i, j]), ha="center", va="center", fontsize=5.4, color="white" if C[i, j] > C.max() * 0.55 else INK)
short = ["asset", "line", "suppr.", "deploy", "retard.", "draft", "burn", "evac.", "warn"]
axes[2].set_xticks(range(9)); axes[2].set_xticklabels(short, fontsize=5.6, rotation=45); axes[2].set_yticks(range(9)); axes[2].set_yticklabels(short, fontsize=5.6); axes[2].grid(False)
for ax, t in zip(axes, ["(a)", "(b)", "(c)"]): ax.text(-0.02, 1.03, t, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
save(fig, "r5_vocabulary")

# ---------------- Fig actuators: effect x sector + archetypes (single column)
adma = [d for d in admp if d.get("kind") == "actuator"]
effects = ["coat", "draft", "wet", "clear", "prime", "ignite", "evacuate"]; sectors = ["head", "flank", "rear", "ring", "at_fire", "assets", "populated"]
M = np.zeros((7, 7))
for d in adma:
    for c in d["clauses"]:
        if c["effect"] in effects and c["sector"] in sectors: M[effects.index(c["effect"]), sectors.index(c["sector"])] += 1
fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.0), gridspec_kw={"height_ratios": [1.1, 1], "hspace": 0.6})
axes[0].imshow(M, cmap="Oranges", aspect="auto")
for i in range(7):
    for j in range(7):
        if M[i, j] > 0: axes[0].text(j, i, int(M[i, j]), ha="center", va="center", fontsize=5.4, color="white" if M[i, j] > M.max() * 0.55 else INK)
axes[0].set_xticks(range(7)); axes[0].set_xticklabels(sectors, fontsize=5.4, rotation=45); axes[0].set_yticks(range(7)); axes[0].set_yticklabels(effects, fontsize=5.6); axes[0].grid(False)
g = collections.Counter()
for d in adma: g[" + ".join(sorted({f"{c['effect']}@{c['sector']}" for c in d['clauses']}))] += 1
top = g.most_common(7); y = np.arange(len(top))
axes[1].barh(y, [v for k, v in top], color=COL["LLM"]); axes[1].set_yticks(y); axes[1].set_yticklabels([k.replace("@", " ") for k, v in top], fontsize=5.2); axes[1].invert_yaxis()
axes[1].set_xlabel("admitted", fontsize=6.5); axes[1].grid(axis="x", color=GRID, lw=0.5); axes[1].grid(axis="y", visible=False)
for ax, t in zip(axes, ["(a)", "(b)"]): ax.text(-0.02, 1.03, t, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
save(fig, "r6_actuators")

# ---------------- Fig attribution + timeline (double column: 3 panels)
gain = {}; admk = {}
for sc in SCEN:
    for s in seeds[sc]: gain[(sc, s)] = runs[(sc, "F5Ev")][s]["burned_ha"] - runs[(sc, "LLM")][s]["burned_ha"]; admk[(sc, s)] = collections.Counter()
for d in props:
    if d.get("accepted"): admk[(d["scenario"], d["seed"])][d.get("kind") or "rule"] += 1
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4), gridspec_kw={"width_ratios": [0.7, 1.1, 1.5], "wspace": 0.45})
z = [g for k, g in gain.items() if sum(admk[k].values()) == 0]; nz = [g for k, g in gain.items() if sum(admk[k].values()) > 0]
bp = axes[0].boxplot([z, nz], widths=0.5, patch_artist=True, showfliers=True, flierprops=dict(markersize=2))
for p, c in zip(bp["boxes"], ["#bbbbbb", COL["LLM"]]): p.set_facecolor(c); p.set_alpha(0.6)
axes[0].set_xticks([1, 2]); axes[0].set_xticklabels([f"no\nadmission\n(n={len(z)})", f"≥1\nadmission\n(n={len(nz)})"], fontsize=6.2); axes[0].set_ylabel("burned area saved vs Layer 1 (ha)"); axes[0].axhline(0, color=INK2, lw=0.6)
xs = np.array([sum(admk[k].values()) for k in gain]); ys = np.array([gain[k] for k in gain]); jit = (np.random.default_rng(1).random(len(xs)) - 0.5) * 0.3
axes[1].scatter(xs + jit, ys, s=9, color=COL["LLM"], alpha=0.6); axes[1].axhline(0, color=INK2, lw=0.6)
for k in sorted(set(xs.tolist())):
    sel = xs == k
    if sel.sum() >= 3: axes[1].plot([k - 0.3, k + 0.3], [np.median(ys[sel])] * 2, color=INK, lw=1.3)
axes[1].set_xlabel("admissions in the world"); axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True)); axes[1].set_ylabel("saved vs Layer 1 (ha)")
# timeline S4 world with max gain
best = max(((sc, s) for sc in ["S4"] for s in seeds[sc]), key=lambda k: gain[k]); sc, s = best
per = collections.defaultdict(dict)
for r in csv.DictReader(open(f"{OUT}/curves.csv", encoding="utf-8"), fieldnames=["exp", "scenario", "arm", "seed", "t_min", "cells", "frac"]):
    try:
        if r["scenario"] == sc and int(r["seed"]) == s and r["arm"] in ARMS: per[r["arm"]][float(r["t_min"])] = float(r["cells"]) * 0.09
    except ValueError: pass
for a in ARMS:
    t = np.array(sorted(per[a])); v = np.array([per[a][k] for k in t])
    if len(t) and t[-1] < 720: t = np.append(t, 720); v = np.append(v, v[-1])
    axes[2].plot(t / 60, v, color=COL[a], lw=1.6, label=f"{NAME[a]}: {runs[(sc,a)][s]['burned_ha']:.0f} ha")
ev = sorted([(d["t_min"], d.get("kind") or "rule") for d in props if d["scenario"] == sc and d["seed"] == s and d.get("accepted")])
ymax = max(max(per[a].values()) for a in ARMS)
for t, k in ev: axes[2].axvline(t / 60, color=COL["LLM"], lw=0.7, ls=":"); axes[2].text(t / 60 + 0.1, ymax * 0.9, f"{k} admitted", fontsize=6, color=COL["LLM"])
axes[2].set_xlabel("time (h)"); axes[2].set_ylabel("burned area (ha)"); axes[2].set_xlim(0, 12); axes[2].legend(fontsize=6, loc="center right"); axes[2].set_title(f"{sc}, world {s}", fontsize=7.5)
for ax, t in zip(axes, ["(a)", "(b)", "(c)"]): ax.text(-0.02, 1.03, t, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
save(fig, "r7_attribution")
json.dump({"timeline_world": [sc, s, ev, {a: runs[(sc, a)][s]["burned_ha"] for a in ARMS}], "zero_adm": [len(z), float(np.mean(z))], "adm": [len(nz), float(np.mean(nz))]}, open(f"{FD}/r7_meta.json", "w"))
print("ok", sc, s, ev, len(z), np.mean(z), len(nz), np.mean(nz))
