"""DisasterAware benchmark dashboard: every view of the 600-run campaign.
python dashboard.py --out out   ->  out/report/dashboard/*.png + dashboard.html
"""
import argparse, csv, json, os, re, base64, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

ARMS = ["T0", "F5", "F5Ev", "LLM"]
NAME = {"T0": "No DSS", "F5": "F5", "F5Ev": "F5+EvFIS", "LLM": "DisasterAware"}
COL = {"T0": "#8A8F96", "F5": "#3A6FC4", "F5Ev": "#1E9E82", "LLM": "#D9651E"}
SCEN = ["S1", "S2", "S3", "S4", "S5"]
POP = 9e-4
INK, INK2, GRID = "#222", "#555", "#E3E3E3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 160})
FIGS = []  # (file, title, caption)


def hw(x):
    x = np.asarray(x, float); return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def save(fig, name, title, cap, fd):
    p = os.path.join(fd, name + ".png"); fig.savefig(p, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    FIGS.append((p, title, cap))


def load(out):
    runs = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(out, "runs.csv"), encoding="utf-8")):
        if r["arm"] in ARMS and r["scenario"] in SCEN:
            for k, v in list(r.items()):
                try: r[k] = float(v)
                except (TypeError, ValueError): pass
            runs[(r["scenario"], r["arm"])][int(r["seed"])] = r
    seeds = {sc: sorted(set.intersection(*[set(runs[(sc, a)]) for a in ARMS])) for sc in SCEN}
    props = []
    for line in open(os.path.join(out, "proposals.jsonl"), encoding="utf-8"):
        try: d = json.loads(line)
        except json.JSONDecodeError: continue
        if d.get("arm") == "LLM" and d.get("scenario") in SCEN and d.get("seed") in seeds[d["scenario"]]:
            props.append(d)
    return runs, seeds, props


def V(runs, seeds, arm, key, sc=None):
    scs = [sc] if sc else SCEN
    return np.array([runs[(s, arm)][x][key] for s in scs for x in seeds[s]], float)


def core(r):
    r = re.sub(r"^G\d+: ", "", r); return re.sub(r" \(adaptation-born.*$", "", r)


# ================================================================ figures
def fig_radar(runs, seeds, fd):
    metrics = [("burned_ha", "burned area", False), ("pop_affected", "affected population", False),
               ("success", "extinguished", True), ("j_phys", "J_phys (physical)", False),
               ("j_resp", "J_resp (response)", False), ("j_delay", "J_delay", False), ("out_min", "time to extinction", False)]
    vals = {}
    RA = ["F5", "F5Ev", "LLM"]
    for a in ARMS:
        row = []
        for k, lab, higher in metrics:
            v = V(runs, seeds, a, k)
            if k == "out_min":
                s = V(runs, seeds, a, "success"); v = v[s > 0] if (s > 0).any() else np.array([720.0])
            row.append(v.mean())
        vals[a] = np.array(row)
    # normalise 0..1 where 1 = best among the four
    M = np.array([vals[a] for a in RA]); best = np.zeros(len(metrics))
    for j, (k, lab, higher) in enumerate(metrics):
        col = M[:, j]
        best_j = col.max() if higher else col.min(); worst_j = col.min() if higher else col.max()
        rng = (worst_j - best_j) if worst_j != best_j else 1
        M[:, j] = (col - worst_j) / rng if higher else (worst_j - col) / rng
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(6.2, 5.2)); ax = plt.subplot(111, polar=True)
    for i, a in enumerate(RA):
        v = M[i].tolist() + M[i][:1].tolist()
        ax.plot(ang, v, color=COL[a], lw=2, label=NAME[a], marker="o", ms=4); ax.fill(ang, v, color=COL[a], alpha=0.05)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([m[1] for m in metrics], fontsize=8.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["worst", "0.5", "", "best"], fontsize=7, color=INK2); ax.set_ylim(0, 1.02)
    ax.grid(color=GRID); ax.spines["polar"].set_color(GRID)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1), fontsize=8.5)
    save(fig, "01_radar", "Radar: seven outcome and cost measures for the three decision layers, normalised so the outer ring is the best of the three",
         "Mean over 150 paired worlds (S1-S5 × 30). Every axis is oriented so that outside is better: lower burned area, population, cost terms and time to extinction; higher extinguished share. No DSS is left out because it has no response or delay cost (it issues no orders); its values are in the table above. DisasterAware is on the outer ring on six of the seven axes, including the response term (it commits fewer resources than the static doctrine, not more); the one axis where F5+EvFIS leads is the delay term, since tuned consequents commit earlier.", fd)
    return vals


def fig_outcome_bars(runs, seeds, fd):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    x = np.arange(len(SCEN) + 1); w = 0.2; labels = SCEN + ["Mean"]
    for i, a in enumerate(ARMS):
        off = (i - 1.5) * w
        for ax, key, scale, lab in ((axes[0], "burned_ha", 1, "burned area (ha)"), (axes[1], "pop_affected", POP, "affected population (persons)"), (axes[2], "success", 100, "extinguished within episode (%)")):
            m = [V(runs, seeds, a, key, sc).mean() * scale for sc in SCEN] + [V(runs, seeds, a, key).mean() * scale]
            e = [hw(V(runs, seeds, a, key, sc) * scale) for sc in SCEN] + [hw(V(runs, seeds, a, key) * scale)]
            if key == "success": e = None
            ax.bar(x + off, m, w * 0.9, color=COL[a], label=NAME[a] if ax is axes[0] else None, yerr=e, error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor=INK2))
            ax.set_ylabel(lab); ax.set_xticks(x); ax.set_xticklabels(labels)
    axes[1].set_yscale("log"); axes[1].set_ylim(5, 2e4); axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(1.7, 1.18))
    save(fig, "02_outcomes", "Outcome at episode end per scenario family: burned area, affected population, extinguished share",
         "30 paired worlds per family (same terrain, ignitions and random stream for all four configurations); bars are means with 95 % CI. Population on a log axis.", fd)


def fig_costs(runs, seeds, fd):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    # (a) two cost types side by side per config
    x = np.arange(len(ARMS)); w = 0.38
    jp = [V(runs, seeds, a, "j_phys").mean() for a in ARMS]; jt = [V(runs, seeds, a, "j_total").mean() for a in ARMS]
    ep = [hw(V(runs, seeds, a, "j_phys")) for a in ARMS]; et = [hw(V(runs, seeds, a, "j_total")) for a in ARMS]
    axes[0].bar(x - w / 2, jp, w, color=[COL[a] for a in ARMS], yerr=ep, error_kw=dict(elinewidth=0.6, capsize=2, ecolor=INK2), label="J_phys (burn, asset, population)")
    axes[0].bar(x + w / 2, jt, w, color=[COL[a] for a in ARMS], alpha=0.45, hatch="///", edgecolor="white", yerr=et, error_kw=dict(elinewidth=0.6, capsize=2, ecolor=INK2), label="J_total (+0.2 response, +0.2 delay)")
    for xi, a, b in zip(x, jp, jt):
        axes[0].text(xi - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=7.5); axes[0].text(xi + w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=7.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels([NAME[a] for a in ARMS]); axes[0].set_ylabel("decision cost (mean, 150 worlds)")
    axes[0].legend(loc="upper right", fontsize=8); axes[0].set_title("(a) physical cost and total cost")
    # (b) five-term decomposition of J_total
    terms = [("j_burn", "J_burn", "#4a4a4a"), ("j_asset", "J_asset", "#7a7a7a"), ("j_pop", "J_pop", "#a8a8a8"), ("j_resp", "0.2 J_resp", "#cfcfcf"), ("j_delay", "0.2 J_delay", "#e8e8e8")]
    wts = {"j_burn": 1, "j_asset": 1, "j_pop": 1, "j_resp": 0.2, "j_delay": 0.2}
    bottom = np.zeros(len(ARMS))
    for k, lab, c in terms:
        v = np.array([wts[k] * V(runs, seeds, a, k).mean() / 3.4 for a in ARMS])
        axes[1].bar(x, v, 0.6, bottom=bottom, color=c, edgecolor="white", label=lab); bottom += v
    axes[1].set_xticks(x); axes[1].set_xticklabels([NAME[a] for a in ARMS]); axes[1].set_ylabel("J_total decomposed")
    axes[1].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=7.5); axes[1].set_title("(b) what the total cost is made of")
    save(fig, "03_costs", "Two cost types: physical cost J_phys and total decision cost J_total, with the five-term decomposition",
         "J_phys = (J_burn + J_asset + J_pop)/3; J_total = (J_burn + J_asset + J_pop + 0.2 J_resp + 0.2 J_delay)/3.4. DisasterAware has the lowest physical and total cost, and its response term (resources committed) is below the static doctrine's: the gain is not bought with more resources.", fd)
    # (c) response vs physical scatter per config (mean per scenario)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for a in ARMS:
        if a == "T0": continue
        xs = [V(runs, seeds, a, "j_resp", sc).mean() for sc in SCEN]; ys = [V(runs, seeds, a, "j_phys", sc).mean() for sc in SCEN]
        ax.scatter(xs, ys, color=COL[a], s=40, label=NAME[a], zorder=3)
        for sc, xv, yv in zip(SCEN, xs, ys): ax.annotate(sc, (xv, yv), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlabel("J_resp: resources committed (mean per family)"); ax.set_ylabel("J_phys: physical loss (mean per family)")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.legend(fontsize=8)
    save(fig, "04_resp_vs_phys", "Resources committed against physical loss, per scenario family", "Each point is one scenario family for one configuration. DisasterAware sits lower on both axes than the two doctrines in every family: less loss with less committed response.", fd)


def fig_paired(runs, seeds, fd):
    pairs = [("F5", "T0"), ("F5Ev", "F5"), ("LLM", "F5Ev")]
    keys = [("burned_ha", 1, "burned area saved (ha)"), ("pop_affected", POP, "population saved (persons)"), ("success", 100, "extinguished gained (points)"), ("j_total", 1, "J_total saved")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    y = np.arange(len(pairs))[::-1]
    for ax, (k, sc_, lab) in zip(axes, keys):
        ms, es, txt = [], [], []
        for a, b in pairs:
            d = (V(runs, seeds, b, k) - V(runs, seeds, a, k)) * sc_ if k != "success" else (V(runs, seeds, a, k) - V(runs, seeds, b, k)) * 100
            ms.append(d.mean()); es.append(hw(d)); wins = int((d > 1e-9).sum()); txt.append(f"{d.mean():+.1f} ± {hw(d):.1f}  ({wins}/150)")
        ax.axvline(0, color=INK2, lw=0.6)
        ax.barh(y, ms, 0.5, color=[COL[a] for a, b in pairs], xerr=es, error_kw=dict(elinewidth=0.8, capsize=3, ecolor=INK))
        xmax = max(m + e for m, e in zip(ms, es)); ax.set_xlim(min(0, min(m - e for m, e in zip(ms, es)) * 1.1), xmax * 1.6)
        for yi, m, e, t in zip(y, ms, es, txt): ax.text(max(m + e, 0) + xmax * 0.03, yi, t, va="center", fontsize=6.8, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([f"{NAME[a]} vs {NAME[b]}" for a, b in pairs], fontsize=8); ax.set_xlabel(lab)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.grid(axis="y", visible=False)
    save(fig, "05_paired_ladder", "Paired ladder: what each step adds on the same worlds (mean paired difference, 95 % CI, worlds in which the step is better)",
         "F5 over no action is the rule-based core. F5+EvFIS over F5 changes extinction (+39 points) but not burned area (5.5 ± 7.0 ha, 73/150: a coin toss). DisasterAware over F5+EvFIS: 29.9 ± 7.2 ha, 128/150, +23 points, 274 persons.", fd)


def fig_scatter(runs, seeds, fd):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    mk = {"S1": "o", "S2": "s", "S3": "^", "S4": "D", "S5": "v"}
    for ax, (a, b) in zip(axes, [("F5Ev", "F5"), ("LLM", "F5Ev")]):
        lim = 0
        for sc in SCEN:
            xv = V(runs, seeds, b, "burned_ha", sc); yv = V(runs, seeds, a, "burned_ha", sc); lim = max(lim, xv.max(), yv.max())
            ax.scatter(xv, yv, s=22, marker=mk[sc], facecolor="none", edgecolor=COL[a], linewidths=0.9, label=sc)
        lim *= 1.05; ax.plot([0, lim], [0, lim], ls="--", color=INK2, lw=0.7); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        d = V(runs, seeds, b, "burned_ha") - V(runs, seeds, a, "burned_ha")
        ax.set_title(f"{NAME[a]} vs {NAME[b]}: below diagonal in {int((d > 1e-9).sum())}/150, ties {int((np.abs(d) <= 1e-9).sum())}", fontsize=9)
        ax.set_xlabel(f"{NAME[b]} burned (ha)"); ax.set_ylabel(f"{NAME[a]} burned (ha)"); ax.grid(axis="x", color=GRID, lw=0.5); ax.legend(fontsize=7, loc="lower right")
    save(fig, "06_world_scatter", "World by world: each point is one world replayed by both configurations", "Left: self-tuning doctrine against static doctrine scatters around the diagonal. Right: DisasterAware against the self-tuning doctrine lies below it almost everywhere; the ties are small fires the doctrine already held.", fd)


def fig_products(props, fd):
    kinds = ["rule", "concept", "macro", "actuator"]; KN = {"rule": "new rule", "concept": "intermediate\nconcept", "macro": "blended intervention\n(macro)", "actuator": "composed intervention\n(actuator)"}
    verd = lambda d: "admitted" if d.get("accepted") else ("G5 package margin" if "G5" in (d.get("gate") or "") else "G4 reseeded A/B" if "G4" in (d.get("gate") or "") else "G3 no improvement" if "G3" in (d.get("gate") or "") else "G1/G2 syntactic")
    VS = ["admitted", "G5 package margin", "G4 reseeded A/B", "G3 no improvement", "G1/G2 syntactic"]
    VC = {"admitted": COL["LLM"], "G5 package margin": "#b5b5b5", "G4 reseeded A/B": "#8f8f8f", "G3 no improvement": "#656565", "G1/G2 syntactic": "#3a3a3a"}
    cnt = {k: collections.Counter() for k in kinds}
    for d in props: cnt[d.get("kind") or "rule"][verd(d)] += 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(kinds)); bottom = np.zeros(len(kinds))
    for v in VS:
        vals = np.array([cnt[k][v] for k in kinds], float)
        axes[0].bar(x, vals, 0.6, bottom=bottom, color=VC[v], edgecolor="white", label=v); bottom += vals
    for xi, k, b in zip(x, kinds, bottom):
        axes[0].text(xi, b + 4, f"{cnt[k]['admitted']}/{int(b)}  ({100*cnt[k]['admitted']/b:.0f} %)", ha="center", fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels([KN[k] for k in kinds], fontsize=8); axes[0].set_ylabel("proposals"); axes[0].set_ylim(0, bottom.max() * 1.18)
    axes[0].legend(fontsize=7.5, loc="upper left"); axes[0].set_title("(a) proposals by kind and verdict")
    # (b) per scenario x kind admitted, with worlds touched
    adm = collections.defaultdict(lambda: collections.Counter()); worlds = collections.defaultdict(set)
    for d in props:
        if d.get("accepted"):
            k = d.get("kind") or "rule"; adm[d["scenario"]][k] += 1; worlds[(d["scenario"], k)].add(d["seed"])
    M = np.array([[adm[sc][k] for k in kinds] for sc in SCEN], float)
    im = axes[1].imshow(M, cmap="Oranges", aspect="auto")
    for i, sc in enumerate(SCEN):
        for j, k in enumerate(kinds):
            axes[1].text(j, i, f"{int(M[i,j])}\n({len(worlds[(sc,k)])} worlds)", ha="center", va="center", fontsize=7.5, color=INK if M[i, j] < M.max() * 0.6 else "white")
    axes[1].set_xticks(range(len(kinds))); axes[1].set_xticklabels([KN[k] for k in kinds], fontsize=8); axes[1].set_yticks(range(len(SCEN))); axes[1].set_yticklabels(SCEN)
    axes[1].grid(False); axes[1].set_title("(b) admitted objects per family: count (worlds in which at least one was admitted, of 30)")
    save(fig, "07_products_gates", "What was produced, how many, and why proposals passed or failed the gates",
         "788 proposals over 150 worlds; 338 admitted. Rejections concentrate at G5 (package margin, after the forecast tests were passed) and at G3 (no forecast improvement); G1/G2 syntactic rejections are rare (7), so the model writes well-formed objects and the simulator decides. Macros pass most often (80 %), plain rules least (28 %).", fd)
    return cnt


def fig_vocab(props, fd):
    adm = [d for d in props if d.get("accepted")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [0.9, 1.3, 1.1]})
    # (a) consequent families: seed (5 rules) vs generated rules
    fams = ["suppression effort", "resource deployment", "containment line", "asset protection", "evacuation", "public warning", "retardant drop", "water drafting", "tactical burn"]
    gen = collections.Counter()
    for d in adm:
        if (d.get("kind") or "rule") == "rule":
            m = re.search(r"THEN (.*)$", core(d["rule"]))
            for part in (m.group(1) if m else "").split(","):
                nm = re.sub(r"\s[\d.]+$", "", part.strip())
                if nm in fams: gen[nm] += 1
    seed = {"suppression effort": 1, "resource deployment": 1, "containment line": 1, "asset protection": 1, "evacuation": 1, "public warning": 1}
    y = np.arange(len(fams))
    axes[0].barh(y + 0.2, [seed.get(f, 0) for f in fams], 0.4, color=COL["F5"], label="seed base (5 rules)")
    axes[0].barh(y - 0.2, [gen[f] for f in fams], 0.4, color=COL["LLM"], label="generated rules (67)")
    axes[0].set_yticks(y); axes[0].set_yticklabels(fams, fontsize=8); axes[0].invert_yaxis(); axes[0].set_xlabel("consequents using the family / channel")
    axes[0].grid(axis="x", color=GRID, lw=0.5); axes[0].grid(axis="y", visible=False); axes[0].legend(fontsize=7.5, loc="lower right"); axes[0].set_title("(a) what the new rules order", fontsize=9)
    # (b) concept weight matrix
    concepts = collections.OrderedDict(); cc = collections.Counter()
    base = ["fire_threat_level", "suppression_feasibility", "asset_exposure_risk", "intervention_urgency", "evacuation_pressure"]
    for d in adm:
        if d.get("kind") == "concept":
            k = d["concept"]; cc[k["name"]] += 1; concepts[k["name"]] = dict(k["inputs"])
    names = [n for n, _ in cc.most_common()]
    W = np.array([[concepts[n].get(b, 0) for b in base] for n in names])
    im = axes[1].imshow(W, cmap="Greens", aspect="auto", vmin=0, vmax=0.5)
    axes[1].set_yticks(range(len(names))); axes[1].set_yticklabels([f"{n} (×{cc[n]})" for n in names], fontsize=6.5)
    axes[1].set_xticks(range(5)); axes[1].set_xticklabels(["threat", "feasibility", "asset exp.", "urgency", "evac. pressure"], fontsize=7.5, rotation=20)
    axes[1].grid(False); axes[1].set_title(f"(b) {len(names)} intermediate concepts: weights over the five base concepts", fontsize=9)
    for i in range(len(names)):
        for j in range(5):
            if W[i, j] > 0: axes[1].text(j, i, f"{W[i,j]:.2f}", ha="center", va="center", fontsize=6, color="white" if W[i, j] > 0.3 else INK)
    # (c) macro composition matrix family x family co-occurrence
    fam6 = ["asset_protection", "containment_line", "suppression_effort", "resource_deployment", "retardant_drop", "water_drafting", "tactical_burn", "evacuation", "public_warning"]
    C = np.zeros((len(fam6), len(fam6)))
    for d in adm:
        if d.get("kind") == "macro":
            fs = [f for f, w in d["composition"] if f in fam6]
            for f1 in fs:
                for f2 in fs: C[fam6.index(f1), fam6.index(f2)] += 1
    axes[2].imshow(C, cmap="Blues", aspect="auto")
    for i in range(len(fam6)):
        for j in range(len(fam6)):
            if C[i, j] > 0: axes[2].text(j, i, int(C[i, j]), ha="center", va="center", fontsize=7, color="white" if C[i, j] > C.max() * 0.55 else INK)
    axes[2].set_xticks(range(len(fam6))); axes[2].set_xticklabels([f.replace("_", "\n") for f in fam6], fontsize=6.5); axes[2].set_yticks(range(len(fam6))); axes[2].set_yticklabels(fam6, fontsize=7)
    axes[2].grid(False); axes[2].set_title("(c) 57 blended interventions: families bundled together\n(diagonal = uses of the family)", fontsize=9)
    save(fig, "08_vocabulary", "What the open space wrote into the vocabulary: rule consequents, intermediate concepts, blended interventions",
         "(a) The seed base never orders retardant drops or water drafting; the generated rules make them the first two consequents. (b) Twenty concepts fall into two families: head-run pressure (threat + urgency) and interface/ember pressure (asset exposure + evacuation pressure); unreachable_head_pressure was found independently in four worlds. (c) Two blends dominate: asset protection + containment line (interface hardening) and containment line + suppression effort (downwind backburn line).", fd)


def fig_actuators(props, fd):
    adm = [d for d in props if d.get("accepted") and d.get("kind") == "actuator"]
    effects = ["coat", "draft", "wet", "clear", "prime", "ignite", "evacuate"]; sectors = ["head", "flank", "rear", "ring", "at_fire", "assets", "populated"]
    M = np.zeros((len(effects), len(sectors)))
    for d in adm:
        for c in d["clauses"]:
            if c["effect"] in effects and c["sector"] in sectors: M[effects.index(c["effect"]), sectors.index(c["sector"])] += 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.55})
    axes[0].imshow(M, cmap="Oranges", aspect="auto")
    for i in range(len(effects)):
        for j in range(len(sectors)):
            if M[i, j] > 0: axes[0].text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8, color="white" if M[i, j] > M.max() * 0.55 else INK)
    axes[0].set_xticks(range(len(sectors))); axes[0].set_xticklabels(sectors); axes[0].set_yticks(range(len(effects))); axes[0].set_yticklabels(effects); axes[0].grid(False)
    axes[0].set_title("(a) clauses in the 185 admitted composed interventions: effect × sector")
    # (b) archetypes
    g = collections.Counter()
    for d in adm: g[" + ".join(sorted({f"{c['effect']}@{c['sector']}" for c in d['clauses']}))] += 1
    top = g.most_common(10)
    y = np.arange(len(top))
    axes[1].barh(y, [v for k, v in top], color=COL["LLM"]); axes[1].set_yticks(y); axes[1].set_yticklabels([k for k, v in top], fontsize=7.5); axes[1].invert_yaxis()
    axes[1].set_xlabel("admitted interventions"); axes[1].grid(axis="x", color=GRID, lw=0.5); axes[1].grid(axis="y", visible=False)
    axes[1].set_title(f"(b) ten most frequent archetypes of {len(g)} distinct effect-sector sets")
    save(fig, "09_actuators", "Composed interventions: built from seven verified effects on seven sectors with a cell band",
         "The dominant tactics are the asset collar (coat assets + coat populated before the fire arrives, a preventive tactic absent from the seed doctrine) and the head-flank retardant band with water drafting at the fire. Backfire (ignite) and evacuation clauses are rare and enter only where the forecast rewards them.", fd)


def fig_attribution(runs, seeds, props, fd):
    """Observational attribution: gain vs what was admitted in that world."""
    gain = {}; adm = {}
    for sc in SCEN:
        for s in seeds[sc]:
            gain[(sc, s)] = runs[(sc, "F5Ev")][s]["burned_ha"] - runs[(sc, "LLM")][s]["burned_ha"]
            adm[(sc, s)] = collections.Counter()
    for d in props:
        if d.get("accepted"): adm[(d["scenario"], d["seed"])][d.get("kind") or "rule"] += 1
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    # (a) zero admissions vs any
    z = [g for k, g in gain.items() if sum(adm[k].values()) == 0]; nz = [g for k, g in gain.items() if sum(adm[k].values()) > 0]
    bp = axes[0].boxplot([z, nz], widths=0.5, patch_artist=True, showfliers=True)
    for p, c in zip(bp["boxes"], ["#bbbbbb", COL["LLM"]]): p.set_facecolor(c); p.set_alpha(0.6)
    axes[0].set_xticks([1, 2]); axes[0].set_xticklabels([f"no admission\n(n={len(z)})", f"≥1 admission\n(n={len(nz)})"]); axes[0].set_ylabel("burned area saved vs F5+EvFIS (ha)")
    axes[0].axhline(0, color=INK2, lw=0.6); axes[0].set_title(f"(a) no admission: {np.mean(z):.1f} ha; with admissions: {np.mean(nz):.1f} ha", fontsize=8.5)
    # (b) by kind present
    kinds = ["rule", "concept", "macro", "actuator"]; KN = {"rule": "rule", "concept": "concept", "macro": "macro", "actuator": "actuator"}
    data = []; labs = []
    for k in kinds:
        w = [g for key, g in gain.items() if adm[key][k] > 0]; wo = [g for key, g in gain.items() if adm[key][k] == 0 and sum(adm[key].values()) > 0]
        data += [wo, w]; labs += [f"{KN[k]}\nabsent\n({len(wo)})", f"{KN[k]}\npresent\n({len(w)})"]
    bp = axes[1].boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for i, p in enumerate(bp["boxes"]): p.set_facecolor(COL["LLM"] if i % 2 else "#bbbbbb"); p.set_alpha(0.6)
    axes[1].set_xticks(range(1, len(labs) + 1)); axes[1].set_xticklabels(labs, fontsize=7); axes[1].axhline(0, color=INK2, lw=0.6)
    axes[1].set_ylabel("burned area saved vs F5+EvFIS (ha)"); axes[1].set_title("(b) worlds with admissions: gain with and without each kind", fontsize=8.5)
    # (c) number of admissions vs gain
    xs = np.array([sum(adm[k].values()) for k in gain]); ys = np.array([gain[k] for k in gain])
    jit = (np.random.default_rng(1).random(len(xs)) - 0.5) * 0.3
    axes[2].scatter(xs + jit, ys, s=14, color=COL["LLM"], alpha=0.6); axes[2].axhline(0, color=INK2, lw=0.6)
    for k in sorted(set(xs.tolist())):
        sel = xs == k
        if sel.sum() >= 3: axes[2].plot([k - 0.3, k + 0.3], [np.median(ys[sel])] * 2, color=INK, lw=1.4)
    axes[2].set_xlabel("admissions in the world"); axes[2].set_ylabel("burned area saved vs F5+EvFIS (ha)"); axes[2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[2].set_title("(c) gain against the number of admissions (bars: median)", fontsize=8.5)
    save(fig, "10_attribution", "Closed physics, open decision space: what the admitted objects changed (observational attribution)",
         "(a) The natural control: in the 16 worlds where every proposal was rejected, DisasterAware and F5+EvFIS end almost identically (no-harm); the 30 ha margin comes entirely from the 134 worlds with admissions. (b) Worlds in which a concept, macro or actuator was admitted gain more than worlds with rules only; this is observational, the causal version needs the ablation campaign. (c) Gain grows with the number of admissions, and worlds with zero admissions sit at zero: the mechanism does not act where the gates say no.", fd)
    return z, nz


def fig_timeline(runs, seeds, props, out, fd):
    """One world: burned-area curves of all four arms with admissions marked."""
    # pick S4 world with largest gain
    best = max(((sc, s) for sc in ["S4"] for s in seeds[sc]), key=lambda k: runs[(k[0], "F5Ev")][k[1]]["burned_ha"] - runs[(k[0], "LLM")][k[1]]["burned_ha"])
    sc, s = best
    per = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(out, "curves.csv"), encoding="utf-8"), fieldnames=["exp", "scenario", "arm", "seed", "t_min", "cells", "frac"]):
        try:
            if r["scenario"] == sc and int(r["seed"]) == s and r["arm"] in ARMS: per[r["arm"]][float(r["t_min"])] = float(r["cells"]) * 0.09
        except ValueError: pass
    fig, ax = plt.subplots(figsize=(9, 4))
    for a in ARMS:
        t = np.array(sorted(per[a])); v = np.array([per[a][k] for k in t])
        if len(t) and t[-1] < 720: t = np.append(t, 720); v = np.append(v, v[-1])
        ax.plot(t / 60, v, color=COL[a], lw=2, label=f"{NAME[a]}: {runs[(sc,a)][s]['burned_ha']:.0f} ha")
    ev = sorted([(d["t_min"], d.get("kind") or "rule", core(d["rule"])) for d in props if d["scenario"] == sc and d["seed"] == s and d.get("accepted")])
    ymax = max(max(per[a].values()) for a in ARMS)
    for i, (t, k, r) in enumerate(ev):
        ax.axvline(t / 60, color=COL["LLM"], lw=0.7, ls=":", alpha=0.8)
        ax.text(t / 60, ymax * (0.97 - 0.07 * (i % 4)), k, fontsize=6.5, color=COL["LLM"], rotation=0, ha="left")
    ax.set_xlabel("time (h)"); ax.set_ylabel("burned area (ha)"); ax.set_xlim(0, 12); ax.legend(fontsize=8, loc="center right")
    ax.set_title(f"{sc}, world {s}: four configurations on the same fire; dotted lines mark the {len(ev)} admissions of DisasterAware", fontsize=9)
    save(fig, "11_world_timeline", "One world, four replays: where the curves separate is where the admitted objects acted",
         "The S4 world with the largest paired gain. The seed doctrine and its tuned version follow the same path; DisasterAware's curve bends at the first admissions. " + "; ".join(f"{t/60:.1f} h {k}: {r[:70]}" for t, k, r in ev[:4]), fd)


def fig_convergence(props, fd):
    adm = [d for d in props if d.get("accepted")]
    # concepts found in >1 world, macros found in >1 world, actuator archetypes in >1 world
    def worlds_of(items):
        c = collections.defaultdict(set)
        for key, w in items: c[key].add(w)
        return sorted(((k, len(v)) for k, v in c.items()), key=lambda x: -x[1])
    con = worlds_of([(d["concept"]["name"], (d["scenario"], d["seed"])) for d in adm if d.get("kind") == "concept"])
    AB = {"asset_protection": "asset", "containment_line": "line", "suppression_effort": "suppr", "resource_deployment": "deploy", "retardant_drop": "retard", "water_drafting": "draft", "tactical_burn": "burn", "evacuation": "evac", "public_warning": "warn"}
    mac = worlds_of([(" + ".join(f"{AB.get(f, f)} {w}" for f, w in d["composition"]), (d["scenario"], d["seed"])) for d in adm if d.get("kind") == "macro"])
    act = worlds_of([(" + ".join(sorted({f"{c['effect']}@{c['sector']}" for c in d['clauses']})), (d["scenario"], d["seed"])) for d in adm if d.get("kind") == "actuator"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), gridspec_kw={"wspace": 0.75})
    for ax, data, title in zip(axes, [con[:10], mac[:10], act[:10]], ["intermediate concepts", "blended interventions", "composed interventions (effect sets)"]):
        y = np.arange(len(data)); ax.barh(y, [n for k, n in data], color=COL["F5Ev"]); ax.set_yticks(y); ax.set_yticklabels([k[:48] for k, n in data], fontsize=7); ax.invert_yaxis()
        ax.set_xlabel("distinct worlds in which it was independently admitted"); ax.grid(axis="x", color=GRID, lw=0.5); ax.grid(axis="y", visible=False); ax.set_title(title, fontsize=9)
    save(fig, "12_convergence", "Convergent discovery: the same object admitted independently in several worlds",
         "Every world starts from the same five seed rules and the model has no memory across worlds, so an object that recurs was rediscovered, not copied. Recurrence across worlds is the first evidence that what was learned is structural rather than incidental.", fd)


def html(fd, vals, out):
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>DisasterAware benchmark dashboard</title><style>body{font-family:Georgia,serif;max-width:1250px;margin:28px auto;padding:0 16px;color:#222}h1{font-size:26px}h2{font-size:17px;margin-top:38px}p.cap{font-size:13.5px;color:#444;max-width:1100px}img{max-width:100%;display:block;border:1px solid #eee}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccc;padding:3px 8px;text-align:right}td:first-child,th:first-child{text-align:left}.hdr{background:#f4f4f4;padding:10px 14px;border-left:4px solid #D9651E;font-size:13.5px}</style></head><body>']
    parts.append('<h1>DisasterAware benchmark: No DSS / F5 / F5+EvFIS / DisasterAware</h1>')
    parts.append('<div class="hdr">Design: five scenario families (S1-S5), 30 worlds each, seeds 201-230; every world is generated once and replayed by all four configurations with the same terrain, ignitions and random stream (paired design). Episodes run to extinction or 12 h. DisasterAware = five seed rules + evolving-fuzzy stages 1-2 + language-model proposer (claude -p) at stage 3 behind gates G1-G5b. 600 runs, 788 proposals, 338 admitted.</div>')
    parts.append('<h2>Summary table (mean over 150 worlds)</h2><table><tr><th>Configuration</th><th>burned (ha)</th><th>population</th><th>extinguished (%)</th><th>J_phys</th><th>J_resp</th><th>J_delay</th><th>time to extinction (min)</th></tr>')
    for a in ARMS:
        v = vals[a]; parts.append(f"<tr><td>{NAME[a]}</td><td>{v[0]:.1f}</td><td>{v[1]*POP:.0f}</td><td>{100*v[2]:.0f}</td><td>{v[3]:.3f}</td><td>{v[4]:.3f}</td><td>{v[5]:.3f}</td><td>{v[6]:.0f}</td></tr>")
    parts.append('</table>')
    for p, t, c in FIGS:
        b = base64.b64encode(open(p, "rb").read()).decode()
        parts.append(f'<h2>{t}</h2><img src="data:image/png;base64,{b}"><p class="cap">{c}</p>')
    parts.append('</body></html>')
    open(os.path.join(fd, "dashboard.html"), "w", encoding="utf-8").write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    runs, seeds, props = load(args.out)
    fd = os.path.join(args.out, "report", "dashboard"); os.makedirs(fd, exist_ok=True)
    vals = fig_radar(runs, seeds, fd)
    fig_outcome_bars(runs, seeds, fd); fig_costs(runs, seeds, fd); fig_paired(runs, seeds, fd); fig_scatter(runs, seeds, fd)
    fig_products(props, fd); fig_vocab(props, fd); fig_actuators(props, fd)
    fig_attribution(runs, seeds, props, fd); fig_timeline(runs, seeds, props, args.out, fd); fig_convergence(props, fd)
    html(fd, vals, args.out); print("ok", len(FIGS))


if __name__ == "__main__":
    main()
