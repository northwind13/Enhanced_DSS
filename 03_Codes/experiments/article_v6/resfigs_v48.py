# -*- coding: utf-8 -*-
"""Section V figures of v45 (Fig. 6 to Fig. 11) from the benchmark output; one spec -> PNG and draw.io page.
Palette: No DSS grey, static seed light blue, closed layer green, VODA blue (Word theme)."""
import csv, json, re, collections, math, sys
import numpy as np
from diag import Diagram, drawio_file
from chartkit import Ax, INK, INK2, GRID, GREYS, GREENS, BLUES, nice_ticks

OUT = "/mnt/user-data/uploads/Enhanced_DSS/03_Codes/experiments/benchmark/out"
OUTN = "/mnt/user-data/uploads/Enhanced_DSS/03_Codes/experiments/benchmark/out_nolib"
FD = "/home/claude/v48/figs/out"
ALL = ["T0", "F5", "F5Ev", "LLM"]; ARMS = ["T0", "F5Ev", "LLM"]; SCEN = ["S1", "S2", "S3", "S4", "S5"]; POP = 9e-4
NAME = {"T0": "No DSS", "F5": "static seed base", "F5Ev": "closed layer", "LLM": "VODA"}
COL = {"T0": "#A5A5A5", "F5": "#9DC3E6", "F5Ev": "#70AD47", "LLM": "#4472C4"}; RED = "#C00000"; TIE = "#BFBFBF"
W1 = 362; F = 7.0


def hw(x):
    x = np.asarray(x, float); return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


runs = collections.defaultdict(dict)
def _rows():
    for r in csv.DictReader(open(f"{OUT}/runs.csv", encoding="utf-8")):
        if r.get("exp") == "E2" and r["arm"] in ("T0", "F5", "F5Ev"): yield r
    for r in csv.DictReader(open(f"{OUTN}/runs.csv", encoding="utf-8")):
        if r.get("exp") == "E5" and r["arm"] == "LLM": yield r
for r in _rows():
    if r["arm"] in ALL and r["scenario"] in SCEN:
        for k, v in list(r.items()):
            try: r[k] = float(v)
            except (TypeError, ValueError): pass
        runs[(r["scenario"], r["arm"])][int(r["seed"])] = r
seeds = {sc: sorted(set.intersection(*[set(runs[(sc, a)]) for a in ALL])) for sc in SCEN}
props = []
for line in open(f"{OUTN}/proposals.jsonl", encoding="utf-8"):
    try: d = json.loads(line)
    except json.JSONDecodeError: continue
    if d.get("arm") == "LLM" and d.get("scenario") in SCEN and d.get("seed") in seeds[d["scenario"]]: props.append(d)
nconcept = sum(1 for d in props if d.get("kind") == "concept")
props = [d for d in props if d.get("kind") != "concept"]
admp = [d for d in props if d.get("accepted")]


def V(arm, key, sc=None):
    scs = [sc] if sc else SCEN
    return np.array([runs[(s, arm)][x][key] for s in scs for x in seeds[s]], float)


def core(r):
    r = re.sub(r"^G\d+: ", "", r); return re.sub(r" \(adaptation-born.*$", "", r)


PAGES = []; STATS = {}


def done(d, name, title):
    d.render(f"{FD}/{name}.png"); PAGES.append((title, d)); print("rendered", name)


# ===================================================================== Fig 6  outcome per family
d = Diagram(W1, 400, font=F); L, R = 46, 8; pw = W1 - L - R
cats = SCEN + ["Mean"]; xl = (-0.6, len(cats) - 0.4); bw = 0.36
panels = [("burned_ha", 1, "burned area (ha)", (0, 235), lambda a, b: f"−{100*(a-b)/a:.0f}%"),
          ("pop_affected", POP, "affected population", (0, 1150), lambda a, b: f"−{100*(a-b)/a:.0f}%"),
          ("success", 100, "extinguished (%)", (0, 118), lambda a, b: f"+{b-a:.0f} pts")]
y0 = 18
for pi, (key, scale, ylab, yl, lab) in enumerate(panels):
    ph = 105
    ax = Ax(d, L, y0, pw, ph, xl, yl); ax.frame(); ax.yticks(grid=True); ax.ylabel(ylab, off=28)
    for j, sc in enumerate(SCEN + [None]):
        vals = {}
        for i, a in enumerate(["F5Ev", "LLM"]):
            v = V(a, key, sc) * scale; m, e = v.mean(), hw(v); vals[a] = (m, e)
            ax.bar(j + (i - 0.5) * bw, m, bw * 0.9, COL[a])
            if key != "success": ax.errbar(j + (i - 0.5) * bw, m, e)
        top = max(vals["F5Ev"][0] + (vals["F5Ev"][1] if key != "success" else 0), vals["LLM"][0] + (vals["LLM"][1] if key != "success" else 0))
        ax.text(j, min(top, yl[1] * 0.93), lab(vals["F5Ev"][0], vals["LLM"][0]), F - 1.2, COL["LLM"], dy=-5, bold=True)
    ax.panel(["(a)", "(b)", "(c)"][pi])
    if pi == 2: ax.xticks(range(len(cats)), cats)
    if pi == 0: ax.text(xl[1], 170, f"No DSS: {V('T0','burned_ha').mean():.0f} ha, 0% extinguished", F - 1.2, "#7F7F7F", ha="right", dx=-3)
    if pi == 1: ax.text(xl[1], 800, f"No DSS: {V('T0','pop_affected').mean()*POP:.0f} persons", F - 1.2, "#7F7F7F", ha="right", dx=-3)
    y0 += ph + 22
Ax(d, L, 0, pw, 1).legend([(COL["F5Ev"], NAME["F5Ev"]), (COL["LLM"], NAME["LLM"])], L + 70, 6, col=2, dx=90)
done(d, "fig6_outcome", "Fig6 outcome per family")

# ===================================================================== Fig 7  paired saving and cost
d = Diagram(W1, 330, font=F); L = 46; pw = W1 - L - 10
diff = V("F5Ev", "burned_ha") - V("LLM", "burned_ha"); order = np.argsort(-diff); ds = diff[order]
ax = Ax(d, L, 16, pw, 130, (-1, 150), (min(ds.min(), -5) - 5, ds.max() + 10)); ax.frame(); ax.yticks(); ax.hline(0)
for i, v in enumerate(ds):
    ax.bar(i, v, 1.0, COL["LLM"] if v > 1e-9 else (TIE if abs(v) <= 1e-9 else RED))
n_pos = int((diff > 1e-9).sum()); n_tie = int((np.abs(diff) <= 1e-9).sum()); n_neg = int((diff < -1e-9).sum())
STATS.update(n_pos=n_pos, n_tie=n_tie, n_neg=n_neg, dmean=diff.mean(), dhw=hw(diff), dmed=float(np.median(diff)))

ax.xlabel("worlds, sorted by saving", off=9); ax.ylabel("saved against the closed layer (ha)", off=28, fs=F - 0.6); ax.panel("(a)")
# (b) physical cost (solid, configuration colour) beside the five-term total cost per configuration
terms = [("j_burn", "burned area", "#404040"), ("j_asset", "asset loss", "#7F7F7F"), ("j_pop", "population", "#A5A5A5"), ("j_resp", "0.2 × response", "#D9D9D9"), ("j_delay", "0.2 × delay", "#F2F2F2")]
wts = {"j_burn": 1, "j_asset": 1, "j_pop": 1, "j_resp": 0.2, "j_delay": 0.2}
ARMS4 = ["T0", "F5", "F5Ev", "LLM"]
ax = Ax(d, L + 10, 185, 200, 125, (-0.6, 3.6), (0, 0.67)); ax.frame(); ax.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6]); ax.ylabel("decision cost, mean of 150 worlds", off=28, fs=F - 0.6)
bw2 = 0.34
for i, a in enumerate(ARMS4):
    jp = V(a, "j_phys").mean(); jt = V(a, "j_total").mean(); STATS[f"jphys_{a}"] = jp; STATS[f"jtot_{a}"] = jt
    ax.bar(i - bw2 / 2 - 0.02, jp, bw2, COL[a]); ax.text(i - bw2 / 2 - 0.02, jp, f"{jp:.3f}", F - 1.5, INK, dy=-5)
    bottom = 0.0
    for k, lab, c in terms:
        v = wts[k] * V(a, k).mean() / 3.4; ax.bar(i + bw2 / 2 + 0.02, v, bw2, c, base=bottom, stroke="#ffffff"); bottom += v
    ax.text(i + bw2 / 2 + 0.02, bottom, f"{jt:.3f}", F - 1.5, INK, dy=-5)
ax.xticks(range(4), ["No DSS", "static seed", "closed layer", "VODA"], fs=F - 1.2); ax.panel("(b)")
ax.legend([("#8FAADC", "physical cost (left bar,\nconfiguration colour)")], ax.x + ax.w + 12, ax.y + 8, fs=F - 1.0)
ax.legend([(c, lab) for k, lab, c in terms], ax.x + ax.w + 12, ax.y + 36, dy=9, box=(7, 5), fs=F - 1.0)
d.text(ax.x + ax.w + 12, ax.y + 84, "right bar: total cost by term", F - 1.2, color=INK2)
done(d, "fig7_saving_cost", "Fig7 paired saving and cost")

# ===================================================================== Fig 8  proposals and admissions
kinds = ["rule", "macro", "actuator"]
verd = lambda d_: "admitted" if d_.get("accepted") else ("G7 margin" if "G5" in (d_.get("gate") or "") else "G6 second seed" if "G4" in (d_.get("gate") or "") else "G5 no gain" if "G3" in (d_.get("gate") or "") else "G1 to G4")
VS = ["admitted", "G7 margin", "G6 second seed", "G5 no gain", "G1 to G4"]; VC = {"admitted": COL["LLM"], "G7 margin": "#BFBFBF", "G6 second seed": "#7F7F7F", "G5 no gain": "#595959", "G1 to G4": "#262626"}
cnt = {k: collections.Counter() for k in kinds}
for p in props: cnt[p.get("kind") or "rule"][verd(p)] += 1
tot = {k: sum(cnt[k].values()) for k in kinds}
STATS["props"] = {k: dict(cnt[k]) for k in kinds}; STATS["nconcept"] = nconcept; STATS["nprops"] = len(props); STATS["nadm"] = len(admp)
d = Diagram(W1, 300, font=F)
ax = Ax(d, 44, 14, W1 - 54, 125, (-0.6, 2.6), (0, max(tot.values()) * 1.22)); ax.frame(); ax.yticks(); ax.ylabel("proposals", off=28)
for i, k in enumerate(kinds):
    b = 0
    for v in VS: ax.bar(i, cnt[k][v], 0.6, VC[v], base=b, stroke="#ffffff"); b += cnt[k][v]
    ax.text(i, b, f"{cnt[k]['admitted']} of {b} admitted ({100*cnt[k]['admitted']/b:.0f}%)", F - 1.2, dy=-5)
ax.xticks(range(3), ["new rule", "macro (blended)", "composed intervention"], fs=F - 1.0); ax.panel("(a)")
ax.legend([(VC[v], v) for v in VS], ax.x + 6, ax.y + 6, col=2, dx=74, dy=8.5, box=(6, 4.5), fs=F - 1.2)
adm = collections.defaultdict(collections.Counter); worlds = collections.defaultdict(set)
for p in admp: k = p.get("kind") or "rule"; adm[p["scenario"]][k] += 1; worlds[(p["scenario"], k)].add(p["seed"])
M = [[adm[sc][k] for k in kinds] for sc in SCEN]
ax = Ax(d, 60, 172, W1 - 76, 110); ax.panel("(b)")
ax.heatmap(M, SCEN, ["new rule", "macro", "composed intervention"], BLUES, rot=0, label=lambda v: "", lab_fs=F - 1.4)
for i, sc in enumerate(SCEN):
    for j, k in enumerate(kinds):
        f = M[i][j] / max(max(r) for r in M)
        d.text(ax.x + (j + 0.5) * ax.w / 3, ax.y + (i + 0.5) * ax.h / 5 - 3.5, f"{M[i][j]}", F - 1.0, color="#ffffff" if f > 0.6 else INK, ha="center")
        d.text(ax.x + (j + 0.5) * ax.w / 3, ax.y + (i + 0.5) * ax.h / 5 + 4.5, f"in {len(worlds[(sc,k)])} of 30 worlds", F - 1.8, color="#ffffff" if f > 0.6 else INK, ha="center")
STATS["adm_by_fam"] = {sc: {k: [adm[sc][k], len(worlds[(sc, k)])] for k in kinds} for sc in SCEN}
done(d, "fig8_proposals", "Fig8 proposals and admissions")

# ===================================================================== Fig 9  what was written (rules, macros, clauses)
fams = ["suppression effort", "resource deployment", "containment line", "asset protection", "evacuation", "public warning"]
gen = collections.Counter()
for p in admp:
    if (p.get("kind") or "rule") == "rule":
        m = re.search(r"THEN (.*)$", core(p["rule"]))
        for part in (m.group(1) if m else "").split(","):
            nm = re.sub(r"\s[\d.]+$", "", part.strip())
            if nm in fams: gen[nm] += 1
STATS["gen_rules_by_family"] = dict(gen); STATS["n_rules"] = sum(1 for p in admp if (p.get("kind") or "rule") == "rule")
d = Diagram(W1, 345, font=F)
ax = Ax(d, 78, 20, 170, 108, (0, max(max(gen.values()), 1) * 1.15), (-0.6, len(fams) - 0.4)); ax.frame()
for t in nice_ticks(0, ax.xlim[1], 4): d.line([(ax.px(t), ax.y), (ax.px(t), ax.y + ax.h)], GRID, 0.5) if t else None
ax.xticks(nice_ticks(0, ax.xlim[1], 4))
for i, f in enumerate(fams):
    yc = len(fams) - 1 - i
    ax.barh(yc + 0.2, 1, 0.4, COL["F5Ev"]); ax.barh(yc - 0.2, gen[f], 0.4, COL["LLM"])
    d.text(ax.x - 3, ax.py(yc), f, F - 1.0, color=INK2, ha="right")
ax.xlabel("rules ordering the family", off=14); ax.panel("(a)")
ax.legend([(COL["F5Ev"], "5 seed rules"), (COL["LLM"], f"{STATS['n_rules']} admitted rules")], ax.x + ax.w * 0.55, ax.y + 68, fs=F - 1.2)
fam9 = ["asset_protection", "containment_line", "suppression_effort", "resource_deployment", "evacuation", "public_warning"]
C = np.zeros((6, 6))
for p in admp:
    if p.get("kind") == "macro":
        fs_ = [f for f, w_ in p["composition"] if f in fam9]
        for f1 in fs_:
            for f2 in fs_: C[fam9.index(f1), fam9.index(f2)] += 1
short = ["asset", "line", "suppr.", "deploy", "evac.", "warn"]
ax = Ax(d, 40, 188, 122, 122); ax.panel("(b)")
ax.heatmap(C.tolist(), short, short, GREYS, rot=45, fs=F - 1.4, label=lambda v: f"{int(v)}" if v else "", lab_fs=F - 1.8)
adma = [p for p in admp if p.get("kind") == "actuator"]
effects = ["coat", "draft", "wet", "clear", "prime", "ignite", "evacuate"]; sectors = ["head", "flank", "rear", "ring", "at fire", "assets", "populated"]
seckeys = ["head", "flank", "rear", "ring", "at_fire", "assets", "populated"]
E = np.zeros((7, 7))
for p in adma:
    for c in p["clauses"]:
        if c["effect"] in effects and c["sector"] in seckeys: E[effects.index(c["effect"]), seckeys.index(c["sector"])] += 1
ax = Ax(d, 206, 188, 116, 122); ax.panel("(c)")
ax.heatmap(E.tolist(), effects, sectors, BLUES, rot=45, fs=F - 1.4, label=lambda v: f"{int(v)}" if v else "", lab_fs=F - 1.8)
STATS["n_macros"] = sum(1 for p in admp if p.get("kind") == "macro"); STATS["n_act"] = len(adma)
g = collections.Counter()
for p in adma: g[" + ".join(sorted({f"{c['effect']} {c['sector']}" for c in p['clauses']}))] += 1
STATS["archetypes"] = g.most_common(8); STATS["distinct_clause_sets"] = len(g)
done(d, "fig9_written", "Fig9 what the open layer wrote")

# ===================================================================== Fig 10  attribution and timeline
gain = {}; admk = {}
for sc in SCEN:
    for s in seeds[sc]: gain[(sc, s)] = runs[(sc, "F5Ev")][s]["burned_ha"] - runs[(sc, "LLM")][s]["burned_ha"]; admk[(sc, s)] = collections.Counter()
for p in admp: admk[(p["scenario"], p["seed"])][p.get("kind") or "rule"] += 1
z = np.array([g for k, g in gain.items() if sum(admk[k].values()) == 0]); nz = np.array([g for k, g in gain.items() if sum(admk[k].values()) > 0])
STATS.update(n_noadm=len(z), n_adm=len(nz), mean_noadm=float(z.mean()), mean_adm=float(nz.mean()), ident_noadm=int((np.abs(z) < 1e-9).sum()), absmean_noadm=float(np.abs(z).mean()))
d = Diagram(W1, 330, font=F)
ylo, yhi = min(z.min(), nz.min()) - 10, max(z.max(), nz.max()) + 10
ax = Ax(d, 46, 14, 90, 125, (0.4, 2.6), (ylo, yhi)); ax.frame(); ax.yticks(); ax.hline(0); ax.ylabel("saved against the closed layer (ha)", off=28, fs=F - 0.6)
for i, (arr, c) in enumerate([(z, TIE), (nz, COL["LLM"])]):
    q1, med, q3 = np.percentile(arr, [25, 50, 75]); iqr = q3 - q1
    lo = arr[arr >= q1 - 1.5 * iqr].min(); hi = arr[arr <= q3 + 1.5 * iqr].max(); xc = i + 1
    ax.bar(xc, q3 - q1, 0.5, c, base=q1, alpha=0.6, stroke=INK2); ax.d.line([(ax.px(xc - 0.25), ax.py(med)), (ax.px(xc + 0.25), ax.py(med))], INK, 1.0)
    ax.d.line([(ax.px(xc), ax.py(q3)), (ax.px(xc), ax.py(hi))], INK2, 0.6); ax.d.line([(ax.px(xc), ax.py(q1)), (ax.px(xc), ax.py(lo))], INK2, 0.6)
    for cap in (lo, hi): ax.d.line([(ax.px(xc - 0.12), ax.py(cap)), (ax.px(xc + 0.12), ax.py(cap))], INK2, 0.6)
    for v in arr[(arr < lo) | (arr > hi)]: ax.dot(xc, v, 1.1, INK2)
ax.xticks([1, 2], [f"no admission\n(n={len(z)})", f"≥1 admission\n(n={len(nz)})"], fs=F - 1.4); ax.panel("(a)")
xs = np.array([sum(admk[k].values()) for k in gain]); ys = np.array([gain[k] for k in gain]); jit = (np.random.default_rng(1).random(len(xs)) - 0.5) * 0.3
ax = Ax(d, 178, 14, 172, 125, (-0.6, xs.max() + 0.6), (ylo, yhi)); ax.frame(); ax.yticks(); ax.hline(0)
for a, b in zip(xs + jit, ys): ax.dot(a, b, 1.3, COL["LLM"], alpha=0.55)
medk = {}
for k in sorted(set(xs.tolist())):
    sel = xs == k
    if sel.sum() >= 3: ax.d.line([(ax.px(k - 0.3), ax.py(np.median(ys[sel]))), (ax.px(k + 0.3), ax.py(np.median(ys[sel])))], INK, 1.2); medk[int(k)] = float(np.median(ys[sel]))
STATS["median_by_admissions"] = medk
ax.xticks(list(range(0, int(xs.max()) + 1))); ax.xlabel("admissions in the world", off=12); ax.panel("(b)")
best = max(((sc, s) for sc in ["S4"] for s in seeds[sc]), key=lambda k: gain[k]); sc, s = best
per = collections.defaultdict(dict)
for src, arms_ in ((OUT, ("T0", "F5Ev")), (OUTN, ("LLM",))):
    for r in csv.DictReader(open(f"{src}/curves.csv", encoding="utf-8"), fieldnames=["exp", "scenario", "arm", "seed", "t_min", "cells", "frac"]):
        try:
            if r["scenario"] == sc and int(r["seed"]) == s and r["arm"] in arms_: per[r["arm"]][float(r["t_min"])] = float(r["cells"]) * 0.09
        except ValueError: pass
ymax = max(max(per[a].values()) for a in ARMS)
ax = Ax(d, 46, 185, W1 - 58, 120, (0, 12), (0, ymax * 1.08)); ax.frame(); ax.yticks(); ax.ylabel("burned area (ha)", off=28)
items = []
for a in ARMS:
    t = np.array(sorted(per[a])); v = np.array([per[a][k] for k in t])
    if len(t) and t[-1] < 720: t = np.append(t, 720); v = np.append(v, v[-1])
    ax.plot(t / 60, v, COL[a], 1.5); items.append((COL[a], f"{NAME[a]}: {runs[(sc,a)][s]['burned_ha']:.0f} ha", "line"))
ev = sorted([(p["t_min"], p.get("kind") or "rule") for p in admp if p["scenario"] == sc and p["seed"] == s])
KN = {"actuator": "composed intervention", "macro": "macro", "rule": "rule"}
for i_, (t, k) in enumerate(ev): ax.vline(t / 60, COL["LLM"], 0.7, dotted=True); ax.text(ev[-1][0] / 60, ymax * (0.90 - 0.08 * i_), f"{KN[k]} admitted, minute {t:.0f}", F - 1.2, COL["LLM"], ha="left", dx=3)
STATS["timeline"] = {"world": [sc, s], "events": ev, "final": {a: runs[(sc, a)][s]["burned_ha"] for a in ARMS}}
ax.xticks(list(range(0, 13, 2))); ax.xlabel("time (h)", off=12); ax.text(12, ymax * 1.02, "one S4 world", F - 1.0, INK2, ha="right"); ax.panel("(c)")
ax.legend(items, ax.x + ax.w - 110, ax.y + ax.h / 2 - 6, fs=F - 1.0)
done(d, "fig10_attribution", "Fig10 attribution and timeline")

# ===================================================================== Fig 11  cost against agents and grid (cost_scale.csv)
SC = "/home/claude/v45/figs/cost_scale.csv"
reg = collections.defaultdict(list); grd = collections.defaultdict(list)
for r in csv.DictReader(open(SC, encoding="utf-8")):
    if r["axis"] == "regions": reg[int(r["regions"])].append(float(r["cycle_ms_med"]))
    else: grd[int(r["cells"])].append(float(r["cycle_ms_med"]))
regm = {k: float(np.median(v)) for k, v in reg.items()}; grdm = {k: float(np.median(v)) for k, v in grd.items()}
STATS["regions_median_ms"] = regm; STATS["grid_median_ms"] = grdm
d = Diagram(W1, 150, font=F)
cyc = [(2, "#9DC3E6", "2 min cycle"), (6, "#2E75B6", "6 min cycle"), (12, "#1F3864", "12 min cycle")]
ks = sorted(regm); xs_ = list(range(len(ks)))
ax = Ax(d, 50, 14, 180, 105, (-0.5, len(ks) - 0.5), (math.log10(0.01), math.log10(1.0))); ax.frame()
for t in [0.01, 0.03, 0.1, 0.3]:
    yy = ax.py(math.log10(t)); d.line([(ax.x, yy), (ax.x + ax.w, yy)], GRID, 0.5); d.text(ax.x - 3, yy, f"{t:g}%", F - 1.0, color=INK2, ha="right")
allsh = []
for cm, c, lab in cyc:
    ys_ = [math.log10(100 * regm[k] / (cm * 60000)) for k in ks]; allsh += [10 ** y for y in ys_]
    ax.plot(xs_, ys_, c, 1.3)
    for x_, y_ in zip(xs_, ys_): ax.dot(x_, y_, 1.6, c)
    ax.text(xs_[0], ys_[0], lab, F - 1.2, c, ha="left", dy=-7)
STATS["share_min"] = min(allsh); STATS["share_max"] = max(allsh); STATS["ratio16"] = regm[16] / regm[1]
ax.xticks(xs_, [str(k) for k in ks]); ax.xlabel("local agents", off=12); ax.ylabel("share of the decision cycle", off=30, fs=F - 0.6); ax.panel("(a)")
gk = sorted(grdm)
ax = Ax(d, 282, 14, 70, 105, (-0.5, 2.5), (0, max(grdm.values()) / 1000 * 1.2)); ax.frame(); ax.yticks(); ax.ylabel("median cycle time (s)", off=22, fs=F - 0.6)
for i, k in enumerate(gk):
    ax.bar(i, grdm[k] / 1000, 0.6, "#7F7F7F"); ax.text(i, grdm[k] / 1000, f"{grdm[k]/1000:.2f}", F - 1.4, dy=-5)
ax.xticks(range(3), ["4.8k", "10.8k", "19.2k"], fs=F - 1.2); ax.xlabel("grid cells", off=12); ax.panel("(b)")
STATS["grid_ratio"] = grdm[gk[-1]] / grdm[gk[0]]
done(d, "fig11_scalability", "Fig11 decision-layer cost")


# ===================================================================== extra statistics for the text
def paired(key, a="F5Ev", b="LLM", sc=None):
    x = V(a, key, sc) - V(b, key, sc); return float(x.mean()), float(hw(x)), int((x > 1e-9).sum()), int((np.abs(x) <= 1e-9).sum()), int((x < -1e-9).sum())
X = {}
for key in ["burned_ha", "pop_affected", "j_phys", "j_total", "j_resp", "j_delay", "rules_final", "success"]:
    X[key] = {a: float(V(a, key).mean()) for a in ALL}
X["paired_burn_all"] = paired("burned_ha"); X["paired_pop_all"] = paired("pop_affected")
X["paired_burn_F5_vs_F5Ev"] = paired("burned_ha", "F5", "F5Ev"); X["paired_burn_T0_vs_F5"] = paired("burned_ha", "T0", "F5")
X["by_family"] = {sc: dict(burn={a: float(V(a, "burned_ha", sc).mean()) for a in ALL}, success={a: float(V(a, "success", sc).mean()) for a in ALL},
                            pop={a: float(V(a, "pop_affected", sc).mean()) for a in ALL}, paired=paired("burned_ha", sc=sc)) for sc in SCEN}
X["trials_closed_in_voda"] = dict(tried=float((V("LLM", "tried_1") + V("LLM", "tried_2")).sum()), kept=float((V("LLM", "acc_1") + V("LLM", "acc_2")).sum()))
X["trials_closed_alone"] = dict(tried=float((V("F5Ev", "tried_1") + V("F5Ev", "tried_2")).sum()), kept=float((V("F5Ev", "acc_1") + V("F5Ev", "acc_2")).sum()))
X["stage3"] = dict(tried=float(V("LLM", "tried_3").sum()), acc=float(V("LLM", "acc_3").sum()), calls_per_world=float(V("LLM", "prop_calls").mean()))
# rediscovery of archetypes across worlds
def has(p, eff, sec): return any(c["effect"] == eff and c["sector"] == sec for c in (p.get("clauses") or []))
coll = set(); band = set(); hard = set(); lna = set()
for p in admp:
    k = (p["scenario"], p["seed"])
    if p.get("kind") == "actuator":
        if has(p, "coat", "assets") and has(p, "coat", "populated"): coll.add(k)
        if has(p, "coat", "head") and has(p, "coat", "flank"): band.add(k)
    if p.get("kind") == "macro":
        fs_ = {f for f, w_ in p["composition"]}
        if {"asset_protection", "containment_line"} <= fs_: hard.add(k)
        if {"containment_line", "suppression_effort"} <= fs_: lna.add(k)
X["rediscovery_worlds"] = dict(asset_collar=len(coll), head_flank_band=len(band), interface_hardening=len(hard), line_and_attack=len(lna))
X["archetype_counts"] = dict(asset_collar=sum(1 for p in admp if p.get("kind") == "actuator" and has(p, "coat", "assets") and has(p, "coat", "populated")),
                             head_flank_band=sum(1 for p in admp if p.get("kind") == "actuator" and has(p, "coat", "head") and has(p, "coat", "flank")),
                             interface_hardening=sum(1 for p in admp if p.get("kind") == "macro" and {"asset_protection", "containment_line"} <= {f for f, w_ in p["composition"]}),
                             line_and_attack=sum(1 for p in admp if p.get("kind") == "macro" and {"containment_line", "suppression_effort"} <= {f for f, w_ in p["composition"]}))
mw = collections.Counter()
for p in admp:
    if p.get("kind") == "macro":
        for f, w_ in p["composition"]: mw[f] += 1
X["macro_family_use"] = dict(mw)
blend = [dict(p["composition"]) for p in admp if p.get("kind") == "macro" and {"asset_protection", "containment_line"} <= {f for f, w_ in p["composition"]}]
X["hardening_line_weights"] = [b["containment_line"] for b in blend]
eff_cnt = collections.Counter(c["effect"] for p in adma for c in p["clauses"]); X["clause_effects"] = dict(eff_cnt)
X["drafting_in_band"] = sum(1 for p in adma if has(p, "coat", "head") and has(p, "coat", "flank") and any(c["effect"] == "draft" for c in p["clauses"]))
# per-world attribution details
X["gain_by_admissions_mean"] = {int(k): float(np.mean(ys[xs == k])) for k in sorted(set(xs.tolist()))}
X["worst_worlds"] = sorted([(k, g) for k, g in gain.items() if g < -5], key=lambda t: t[1])[:5]
STATS["X"] = X

drawio_file(PAGES, f"{FD}/results_pages.drawio")
json.dump(STATS, open(f"{FD}/stats.json", "w"), indent=1, default=str)
print(json.dumps(STATS, indent=1, default=str))
