"""Result figures as primitive shapes: one spec -> PNG for the manuscript and an editable draw.io page.
Reads the benchmark output (runs.csv, curves.csv, proposals.jsonl)."""
import csv, json, re, collections, math
import numpy as np
from diag import Diagram
from chartkit import Ax, INK, INK2, GRID, GREYS, GREENS, BLUES, nice_ticks

OUT = "/home/claude/bench/out"; FD = "/home/claude/v6/fig"
ALL = ["T0", "F5", "F5Ev", "LLM"]; ARMS = ["T0", "F5Ev", "LLM"]; SCEN = ["S1", "S2", "S3", "S4", "S5"]; POP = 9e-4
NAME = {"T0": "No DSS", "F5": "static seed", "F5Ev": "Layer 1 (closed)", "LLM": "VODA"}
COL = {"T0": "#A5A5A5", "F5": "#9DC3E6", "F5Ev": "#70AD47", "LLM": "#4472C4"}; RED = "#C00000"
W1 = 362  # single-column canvas width (px); double column 750


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
props = [d for d in props if d.get("kind") != "concept"]   # intermediate concepts are not part of the article
admp = [d for d in props if d.get("accepted")]


def V(arm, key, sc=None):
    scs = [sc] if sc else SCEN
    return np.array([runs[(s, arm)][x][key] for s in scs for x in seeds[s]], float)


def core(r):
    r = re.sub(r"^G\d+: ", "", r); return re.sub(r" \(adaptation-born.*$", "", r)


PAGES = []


def done(d, name, title):
    d.render(f"{FD}/{name}.png"); PAGES.append((title, d))


# ===================================================================== Fig 6  outcomes (r1b)
d = Diagram(W1, 540, font=6.4); L, R = 48, 10; pw = W1 - L - R
cats = SCEN + ["Mean"]; xl = (-0.6, len(cats) - 0.4); bw = 0.36
panels = [("burned_ha", 1, "burned area (ha)", (0, 240), lambda a, b: f"−{100*(a-b)/a:.0f} %"),
          ("pop_affected", POP, "affected population (persons)", (0, 1150), lambda a, b: f"−{100*(a-b)/a:.0f} %"),
          ("success", 100, "extinguished within episode (%)", (0, 118), lambda a, b: f"+{b-a:.0f} pts")]
y0 = 26
for pi, (key, scale, ylab, yl, lab) in enumerate(panels):
    ax = Ax(d, L, y0, pw, 140, xl, yl); ax.frame(); ax.yticks(grid=True); ax.ylabel(ylab, off=30)
    for j, sc in enumerate(SCEN + [None]):
        vals = {}
        for i, a in enumerate(["F5Ev", "LLM"]):
            v = V(a, key, sc) * scale; m, e = v.mean(), hw(v); vals[a] = (m, e)
            ax.bar(j + (i - 0.5) * bw, m, bw * 0.9, COL[a])
            if key != "success": ax.errbar(j + (i - 0.5) * bw, m, e)
        top = max(vals["F5Ev"][0], vals["LLM"][0]) + (vals["F5Ev"][1] if key != "success" else 0)
        ax.text(j, top, lab(vals["F5Ev"][0], vals["LLM"][0]), 5.8, COL["LLM"], dy=-6, bold=True)
    ax.panel(["(a)", "(b)", "(c)"][pi])
    if pi == 2: ax.xticks(range(len(cats)), cats)
    if pi == 0: ax.text(xl[1], 150, f"No DSS: {V('T0','burned_ha').mean():.0f} ha (mean),\n0 % extinguished", 5.8, COL["T0"], ha="right", dx=-3)
    if pi == 1: ax.text(xl[1], 700, f"No DSS: {V('T0','pop_affected').mean()*POP:.0f} persons (mean)", 5.8, COL["T0"], ha="right", dx=-3)
    y0 += 168
Ax(d, L, 0, pw, 1).legend([(COL["F5Ev"], NAME["F5Ev"]), (COL["LLM"], NAME["LLM"])], L + 60, 10, col=2, dx=110)
done(d, "r1b_outcomes", "Fig6 outcomes")

# ===================================================================== Fig 7  paired differences (r2b)
d = Diagram(W1, 520, font=6.4); L = 48; pw = W1 - L - 10
diff = V("F5Ev", "burned_ha") - V("LLM", "burned_ha"); order = np.argsort(-diff); ds = diff[order]
ax = Ax(d, L, 26, pw, 230, (-1, 150), (min(ds.min(), -5) - 5, ds.max() + 10)); ax.frame(); ax.yticks(); ax.hline(0)
for i, v in enumerate(ds):
    ax.bar(i, v, 1.0, COL["LLM"] if v > 1e-9 else ("#BFBFBF" if abs(v) <= 1e-9 else RED))
n_pos = int((diff > 1e-9).sum()); n_tie = int((np.abs(diff) <= 1e-9).sum()); n_neg = int((diff < -1e-9).sum())
d.text(L + 95, 36, f"VODA burned less in {n_pos} of 150 worlds", 6.0, color=INK)
d.text(L + 95, 46, f"ties {n_tie} (small fires already held), worse in {n_neg}", 6.0, color=INK)
d.text(L + 95, 56, f"mean {diff.mean():.1f} ± {hw(diff):.1f} ha, median {np.median(diff):.1f} ha", 6.0, color=INK)
ax.xlabel("worlds, sorted by saving", off=10); ax.ylabel("burned area saved vs Layer 1 (ha)", off=30); ax.panel("(a)")
meas = [("burned_ha", "burned area"), ("pop_affected", "affected population"), ("j_phys", "$J_{phys}$"), ("j_total", "$J_{total}$"), ("j_resp", "$J_{resp}$ (resources)"), ("j_delay", "$J_{delay}$")]
rel, relf5 = [], []
for k, lab in meas:
    l1 = V("F5Ev", k).mean(); da = V("LLM", k).mean(); f5 = V("F5", k).mean(); rel.append(100 * (l1 - da) / l1); relf5.append(100 * (f5 - l1) / f5)
ax = Ax(d, L + 50, 300, pw - 50, 190, (-32, 100), (-0.6, len(meas) - 0.4)); ax.frame(); ax.vline(0)
for t in nice_ticks(-30, 100, 6):
    ax.d.line([(ax.px(t), ax.y), (ax.px(t), ax.y + ax.h)], GRID, 0.5) if t else None
ax.xticks([-25, 0, 25, 50, 75, 100], [f"{t}" for t in [-25, 0, 25, 50, 75, 100]])
for i, (k, lab) in enumerate(meas):
    yc = len(meas) - 1 - i
    ax.barh(yc + 0.2, relf5[i], 0.38, COL["F5Ev"]); ax.barh(yc - 0.2, rel[i], 0.38, COL["LLM"])
    for v, off, c in ((rel[i], -0.2, COL["LLM"]), (relf5[i], 0.2, COL["F5Ev"])):
        ax.text(v, yc + off, f"{v:+.0f} %", 5.6, c, ha="left" if v >= 0 else "right", dx=2 if v >= 0 else -2)
    d.text(ax.x - 3, ax.py(yc), lab, 6.0, color=INK2, ha="right")
ax.xlabel("reduction of the mean (%)", off=16); ax.panel("(b)")
ax.legend([(COL["F5Ev"], "Layer 1 vs static seed"), (COL["LLM"], "VODA vs Layer 1")], ax.x + ax.w - 110, ax.y + ax.h - 22)
done(d, "r2b_paired", "Fig7 paired differences")

# ===================================================================== Fig 8  costs (r3)
d = Diagram(W1, 230, font=6.4)
jp = [V(a, "j_phys").mean() for a in ARMS]; jt = [V(a, "j_total").mean() for a in ARMS]
ax = Ax(d, 44, 24, 130, 160, (-0.6, 2.6), (0, 0.68)); ax.frame(); ax.yticks(); ax.ylabel("decision cost (mean)", off=28)
for i, a in enumerate(ARMS):
    ax.bar(i - 0.19, jp[i], 0.36, COL[a]); ax.bar(i + 0.19, jt[i], 0.36, COL[a], alpha=0.45, hatch="////")
    ax.text(i - 0.19, jp[i], f"{jp[i]:.3f}", 5.2, dy=-5); ax.text(i + 0.19, jt[i], f"{jt[i]:.3f}", 5.2, dy=-5)
ax.xticks(range(3), ["No DSS", "Layer 1", "VODA"], fs=5.6); ax.panel("(a)")
ax.legend([("#555555", "$J_{phys}$"), ("#555555", "$J_{total}$", "hatch")], ax.x + ax.w - 48, ax.y + 8)
terms = [("j_burn", "$J_{burn}$", "#404040"), ("j_asset", "$J_{asset}$", "#7F7F7F"), ("j_pop", "$J_{pop}$", "#A5A5A5"), ("j_resp", "$0.2\\,J_{resp}$", "#D9D9D9"), ("j_delay", "$0.2\\,J_{delay}$", "#F2F2F2")]
wts = {"j_burn": 1, "j_asset": 1, "j_pop": 1, "j_resp": 0.2, "j_delay": 0.2}
ax = Ax(d, 218, 24, 130, 160, (-0.6, 2.6), (0, 0.68)); ax.frame(); ax.yticks(); ax.ylabel("$J_{total}$ by term", off=28)
bottom = [0.0] * 3
for k, lab, c in terms:
    for i, a in enumerate(ARMS):
        v = wts[k] * V(a, k).mean() / 3.4; ax.bar(i, v, 0.6, c, base=bottom[i], stroke="#ffffff"); bottom[i] += v
ax.xticks(range(3), ["No DSS", "Layer 1", "VODA"], fs=5.6); ax.panel("(b)")
ax.legend([(c, lab) for k, lab, c in terms], ax.x + ax.w - 50, ax.y + 8, dy=8, box=(6, 4.5))
done(d, "r3_costs", "Fig8 costs")

# ===================================================================== Fig 9  products (r4)
kinds = ["rule", "macro", "actuator"]
verd = lambda d_: "admitted" if d_.get("accepted") else ("G7 margin" if "G5" in (d_.get("gate") or "") else "G6 second seed" if "G4" in (d_.get("gate") or "") else "G5 no gain" if "G3" in (d_.get("gate") or "") else "G1 to G4")
VS = ["admitted", "G7 margin", "G6 second seed", "G5 no gain", "G1 to G4"]; VC = {"admitted": COL["LLM"], "G7 margin": "#BFBFBF", "G6 second seed": "#7F7F7F", "G5 no gain": "#595959", "G1 to G4": "#262626"}
cnt = {k: collections.Counter() for k in kinds}
for p in props: cnt[p.get("kind") or "rule"][verd(p)] += 1
tot = {k: sum(cnt[k].values()) for k in kinds}
d = Diagram(W1, 440, font=6.4)
ax = Ax(d, 44, 24, W1 - 54, 180, (-0.6, 2.6), (0, max(tot.values()) * 1.22)); ax.frame(); ax.yticks(); ax.ylabel("proposals", off=28)
for i, k in enumerate(kinds):
    b = 0
    for v in VS: ax.bar(i, cnt[k][v], 0.6, VC[v], base=b, stroke="#ffffff"); b += cnt[k][v]
    ax.text(i, b, f"{cnt[k]['admitted']}/{b} ({100*cnt[k]['admitted']/b:.0f} %)", 5.6, dy=-6)
ax.xticks(range(3), ["new rule", "macro (blended)", "actuator (composed)"], fs=5.6); ax.panel("(a)")
ax.legend([(VC[v], v) for v in VS], ax.x + 8, ax.y + 8, col=2, dx=70, dy=8, box=(6, 4.5))
adm = collections.defaultdict(collections.Counter); worlds = collections.defaultdict(set)
for p in admp: k = p.get("kind") or "rule"; adm[p["scenario"]][k] += 1; worlds[(p["scenario"], k)].add(p["seed"])
M = [[adm[sc][k] for k in kinds] for sc in SCEN]
ax = Ax(d, 60, 250, W1 - 76, 160); ax.panel("(b)")
ax.heatmap(M, SCEN, ["new rule", "macro", "actuator"], BLUES, rot=0, label=lambda v: "", lab_fs=5.4)
for i, sc in enumerate(SCEN):
    for j, k in enumerate(kinds):
        f = M[i][j] / max(max(r) for r in M)
        d.text(ax.x + (j + 0.5) * ax.w / 3, ax.y + (i + 0.5) * ax.h / 5 - 4, f"{M[i][j]}", 5.6, color="#ffffff" if f > 0.6 else INK, ha="center")
        d.text(ax.x + (j + 0.5) * ax.w / 3, ax.y + (i + 0.5) * ax.h / 5 + 5, f"({len(worlds[(sc,k)])} w.)", 5.0, color="#ffffff" if f > 0.6 else INK, ha="center")
done(d, "r4_products", "Fig9 proposals and admissions")

# ===================================================================== Fig 10  vocabulary (r5), double column
fams = ["suppression effort", "resource deployment", "containment line", "asset protection", "evacuation", "public warning", "retardant drop", "water drafting", "tactical burn"]
gen = collections.Counter()
for p in admp:
    if (p.get("kind") or "rule") == "rule":
        m = re.search(r"THEN (.*)$", core(p["rule"]))
        for part in (m.group(1) if m else "").split(","):
            nm = re.sub(r"\s[\d.]+$", "", part.strip())
            if nm in fams: gen[nm] += 1
d = Diagram(560, 300, font=6.4)
ax = Ax(d, 90, 24, 170, 220, (0, max(max(gen.values()), 1) * 1.15), (-0.6, len(fams) - 0.4)); ax.frame()
for t in nice_ticks(0, ax.xlim[1], 4): d.line([(ax.px(t), ax.y), (ax.px(t), ax.y + ax.h)], GRID, 0.5) if t else None
ax.xticks(nice_ticks(0, ax.xlim[1], 4))
for i, f in enumerate(fams):
    yc = len(fams) - 1 - i
    ax.barh(yc + 0.2, 1 if i < 6 else 0, 0.4, COL["F5Ev"]); ax.barh(yc - 0.2, gen[f], 0.4, COL["LLM"])
    d.text(ax.x - 3, ax.py(yc), f, 5.8, color=INK2, ha="right")
ax.xlabel("rules ordering the family", off=16); ax.panel("(a)")
ax.legend([(COL["F5Ev"], "seed base"), (COL["LLM"], "generated rules")], ax.x + ax.w - 70, ax.y + ax.h - 18)
fam6 = ["asset_protection", "containment_line", "suppression_effort", "resource_deployment", "retardant_drop", "water_drafting", "tactical_burn", "evacuation", "public_warning"]
C = np.zeros((9, 9))
for p in admp:
    if p.get("kind") == "macro":
        fs_ = [f for f, w_ in p["composition"] if f in fam6]
        for f1 in fs_:
            for f2 in fs_: C[fam6.index(f1), fam6.index(f2)] += 1
short = ["asset", "line", "suppr.", "deploy", "retard.", "draft", "burn", "evac.", "warn"]
ax = Ax(d, 350, 24, 180, 180); ax.panel("(b)")
ax.heatmap(C.tolist(), short, short, GREYS, rot=45, fs=5.4, label=lambda v: f"{int(v)}" if v else "", lab_fs=5.0)
done(d, "r5_vocabulary", "Fig10 vocabulary")

# ===================================================================== Fig 11  composed interventions (r6)
adma = [p for p in admp if p.get("kind") == "actuator"]
effects = ["coat", "draft", "wet", "clear", "prime", "ignite", "evacuate"]; sectors = ["head", "flank", "rear", "ring", "at_fire", "assets", "populated"]
M = np.zeros((7, 7))
for p in adma:
    for c in p["clauses"]:
        if c["effect"] in effects and c["sector"] in sectors: M[effects.index(c["effect"]), sectors.index(c["sector"])] += 1
d = Diagram(W1, 420, font=6.4)
ax = Ax(d, 60, 24, W1 - 80, 160); ax.panel("(a)")
ax.heatmap(M.tolist(), effects, sectors, BLUES, rot=45, fs=5.6, label=lambda v: f"{int(v)}" if v else "", lab_fs=5.2)
g = collections.Counter()
for p in adma: g[" + ".join(sorted({f"{c['effect']} {c['sector']}" for c in p['clauses']}))] += 1
top = g.most_common(7)
ax = Ax(d, 150, 250, W1 - 170, 150, (0, max(v for k, v in top) * 1.1), (-0.6, len(top) - 0.4)); ax.frame()
for t in nice_ticks(0, ax.xlim[1], 4): d.line([(ax.px(t), ax.y), (ax.px(t), ax.y + ax.h)], GRID, 0.5) if t else None
ax.xticks(nice_ticks(0, ax.xlim[1], 4))
for i, (k, v) in enumerate(top):
    yc = len(top) - 1 - i; ax.barh(yc, v, 0.6, COL["LLM"]); d.text(ax.x - 3, ax.py(yc), k, 5.0, color=INK2, ha="right")
ax.xlabel("admitted", off=16); ax.panel("(b)")
done(d, "r6_actuators", "Fig11 composed interventions")

# ===================================================================== Fig 12  attribution + timeline (r7), double column
gain = {}; admk = {}
for sc in SCEN:
    for s in seeds[sc]: gain[(sc, s)] = runs[(sc, "F5Ev")][s]["burned_ha"] - runs[(sc, "LLM")][s]["burned_ha"]; admk[(sc, s)] = collections.Counter()
for p in admp: admk[(p["scenario"], p["seed"])][p.get("kind") or "rule"] += 1
z = np.array([g for k, g in gain.items() if sum(admk[k].values()) == 0]); nz = np.array([g for k, g in gain.items() if sum(admk[k].values()) > 0])
d = Diagram(750, 250, font=6.4)
ylo, yhi = min(z.min(), nz.min()) - 10, max(z.max(), nz.max()) + 10
ax = Ax(d, 50, 24, 130, 170, (0.4, 2.6), (ylo, yhi)); ax.frame(); ax.yticks(); ax.hline(0); ax.ylabel("burned area saved vs Layer 1 (ha)", off=30)
for i, (arr, c) in enumerate([(z, "#BFBFBF"), (nz, COL["LLM"])]):
    q1, med, q3 = np.percentile(arr, [25, 50, 75]); iqr = q3 - q1
    lo = arr[arr >= q1 - 1.5 * iqr].min(); hi = arr[arr <= q3 + 1.5 * iqr].max(); xc = i + 1
    ax.bar(xc, q3 - q1, 0.5, c, base=q1, alpha=0.6, stroke=INK2); ax.d.line([(ax.px(xc - 0.25), ax.py(med)), (ax.px(xc + 0.25), ax.py(med))], INK, 1.0)
    ax.d.line([(ax.px(xc), ax.py(q3)), (ax.px(xc), ax.py(hi))], INK2, 0.6); ax.d.line([(ax.px(xc), ax.py(q1)), (ax.px(xc), ax.py(lo))], INK2, 0.6)
    for cap in (lo, hi): ax.d.line([(ax.px(xc - 0.12), ax.py(cap)), (ax.px(xc + 0.12), ax.py(cap))], INK2, 0.6)
    for v in arr[(arr < lo) | (arr > hi)]: ax.dot(xc, v, 1.2, INK2)
ax.xticks([1, 2], [f"no admission\n(n={len(z)})", f"≥1 admission\n(n={len(nz)})"], fs=5.4); ax.panel("(a)")
xs = np.array([sum(admk[k].values()) for k in gain]); ys = np.array([gain[k] for k in gain]); jit = (np.random.default_rng(1).random(len(xs)) - 0.5) * 0.3
ax = Ax(d, 230, 24, 200, 170, (-0.6, xs.max() + 0.6), (ylo, yhi)); ax.frame(); ax.yticks(); ax.hline(0); ax.ylabel("saved vs Layer 1 (ha)", off=28)
for a, b in zip(xs + jit, ys): ax.dot(a, b, 1.5, COL["LLM"], alpha=0.6)
for k in sorted(set(xs.tolist())):
    sel = xs == k
    if sel.sum() >= 3: ax.d.line([(ax.px(k - 0.3), ax.py(np.median(ys[sel]))), (ax.px(k + 0.3), ax.py(np.median(ys[sel])))], INK, 1.2)
ax.xticks(list(range(0, int(xs.max()) + 1))); ax.xlabel("admissions in the world", off=14); ax.panel("(b)")
best = max(((sc, s) for sc in ["S4"] for s in seeds[sc]), key=lambda k: gain[k]); sc, s = best
per = collections.defaultdict(dict)
for r in csv.DictReader(open(f"{OUT}/curves.csv", encoding="utf-8"), fieldnames=["exp", "scenario", "arm", "seed", "t_min", "cells", "frac"]):
    try:
        if r["scenario"] == sc and int(r["seed"]) == s and r["arm"] in ARMS: per[r["arm"]][float(r["t_min"])] = float(r["cells"]) * 0.09
    except ValueError: pass
ymax = max(max(per[a].values()) for a in ARMS)
ax = Ax(d, 480, 24, 250, 170, (0, 12), (0, ymax * 1.08)); ax.frame(); ax.yticks(); ax.ylabel("burned area (ha)", off=28)
items = []
for a in ARMS:
    t = np.array(sorted(per[a])); v = np.array([per[a][k] for k in t])
    if len(t) and t[-1] < 720: t = np.append(t, 720); v = np.append(v, v[-1])
    ax.plot(t / 60, v, COL[a], 1.5); items.append((COL[a], f"{NAME[a]}: {runs[(sc,a)][s]['burned_ha']:.0f} ha", "line"))
ev = sorted([(p["t_min"], p.get("kind") or "rule") for p in admp if p["scenario"] == sc and p["seed"] == s])
for t, k in ev: ax.vline(t / 60, COL["LLM"], 0.7, dotted=True); ax.text(t / 60, ymax * 0.9, f"{k} admitted", 5.6, COL["LLM"], ha="left", dx=2)
ax.xticks(list(range(0, 13, 2))); ax.xlabel("time (h)", off=14); ax.title(f"{sc}, world {s}"); ax.panel("(c)")
ax.legend(items, ax.x + ax.w - 120, ax.y + ax.h / 2 - 10)
done(d, "r7_attribution", "Fig12 attribution and timeline")

# ===================================================================== raster figures (hindcast maps, scalability): embedded images
MEDIA = "/home/claude/v6/media/word/media"
from PIL import Image
for name, path, title, wpx in [("hindcast", f"{MEDIA}/image12.jpeg", "Fig5 hindcast maps (raster)", 750), ("scalability", f"{MEDIA}/image17.png", "Fig13 scalability (raster)", W1)]:
    iw, ih = Image.open(path).size; h = wpx * ih / iw
    d = Diagram(wpx, h, font=6.4); d.image(0, 0, wpx, h, path); PAGES.append((title, d))

if __name__ == "__main__":
    print("ok", [p[0] for p in PAGES])
