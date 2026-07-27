"""Aggregate the campaign into the thesis tables and figures.

Reads  experiments/out/ladder_runs.csv + ladder_curves.csv
Writes experiments/out/table58_phys.csv     (Table 5.8 rows)
       experiments/out/table59_cost.csv     (Table 5.9 rows)
       experiments/out/claim_chain.json     (x1, x2, x3, d, ...)
       01_Thesis/figures/fig57_burned_curves.png
       01_Thesis/figures/fig58_cost_decomposition.png

Every figure is drawn from the CSVs by THIS script, so a rename or a
style change is a one-line edit here and a re-run; the data never has
to be regenerated. Edit the STYLE block below for cosmetics.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGDIR = os.path.join(HERE, "..", "..", "01_Thesis", "figures")

# ----------------------------------------------------------- STYLE
ARM_ORDER = ["Test0", "F5", "F5Ev", "F5AI", "F5EvAI", "F22", "F40"]
ARM_LABEL = {                       # edit freely; used in figures
    "Test0": "Test$_0$ (no DSS)",
    "F5": "Test$_{F5}$",
    "F5Ev": "Test$_{F5+Ev}$",
    "F5AI": "Test$_{F5+AI}$",
    "F5EvAI": "Test$_{F5+Ev+AI}$",
    "F22": "Test$_{F22}$",
    "F40": "Test$_{F40}$",
}
ARM_COLOR = {
    "Test0": "#7f8c8d", "F5": "#2980b9", "F5Ev": "#27ae60",
    "F5AI": "#8e44ad", "F5EvAI": "#c0392b", "F22": "#f39c12",
    "F40": "#16a085",
}
SCEN_ORDER = ["S1", "S2", "S3", "S4", "S5"]
COST_TERMS = [("j_burn", "burned area", "#e67e22"),
              ("j_asset", "asset loss", "#c0392b"),
              ("j_pop", "population", "#2980b9"),
              ("j_resp", "response", "#8e44ad"),
              ("j_delay", "delay", "#95a5a6")]
# J weights (Balanced profile): the FIGURE stacks weighted
# contributions so the bar height equals the decision cost J itself;
# the TABLE keeps the raw normalized terms.
def _weights():
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from disaster_phyengine.config import SimConfig
    c = SimConfig().cost
    w = {"j_burn": c.w_burn, "j_asset": c.w_asset,
         "j_pop": c.w_pop, "j_resp": c.w_resp,
         "j_delay": getattr(c, "w_delay", 0.2)}
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


def _mean_ci(vals):
    v = [x for x in vals if isinstance(x, float) and not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    m = float(np.mean(v))
    if len(v) < 2:
        return m, 0.0
    s = float(np.std(v, ddof=1)) / math.sqrt(len(v))
    return m, 1.96 * s


def load_runs():
    rows = list(csv.DictReader(open(os.path.join(OUT,
                                                 "ladder_runs.csv"))))
    cell = defaultdict(list)
    for r in rows:
        cell[(r["scenario"], r["arm"])].append(r)
    return rows, cell


def fnum(r, k):
    try:
        return float(r[k])
    except Exception:
        return float("nan")


def table_phys(cell):
    out = []
    for sc in SCEN_ORDER:
        for arm in ARM_ORDER:
            rs = cell[(sc, arm)]
            n = len(rs)
            bm, bc = _mean_ci([fnum(r, "burned_ha") for r in rs])
            fm, _ = _mean_ci([fnum(r, "forest_ha") for r in rs])
            # vpop is a density per km2; one 30 m cell is 9e-4 km2,
            # so the sum over burned cells converts to PERSONS here
            pm, _ = _mean_ci([fnum(r, "pop_affected") * 9e-4
                              for r in rs])
            em, _ = _mean_ci([fnum(r, "evacuated") for r in rs])
            outs = [fnum(r, "out_min") for r in rs
                    if fnum(r, "out_min") > 0]
            om = float(np.mean(outs)) if outs else float("nan")
            sm = 100.0 * np.mean([fnum(r, "success") for r in rs])
            out.append(dict(scenario=sc, arm=arm, n=n,
                            burned_ha=round(bm, 1),
                            burned_ci=round(bc, 1),
                            forest_ha=round(fm, 1),
                            pop_affected=round(pm, 0),
                            evacuated=round(em, 0),
                            out_min=("-" if math.isnan(om)
                                     else round(om, 0)),
                            success_pct=round(sm, 0)))
    return out


def table_cost(cell):
    out = []
    for sc in SCEN_ORDER:
        for arm in ARM_ORDER:
            rs = cell[(sc, arm)]
            row = dict(scenario=sc, arm=arm, n=len(rs))
            for k, _lab, _c in COST_TERMS:
                m, _ = _mean_ci([fnum(r, f"t6h_{k}") for r in rs])
                row[k] = "" if math.isnan(m) else round(m, 3)
            m, _ = _mean_ci([fnum(r, "t6h_j_total") for r in rs])
            row["j_total"] = "" if math.isnan(m) else round(m, 3)
            m, _ = _mean_ci([fnum(r, "t6h_j_phys") for r in rs])
            row["j_phys"] = "" if math.isnan(m) else round(m, 3)
            out.append(row)
    return out


def claim_chain(rows):
    def pool(arm, key="burned_ha"):
        return float(np.mean([fnum(r, key) for r in rows
                              if r["arm"] == arm]))
    b = {a: pool(a) for a in ARM_ORDER}
    jt = {a: pool(a, "end_j_total") for a in ARM_ORDER}
    n_per_cell = len([r for r in rows
                      if r["arm"] == "F5" and r["scenario"] == "S1"])
    return dict(
        n_per_cell=n_per_cell,
        burned_mean=b, j_total_mean=jt,
        x1_pct=round(100 * (b["Test0"] - b["F5"]) / b["Test0"], 0),
        x2_pct=round(100 * (b["F5"] - b["F5Ev"]) / b["F5"], 0),
        x3_pct=round(100 * (b["F5"] - b["F5AI"]) / b["F5"], 0),
        full_pct=round(100 * (b["F5"] - b["F5EvAI"]) / b["F5"], 0),
        d_pct=round(100 * (b["F5EvAI"] - b["F40"]) / b["F40"], 1),
        f22_worse_than_open=bool(b["F22"] > b["F5EvAI"]),
        jtotal_open_vs_test0=(round(jt["F5EvAI"], 3),
                              round(jt["Test0"], 3)))


def fig57(rows_curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ha = (30.0 ** 2) / 1e4
    series = defaultdict(lambda: defaultdict(list))
    for r in rows_curves:
        series[(r["scenario"], r["arm"])][float(r["t_min"])].append(
            float(r["burned_cells"]) * ha)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=False)
    for ax, sc in zip(axes, SCEN_ORDER):
        for arm in ARM_ORDER:
            pts = series[(sc, arm)]
            if not pts:
                continue
            ts = sorted(pts)
            mean = [np.mean(pts[t]) for t in ts]
            lo = [np.min(pts[t]) for t in ts]
            hi = [np.max(pts[t]) for t in ts]
            ax.plot([t / 60 for t in ts], mean, lw=1.6,
                    color=ARM_COLOR[arm], label=ARM_LABEL[arm])
            ax.fill_between([t / 60 for t in ts], lo, hi,
                            color=ARM_COLOR[arm], alpha=0.10)
        ax.set_title(sc, fontsize=11)
        ax.set_xlabel("hours")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("burned area (ha)")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig57_burned_curves.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def fig58(tcost):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
    for ax, sc in zip(axes, SCEN_ORDER):
        xs = np.arange(len(ARM_ORDER))
        bottom = np.zeros(len(ARM_ORDER))
        _w = _weights()
        for k, lab, col in COST_TERMS:
            vals = []
            for arm in ARM_ORDER:
                row = next(r for r in tcost
                           if r["scenario"] == sc and r["arm"] == arm)
                v = row[k]
                vals.append(_w[k] * float(v) if v != "" else 0.0)
            ax.bar(xs, vals, bottom=bottom, color=col, width=0.7,
                   label=lab)
            bottom += np.array(vals)
        ax.set_title(sc, fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER],
                           rotation=60, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("decision cost $J$ at 6 h\n(weighted term contributions)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig58_cost_decomposition.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


# ------------------- rediscovery + vocabulary ground truth
def _rule_cell(ants):
    return frozenset((str(v), str(t).replace(">=", ""))
                     for v, t in ants)


def _withheld():
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from dss.adapt import make_runtime_rules
    full = {r.name: r for r in make_runtime_rules("full")}
    mini = {r.name for r in make_runtime_rules("minimal")}
    with_ = {n: r for n, r in full.items() if n not in mini}
    return {n: (_rule_cell(r.antecedents),
                dict((c, float(v)) for c, v in r.consequents))
            for n, r in with_.items()}


def table_rediscovery():
    import json as js
    wh = _withheld()
    out = []
    prods = [js.loads(l) for l in open(os.path.join(
        OUT, "ladder_products.jsonl"), encoding="utf-8")]
    for arm in ("F5Ev", "F5AI", "F5EvAI"):
        learned = []
        for pr in prods:
            if pr["arm"] != arm:
                continue
            learned += pr["learned_rules"]
        n_learn = len(learned)
        matched = []
        hit_wh = set()
        for lr in learned:
            cell = _rule_cell(lr["antecedents"])
            for wn, (wcell, wcons) in wh.items():
                if cell == wcell:
                    matched.append((lr, wn, wcons))
                    hit_wh.add(wn)
                    break
        prec = len(matched) / n_learn if n_learn else 0.0
        rec = len(hit_wh) / max(len(wh), 1)
        errs = []
        for lr, wn, wcons in matched:
            lc = dict((c, float(v)) for c, v in lr["consequents"])
            common = set(lc) & set(wcons)
            for c in common:
                errs.append(abs(lc[c] - wcons[c]))
        cerr = (sum(errs) / len(errs)) if errs else float("nan")
        out.append(dict(arm=arm, learned=n_learn,
                        on_withheld=len(matched),
                        precision=round(prec, 2),
                        recall=round(rec, 2),
                        cons_err=("-" if errs == [] else
                                  round(cerr, 2)),
                        withheld_total=len(wh)))
    with open(os.path.join(OUT, "table512_rediscovery.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    return out


def _cos(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    den = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / den) if den > 1e-12 else 0.0


BACKBURN_T = {"containment_line": 0.9, "suppression_effort": 0.4}
EMBER_T = {"weather_severity": 0.40, "spread_potential": 0.35,
           "fuel_load": 0.25}


def table_vocab():
    import json as js
    prods = [js.loads(l) for l in open(os.path.join(
        OUT, "ladder_products.jsonl"), encoding="utf-8")]
    camp_macros = {}
    for pr in prods:
        for mn, spec in (pr.get("macros") or {}).items():
            comp = {}
            for d in (spec.get("composition") or []):
                if isinstance(d, dict):
                    comp[d["channel"]] = float(d["weight"])
                elif isinstance(d, (list, tuple)) and len(d) == 2:
                    comp[str(d[0])] = float(d[1])
            if comp:
                camp_macros[mn] = comp
    # interactive live-model store
    gpath = os.path.join(HERE, "..", "logs",
                         "dss_generated_state.json")
    live_macros, live_concepts = {}, {}
    try:
        gg = js.load(open(gpath, encoding="utf-8"))
        for m in gg.get("genai_interventions") or []:
            live_macros[m["name"]] = {d["channel"]: float(d["weight"])
                                      for d in m.get("composition")
                                      or []}
        for c in gg.get("genai_concepts") or []:
            live_concepts[c["name"]] = {d["name"]: float(d["weight"])
                                        for d in c.get("inputs") or []}
    except Exception:
        pass
    rows = []
    best_m = max(list(camp_macros.items()) or [("-", {})],
                 key=lambda kv: _cos(kv[1], BACKBURN_T))
    best_lm = max(list(live_macros.items()) or [("-", {})],
                  key=lambda kv: _cos(kv[1], BACKBURN_T))
    rows.append(dict(
        target="macro_backburn",
        campaign_props=len(camp_macros),
        campaign_best=best_m[0],
        campaign_cos=round(_cos(best_m[1], BACKBURN_T), 2),
        live_props=len(live_macros),
        live_best=best_lm[0],
        live_cos=round(_cos(best_lm[1], BACKBURN_T), 2),
        rediscovered=("yes (live)" if _cos(best_lm[1],
                                           BACKBURN_T) >= 0.9
                      else "no")))
    best_lc = max(list(live_concepts.items()) or [("-", {})],
                  key=lambda kv: _cos(kv[1], EMBER_T))
    rows.append(dict(
        target="concept_ember_exposure",
        campaign_props=0, campaign_best="-", campaign_cos=0.0,
        live_props=len(live_concepts),
        live_best=best_lc[0],
        live_cos=round(_cos(best_lc[1], EMBER_T), 2),
        rediscovered="no (a different gap was named)"))
    with open(os.path.join(OUT, "table513_vocab.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


def fig59():
    """Engagement timeline of the representative S4 run."""
    import json as js
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p_in = os.path.join(OUT, "rep_cycles_F5EvAI.json")
    cyc = js.load(open(p_in, encoding="utf-8"))
    ha = (30.0 ** 2) / 1e4
    t = [c["t_min"] / 60 for c in cyc]
    burn = [c["burning"] * ha if c["burning"] is not None else 0
            for c in cyc]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, burn, color="#7f8c8d", lw=1.6,
            label="actively burning (ha)")
    mk = {1: ("o", "#27ae60", "stage 1 tuning"),
          2: ("s", "#2980b9", "stage 2 resolution"),
          3: ("^", "#c0392b", "stage 3 generative")}
    seen = set()
    for c in cyc:
        st_ = c.get("tried") or 0
        if st_ not in mk:
            continue
        m, col, lab = mk[st_]
        acc = bool(c.get("accepted"))
        ax.plot(c["t_min"] / 60,
                (c["burning"] or 0) * ha,
                marker=m, ms=8 if acc else 6,
                mfc=(col if acc else "none"), mec=col,
                ls="none",
                label=(lab + " (filled = accepted)"
                       if st_ not in seen else None))
        seen.add(st_)
    ax.set_xlabel("hours")
    ax.set_ylabel("actively burning (ha)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig59_engagement_timeline.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def fig510():
    """Tuned-consequent trajectory (live store lineage) + realized
    cumulative gain per stage over the campaign."""
    import json as js
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gpath = os.path.join(HERE, "..", "logs",
                         "dss_generated_state.json")
    gg = js.load(open(gpath, encoding="utf-8"))
    traj = []
    for m in gg.get("evfis_rule_modifications") or []:
        if m.get("base_rule_id") != "A1":
            continue
        aft = dict((c, v) for c, v in
                   (m.get("after", {}).get("consequents") or []))
        if "evacuation" in aft:
            traj.append(float(aft["evacuation"]))
    runs = list(csv.DictReader(open(os.path.join(
        OUT, "ladder_runs.csv"))))
    def dsum(arm, key):
        return sum(float(r[key] or 0) for r in runs
                   if r["arm"] == arm)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.2))
    a1.plot(range(1, len(traj) + 1), traj, "-o", ms=4,
            color="#27ae60")
    a1.set_xlabel("accepted tuning step (rule A1)")
    a1.set_ylabel("evacuation consequent")
    a1.grid(alpha=0.25)
    a1.set_title("kept consequent trajectory (live lineage)",
                 fontsize=10)
    arms = ["F5Ev", "F5AI", "F5EvAI"]
    x = np.arange(len(arms))
    for i, (key, lab, col) in enumerate(
            (("dj_1", "stage 1", "#27ae60"),
             ("dj_2", "stage 2", "#2980b9"),
             ("dj_3", "stage 3", "#c0392b"))):
        a2.bar(x + (i - 1) * 0.25,
               [-dsum(a, key) for a in arms], 0.25,
               color=col, label=lab)
    a2.set_xticks(x)
    a2.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=8)
    a2.set_ylabel("summed realized forecast gain (−ΔĴ)")
    a2.grid(axis="y", alpha=0.25)
    a2.legend(fontsize=8)
    a2.set_title("realized gain per stage over the campaign",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig510_evfis_mechanism.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rows, cell = load_runs()
    tp = table_phys(cell)
    tc = table_cost(cell)
    with open(os.path.join(OUT, "table58_phys.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tp[0]))
        w.writeheader()
        w.writerows(tp)
    with open(os.path.join(OUT, "table59_cost.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tc[0]))
        w.writeheader()
        w.writerows(tc)
    cc = claim_chain(rows)
    with open(os.path.join(OUT, "claim_chain.json"), "w",
              encoding="utf-8") as f:
        json.dump(cc, f, indent=1)
    curves = list(csv.DictReader(open(os.path.join(
        OUT, "ladder_curves.csv"))))
    p1 = fig57(curves)
    p2 = fig58(tc)
    print("rediscovery:", table_rediscovery())
    print("vocab:", table_vocab())
    print("fig59:", fig59())
    print("fig510:", fig510())
    print("tables + claim chain written; figures:", p1, p2)
    print(json.dumps({k: v for k, v in cc.items()
                      if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
