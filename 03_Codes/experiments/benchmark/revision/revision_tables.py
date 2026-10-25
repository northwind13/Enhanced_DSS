# -*- coding: utf-8 -*-
r"""Summary of the revision runs K1 to K4 against the article campaign.

Paired statistics on the same worlds (scenario, seed): mean relative change
of burned area, 95% half-width, worlds better / tied / worse, share
extinguished, mean affected population (persons, see note).

NOTE on population: runs.csv stores pop_affected as the SUM OF DENSITY
(persons per km2) over burned cells; persons = pop_affected x cell area in
km2 (0.0009 at 30 m). Table V of the article printed the density sum under
"affected population (x1000)"; the figures here are persons.

  python experiments\benchmark\revision\revision_tables.py
"""
import csv, glob, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
FINAL = os.path.join(BENCH, "final")
CELL_KM2 = 0.0009
OUT = os.path.join(HERE, "summary.md")


def load(path, **flt):
    """path may hold a glob (several windows wrote several folders)."""
    rows = {}
    for pth in sorted(glob.glob(path)):
        with open(pth, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if all(str(r.get(k, "")) == str(v) for k, v in flt.items()):
                    rows[(r["scenario"], int(r["seed"]))] = r
    return rows


def paired(a, b, key="burned_ha"):
    """b against a on shared worlds: mean relative change of the mean,
    95% half-width of the mean difference relative to mean(a), counts."""
    ks = sorted(set(a) & set(b))
    if not ks:
        return None
    da = [float(a[k][key]) for k in ks]
    db = [float(b[k][key]) for k in ks]
    d = [y - x for x, y in zip(da, db)]
    n = len(d)
    ma = sum(da) / n
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / max(n - 1, 1))
    hw = 1.96 * sd / math.sqrt(n)
    better = sum(1 for x in d if x < -1e-9)
    tie = sum(1 for x in d if abs(x) <= 1e-9)
    worse = n - better - tie
    return dict(n=n, mean_a=ma, mean_b=sum(db) / n,
                rel=100.0 * md / ma if ma else float("nan"),
                hw=100.0 * hw / ma if ma else float("nan"),
                better=better, tie=tie, worse=worse)


def succ(rows):
    ks = list(rows)
    return 100.0 * sum(float(rows[k]["success"]) for k in ks) / len(ks) \
        if ks else float("nan")


def pop(rows):
    ks = list(rows)
    return CELL_KM2 * sum(float(rows[k]["pop_affected"]) for k in ks) / len(ks) \
        if ks else float("nan")


def line(name, ref, rows, L):
    p = paired(ref, rows)
    if p is None:
        L.append(f"| {name} | (no runs yet) | | | | |")
        return
    L.append(f"| {name} | {p['mean_b']:.1f} | {p['rel']:+.0f} ± {p['hw']:.0f}% "
             f"| {p['better']}/{p['tie']}/{p['worse']} | {succ(rows):.0f}% "
             f"| {pop(rows):.0f} |")


def main():
    closed = load(os.path.join(FINAL, "t1_reference", "runs.csv"), arm="F5Ev")
    none_ = load(os.path.join(FINAL, "t1_reference", "runs.csv"), arm="T0")
    voda = load(os.path.join(FINAL, "t6_voda", "runs.csv"), arm="LLMScreen")
    L = ["# Revision runs against the article campaign", ""]
    L.append(f"Reference: closed layer F5Ev, {len(closed)} worlds, mean burned "
             f"{sum(float(r['burned_ha']) for r in closed.values())/max(len(closed),1):.1f} ha, "
             f"extinguished {succ(closed):.0f}%, affected {pop(closed):.0f} persons.")
    L.append("")
    hdr = ["| Configuration | Burned (ha) | vs closed layer | better/tie/worse | Extinguished | Affected (persons) |",
           "|---|---|---|---|---|---|"]

    # K1
    L += ["## K1 Strong closed reference (whole written doctrine, 40 rules)", ""] + hdr
    line("No decision support", closed, none_, L)
    line("Closed layer, 5 seed rules (article)", closed, closed, L)
    f40 = load(os.path.join(FINAL, "t13_full_doctrine", "runs.csv"), arm="F40Ev")
    line("Closed layer, 40 rules (K1)", closed, f40, L)
    line("VODA (article)", closed, voda, L)
    p = paired(f40, voda)
    if p:
        L += ["", f"VODA against the 40-rule closed layer: {p['rel']:+.0f} ± {p['hw']:.0f}%, "
              f"better in {p['better']} of {p['n']} worlds."]
    L.append("")

    # K2
    L += ["## K2 Discovered tactics given to the closed layer in advance", ""] + hdr
    k2 = os.path.join(HERE, "k2_predefined*", "runs.csv")
    canon = load(k2, mode="canon")
    own = load(k2, mode="own")
    line("Closed + 4 canonical tactics at minute 0 (K2 canon)", closed, canon, L)
    line("Closed + own admitted objects at minute 0 (K2 own)", closed, own, L)
    line("VODA (article)", closed, voda, L)
    k2b = load(os.path.join(HERE, "k2b_generalization", "runs.csv"), mode="canon")
    if k2b:
        ks = set(k2b)
        cl = {k: v for k, v in closed.items() if k in ks}; vo = {k: v for k, v in voda.items() if k in ks}
        L += ["", "### K2b Tactics derived from seeds 201 to 215, frozen, evaluated on seeds 216 to 230 (unseen worlds)", ""] + hdr
        line("Closed layer (same 75 worlds)", cl, cl, L)
        line("Closed + frozen tactics from the other 75 worlds (K2b)", cl, k2b, L)
        line("VODA (same 75 worlds)", cl, vo, L)
        p = paired(k2b, vo)
        if p:
            L.append(f"\nVODA against the frozen library on unseen worlds: {p['rel']:+.0f} ± {p['hw']:.0f}%, better in {p['better']} of {p['n']} worlds.")
    for nm, rows in (("canonical tactics", canon), ("own objects", own)):
        p = paired(rows, voda)
        if p:
            L.append(f"\nVODA against closed + {nm}: {p['rel']:+.0f} ± {p['hw']:.0f}%, "
                     f"better in {p['better']} of {p['n']} worlds.")
    L.append("")

    # K3
    L += ["## K3 Planning on the nominal model, evaluated under model error", ""]
    k3 = os.path.join(HERE, "k3_model_error*", "runs.csv")
    L += ["| Perturbation of the live world | Closed (ha) | VODA replay (ha) | VODA vs closed | better/tie/worse | Extinguished closed / VODA |",
          "|---|---|---|---|---|---|"]
    loo = load(os.path.join(FINAL, "t10_benefit", "loo.csv"), removed="none")
    # nominal row: closed layer and the T10 replay reference (worlds with admissions,
    # the others are the closed layer itself)
    ref_voda = dict(closed); ref_voda.update(loo)
    p = paired(closed, ref_voda)
    if p:
        L.append(f"| none (nominal, article) | {p['mean_a']:.1f} | {p['mean_b']:.1f} | "
                 f"{p['rel']:+.0f} ± {p['hw']:.0f}% | {p['better']}/{p['tie']}/{p['worse']} "
                 f"| {succ(closed):.0f}% / {succ(ref_voda):.0f}% |")
    for name in ("gammaI_x1.5", "betat_x1.5", "knock_x1.5", "gammaI_x0.67"):
        c = load(k3, perturb=name, arm="closed")
        v = load(k3, perturb=name, arm="voda")
        p = paired(c, v)
        if p is None:
            if c or v:
                L.append(f"| {name} | (partial: {len(c)} closed, {len(v)} voda) | | | | |")
            continue
        L.append(f"| {name} | {p['mean_a']:.1f} | {p['mean_b']:.1f} | "
                 f"{p['rel']:+.0f} ± {p['hw']:.0f}% | {p['better']}/{p['tie']}/{p['worse']} "
                 f"| {succ(c):.0f}% / {succ(v):.0f}% |")
    L.append("")

    # K6
    L += ["## K6 Coating weaker in the live world, planning on the nominal coating", ""]
    k6 = os.path.join(HERE, "k6_coating*", "runs.csv")
    voda_live = load(os.path.join(FINAL, "t6_voda", "runs.csv"), exp="E9", arm="LLMScreen")
    L += ["| Live-world coating | Closed (ha) | Replay (ha) | Replay vs closed | b/t/w | Live VODA (ha) | Live vs closed | b/t/w | Coated ha closed / replay / live |",
          "|---|---|---|---|---|---|---|---|---|"]
    def _coat(rows):
        ks = list(rows)
        try:
            return sum(float(rows[k].get("coated_ha") or 0) for k in ks) / len(ks) if ks else float("nan")
        except ValueError:
            return float("nan")
    p1 = paired(closed, ref_voda); p2 = paired(closed, voda_live)
    if p1 and p2:
        L.append(f"| nominal (article) | {p1['mean_a']:.1f} | {p1['mean_b']:.1f} | {p1['rel']:+.0f} ± {p1['hw']:.0f}% | {p1['better']}/{p1['tie']}/{p1['worse']} "
                 f"| {p2['mean_b']:.1f} | {p2['rel']:+.0f} ± {p2['hw']:.0f}% | {p2['better']}/{p2['tie']}/{p2['worse']} | {_coat(closed):.1f} / {_coat(ref_voda):.1f} / {_coat(voda_live):.1f} |")
    for name in ("coat_half", "coat_off", "coat_price_x2", "coat_stock_20ha", "coat_delay_12min"):
        c = load(k6, perturb=name, arm="closed")
        r = load(k6, perturb=name, arm="replay")
        v = load(k6, perturb=name, arm="live")
        p1 = paired(c, r); p2 = paired(c, v)
        if p1 is None and p2 is None:
            if c or r or v:
                L.append(f"| {name} | (partial: {len(c)} closed, {len(r)} replay, {len(v)} live) | | | | | | | |")
            continue
        f1 = (f"{p1['mean_b']:.1f} | {p1['rel']:+.0f} ± {p1['hw']:.0f}% | {p1['better']}/{p1['tie']}/{p1['worse']}"
              if p1 else f"({len(r)} runs) | |")
        f2 = (f"{p2['mean_b']:.1f} | {p2['rel']:+.0f} ± {p2['hw']:.0f}% | {p2['better']}/{p2['tie']}/{p2['worse']}"
              if p2 else f"({len(v)} runs) | |")
        L.append(f"| {name} | {(p1 or p2)['mean_a']:.1f} | {f1} | {f2} | {_coat(c):.1f} / {_coat(r):.1f} / {_coat(v):.1f} |")
    L.append("")

    # K7
    L += ["## K7 Wind different in the live world, planning on the nominal wind", ""]
    k7 = os.path.join(HERE, "k7_spread*", "runs.csv")
    L += ["| Live-world wind | Closed (ha) | Replay (ha) | Replay vs closed | b/t/w | Live VODA (ha) | Live vs closed | b/t/w | Ext. closed / replay / live |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name in ("wind_x1.5", "wind_rot_p30", "wind_rot_m30"):
        c = load(k7, perturb=name, arm="closed"); r = load(k7, perturb=name, arm="replay"); v = load(k7, perturb=name, arm="live")
        p1 = paired(c, r); p2 = paired(c, v)
        if p1 is None and p2 is None:
            if c or r or v: L.append(f"| {name} | (partial: {len(c)} closed, {len(r)} replay, {len(v)} live) | | | | | | | |")
            continue
        f1 = (f"{p1['mean_b']:.1f} | {p1['rel']:+.0f} ± {p1['hw']:.0f}% | {p1['better']}/{p1['tie']}/{p1['worse']}" if p1 else f"({len(r)} runs) | |")
        f2 = (f"{p2['mean_b']:.1f} | {p2['rel']:+.0f} ± {p2['hw']:.0f}% | {p2['better']}/{p2['tie']}/{p2['worse']}" if p2 else f"({len(v)} runs) | |")
        L.append(f"| {name} | {(p1 or p2)['mean_a']:.1f} | {f1} | {f2} | {succ(c):.0f}% / {succ(r):.0f}% / {succ(v):.0f}% |")
    L.append("")
    # K8
    L += ["## K8 Gates on independent futures (seeds 303, 404), VODA run anew", ""]
    k8 = load(os.path.join(HERE, "k8_gate_seeds", "runs.csv"), exp="E9", arm="LLMScreen")
    p = paired(closed, k8); pv = paired(voda_live, k8)
    if p:
        L.append(f"VODA with independent gate futures: {p['mean_b']:.1f} ha, {p['rel']:+.0f} ± {p['hw']:.0f}% against the closed layer, {p['better']}/{p['tie']}/{p['worse']}, extinguished {succ(k8):.0f}%; against nominal VODA {pv['rel']:+.0f} ± {pv['hw']:.0f}%, {pv['better']}/{pv['tie']}/{pv['worse']}")
        try:
            import json as _json
            n_att = n_adm = 0
            pj = os.path.join(HERE, "k8_gate_seeds", "proposals.jsonl")
            if os.path.exists(pj):
                for ln in open(pj, encoding="utf-8"):
                    try: o = _json.loads(ln)
                    except Exception: continue
                    if str(o.get("gate", "")).startswith("S0"): continue
                    n_att += 1; n_adm += int(bool(o.get("accepted")))
            L.append(f"candidates handed to the gates {n_att}, admitted {n_adm}")
        except Exception as exc:
            L.append(f"(proposal log not read: {exc})")
    else:
        L.append(f"no K8 runs yet ({len(k8)} rows)")
    L.append("")

    # K4
    L += ["## K4 Hindcast, 50 replicates, stated stopping rule", ""]
    for proto, stop in (("window12", "area"), ("thesis", "area"),
                        ("clean", "area"), ("clean", "none")):
        pths = sorted(glob.glob(os.path.join(HERE, "k4_hindcast",
                                             f"{proto}_stop_{stop}",
                                             "hindcast_errors_*.csv")))
        if not pths:
            L.append(f"protocol {proto}, stop rule {stop}: no run yet")
            continue
        rows = [r for pth in pths
                for r in csv.DictReader(open(pth, encoding="utf-8"))]
        rows.sort(key=lambda r: r["fire"])
        L += [f"Protocol {proto}, stop rule {stop}", "",
              "| Fire | Wind offset | Footprint (h) | Reached in (h) | POD | Front mean / p90 (m) | Arrival order rho | Arrival MAE (h) | Area sim/obs |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['fire']} | {float(r.get('wind_offset_deg', 0)):+.0f} | {float(r['duration_h']):.0f} | {r['window_h']} | {r['pod']} ± {r['pod_sd']} "
                     f"| {r['front_mean_m']} ± {r['front_mean_sd']} / {r['front_p90_m']} | {r['arrival_rho']} "
                     f"| {r['arrival_mae_h']} ± {r['arrival_mae_sd']} | {r['area_ratio_sim_obs']} |")
        L.append("")
    txt = "\n".join(L)
    open(OUT, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
