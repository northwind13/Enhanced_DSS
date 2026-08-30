"""Chapter 5 Monte Carlo campaign on the THESIS scenario grid.

Arms (thesis 5.4.2, Table 5.6):
  Test0    no DSS: the cost of inaction              (T_0)
  F5       five seed rules, static (adaptation off)  (T_F5)
  F5Ev     F5 + the evolving-fuzzy stages            (T_F5+Ev)
  F5EvAI   F5 + full staged adaptation (evolving stages + gated
           generative stage together)                (T_DisasterAware)

Scenarios (thesis Table 5.7):
  S1  remote forest (~2 settlements), sufficient pool, 1 remote ignition
  S2  remote forest, capacity-limited pool (x0.6), 2 remote ignitions
  S3  WUI (~12 settlements, high asset density), sufficient, 1 near town
  S4  WUI, capacity-limited, 2 near towns
  S5  S4 under degraded observation (outages + noise)

Weather is critical fire weather in every scenario (dry fuel, strong
wind), so the scenario axes are the only controlled difference. Every
(scenario, seed) pair generates ONE world; every arm replays the
identical world, which is what makes the paired comparison legal.

Resumable: finished (scenario, arm, seed) rows are skipped on restart.

  python experiments/campaign5.py --seeds 50 --workers 8   # full
  python experiments/campaign5.py --seeds 3                # pilot
  python experiments/campaign5.py --budget-s 40            # one slice

Outputs in experiments/out/:
  ladder_runs.csv    one row per run (all cost terms + mechanism)
  ladder_curves.csv  burned-area / J_phys time series (Figure 5.7)
  ladder_funnel.csv  stage-3 gate funnel events (Table 5.11)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import dss                                            # noqa: E402
from dss import loop as dss_loop                      # noqa: E402
from disaster_phyengine import terrain                # noqa: E402
from disaster_phyengine.config import SimConfig       # noqa: E402
from disaster_phyengine.core import Simulator         # noqa: E402
from disaster_phyengine.costs import compute_costs    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CHECKPOINTS_MIN = (120.0, 360.0)
# Episodes run to extinction or 12 h (thesis 5.4: costs are READ at
# the 6 h checkpoint; success and time to extinction over the full
# episode)
MAX_HOURS = 12.0
COVER_THR = 0.45
# quality gate of the decision engine (graduated fail-safe Q/eta).
# Read from the environment so multiprocessing workers, which import
# this module afresh under spawn, see the SAME value the main process
# was started with; --eta sets it for both.
ETA = float(os.environ.get("DSS_ETA", "0.60"))

ARMS = {
    "Test0":  ("minimal", None),
    # the STATIC control: the same five-rule seed base with no adaptation.
    # Without it the campaign has nothing to attribute the growth to.
    "F5":     ("minimal", dict(adapt_on=False)),
    # evolving stages WITHOUT the generative stage: the evFIS rung of
    # the four-configuration ladder (thesis Table 5.6)
    "F5Ev":   ("minimal", dict(adapt_on=True, evfis_on=True,
                               genai_on=False)),
    # full staged adaptation ON TOP of the static control: evolving
    # stages and the gated generative stage together
    "F5EvAI": ("minimal", dict(adapt_on=True, evfis_on=True,
                               genai_on=True)),
    # the upper reference: the whole written doctrine, run statically. The
    # 22-rule "core" block is retired - a middle setting nobody used - so
    # the campaign runs between the five seeds and the forty.
    "F40":    ("full",    dict(adapt_on=False)),
}
# WHAT A PLAIN RUN COVERS. The doctrine arm is defined above and can
# still be asked for by name, but it is not run by default: the
# thesis tables and figures no longer report it, so a fifth of the
# campaign's time was being spent on numbers nobody reads.
DEFAULT_ARMS = ("Test0", "F5", "F5Ev", "F5EvAI")
SCENARIOS = ("S1", "S2", "S3", "S4", "S5")
LIMITED = {"S2", "S4", "S5"}          # pool x0.6
WUI = {"S3", "S4", "S5"}


def build_world(seed: int, scenario: str):
    cfg = SimConfig(nx=72, ny=50, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    if scenario in WUI:
        w = terrain.generate_landscape(
            cfg, seed=seed, preset="Rolling hills", n_settlements=12,
            population_per_settlement=8000)
    else:
        w = terrain.generate_landscape(
            cfg, seed=seed, preset="Mountain forest", n_settlements=2,
            population_per_settlement=4000)
    # critical fire weather everywhere (thesis 5.4.1)
    w.fuel.fmoist[:] = 0.08
    w.meteo.wws[:] = 8.0
    return w


def pick_ignitions(w, base, scenario: str):
    """Forest scenarios ignite REMOTE fuel (longest travel time); WUI
    scenarios ignite NEAR the interface (burnable wildland within a
    short walk of the built-up edge, where the wind carries the fire
    toward people)."""
    ok = (w.fuel.fload > 0.4) & (w.fuel.ftype >= 1) \
        & (w.fuel.ftype <= 4)
    ys, xs = np.where(ok)
    n_ign = 2 if scenario in LIMITED else 1
    if scenario not in WUI:
        order = np.argsort(-base.rtime[ys, xs])
    else:
        urban = (w.fuel.ftype == 6)
        if urban.any():
            uy, ux = np.where(urban)
            d2 = ((xs[:, None] - ux[None, :]) ** 2
                  + (ys[:, None] - uy[None, :]) ** 2).min(axis=1)
            near = (d2 >= 4 ** 2) & (d2 <= 12 ** 2)
            if near.any():
                ys, xs = ys[near], xs[near]
            order = np.argsort(-base.rtime[ys, xs])
        else:
            order = np.argsort(-base.rtime[ys, xs])
    spots = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all((x - a) ** 2 + (y - b) ** 2 > 20 ** 2 for a, b in spots):
            spots.append((x, y))
        if len(spots) == n_ign:
            break
    return spots or [(int(xs[0]), int(ys[0]))]


# --------------------------------------- degraded observation (S5)
_ORIG_FEATURES = dss_loop.ten_features
_ORIG_CONF = dss_loop.feature_confidence


def _install_outage(rng, outage=0.25, noise=0.10):
    def feats(sim, r, network=None, pool=None):
        f = _ORIG_FEATURES(sim, r, network=network, pool=pool)
        if rng.random() < outage:
            return {k: float(np.clip(v + rng.normal(0.0, noise), 0, 1))
                    for k, v in f.items()}
        return f

    def conf(network, r):
        c = _ORIG_CONF(network, r)
        return {k: float(v) * (1.0 - outage * 0.8)
                for k, v in c.items()}

    dss_loop.ten_features = feats
    dss_loop.feature_confidence = conf


def _remove_outage():
    dss_loop.ten_features = _ORIG_FEATURES
    dss_loop.feature_confidence = _ORIG_CONF


# ------------------------------------ offline stage-3 proposer
def _make_template_proposer(seed: int):
    """Deterministic stand-in for the live model in stage 3.

    The campaign must run offline; what is under test is not the
    language model but the GATE CHAIN, so the proposer only has to
    behave like a competent officer: read the CURRENT dominant terms
    from the situation brief, aim a strong rule at them, and react to
    a named rejection the way the revision protocol asks. Every
    proposal still has to survive G1-G5 exactly like a live one, and
    the funnel records the source as 'template'."""
    state = {"n": 0}

    def _dom(text):
        import re
        m = re.search(r"Current dominant terms: ([^\n]+)", text)
        out = {}
        if m:
            for kv in m.group(1).split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    out[k.strip()] = v.strip()
        return out

    def propose(situation, timeout=None, engine=None, mission=""):
        state["n"] += 1
        d = _dom(situation)
        thr = d.get("fire_threat_level", "M")
        aer = d.get("asset_exposure_risk", "M")
        feas = d.get("suppression_feasibility", "M")
        urg = d.get("intervention_urgency", "M")
        # POOL SATURATED (brief line or a G2d rejection): a competent
        # officer stops ordering more physical work the budget cannot
        # fund. What still helps is sustained capacity from water when
        # the map carries it, otherwise the non-spending channels.
        if ("POOL SATURATED" in situation or "G2d" in situation):
            ants = [["fire_threat_level",
                     (">=" + thr) if thr in ("L", "M", "H") else thr],
                    ["suppression_feasibility", feas]]
            if "a lake/sea lies" in situation:
                return {"antecedents": ants,
                        "consequents": [["water_drafting", 0.9]],
                        "note": "template: pool saturated; raise "
                                "sustained capacity from the water "
                                "body instead of re-dividing the "
                                "budget"}
            return {"antecedents": [["asset_exposure_risk",
                                     (">=" + aer) if aer in
                                     ("L", "M", "H") else aer],
                                    ["intervention_urgency", urg]],
                    "consequents": [["public_warning", 1.0],
                                    ["evacuation", 0.7]],
                    "note": "template: pool saturated and no water "
                            "body; spend nothing, protect people"}
        dup = ("duplicate cell" in situation
               or "G2 duplicate" in situation)
        pkg = dup or (state["n"] % 3 == 0)
        if pkg:
            # a composite package: defense as ONE act (the escape the
            # duplicate-cell gate names)
            return {
                "antecedents": [["asset_exposure_risk", aer],
                                ["fire_threat_level", ">=" + thr
                                 if thr in ("L", "M", "H") else thr]],
                "consequents": [["town_shield", 1.0]],
                "new_intervention": {
                    "name": "town_shield",
                    "composition": [["asset_protection", 1.0],
                                    ["containment_line", 0.8]]},
                "note": "template: defense composite for the exposed "
                        "settlement edge"}
        if feas in ("VL", "L"):
            cons = [["containment_line", 0.9],
                    ["resource_deployment", 0.8],
                    ["asset_protection", 0.7]]
        elif aer in ("H", "VH"):
            cons = [["asset_protection", 0.9],
                    ["evacuation", 0.8],
                    ["containment_line", 0.7]]
        else:
            cons = [["suppression_effort", 0.95],
                    ["resource_deployment", 0.85]]
        ants = [["fire_threat_level",
                 (">=" + thr) if thr in ("L", "M", "H") else thr],
                ["intervention_urgency", urg]]
        return {"antecedents": ants, "consequents": cons,
                "note": "template: strongest sensible answer to the "
                        "present dominant terms"}

    return propose


# ----------------------------------------------------------- one run
def run_once(scenario: str, arm: str, seed: int):
    profile, ekw = ARMS[arm]
    w = build_world(seed, scenario)
    base, _ = dss.resource_suggestion(w)
    if scenario in LIMITED:
        base.rcap *= 0.6
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    for x, y in pick_ignitions(w, base, scenario):
        w.add_ignition(x, y, step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    rng = np.random.default_rng(10_000 + seed)
    if scenario == "S5":
        _install_outage(rng)
    from dss import adapt as _adapt
    _orig_prop = _adapt._genai_propose
    _use_tmpl = ekw is not None and ekw.get("genai_on")
    if _use_tmpl:
        _adapt._genai_propose = _make_template_proposer(seed)
    eng = None
    if ekw is not None:
        eng = dss.DecisionEngine(
            dss.partition_n(w.config.nx, w.config.ny, 4),
            base_pool=base, seed_profile=profile,
            # ONE STORE PER RUN. The generated-knowledge store is the
            # DSS's memory, and every arm of this campaign used to share
            # the field file: a run inherited what the previous run had
            # learned, so the arms were not independent. See
            # dss.isolated_store_path.
            state_path=dss.isolated_store_path("campaign5"),
            cycle_min=12.0, horizon_min=24.0, eta=ETA, **ekw)
        # campaign parameter: adaptation windows every 24 min, not
        # every cycle. Documented with the results; also what keeps a
        # 6 h run from spending most of its wall time on trial
        # forecasts instead of on the fire.
        eng.adapt_cooldown_min = max(
            24.0, float(getattr(eng, "adapt_cooldown_min", 5.0)))
    max_steps = int(round(MAX_HOURS * 60.0 / w.config.step_minutes))
    out_at = None
    cp = {}
    curves = []
    try:
        for i in range(max_steps):
            ov = eng.maybe_decide(sim) if eng is not None else None
            sim.step(resource_override=ov)
            t_min = (i + 1) * w.config.step_minutes
            if i % 6 == 0:
                rep_i = compute_costs(sim)
                curves.append([scenario, arm, seed, round(t_min, 1),
                               int(sim.ever_burned.sum()),
                               round(float(getattr(rep_i, "j_physical",
                                                   rep_i.j_total)), 5)])
            for c in CHECKPOINTS_MIN:
                if c not in cp and t_min >= c:
                    cp[c] = compute_costs(sim)
            if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
                out_at = t_min
                break
    finally:
        if scenario == "S5":
            _remove_outage()
        _adapt._genai_propose = _orig_prop
    rep = compute_costs(sim)

    def costrow(r, tag):
        if r is None:
            return {f"{tag}_{k}": "" for k in
                    ("j_burn", "j_asset", "j_pop", "j_resp", "j_delay",
                     "j_total", "j_phys")}
        return {f"{tag}_j_burn": round(float(r.j_burn), 5),
                f"{tag}_j_asset": round(float(r.j_asset), 5),
                f"{tag}_j_pop": round(float(r.j_pop), 5),
                f"{tag}_j_resp": round(float(r.j_resp), 5),
                f"{tag}_j_delay": round(float(getattr(
                    r, "j_delay", float("nan"))), 5),
                f"{tag}_j_total": round(float(r.j_total), 5),
                f"{tag}_j_phys": round(float(getattr(
                    r, "j_physical", float("nan"))), 5)}

    # forest burned = burned cells that were forest classes (3, 4)
    ft0 = np.asarray(w.fuel.ftype)
    forest_burn = int((sim.ever_burned
                       & ((ft0 == 3) | (ft0 == 4))).sum())
    ha = (w.config.cell_size_m ** 2) / 1e4
    # population affected = people whose HOME cell burned (of the
    # initial density map), evacuated = actually moved out
    vp0 = np.asarray(getattr(sim, "_vpop0", w.value.vpop))
    pop_aff = float(vp0[sim.ever_burned].sum())
    row = dict(scenario=scenario, arm=arm, seed=seed, profile=profile,
               burned_ha=round(int(sim.ever_burned.sum()) * ha, 2),
               forest_ha=round(forest_burn * ha, 2),
               pop_affected=round(pop_aff, 0),
               evacuated=round(float(sim.population_evacuated), 0),
               out_min=(out_at if out_at is not None else -1),
               success=int(out_at is not None))
    row.update(costrow(cp.get(CHECKPOINTS_MIN[0]), "t2h"))
    row.update(costrow(cp.get(CHECKPOINTS_MIN[1]), "t6h"))
    row.update(costrow(rep, "end"))

    tried = {1: 0, 2: 0, 3: 0}
    acc = {1: 0, 2: 0, 3: 0}
    dj = {1: 0.0, 2: 0.0, 3: 0.0}
    fs_hits = fs_all = cov_hits = cov_all = 0
    share_num = share_den = 0.0
    genai_source = ""
    funnel = []
    if eng is not None:
        for c in eng.cycles:
            ad = c.get("adaptation") or {}
            st = int(ad.get("tried") or 0)
            if st in tried:
                tried[st] += 1
                if ad.get("accepted"):
                    acc[st] += 1
                    dj[st] += float(ad.get("dJ") or 0.0)
                info = ad.get("info") or {}
                if st == 3:
                    genai_source = ("template" if _use_tmpl
                                    else info.get("source",
                                                  genai_source))
                    funnel.append([scenario, arm, seed,
                                   ("template" if _use_tmpl
                                    else info.get("source", "")),
                                   bool(info.get("package")),
                                   info.get("gate", ""),
                                   bool(ad.get("accepted")),
                                   round(float(ad.get("dJ") or 0.0), 5)])
            for rd in (c.get("regions") or {}).values():
                fs_all += 1
                fs_hits += 1 if rd.get("failsafe") else 0
                fired = rd.get("fired") or []
                wmax = max((fw for _n, fw in fired), default=0.0)
                cov_all += 1
                cov_hits += 1 if wmax >= COVER_THR else 0
                for name, fw in fired:
                    share_den += fw
                    if name and name[0] in ("A", "G"):
                        share_num += fw
    products = None
    if eng is not None and (ekw or {}).get("adapt_on"):
        products = dict(
            scenario=scenario, arm=arm, seed=seed,
            learned_rules=[dict(name=r.name,
                                antecedents=[list(a) for a in
                                             r.antecedents],
                                consequents=[[c0, float(v)] for c0, v
                                             in r.consequents])
                           for r in eng.rules
                           if r.active and r.name[:1] in ("A", "G")],
            macros={m: dict(spec) for m, spec in
                    (eng.macros or {}).items()})
    rep_cycles = None
    if (eng is not None and scenario == "S4" and seed == 101
            and arm in ("F5Ev", "F5EvAI")):
        rep_cycles = [dict(step=c.get("step"),
                           t_min=c.get("t_min"),
                           burning=(c.get("sim") or {}).get("burning"),
                           tried=(c.get("adaptation") or {}).get(
                               "tried"),
                           accepted=(c.get("adaptation") or {}).get(
                               "accepted"),
                           dJ=(c.get("adaptation") or {}).get("dJ"))
                      for c in eng.cycles]
    row.update(rules_final=(len([r for r in eng.rules if r.active])
                            if eng is not None else 0),
               coverage=round(cov_hits / cov_all, 3) if cov_all else "",
               fs_frac=round(fs_hits / fs_all, 3) if fs_all else "",
               adapt_share=(round(share_num / share_den, 4)
                            if share_den else 0.0),
               tried_1=tried[1], tried_2=tried[2], tried_3=tried[3],
               acc_1=acc[1], acc_2=acc[2], acc_3=acc[3],
               dj_1=round(dj[1], 4), dj_2=round(dj[2], 4),
               dj_3=round(dj[3], 4), genai_source=genai_source)
    return row, curves, funnel, products, rep_cycles


def _worker(job):
    sc, arm, seed = job
    t0 = time.time()
    try:
        row, curves, funnel, products, repc = run_once(sc, arm, seed)
        row["wall_s"] = round(time.time() - t0, 1)
        return row, curves, funnel, products, repc, None
    except Exception as exc:
        return (None, [], [], None, None,
                f"{sc}/{arm}/{seed}: {type(exc).__name__}: {exc}")


def _purge_arms(redo, runs_p, curves_p, funnel_p):
    """Drop every finished row of the named arms from the resumable
    outputs, so the scheduler sees them as not-done and re-runs them
    under the current code. Nothing else is touched."""
    import json as _js
    n = 0
    for path in (runs_p, curves_p, funnel_p):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            fields = rows[0].keys() if rows else None
        if fields is None:
            continue
        keep = [r for r in rows if r.get("arm") not in redo]
        n += len(rows) - len(keep)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fields))
            w.writeheader()
            w.writerows(keep)
    prod_p = os.path.join(OUT, "ladder_products.jsonl")
    if os.path.exists(prod_p):
        with open(prod_p, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        keep = [ln for ln in lines
                if _js.loads(ln).get("arm") not in redo]
        with open(prod_p, "w", encoding="utf-8") as f:
            f.writelines(keep)
    for a in redo:
        rp = os.path.join(OUT, f"rep_cycles_{a}.json")
        if os.path.exists(rp):
            os.remove(rp)
    print(f"purged {n} finished rows for re-run: "
          + ", ".join(sorted(redo)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    # evaluation seeds are DISJOINT from every seed used during
    # development and calibration (101-120): the final campaign
    # starts at 201
    ap.add_argument("--seed0", type=int, default=201)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget-s", type=float, default=0.0,
                    help="stop cleanly after this many seconds")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma list; F40 is defined but not run by "
                         "default (--arms Test0,F5,F5Ev,F5EvAI,F40)")
    ap.add_argument("--redo", default="",
                    help="comma list of arms whose finished runs are "
                         "PURGED from the outputs first, so they are "
                         "re-run under the current code (e.g. after "
                         "a gate change: --redo F5EvAI)")
    ap.add_argument("--eta", type=float, default=ETA,
                    help="quality gate of the decision engine "
                         "(default 0.60)")
    args = ap.parse_args()
    globals()["ETA"] = args.eta
    os.environ["DSS_ETA"] = str(args.eta)    # workers inherit this
    os.makedirs(OUT, exist_ok=True)
    runs_p = os.path.join(OUT, "ladder_runs.csv")
    curves_p = os.path.join(OUT, "ladder_curves.csv")
    funnel_p = os.path.join(OUT, "ladder_funnel.csv")

    redo = {a for a in args.redo.split(",") if a}
    if redo:
        _purge_arms(redo, runs_p, curves_p, funnel_p)

    done = set()
    if os.path.exists(runs_p):
        with open(runs_p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["scenario"], r["arm"], int(r["seed"])))
    scen = [s for s in args.scenarios.split(",") if s]
    arms = [a for a in args.arms.split(",") if a]
    jobs = [(s, a, args.seed0 + k)
            for k in range(args.seeds)
            for s in scen for a in arms
            if (s, a, args.seed0 + k) not in done]
    if not jobs:
        print("campaign complete:", len(done), "runs")
        return
    print(f"todo {len(jobs)} runs (done {len(done)})")

    t_start = time.time()
    import multiprocessing as mp
    new_runs = not os.path.exists(runs_p)
    new_curves = not os.path.exists(curves_p)
    new_funnel = not os.path.exists(funnel_p)
    with open(runs_p, "a", newline="", encoding="utf-8") as fr, \
            open(curves_p, "a", newline="", encoding="utf-8") as fc, \
            open(funnel_p, "a", newline="", encoding="utf-8") as ff:
        cw = csv.writer(fc)
        fw = csv.writer(ff)
        if new_curves:
            cw.writerow(["scenario", "arm", "seed", "t_min",
                         "burned_cells", "j_phys"])
        if new_funnel:
            fw.writerow(["scenario", "arm", "seed", "source",
                         "package", "gate", "accepted", "dJ"])
        rw = None
        n_done = 0
        # SUBMIT WINDOW, not a kill switch: after budget_s no NEW job
        # starts, in-flight jobs are allowed to finish and be written.
        # A slice can then never waste a started run.
        pool = mp.Pool(args.workers)
        pending = []
        ji = 0

        def _submit():
            nonlocal ji
            if ji < len(jobs):
                pending.append(pool.apply_async(_worker, (jobs[ji],)))
                ji += 1

        for _ in range(args.workers):
            _submit()
        while pending:
            k_done = None
            while k_done is None:
                for k, ar in enumerate(pending):
                    if ar.ready():
                        k_done = k
                        break
                if k_done is None:
                    time.sleep(0.25)
            (row, curves, funnel, products, repc,
             err) = pending.pop(k_done).get()
            if err:
                print("FAIL", err)
            else:
                if rw is None:
                    rw = csv.DictWriter(fr, fieldnames=list(row))
                    if new_runs:
                        rw.writeheader()
                        new_runs = False
                rw.writerow(row)
                fr.flush()
                for c in curves:
                    cw.writerow(c)
                for fl in funnel:
                    fw.writerow(fl)
                fc.flush(); ff.flush()
                if products is not None:
                    import json as _js
                    with open(os.path.join(OUT,
                                           "ladder_products.jsonl"),
                              "a", encoding="utf-8") as fp_:
                        fp_.write(_js.dumps(products) + "\n")
                if repc is not None:
                    import json as _js
                    with open(os.path.join(
                            OUT, f"rep_cycles_{row['arm']}.json"),
                            "w", encoding="utf-8") as fp_:
                        _js.dump(repc, fp_)
                n_done += 1
                print(f"[{n_done}] {row['scenario']} {row['arm']} "
                      f"s{row['seed']} {row['burned_ha']}ha "
                      f"({row['wall_s']}s)")
            if (not args.budget_s
                    or time.time() - t_start < args.budget_s):
                _submit()
        pool.close()
        pool.join()
    print("slice done:", n_done, "runs; kalan",
          len(jobs) - ji)


if __name__ == "__main__":
    main()
