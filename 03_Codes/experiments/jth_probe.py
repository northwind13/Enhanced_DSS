"""Diagnose the scale of the satisficing bound and sweep J_TH / eta.

Phase A (probe): one world, base config, records per decision cycle
    j_c, j_0, need, phys_c, phys_0 and which of the three adaptation
    symptoms fired. Answers: at what scale does j_total actually live,
    and is the cost deficit saturated on or off.

Phase B (sweep): J_TH over a range that spans the observed scale of
    j_0, and eta over its configured range, paired by world.

    python experiments/jth_probe.py --phase probe
    python experiments/jth_probe.py --phase sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import dss                                            # noqa: E402
from disaster_phyengine.core import Simulator         # noqa: E402
from disaster_phyengine.costs import compute_costs    # noqa: E402
from sensitivity2 import build_world, pick_ignitions  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

MAX_MIN = 360.0
BASE_TUNE = dict(j_threshold=0.35, eta=0.60, attention_thr=0.35,
                 cycle_min=12.0, horizon_min=24.0, revision_budget=3)
ENV = dict(n_ign=4, pool=0.25, n_regions=4)


def make_run(seed, pool, n_ign, n_regions):
    w = build_world(seed)
    base, _ = dss.resource_suggestion(w)
    if pool != 1.0:
        base.ravail = base.ravail * float(pool)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    for x, y in pick_ignitions(w, base, seed, int(n_ign)):
        w.add_ignition(x, y, step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    return w, base, sim


def run_one(seed, tune, env=ENV, adaptive=True, trace=None,
            tighten=(1.0, 1.0), tag="jthprobe", genai=True):
    """One run. trace, if a list, receives one dict per decision cycle."""
    t0 = time.time()
    w, base, sim = make_run(seed, env["pool"], env["n_ign"],
                            env["n_regions"])
    eng = dss.DecisionEngine(
        dss.partition_n(w.config.nx, w.config.ny, int(env["n_regions"])),
        base_pool=base, network=None, seed_profile="minimal",
        state_path=dss.isolated_store_path(tag),
        spread_tighten=float(tighten[0]),
        void_tighten=float(tighten[1]),
        j_threshold=float(tune["j_threshold"]),
        eta=float(tune["eta"]),
        attention_thr=float(tune["attention_thr"]),
        cycle_min=float(tune["cycle_min"]),
        horizon_min=float(tune["horizon_min"]),
        revision_budget=int(tune["revision_budget"]),
        adapt_on=adaptive, evfis_on=adaptive,
        genai_on=bool(adaptive and genai))
    if trace is not None:
        _install_trace(eng, trace)
    from dss import adapt as _adapt
    _orig = _adapt._genai_propose
    if adaptive and genai:
        from campaign5 import _make_template_proposer
        _adapt._genai_propose = _make_template_proposer(seed)
    max_steps = int(round(MAX_MIN / w.config.step_minutes))
    try:
        for i in range(max_steps):
            sim.step(resource_override=eng.maybe_decide(sim))
            if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
                break
    finally:
        _adapt._genai_propose = _orig
    rep = compute_costs(sim)
    fs_hits = fs_all = 0
    tried = {1: 0, 2: 0, 3: 0}
    acc = {1: 0, 2: 0, 3: 0}
    engaged = 0
    for c in eng.cycles:
        ad = c.get("adaptation") or {}
        t = int(ad.get("tried") or 0)
        if t:
            engaged += 1
            tried[t] = tried.get(t, 0) + 1
            if ad.get("accepted"):
                acc[t] = acc.get(t, 0) + 1
        for rd in (c.get("regions") or {}).values():
            fs_all += 1
            fs_hits += 1 if rd.get("failsafe") else 0
    n = max(len(eng.cycles), 1)
    return dict(seed=seed, j_phys=float(rep.j_physical),
                j_total=float(rep.j_total),
                burned=int(sim.ever_burned.sum()),
                cycles=len(eng.cycles),
                adapt_frac=engaged / n,
                tried1=tried[1], tried2=tried[2], tried3=tried[3],
                acc1=acc[1], acc2=acc[2], acc3=acc[3],
                rules=len(getattr(eng, "rules", []) or []),
                fs_frac=(fs_hits / fs_all) if fs_all else 0.0,
                seconds=round(time.time() - t0, 1))


def freeburn(seed, env=ENV):
    w, base, sim = make_run(seed, env["pool"], env["n_ign"],
                            env["n_regions"])
    max_steps = int(round(MAX_MIN / w.config.step_minutes))
    for i in range(max_steps):
        sim.step()
        if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
            break
    rep = compute_costs(sim)
    return float(rep.j_physical), int(sim.ever_burned.sum())


def _install_trace(eng, trace):
    """Wrap _tally_cycle so every gate evaluation is recorded."""
    orig = eng._tally_cycle

    def wrapped(step, j_c, j_0, bound, deficit_on, gap, adapt_due, menu):
        trace.append(dict(step=int(step), j_c=float(j_c), j_0=float(j_0),
                          bound=float(bound), deficit_on=bool(deficit_on),
                          gap=bool(gap), adapt_due=bool(adapt_due)))
        return orig(step, j_c, j_0, bound, deficit_on, gap, adapt_due, menu)
    eng._tally_cycle = wrapped


# ------------------------------------------------------------- phase A
def probe(seeds):
    rows = []
    for s in seeds:
        tr = []
        r = run_one(s, BASE_TUNE, trace=tr)
        for t in tr:
            t["seed"] = s
        rows.extend(tr)
        print(f"seed {s}: cycles={r['cycles']} j_phys={r['j_phys']:.5f} "
              f"burned={r['burned']} {r['seconds']}s", flush=True)
    p = os.path.join(OUT, "jth_probe_trace.csv")
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    j0 = np.array([r["j_0"] for r in rows])
    jc = np.array([r["j_c"] for r in rows])
    bd = np.array([r["bound"] for r in rows])
    df = np.array([r["deficit_on"] for r in rows])
    gp = np.array([r["gap"] for r in rows])
    print("\n--- probe summary over", len(rows), "cycles ---")
    print("j_0    min %.5f  med %.5f  max %.5f" % (j0.min(), np.median(j0), j0.max()))
    print("j_c    min %.5f  med %.5f  max %.5f" % (jc.min(), np.median(jc), jc.max()))
    print("bound  min %.5f  med %.5f  max %.5f" % (bd.min(), np.median(bd), bd.max()))
    print("deficit_on %d/%d   gap %d/%d   deficit ALONE %d"
          % (df.sum(), len(df), gp.sum(), len(gp), (df & ~gp).sum()))
    print("cycles where J_TH=0.35 is the binding min:",
          int((0.95 * j0 > 0.35).sum()))
    print("J_TH would have to be below %.5f to ever bind" % (0.95 * j0.max()))
    json.dump(dict(j0_max=float(j0.max()), j0_med=float(np.median(j0)),
                   n=len(rows)),
              open(os.path.join(OUT, "jth_probe.json"), "w"), indent=1)


# ------------------------------------------------------------- phase B
def _job(a):
    dial, lv, s, tighten = a
    t = dict(BASE_TUNE)
    t[dial] = lv
    tag = f"jth_{dial}_{lv}_{s}_{tighten[0]}"
    r = run_one(s, t, tighten=tighten, tag=tag)
    return dial, lv, s, tighten[0], r


def sweep(seeds, jth_levels, eta_levels, tightens, workers=2):
    p = os.path.join(OUT, "jth_sweep_genai.csv")
    done = set()
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            done.add((r["dial"], r["level"], r["seed"],
                      r["spread_tighten"]))
    f = open(p, "a", newline="")
    wr = csv.writer(f)
    if not done:
        wr.writerow(["dial", "level", "seed", "spread_tighten", "j_phys",
                     "j_total", "burned", "cycles", "adapt_frac",
                     "tried1", "tried2", "tried3", "acc1", "acc2", "acc3",
                     "rules", "fs_frac", "seconds"])
    fb = {}
    for s in seeds:
        fb[s] = freeburn(s)
        print(f"freeburn seed {s}: j_phys={fb[s][0]:.5f} burned={fb[s][1]}",
              flush=True)
    json.dump({str(k): v for k, v in fb.items()},
              open(os.path.join(OUT, "jth_freeburn.json"), "w"))
    jobs = []
    for tg in tightens:
        for lv in jth_levels:
            for s in seeds:
                jobs.append(("j_threshold", lv, s, (tg, tg)))
        for lv in eta_levels:
            for s in seeds:
                jobs.append(("eta", lv, s, (tg, tg)))
    jobs = [j for j in jobs
            if (j[0], str(j[1]), str(j[2]), str(j[3][0])) not in done]
    print(f"{len(jobs)} runs to do", flush=True)
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for dial, lv, s, tg, r in pool.imap_unordered(_job, jobs):
            wr.writerow([dial, lv, s, tg, round(r["j_phys"], 5),
                         round(r["j_total"], 5), r["burned"],
                         r["cycles"], round(r["adapt_frac"], 3),
                         r["tried1"], r["tried2"], r["tried3"],
                         r["acc1"], r["acc2"], r["acc3"], r["rules"],
                         round(r["fs_frac"], 3), r["seconds"]])
            f.flush()
            print(f"{dial}={lv} tg={tg} seed={s}: "
                  f"j_phys={r['j_phys']:.5f} adapt={r['adapt_frac']:.2f} "
                  f"s3={r['tried3']}/{r['acc3']} rules={r['rules']} "
                  f"{r['seconds']}s", flush=True)
    f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="probe",
                    choices=("probe", "sweep"))
    ap.add_argument("--seeds", default="201,202,203")
    ap.add_argument("--jth", default="")
    ap.add_argument("--eta", default="0.30,0.45,0.60,0.75,0.90")
    ap.add_argument("--tighten", default="1.0,0.0")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.phase == "probe":
        probe(seeds)
    else:
        jth = [float(x) for x in a.jth.split(",")] if a.jth else []
        eta = [float(x) for x in a.eta.split(",")] if a.eta else []
        tgs = [float(x) for x in a.tighten.split(",")]
        sweep(seeds, jth, eta, tgs, workers=a.workers)
