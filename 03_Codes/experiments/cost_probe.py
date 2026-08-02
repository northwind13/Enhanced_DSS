"""Decision latency and computational cost (criteria S3 and S8).

S3, timeliness: the wall time of one decision cycle, and the simulated
time from ignition to the first fielded offensive order.

S8, scalability: the wall time of one decision cycle against the size of
the grid and against the number of local regions.

The timing is taken in the harness, around the call to maybe_decide, so
nothing inside the decision layer is changed for the measurement.

    python experiments/cost_probe.py --phase latency
    python experiments/cost_probe.py --phase scale
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import dss                                            # noqa: E402
from disaster_phyengine import terrain                # noqa: E402
from disaster_phyengine.config import SimConfig       # noqa: E402
from disaster_phyengine.core import Simulator         # noqa: E402
from sensitivity2 import pick_ignitions               # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

MAX_MIN = 360.0
TUNE = dict(j_threshold=0.35, eta=0.60, attention_thr=0.35,
            cycle_min=12.0, horizon_min=45.0, revision_budget=3)


def build(seed, nx, ny):
    cfg = SimConfig(nx=nx, ny=ny, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=seed, preset="Rolling hills", n_settlements=5,
        population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.08
    w.meteo.wws[:] = 8.0
    return w


def run(seed, nx=80, ny=60, n_regions=4, n_ign=4, pool=0.25, genai=True):
    w = build(seed, nx, ny)
    base, _ = dss.resource_suggestion(w)
    base.ravail = base.ravail * float(pool)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    for x, y in pick_ignitions(w, base, seed, int(n_ign)):
        w.add_ignition(x, y, step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False

    eng = dss.DecisionEngine(
        dss.partition_n(w.config.nx, w.config.ny, int(n_regions)),
        base_pool=base, network=None, seed_profile="minimal",
        state_path=dss.isolated_store_path(f"cost_{nx}_{n_regions}_{seed}"),
        spread_tighten=0.0, void_tighten=0.0, rel_physical=True,
        j_threshold=TUNE["j_threshold"], eta=TUNE["eta"],
        attention_thr=TUNE["attention_thr"], cycle_min=TUNE["cycle_min"],
        horizon_min=TUNE["horizon_min"],
        revision_budget=TUNE["revision_budget"],
        adapt_on=True, evfis_on=True, genai_on=genai)

    from dss import adapt as _adapt
    _orig = _adapt._genai_propose
    if genai:
        from campaign5 import _make_template_proposer
        _adapt._genai_propose = _make_template_proposer(seed)

    dt_min = float(w.config.step_minutes)
    max_steps = int(round(MAX_MIN / dt_min))
    cyc, first_order = [], None
    t_wall = time.perf_counter()
    _nc = 0
    try:
        for i in range(max_steps):
            t0 = time.perf_counter()
            ov = eng.maybe_decide(sim)
            dt = time.perf_counter() - t0
            # a tick carries a DECISION only when the engine appended a
            # cycle record; between cycles maybe_decide returns the held
            # override and costs nothing
            _decided = len(eng.cycles) > _nc
            _nc = len(eng.cycles)
            if _decided:
                cyc.append(dt)
                if first_order is None:
                    _u = eng.last_global or {}
                    _rows = getattr(eng, "last_actions", None) or {}
                    _fired = False
                    for _r in (_rows.get("regions") or []):
                        _uu = _r.get("u") or {}
                        if (float(_uu.get("suppression_effort", 0.0)) > 0.05
                                or float(_uu.get("resource_deployment",
                                                 0.0)) > 0.05):
                            _fired = True
                    if _fired:
                        first_order = (i + 1) * dt_min
            sim.step(resource_override=ov)
            if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
                break
    finally:
        _adapt._genai_propose = _orig
    wall = time.perf_counter() - t_wall
    cyc_ms = sorted(1000.0 * c for c in cyc)
    return dict(
        seed=seed, nx=nx, ny=ny, cells=nx * ny, regions=n_regions,
        cycles=len(cyc),
        cycle_ms_med=round(st.median(cyc_ms), 1) if cyc_ms else 0.0,
        cycle_ms_p90=round(cyc_ms[int(0.9 * (len(cyc_ms) - 1))], 1)
        if cyc_ms else 0.0,
        cycle_ms_max=round(cyc_ms[-1], 1) if cyc_ms else 0.0,
        first_order_min=first_order if first_order is not None else -1,
        run_wall_s=round(wall, 1))


def write(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("written", path)


def phase_latency(seeds):
    rows = []
    for s in seeds:
        r = run(s)
        rows.append(r)
        print(f"seed {s}: cycles={r['cycles']} med={r['cycle_ms_med']} ms "
              f"p90={r['cycle_ms_p90']} ms first_order={r['first_order_min']} "
              f"min  run={r['run_wall_s']}s", flush=True)
    write(os.path.join(OUT, "cost_latency.csv"), rows)
    med = st.median([r["cycle_ms_med"] for r in rows])
    p90 = st.median([r["cycle_ms_p90"] for r in rows])
    fo = [r["first_order_min"] for r in rows if r["first_order_min"] > 0]
    print(f"\nmedian cycle {med:.0f} ms, p90 {p90:.0f} ms, "
          f"cycle interval {TUNE['cycle_min']*60000:.0f} ms, "
          f"first order median {st.median(fo) if fo else -1} min")


def phase_scale(seeds, grids, regions):
    rows = []
    for nx, ny in grids:
        for s in seeds:
            r = run(s, nx=nx, ny=ny, n_regions=4)
            r["axis"] = "grid"
            rows.append(r)
            print(f"grid {nx}x{ny} seed {s}: med={r['cycle_ms_med']} ms "
                  f"run={r['run_wall_s']}s", flush=True)
    for k in regions:
        for s in seeds:
            r = run(s, n_regions=k)
            r["axis"] = "regions"
            rows.append(r)
            print(f"regions {k} seed {s}: med={r['cycle_ms_med']} ms "
                  f"run={r['run_wall_s']}s", flush=True)
    write(os.path.join(OUT, "cost_scale.csv"), rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="latency",
                    choices=("latency", "scale"))
    ap.add_argument("--seeds", default="201,202,203,204,205")
    ap.add_argument("--grids", default="80x60,120x90,160x120")
    ap.add_argument("--regions", default="1,2,4,8,16")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.phase == "latency":
        phase_latency(seeds)
    else:
        grids = [tuple(int(v) for v in g.split("x"))
                 for g in a.grids.split(",")]
        regions = [int(x) for x in a.regions.split(",")]
        phase_scale(seeds, grids, regions)
