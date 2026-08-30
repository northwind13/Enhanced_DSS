"""External baseline: value-weighted greedy capacity allocation.

The policy answers the article's external-comparison requirement. It
shares the simulator, the worlds, the seeds, the resource pool, and the
actuation mapping (decision_to_resources) with every other arm of the
campaign, and differs only in how it decides:

  every cycle it commits the full pool toward the regions with the
  highest value at risk near the fire, in proportion to that value,
  with the highest-value region always at full intensity.

No concepts, no rules, no forecasts, no satisficing test, no quality
gate, no fail-safe: the classic dispatch heuristic. It is GENEROUS to
the baseline: it reads the TRUE burning mask (perfect observation),
while the DSS arms act on the observed fire.

  python experiments/greedy_baseline.py --budget-s 150

Resumable; rows land in experiments/out/greedy_runs.csv,
curves in greedy_curves.csv (same schema as ladder_curves).
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dss                                            # noqa: E402
from dss.actions import decision_to_resources         # noqa: E402
from disaster_phyengine.core import Simulator         # noqa: E402
from disaster_phyengine.costs import compute_costs    # noqa: E402

from campaign5 import (build_world, pick_ignitions,   # noqa: E402
                       LIMITED, SCENARIOS, CHECKPOINTS_MIN, MAX_HOURS)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
RUNS = os.path.join(OUT, "greedy_runs.csv")
CURVES = os.path.join(OUT, "greedy_curves.csv")
SEEDS = list(range(101, 111))
CYCLE_MIN = 12.0
THREAT_R = 6      # cells: how far ahead of the front value counts


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        m = out
        out = m.copy()
        out[1:, :] |= m[:-1, :]
        out[:-1, :] |= m[1:, :]
        out[:, 1:] |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
    return out


def _value_field(w):
    vpop = np.asarray(w.value.vpop, dtype=float)
    vbld = np.asarray(w.value.vbld, dtype=float)
    vcrit = np.asarray(w.value.vcrit, dtype=float)
    fload = np.asarray(w.fuel.fload, dtype=float)
    def _n(a):
        m = float(a.max())
        return a / m if m > 1e-9 else a
    return _n(vpop) + _n(vbld) + _n(vcrit) + 0.2 * _n(fload)


def greedy_pairs(w, sim, regions, value):
    burning = np.asarray(sim.state.burning) > 0.5
    if not burning.any():
        return None
    threat = _dilate(burning, THREAT_R) & ~sim.ever_burned
    vr = []
    for r in regions:
        m = np.zeros_like(burning)
        m[r.y0:r.y1, r.x0:r.x1] = True
        vr.append(float(value[threat & m].sum())
                  + 1e-6 * float(burning[m].sum()))
    vmax = max(vr)
    if vmax <= 0.0:
        return None
    pairs = []
    for r, v in zip(regions, vr):
        u = v / vmax
        if u < 0.05:
            continue
        pairs.append((r, {"suppression_effort": u,
                          "resource_deployment": u,
                          "containment_line": 0.6 * u}))
    return pairs or None


def run_once(scenario: str, seed: int):
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
    regions = dss.partition_n(w.config.nx, w.config.ny, 4)
    value = _value_field(w)
    steps_per_cycle = int(round(CYCLE_MIN / w.config.step_minutes))
    max_steps = int(round(MAX_HOURS * 60.0 / w.config.step_minutes))
    ov = None
    out_at = None
    cp = {}
    curves = []
    for i in range(max_steps):
        if i % steps_per_cycle == 0:
            pairs = greedy_pairs(w, sim, regions, value)
            ov = (decision_to_resources(
                      w, np.asarray(sim.state.burning) > 0.5,
                      pairs, base)
                  if pairs is not None else None)
        sim.step(resource_override=ov)
        t_min = (i + 1) * w.config.step_minutes
        if i % 6 == 0:
            rep_i = compute_costs(sim)
            curves.append([scenario, "Greedy", seed, round(t_min, 1),
                           int(sim.ever_burned.sum()),
                           round(float(getattr(rep_i, "j_physical",
                                               rep_i.j_total)), 5)])
        for c in CHECKPOINTS_MIN:
            if c not in cp and t_min >= c:
                cp[c] = compute_costs(sim)
        if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
            out_at = t_min
            break
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

    ft0 = np.asarray(w.fuel.ftype)
    forest_burn = int((sim.ever_burned
                       & ((ft0 == 3) | (ft0 == 4))).sum())
    ha = (w.config.cell_size_m ** 2) / 1e4
    vp0 = np.asarray(getattr(sim, "_vpop0", w.value.vpop))
    row = dict(scenario=scenario, arm="Greedy", seed=seed,
               profile="greedy",
               burned_ha=round(int(sim.ever_burned.sum()) * ha, 2),
               forest_ha=round(forest_burn * ha, 2),
               pop_affected=round(float(vp0[sim.ever_burned].sum()), 0),
               evacuated=round(float(sim.population_evacuated), 0),
               out_min=(out_at if out_at is not None else -1),
               success=int(out_at is not None))
    row.update(costrow(cp.get(CHECKPOINTS_MIN[0]), "t2h"))
    row.update(costrow(cp.get(CHECKPOINTS_MIN[1]), "t6h"))
    row.update(costrow(rep, "end"))
    return row, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-s", type=float, default=1e9)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(RUNS):
        with open(RUNS, encoding="utf-8") as f:
            done = {(r["scenario"], int(r["seed"]))
                    for r in csv.DictReader(f)}
    t0 = time.time()
    for sc in SCENARIOS:
        for seed in SEEDS:
            if (sc, seed) in done:
                continue
            if time.time() - t0 > args.budget_s:
                print("budget reached; resumable")
                return
            t1 = time.time()
            row, curves = run_once(sc, seed)
            row["wall_s"] = round(time.time() - t1, 1)
            new = not os.path.exists(RUNS)
            with open(RUNS, "a", newline="", encoding="utf-8") as f:
                wtr = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new:
                    wtr.writeheader()
                wtr.writerow(row)
            newc = not os.path.exists(CURVES)
            with open(CURVES, "a", newline="", encoding="utf-8") as f:
                wtr = csv.writer(f)
                if newc:
                    wtr.writerow(["scenario", "arm", "seed", "t_min",
                                  "burned_cells", "j_phys"])
                wtr.writerows(curves)
            print(f"{sc}/{seed}: burned={row['burned_ha']} "
                  f"t6h_j_phys={row['t6h_j_phys']} "
                  f"wall={row['wall_s']}s", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
