"""One-at-a-time sensitivity analysis of the DSS parameters.

Design
------
A fixed scenario (map seed, ignition, weather, base pool) is run once
per parameter value while every other parameter stays at its default.
The DSS runs with adaptation OFF so that the sweep measures the
decision layer itself, not the learning path. Reported per run:

  j_physical   fair outcome metric (burn + asset + population only)
  burned       ever-burned cells at the end
  out_min      minutes to extinguish (-1 = still burning at the cap)
  fs_frac      fraction of decision cycles with the fail-safe engaged

Swept parameters (defaults in brackets):
  j_threshold   satisficing bound            [0.35]
  eta           quality gate of fail-safe    [0.60]
  attention_thr global attention threshold   [0.35]
  rho           persistence prior decay      [0.90]
  horizon_min   no-harm forecast horizon     [15]
  cycle_min     decision cycle period        [1]
  w_burn/w_asset/w_pop  cost priority weights [1.0]

Run on the workstation (no time cap):
  python experiments/sensitivity.py --seed 11
  python experiments/sensitivity.py --seed 11 --params eta,rho
Outputs: experiments/out/sensitivity_<seed>.csv + console summary.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import dss                                            # noqa: E402
from disaster_phyengine import terrain                # noqa: E402
from disaster_phyengine.config import SimConfig       # noqa: E402
from disaster_phyengine.core import Simulator         # noqa: E402
from disaster_phyengine.costs import compute_costs    # noqa: E402


SWEEPS = {
    "j_threshold":   [0.15, 0.25, 0.35, 0.45, 0.60],
    "eta":           [0.30, 0.45, 0.60, 0.75, 0.90],
    "attention_thr": [0.15, 0.25, 0.35, 0.50, 0.70],
    "rho":           [0.70, 0.80, 0.90, 0.95, 0.99],
    "horizon_min":   [5.0, 10.0, 15.0, 30.0, 45.0],
    "cycle_min":     [1.0, 2.0, 4.0, 8.0],
    "w_burn":        [0.5, 1.0, 2.0],
    "w_asset":       [0.5, 1.0, 2.0],
    "w_pop":         [0.5, 1.0, 2.0],
}
DEFAULTS = dict(j_threshold=0.35, eta=0.60, attention_thr=0.35,
                rho=0.90, horizon_min=15.0, cycle_min=1.0,
                w_burn=1.0, w_asset=1.0, w_pop=1.0)


def build_world(seed: int):
    cfg = SimConfig(nx=80, ny=60, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=seed, preset="Rolling hills", n_settlements=5,
        population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.08
    w.meteo.wws[:] = 8.0
    return w


def pick_ignitions(w, base, seed: int, n: int = 2):
    """The n burnable spots FARTHEST from the resource bases, mutually
    apart; a trivially reachable fire is out in minutes and every
    sweep row reads the same, so the scenario must be capacity-
    limited to expose the parameters."""
    ok = (w.fuel.fload > 0.4) & (w.fuel.ftype != 0) \
        & (w.fuel.ftype != 5) & (w.fuel.ftype != 6)
    ys, xs = np.where(ok)
    order = np.argsort(-base.rtime[ys, xs])
    spots = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all((x - a) ** 2 + (y - b) ** 2 > 20 ** 2
               for a, b in spots):
            spots.append((x, y))
        if len(spots) == n:
            break
    return spots


def run_once(seed: int, overrides: dict, max_steps: int = 150):
    w = build_world(seed)
    base, _ = dss.resource_suggestion(w)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    p = dict(DEFAULTS)
    p.update(overrides)
    for key in ("w_burn", "w_asset", "w_pop"):
        setattr(w.config.cost, key, float(p[key]))
    for x, y in pick_ignitions(w, base, seed):
        w.add_ignition(x, y, step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    eng = dss.DecisionEngine(
        dss.partition_n(w.config.nx, w.config.ny, 4), base_pool=base,
        j_threshold=p["j_threshold"], eta=p["eta"],
        attention_thr=p["attention_thr"],
        cycle_min=p["cycle_min"], horizon_min=p["horizon_min"],
        # a sensitivity sweep varies ONE parameter at a time; a shared
        # store would vary the memory too (dss.isolated_store_path)
        state_path=dss.isolated_store_path("sensitivity"),
        adapt_on=False)
    for g in eng.gaters.values():
        g.rho = float(p["rho"])
    out_at = None
    for i in range(max_steps):
        sim.step(resource_override=eng.maybe_decide(sim))
        if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
            out_at = (i + 1) * w.config.step_minutes
            break
    fs_hits = fs_all = 0
    for c in eng.cycles:
        for rd in (c.get("regions") or {}).values():
            fs_all += 1
            if rd.get("failsafe"):
                fs_hits += 1
    rep = compute_costs(sim)
    return dict(
        j_physical=round(float(rep.j_physical), 4),
        burned=int(sim.ever_burned.sum()),
        out_min=(out_at if out_at is not None else -1),
        fs_frac=round(fs_hits / fs_all, 3) if fs_all else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--params", default=",".join(SWEEPS))
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args()

    names = [n.strip() for n in args.params.split(",") if n.strip()]
    for n in names:
        if n not in SWEEPS:
            sys.exit(f"unknown parameter {n!r}; "
                     f"choose from {', '.join(SWEEPS)}")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "out")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    print("== baseline (all defaults) ==")
    ref = run_once(args.seed, {}, args.max_steps)
    print(f"  j_phys={ref['j_physical']:.4f} burned={ref['burned']} "
          f"out={ref['out_min']:.0f} min fs={ref['fs_frac']:.2f}")
    rows.append(dict(param="baseline", value="", **ref))
    for name in names:
        print(f"== sweep {name} (default {DEFAULTS[name]}) ==")
        for v in SWEEPS[name]:
            r = run_once(args.seed, {name: v}, args.max_steps)
            rows.append(dict(param=name, value=v, **r))
            mark = " *" if v == DEFAULTS[name] else ""
            print(f"  {name}={v:<6} j_phys={r['j_physical']:.4f} "
                  f"burned={r['burned']:4d} out={r['out_min']:5.0f} "
                  f"min fs={r['fs_frac']:.2f}{mark}")
    path = os.path.join(outdir, f"sensitivity_{args.seed}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                              delimiter=";")
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"\nCSV: {path}")
    # one-line verdict per parameter: relative spread of j_physical
    print("\nspread of j_physical across each sweep "
          "(high spread = sensitive):")
    for name in names:
        vs = [r["j_physical"] for r in rows if r["param"] == name]
        if vs:
            spread = max(vs) - min(vs)
            print(f"  {name:<14} {spread:.4f}")


if __name__ == "__main__":
    main()


