"""Controlled experiment: does the staged adaptation (evFIS +
resolution + GenAI) REBUILD the rule base from a thin seed profile?

Design
------
One map, ONE persistent decision engine, a SEQUENCE of fire episodes:
after every episode the fire is reset but the engine keeps everything
it has learned (rules, memberships, controller value table). Each episode ignites
a different spot, so the engine visits new situation cells and the
growth stages get material to work on.

Arms (same map, same ignitions, same weather, same rng):
  A  profile=minimal, adaptation OFF   (static thin base, control)
  B  profile=minimal, adaptation ON    (must LEARN the missing rules)
  C  profile=full,    adaptation OFF   (the thesis doctrine, upper ref)

Reported per arm: burned cells per episode, time-to-out, final rule
count, coverage (fraction of episodes' decision cycles where a rule
fired above 0.45), and for arm B the REDISCOVERY table: which learned
rules sit on antecedent cells of the withheld doctrine rules.

Run on the workstation (no time cap):
  python experiments/rule_growth.py --episodes 8 --seed 11
  python experiments/rule_growth.py --episodes 8 --seed 11 --arms B
Outputs: experiments/out/rule_growth_<seed>.csv + console summary.
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


def build_world(seed: int):
    cfg = SimConfig(nx=80, ny=60, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=seed, preset="Rolling hills", n_settlements=5,
        population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.09
    w.meteo.wws[:] = 6.0
    return w


def ignition_spots(w, n: int, seed: int):
    """n diverse burnable spots, spread over the map (same for all
    arms so the episode sequence is identical)."""
    rng = np.random.default_rng(seed + 777)
    ok = (w.fuel.fload > 0.35) & (w.fuel.ftype != 0) \
        & (w.fuel.ftype != 5) & (w.fuel.ftype != 6)
    ys, xs = np.where(ok)
    order = rng.permutation(len(xs))
    spots, taken = [], []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all((x - a) ** 2 + (y - b) ** 2 > 15 ** 2 for a, b in taken):
            spots.append((x, y))
            taken.append((x, y))
        if len(spots) == n:
            break
    return spots


def run_arm(arm: str, seed: int, episodes: int, max_steps: int = 150):
    w = build_world(seed)
    base, _ = dss.resource_suggestion(w)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    profile = "full" if arm == "C" else "minimal"
    adapt = arm == "B"
    eng = dss.DecisionEngine(
        dss.partition_n(w.config.nx, w.config.ny, 1), base_pool=base,
        cycle_min=8.0, horizon_min=15.0, adapt_on=adapt,
        genai_on=adapt, evfis_on=adapt, seed_profile=profile,
        # the point of this experiment is how a rule base GROWS, so it has
        # to start from the seed profile and not from what an earlier run
        # left in the field store (dss.isolated_store_path)
        state_path=dss.isolated_store_path("rule_growth"))
    eng.adapt_cooldown_min = 8.0
    spots = ignition_spots(w, episodes, seed)
    rows = []
    for ep, (x, y) in enumerate(spots, 1):
        sim = Simulator(w)          # fresh fire, SAME engine
        sim.record_states = False
        w.ignitions.clear()
        w.add_ignition(x, y, step=0, radius=1)
        n0 = len(eng.rules)
        out_at = None
        cov_hits = cov_all = 0
        for i in range(max_steps):
            ov = eng.maybe_decide(sim)
            sim.step(resource_override=ov)
            if eng.cycles and eng.cycles[-1].get("step") is not None:
                pass
            b = int((sim.state.burning > 0.5).sum())
            if b == 0 and i > 5:
                out_at = (i + 1) * w.config.step_minutes
                break
        # coverage over this episode's logged cycles
        for c in eng.cycles:
            regs = c.get("regions") or {}
            for rname, rd in regs.items():
                fired = rd.get("fired") or []
                cov_all += 1
                if any(wgt > 0.45 for _n, wgt in fired):
                    cov_hits += 1
        eng.cycles.clear()
        rows.append(dict(
            arm=arm, episode=ep, ignition=f"{x},{y}",
            burned=int(sim.ever_burned.sum()),
            out_min=(out_at if out_at is not None else -1),
            rules_before=n0, rules_after=len(eng.rules),
            coverage=(cov_hits / cov_all if cov_all else 0.0)))
        print(f"  {arm} ep{ep:02d} ignite=({x},{y}) "
              f"burned={rows[-1]['burned']:4d} "
              f"out={rows[-1]['out_min']:5.0f} min "
              f"rules {n0}->{len(eng.rules)} "
              f"cov={rows[-1]['coverage']:.2f}")
    return rows, eng


def rediscovery_table(eng):
    """Which withheld doctrine cells did the learned rules re-occupy?"""
    full = dss.make_runtime_rules("full")
    have = {r.name for r in eng.rules}
    withheld = [r for r in full if r.name not in have and r.active]
    learned = [r for r in eng.rules if r.name[0] in "AG"]
    out = []
    for lr in learned:
        la = set(lr.antecedents)
        for wr in withheld:
            if la & set(wr.antecedents):
                shared = la & set(wr.antecedents)
                out.append((lr.name, wr.name,
                            ", ".join(f"{v}={t}" for v, t in shared)))
    return learned, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--arms", default="ABC")
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args()

    os.makedirs(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "out"), exist_ok=True)
    all_rows = []
    engB = None
    for arm in args.arms:
        print(f"== arm {arm} "
              f"({'minimal+adapt' if arm == 'B' else 'minimal static' if arm == 'A' else 'full static'}) ==")
        rows, eng = run_arm(arm, args.seed, args.episodes,
                            args.max_steps)
        all_rows.extend(rows)
        if arm == "B":
            engB = eng
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "out", f"rule_growth_{args.seed}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()),
                              delimiter=";")
        wcsv.writeheader()
        wcsv.writerows(all_rows)
    print(f"\nCSV: {path}")

    for arm in args.arms:
        rs = [r for r in all_rows if r["arm"] == arm]
        if not rs:
            continue
        print(f"arm {arm}: mean burned="
              f"{np.mean([r['burned'] for r in rs]):.0f} "
              f"mean cov={np.mean([r['coverage'] for r in rs]):.2f} "
              f"rules end={rs[-1]['rules_after']}")
    if engB is not None:
        learned, redisc = rediscovery_table(engB)
        print(f"\narm B learned {len(learned)} rules:")
        for r in learned:
            print("  " + r.text())
        if redisc:
            print("\nrediscovery (learned rule <-> withheld doctrine "
                  "cell overlap):")
            for ln, wn, sh in redisc:
                print(f"  {ln} ↔ {wn}  [{sh}]")


if __name__ == "__main__":
    main()
