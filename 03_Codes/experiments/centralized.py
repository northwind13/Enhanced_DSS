"""Centralized against distributed: the experiment behind H1.

H1 claims that a hierarchically distributed, open architecture carries a
LOWER RULE-BASE COMPLEXITY and a FASTER RESPONSE than a closed,
centralized configuration of the same idea. That is a comparative claim
and until now the thesis measured only one side of it. This script runs
both sides on the same worlds, the same gates and the same simulator, so
the comparison is an experiment rather than an assertion.

  A  distributed-open      one local fuzzy DSS per region, the
                           non-inferential Global DSS above them,
                           staged adaptation on. The configuration of
                           Section 5.5.
  B  centralized-closed     one inferential core that receives every
                           region's observation each cycle and runs the
                           whole reasoning for each of them in sequence,
                           on the seed rule base alone, adaptation off.
  C  centralized-open       B with adaptation on. It exists to separate
                           the two properties: without it a difference
                           between A and B could be blamed on either
                           centralization or closure, and the obvious
                           question from the floor has no answer.

HOW LATENCY IS MEASURED, AND WHY IT IS NOT RAW WALL TIME. Both
configurations execute in one Python process on one machine, so a
stopwatch around the cycle would time the harness and not the
architecture: the distributed configuration would be charged for
reasoning that a deployment runs on separate nodes at the same time. The
per-region reasoning is therefore timed region by region and composed
under an explicit deployment model,

    distributed  t_cycle = max over regions + shared
    centralized  t_cycle = sum over regions + shared

where "shared" is everything that is not per-region reasoning: the
coordination, the composition and the two shadow forecasts of the
acceptance test, which a centralized core must also perform. The model
is stated rather than hidden, and the assumption it makes is the one the
architecture makes, that local agents reason concurrently.

THE COMPLEXITY METRIC IS IMPLEMENTATION-INDEPENDENT. Wall time depends
on the language and the machine; antecedent evaluations per cycle do
not. Both are reported, and the complexity claim rests on the second.

    python experiments/centralized.py --phase latency
    python experiments/centralized.py --phase outcome
    python experiments/centralized.py --phase outcome --workers 2

Both phases are resumable: a finished row is skipped, so an interrupted
campaign continues where it stopped.
Output: experiments/out/central_latency.csv, central_outcome.csv
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
from dss import loop as _loop                         # noqa: E402
from dss import rules as _rules                       # noqa: E402
from disaster_phyengine import terrain                # noqa: E402
from disaster_phyengine.config import SimConfig       # noqa: E402
from disaster_phyengine.core import Simulator         # noqa: E402
from scenario import pick_ignitions                   # noqa: E402
import closed_catalogue                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
LAT_CSV = os.path.join(OUT, "central_latency.csv")
OUTC_CSV = os.path.join(OUT, "central_outcome.csv")

#: THE OPERATING POINT OF SECTION 5.5, unchanged. A comparison run at a
#: different point would answer a different question, and the reader is
#: entitled to place this table beside Table 5.13 without a footnote.
CYCLE_MIN = 12.0
HORIZON_MIN = 24.0
J_TH = 0.35
ETA = 0.60
TAU_ATT = 0.35
POOL = 0.25
N_IGN = 4

#: The scale sweep of Section 5.5.4. The grid grows with the number of
#: regions so that the DOMAIN PER AGENT stays fixed: raising the agent
#: count on a fixed grid shrinks each region and makes the architecture
#: look cheap for a reason that has nothing to do with the architecture.
SCALE = {1: (80, 60), 2: (80, 60), 4: (80, 60),
         8: (120, 90), 16: (160, 120)}

#: name -> (centralized?, adaptation on?, rule base)
#:
#: "closed" means COMPLETE, not small. A closed configuration cannot
#: write a rule at run time, so it has to carry an answer for every cell
#: of the antecedent space in advance: 3125 of them. Running the closed
#: baseline on the 42 seed rules instead would make it FASTER than the
#: distributed system, which reads as a refutation of H1 and is in fact
#: a measurement of a base that does not cover the space. That
#: configuration is kept as D, reported for what it is.
CONFIGS = {
    "A_distributed_open": (False, True, "seed"),
    "B_centralized_closed": (True, False, "catalogue"),
    "C_centralized_open": (True, True, "seed"),
    "D_centralized_seedonly": (True, False, "seed"),
}

MAX_MIN_LATENCY = 180.0
# THE OUTCOME HORIZON IS 240 MINUTES, not the 360 of the 5.5 campaign.
# Four hours is past the point where the arms have separated on these
# worlds, and the reduction buys a third of the compute back. The number
# is stated in the section so no reader compares a 240-minute burned
# area with a 360-minute one.
MAX_MIN_OUTCOME = 240.0


# ------------------------------------------------------- instrumentation
class _Probe:
    """Per-region timing and rule-evaluation counting for one run.

    The engine calls evaluate_rules once per region per pass, so wrapping
    that function is enough to attribute reasoning to regions without
    touching the decision layer itself. The experiment must not edit the
    thing it measures.
    """

    def __init__(self):
        self.calls = []          # (seconds, rules_seen, antecedent_evals)
        self._orig = None

    def install(self):
        self._orig = _loop.evaluate_rules

        def _timed(concepts, features, rules=None, macros=None):
            rs = _rules.SEED_RULES if rules is None else rules
            n_rules = sum(1 for r in rs if r.active)
            n_ante = sum(len(r.antecedents) for r in rs if r.active)
            t0 = time.perf_counter()
            out = self._orig(concepts, features, rules, macros)
            self.calls.append((time.perf_counter() - t0, n_rules, n_ante))
            return out

        _loop.evaluate_rules = _timed

    def remove(self):
        if self._orig is not None:
            _loop.evaluate_rules = self._orig

    def take(self):
        c = self.calls
        self.calls = []
        return c


def build_world(nx: int, ny: int, seed: int):
    """The testbed of scenario.py at an arbitrary grid size.

    The scale sweep needs the grid as a parameter, which the fixed
    testbed does not offer; everything else about the world is the same,
    so a point of this sweep and a point of the 5.5 campaign describe
    the same landscape family under the same weather.
    """
    cfg = SimConfig(nx=nx, ny=ny, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=seed, preset="Rolling hills", n_settlements=5,
        population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.08
    w.meteo.wws[:] = 8.0
    return w


def _engine(w, base, n_regions, adaptive, tag, rule_base):
    eng = dss.DecisionEngine(
        dss.partition_n(w.config.nx, w.config.ny, n_regions),
        base_pool=base,
        state_path=dss.isolated_store_path(tag),
        cycle_min=CYCLE_MIN, horizon_min=HORIZON_MIN,
        j_threshold=J_TH, eta=ETA, attention_thr=TAU_ATT,
        # CLOSED MEANS CLOSED, on every route. Turning the live stages
        # off while still loading what earlier runs learned would leave
        # the closed configuration reasoning on an open vocabulary, and
        # the comparison would be worthless.
        adapt_on=adaptive, evfis_on=adaptive, genai_on=False,
        use_evfis=adaptive, use_genai=adaptive,
        seed_profile="full")
    if rule_base == "catalogue":
        # THE ACTIVE SET IS DERIVED EVERY CYCLE, so assigning the rules
        # once is not enough: _sync_active_set rebuilds them from the
        # seed profile and the store at the top of each decision and the
        # catalogue would be gone by the first cycle. A closed
        # configuration has nothing to derive, since nothing may change
        # at run time, so the rebuild is retired on this instance and
        # the catalogue stands for the whole run. The hierarchy and the
        # concept set were already resolved by the constructor.
        eng.rules = closed_catalogue.build()
        eng._sync_active_set = lambda: None
    return eng


def run_point(config: str, n_regions: int, seed: int, max_min: float,
              collect_latency: bool):
    """One run. Returns the row that describes it."""
    centralized, adaptive, rule_base = CONFIGS[config]
    nx, ny = SCALE[n_regions]
    w = build_world(nx, ny, seed)
    base, _ = dss.resource_suggestion(w)
    base.ravail = base.ravail * POOL
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    for x, y in pick_ignitions(w, base, seed, N_IGN):
        w.add_ignition(x, y, step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    eng = _engine(w, base, n_regions, adaptive,
                  f"central_{config}_{n_regions}_{seed}", rule_base)

    probe = _Probe() if collect_latency else None
    if probe:
        probe.install()

    lat_ms, ante_per_cycle, rules_per_cycle = [], [], []
    steps = int(round(max_min / w.config.step_minutes))
    t_run = time.time()
    try:
        for _ in range(steps):
            t0 = time.perf_counter()
            ov = eng.maybe_decide(sim)
            t_cycle = time.perf_counter() - t0
            if probe is not None:
                calls = probe.take()
                # A step that was not a decision cycle makes no calls.
                if calls:
                    per_region = [c[0] for c in calls]
                    shared = max(0.0, t_cycle - sum(per_region))
                    if centralized:
                        model = sum(per_region) + shared
                    else:
                        model = max(per_region) + shared
                    lat_ms.append(model * 1000.0)
                    # One PASS over the regions is one decision. The
                    # engine also evaluates rules inside its trial
                    # forecasts, and counting those as decision work
                    # would inflate the adaptive arms for doing the
                    # extra thinking the acceptance test asks for.
                    n_pass = min(len(calls), n_regions)
                    if centralized:
                        ante_per_cycle.append(
                            sum(c[2] for c in calls[:n_pass]))
                        rules_per_cycle.append(
                            sum(c[1] for c in calls[:n_pass]))
                    else:
                        ante_per_cycle.append(
                            max(c[2] for c in calls[:n_pass]))
                        rules_per_cycle.append(
                            max(c[1] for c in calls[:n_pass]))
            sim.step(resource_override=ov)
            if int((sim.state.burning > 0.5).sum()) == 0:
                break
    finally:
        if probe:
            probe.remove()

    from disaster_phyengine.costs import compute_costs
    cell_ha = (w.config.cell_size_m ** 2) / 10000.0
    burned_ha = float(sim.ever_burned.sum()) * cell_ha
    rep = compute_costs(sim)
    j_total = float(rep.j_total)
    j_phys = float(rep.j_physical)

    row = dict(config=config, centralized=int(centralized),
               adaptive=int(adaptive), rule_base=rule_base,
               n_regions=n_regions,
               nx=nx, ny=ny, seed=seed,
               burned_ha=round(burned_ha, 2),
               j_total=round(j_total, 6), j_phys=round(j_phys, 6),
               cycles=len(lat_ms), seconds=round(time.time() - t_run, 1))
    if lat_ms:
        a = np.array(lat_ms)
        budget = CYCLE_MIN * 60_000.0
        row.update(
            lat_median_ms=round(float(np.median(a)), 2),
            lat_p90_ms=round(float(np.percentile(a, 90)), 2),
            lat_max_ms=round(float(a.max()), 2),
            duty_median=round(float(np.median(a)) / budget, 8),
            missed=int((a > budget).sum()),
            missed_rate=round(float((a > budget).mean()), 4),
            ante_median=round(float(np.median(ante_per_cycle)), 1),
            rules_median=round(float(np.median(rules_per_cycle)), 1))
    return row


FIELDS = ["config", "centralized", "adaptive", "rule_base",
          "n_regions", "nx", "ny",
          "seed", "burned_ha", "j_total", "j_phys", "cycles", "seconds",
          "lat_median_ms", "lat_p90_ms", "lat_max_ms", "duty_median",
          "missed", "missed_rate", "ante_median", "rules_median"]


def _done(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["config"], r["n_regions"], r["seed"])
                for r in csv.DictReader(f)}


def _append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            wtr.writeheader()
        wtr.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["latency", "outcome"],
                    default="latency")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=0.0,
                    help="stop after this many seconds (0 = no limit)")
    ap.add_argument("--configs", default="")
    ap.add_argument("--csv", default="",
                    help="write to this file instead of the default, so "
                         "two workers can run without racing on one CSV")
    ap.add_argument("--regions", default="")
    a = ap.parse_args()

    latency = a.phase == "latency"
    path = a.csv or (LAT_CSV if latency else OUTC_CSV)
    max_min = MAX_MIN_LATENCY if latency else MAX_MIN_OUTCOME
    configs = ([c for c in a.configs.split(",") if c] or list(CONFIGS))
    regions = ([int(x) for x in a.regions.split(",") if x]
               or sorted(SCALE))
    seeds = [201 + i for i in range(a.seeds)]

    done = _done(path)
    t0 = time.time()
    for n in regions:
        for cfg in configs:
            for s in seeds:
                key = (cfg, str(n), str(s))
                if key in done:
                    continue
                if a.budget and time.time() - t0 > a.budget:
                    print("budget reached, stopping cleanly")
                    return
                row = run_point(cfg, n, s, max_min, latency)
                _append(path, row)
                print(f"{cfg} N={n} seed={s} "
                      f"burned={row['burned_ha']} "
                      f"lat={row.get('lat_median_ms', '-')} "
                      f"ante={row.get('ante_median', '-')} "
                      f"({row['seconds']}s)", flush=True)
    print("phase complete:", path)


if __name__ == "__main__":
    main()
