"""How a perturbation travels from the observation to the decision.

C2 and H2 claim that the framework's sensitivity is understood: that it
is known which of the ten observation features, which membership
parameters and which cost weights actually move a decision, and by how
much. This script measures that, so the claim rests on a sweep rather
than on an argument.

Three factor groups, twenty factors in all:

  features (10)   a multiplicative bias on one feature at the output of
                  the observation model, before the concepts see it
  membership (5)  the term partition of one decision concept, widened
                  or narrowed about its centre
  weights (5)     one weight of the cost decomposition, the rest left
                  as they are and the sum renormalised by the cost
                  function itself

Two designs over the same factors. One-at-a-time gives the size and the
direction of each effect on its own; Morris screening gives mu* and
sigma over the combined space, which is what separates a factor that
matters everywhere from one that matters only in company. Morris is the
screening design the chapter already cites.

WHAT IS REPORTED IS THE DECISION, NOT ONLY THE OUTCOME. A perturbation
that changes the burned area by nothing may still have changed which
intervention was ordered, and a framework that claims traceability has
to answer for that. Every run is therefore compared with the unperturbed
run of the same seed cycle by cycle: the flip rate counts the region
cycles whose dominant intervention family changed, and the activation
shift measures how far the concepts moved.

    python experiments/propagation.py --phase oat
    python experiments/propagation.py --phase morris

Resumable: a finished row is skipped.
Output: experiments/out/prop_oat.csv, prop_morris.csv
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
from dss.fuzzy import REGISTRY, TERMS, default_partition   # noqa: E402
from disaster_phyengine.core import Simulator         # noqa: E402
from disaster_phyengine.costs import compute_costs    # noqa: E402
from scenario import build_world, pick_ignitions      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
OAT_CSV = os.path.join(OUT, "prop_oat.csv")
MORRIS_CSV = os.path.join(OUT, "prop_morris.csv")

CYCLE_MIN = 12.0
HORIZON_MIN = 24.0
J_TH = 0.35
ETA = 0.60
TAU_ATT = 0.35
POOL = 0.25
N_IGN = 4
N_REGIONS = 4
MAX_MIN = 150.0

FEATURES = list(dss.FEATURE_ORDER)
CONCEPTS = list(dss.DECISION_CONCEPTS)
WEIGHTS = ["w_burn", "w_asset", "w_pop", "w_resp", "w_delay"]

#: the perturbation each group is swept over, as a signed fraction
BIAS_FEATURE = 0.25
BIAS_WEIGHT = 0.25
BIAS_MF = 0.20

#: THE MEMBERSHIP GROUP ACTS ON THE FEATURES, NOT ON THE CONCEPTS. The
#: five decision concepts are not fuzzified: their term activations come
#: out of the hierarchy already as vectors, and the rule evaluator reads
#: that vector directly. Perturbing a concept's stored partition
#: therefore changes nothing, which a first sweep confirmed by returning
#: exactly zero for all five. The membership parameters that a decision
#: really depends on are the ten feature partitions, which is also the
#: surface the evolving-FIS stage edits.
FACTORS = ([("feature", f) for f in FEATURES]
           + [("mf", f) for f in FEATURES]
           + [("weight", w) for w in WEIGHTS])


def _factor_span(kind):
    return {"feature": BIAS_FEATURE, "mf": BIAS_MF,
            "weight": BIAS_WEIGHT}[kind]


# ---------------------------------------------------------- perturbations
class _Perturbation:
    """Everything one run needs to be different from the baseline.

    `deltas` maps (kind, name) to a signed fraction. The object installs
    itself, runs, and removes itself, because the membership registry and
    the feature function are process-wide: a perturbation left in place
    would silently contaminate every run after it.
    """

    def __init__(self, deltas):
        self.deltas = dict(deltas)
        self._orig_features = None
        self._orig_parts = {}

    # -- features: a bias at the output of the observation model
    def _install_features(self):
        biases = {n: d for (k, n), d in self.deltas.items()
                  if k == "feature" and d}
        if not biases:
            return
        self._orig_features = _loop.ten_features

        def _biased(sim, region, network=None, pool=None):
            f = self._orig_features(sim, region, network=network,
                                    pool=pool)
            for name, d in biases.items():
                if name in f:
                    f[name] = float(np.clip(f[name] * (1.0 + d),
                                            0.0, 1.0))
            return f

        _loop.ten_features = _biased

    # -- membership: widen or narrow a concept's partition about 0.5
    def reapply_mf(self):
        """Put the perturbed partitions back after the engine resets them.

        THE ACTIVE SET IS DERIVED EVERY CYCLE, and deriving it calls
        reset_partitions, which returns every membership to the default.
        A perturbation installed once therefore survives until the first
        decision and no further: the sweep would report that membership
        parameters do not matter, having measured a partition that was
        no longer perturbed. The engine re-derives, so this re-applies.
        """
        for (kind, name), d in self.deltas.items():
            if kind == "mf" and d:
                self._set_partition(name, d)

    def _set_partition(self, name, d):
        part = default_partition()
        for term in TERMS:
            a, b, c, e = part[term]
            # SCALE ABOUT THE CENTRE, so the partition stays ordered and
            # still covers the interval. Shifting every corner by a
            # constant would push the outer terms off the unit interval
            # and leave a hole at one end.
            new = tuple(sorted(
                float(np.clip(0.5 + (x - 0.5) * (1.0 + d), 0.0, 1.0))
                for x in (a, b, c, e)))
            REGISTRY.set_term(name, term, new)

    def _install_mf(self):
        for (kind, name), d in self.deltas.items():
            if kind != "mf" or not d:
                continue
            self._orig_parts[name] = default_partition()
            self._set_partition(name, d)

    def _restore_mf(self):
        for name, part in self._orig_parts.items():
            for term in TERMS:
                REGISTRY.set_term(name, term, part[term])

    def apply_weights(self, cost):
        for (kind, name), d in self.deltas.items():
            if kind == "weight" and d:
                setattr(cost, name,
                        max(0.0, float(getattr(cost, name)) * (1.0 + d)))

    def __enter__(self):
        self._install_features()
        self._install_mf()
        return self

    def __exit__(self, *exc):
        if self._orig_features is not None:
            _loop.ten_features = self._orig_features
        self._restore_mf()
        return False


# ------------------------------------------------------------------ a run
def _dominant(orders):
    """The intervention family a region is actually being told to do."""
    if not orders:
        return ""
    return max(orders.items(), key=lambda kv: kv[1])[0]


def run_once(seed: int, perturbation: _Perturbation, tag: str):
    w = build_world(seed)
    base, _ = dss.resource_suggestion(w)
    base.ravail = base.ravail * POOL
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    perturbation.apply_weights(w.config.cost)
    for x, y in pick_ignitions(w, base, seed, N_IGN):
        w.add_ignition(x, y, step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    eng = dss.DecisionEngine(
        dss.partition_n(w.config.nx, w.config.ny, N_REGIONS),
        base_pool=base, state_path=dss.isolated_store_path(tag),
        cycle_min=CYCLE_MIN, horizon_min=HORIZON_MIN,
        j_threshold=J_TH, eta=ETA, attention_thr=TAU_ATT,
        adapt_on=True, evfis_on=True, genai_on=False,
        seed_profile="full")
    if any(k == "mf" for k, _ in perturbation.deltas):
        _orig_sync = eng._sync_active_set

        def _sync_then_perturb():
            _orig_sync()
            perturbation.reapply_mf()

        eng._sync_active_set = _sync_then_perturb
        perturbation.reapply_mf()
    steps = int(round(MAX_MIN / w.config.step_minutes))
    for _ in range(steps):
        sim.step(resource_override=eng.maybe_decide(sim))
        if int((sim.state.burning > 0.5).sum()) == 0:
            break
    rep = compute_costs(sim)
    cell_ha = (w.config.cell_size_m ** 2) / 10000.0
    trace = [{n: (dict(r["concepts_effective"]), dict(r["orders_final"]))
              for n, r in c["regions"].items()}
             for c in eng.cycles]
    return dict(burned_ha=float(sim.ever_burned.sum()) * cell_ha,
                j_total=float(rep.j_total),
                j_phys=float(rep.j_physical),
                j_burn=float(rep.j_burn), j_asset=float(rep.j_asset),
                j_pop=float(rep.j_pop), trace=trace)


def compare(basel, pert):
    """Flip rate and activation shift between two aligned runs."""
    flips = seen = 0
    shifts = []
    per_concept = {c: [] for c in CONCEPTS}
    for cb, cp in zip(basel["trace"], pert["trace"]):
        for name in cb:
            if name not in cp:
                continue
            (ab, ob), (ap, op) = cb[name], cp[name]
            seen += 1
            if _dominant(ob) != _dominant(op):
                flips += 1
            for k in ab:
                if k in ap:
                    d = abs(float(ab[k]) - float(ap[k]))
                    shifts.append(d)
                    if k in per_concept:
                        per_concept[k].append(d)
    out = dict(
        flip_rate=round(flips / seen, 4) if seen else float("nan"),
        region_cycles=seen,
        act_shift=round(float(np.mean(shifts)), 5) if shifts else 0.0,
        d_burned_ha=round(pert["burned_ha"] - basel["burned_ha"], 2),
        d_j_total=round(pert["j_total"] - basel["j_total"], 6),
        d_j_phys=round(pert["j_phys"] - basel["j_phys"], 6),
        d_j_burn=round(pert["j_burn"] - basel["j_burn"], 6),
        d_j_asset=round(pert["j_asset"] - basel["j_asset"], 6),
        d_j_pop=round(pert["j_pop"] - basel["j_pop"], 6))
    for c in CONCEPTS:
        out["shift_" + c] = (round(float(np.mean(per_concept[c])), 5)
                             if per_concept[c] else 0.0)
    return out


FIELDS = (["design", "kind", "factor", "delta", "seed", "traj",
           "flip_rate", "region_cycles", "act_shift", "d_burned_ha",
           "d_j_total", "d_j_phys", "d_j_burn", "d_j_asset", "d_j_pop",
           "burned_ha", "j_total", "seconds"]
          + ["shift_" + c for c in CONCEPTS])


def _done(path, keys):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {tuple(r[k] for k in keys) for r in csv.DictReader(f)}


def _append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            wtr.writeheader()
        wtr.writerow(row)


_BASELINE = {}


def baseline(seed):
    """The unperturbed run of a seed, cached in memory and on disk.

    EVERY PERTURBED RUN IS SCORED AGAINST THIS ONE, so a campaign split
    across processes would otherwise recompute it in each of them. On a
    budgeted worker that is not merely wasteful: two baselines cost more
    than the budget leaves for real runs, and the sweep stops making
    progress. The trace is deterministic given the seed, so it caches.
    """
    if seed in _BASELINE:
        return _BASELINE[seed]
    import pickle
    path = os.path.join(OUT, f"prop_baseline_{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            _BASELINE[seed] = pickle.load(fh)
            return _BASELINE[seed]
    with _Perturbation({}) as p:
        _BASELINE[seed] = run_once(seed, p, f"prop_base_{seed}")
    with open(path, "wb") as fh:
        pickle.dump(_BASELINE[seed], fh)
    return _BASELINE[seed]


def phase_oat(seeds, budget, path):
    done = _done(path, ("kind", "factor", "delta", "seed"))
    t0 = time.time()
    for seed in seeds:
        todo = [(k, n, s) for (k, n) in FACTORS for s in (+1, -1)
                if (k, n, f"{s * _factor_span(k):+.3f}",
                    str(seed)) not in done]
        if not todo:
            continue
        b = baseline(seed)
        for kind, name, sign in todo:
            if budget and time.time() - t0 > budget:
                print("budget reached, stopping cleanly")
                return
            d = sign * _factor_span(kind)
            ts = time.time()
            with _Perturbation({(kind, name): d}) as p:
                r = run_once(seed, p, f"prop_{kind}_{name}_{sign}_{seed}")
            row = dict(design="oat", kind=kind, factor=name,
                       delta=f"{d:+.3f}", seed=seed, traj="",
                       burned_ha=round(r["burned_ha"], 2),
                       j_total=round(r["j_total"], 6),
                       seconds=round(time.time() - ts, 1))
            row.update(compare(b, r))
            _append(path, row)
            print(f"{kind}/{name} {d:+.2f} seed={seed} "
                  f"flip={row['flip_rate']} dJ={row['d_j_total']:+.4f} "
                  f"({row['seconds']}s)", flush=True)


def phase_morris(seeds, budget, path, r_traj, levels, rng_seed=7):
    """Elementary effects over the combined space.

    Each trajectory starts at a random level vector and moves one factor
    at a time, so k+1 runs give one elementary effect per factor. mu* and
    sigma are computed by the plotting script from these rows.
    """
    done = _done(path, ("kind", "factor", "seed", "traj"))
    t0 = time.time()
    # THE TRAJECTORY STREAM IS A PARAMETER so that two workers can draw
    # DIFFERENT trajectories. Sharing the default stream would have both
    # of them walk the same points and the campaign would take twice as
    # long to cover half as much.
    rng = np.random.default_rng(rng_seed)
    grid = np.linspace(-1.0, 1.0, levels)
    for seed in seeds:
        b = baseline(seed)
        for t in range(r_traj):
            pt = {f: float(rng.choice(grid)) for f in FACTORS}
            order = list(FACTORS)
            rng.shuffle(order)
            prev = None
            for kind, name in order:
                tid = f"{rng_seed}_{t}"
                key = (kind, name, str(seed), tid)
                if key in done:
                    continue
                if budget and time.time() - t0 > budget:
                    print("budget reached, stopping cleanly")
                    return
                step = float(rng.choice([-1, 1])) * (2.0 / (levels - 1))
                pt[(kind, name)] = float(
                    np.clip(pt[(kind, name)] + step, -1.0, 1.0))
                deltas = {f: v * _factor_span(f[0])
                          for f, v in pt.items() if v}
                ts = time.time()
                with _Perturbation(deltas) as p:
                    r = run_once(seed, p, f"prop_m_{t}_{kind}_{name}_{seed}")
                row = dict(design="morris", kind=kind, factor=name,
                           delta=f"{step:+.3f}", seed=seed, traj=tid,
                           burned_ha=round(r["burned_ha"], 2),
                           j_total=round(r["j_total"], 6),
                           seconds=round(time.time() - ts, 1))
                row.update(compare(b, r))
                _append(path, row)
                prev = r
                print(f"morris t{tid} {kind}/{name} seed={seed} "
                      f"flip={row['flip_rate']} "
                      f"dJ={row['d_j_total']:+.4f} "
                      f"({row['seconds']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["oat", "morris"], default="oat")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--traj", type=int, default=4)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--rng", type=int, default=7)
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--csv", default="")
    ap.add_argument("--only", default="",
                    help="restrict to these factor kinds, comma separated")
    a = ap.parse_args()
    seeds = [201 + i for i in range(a.seeds)]
    if a.only:
        keep = set(a.only.split(","))
        FACTORS[:] = [f for f in FACTORS if f[0] in keep]
    if a.phase == "oat":
        phase_oat(seeds, a.budget, a.csv or OAT_CSV)
    else:
        phase_morris(seeds, a.budget, a.csv or MORRIS_CSV,
                     a.traj, a.levels, a.rng)


if __name__ == "__main__":
    main()
