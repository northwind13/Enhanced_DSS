"""One-at-a-time sensitivity study of the decision layer.

WHY THIS REPLACES THE EARLIER SWEEP. The first version ran every sweep
at one operating point and never checked whether that point could show
anything. Measured on its own testbed, the free-burning fire covers
3905 of 4800 cells while the decision layer holds it to 15, so the
physical cost sits at 0.0025 against 0.5865 and no threshold in the
system can move it. A sweep run there reports that everything is
robust, which is a property of the operating point rather than of the
decision layer. This version therefore calibrates first and sweeps
second.

  phase 1, calibration   fire load x resource level, with the free
                         burn of the same world as the reference. The
                         operating point is the cell where the decision
                         layer neither wins nor loses outright, which
                         is the only place a threshold can be seen to
                         matter.
  phase 2, environment   fire load, resource level, sensor coverage and
                         the number of local regions, each run with the
                         static and the adaptive configuration, so the
                         question "does adaptation help more as the
                         situation gets harder" has an answer.
  phase 2, tuning        the decision layer's own parameters, swept at
                         the calibrated point with the adaptive
                         configuration.
  phase 2, weights       the cost weights, reported against the
                         PHYSICAL outcome. Sweeping them against the
                         cost they define answers itself; the real
                         question is whether the priority profile
                         changes what the system does.

Every point is repeated over several seeds, because one world cannot
separate the effect of a parameter from the noise of the map.

    python experiments/sensitivity2.py --phase calibrate
    python experiments/sensitivity2.py --phase sweep
    python experiments/sensitivity2.py --workers 8

Resumable: finished rows are skipped, so an interrupted run continues.
Output: experiments/out/sens_runs.csv, experiments/out/sens_point.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dss                                            # noqa: E402
from disaster_phyengine import terrain                # noqa: E402
from disaster_phyengine.config import SimConfig       # noqa: E402
from disaster_phyengine.core import Simulator         # noqa: E402
from disaster_phyengine.costs import compute_costs    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
RUNS = os.path.join(OUT, "sens_runs.csv")
POINT = os.path.join(OUT, "sens_point.json")

MAX_MIN = 360.0            # six hours, the horizon Chapter 5 reports on
# REPETITION IS NOT OPTIONAL. One world cannot separate the effect of a
# parameter from the noise of the map, which is the standing weakness
# of a one-at-a-time design. Three seeds is the smallest number that
# still yields an interval, and it was the first setting used here, but
# it is not enough to report one. The half width of an interval is
# t(n-1) * sd / sqrt(n). At three worlds the Student factor is 4.303 and
# the half width is 2.48 sd; at five worlds the factor falls to 2.776
# and the half width to 1.24 sd, which is exactly half. Most of that
# gain comes from the Student factor and not from sqrt(n), which is why
# the step from three worlds to five buys far more than the step from
# five to seven, where the half width only falls to 0.93 sd. Five is
# also the number the calibration grid already carries, so every sweep
# run divides by a free burn of its own world that is already on disk,
# the operating point does not move, and the rows already written stay
# valid because run_jobs appends only what is missing.
SEEDS = [201, 202, 203, 204, 205]
# THE GRID IS REPEATED AS OFTEN AS THE SWEEPS. It ran on two seeds to
# save time, which cost more than it saved: an interval cannot be built
# from two observations, and the cell that was chosen as the operating
# point turned out to be the average of one world in which the fire was
# lost and one in which it was largely held. The grid is what justifies
# every later sweep, so it is the last place to economise. Five
# observations keep one unusual map from carrying a third of every cell
# mean, which matters most in the cells where the free burn itself is
# small and the reported quantity is a ratio. The grid and the sweeps
# now stand on the same five worlds, so the denominator of every
# reported ratio is a world that the grid also reports.
CAL_SEEDS = list(SEEDS)

# THE DECISION LAYER IS CONFIGURED AS IT IS IN THE CAMPAIGN. A
# sensitivity study of a differently configured system would not
# describe the system the rest of the chapter reports.
TUNE_BASE = dict(j_threshold=0.35, eta=0.60, attention_thr=0.35,
                 cycle_min=12.0, horizon_min=24.0, revision_budget=3)
ENV_BASE = dict(n_ign=3, pool=1.0, n_sensors=None, n_regions=4)
W_BASE = dict(w_burn=1.0, w_asset=1.0, w_pop=1.0)

CAL_IGN = [1, 2, 4, 8, 12]
# THE GRID CARRIES THE SAME RESOURCE LEVELS AS THE SWEEP. With 0.75
# missing, the calibration answered the question at four levels while
# the sweep asked it at five, and the widest gap in the grid sat exactly
# where the outcome turns over.
CAL_POOL = [0.10, 0.25, 0.50, 0.75, 1.00]

ENV_SWEEPS = {
    "n_ign":     [1, 2, 4, 8, 12],
    "pool":      [0.10, 0.25, 0.50, 0.75, 1.00],
    "n_sensors": [1, 2, 3, 5, 9],
    # THE REGION COUNT IS SWEPT AS A POWER OF TWO UP TO SIXTEEN. The
    # expectation taken from the coordination literature is that the
    # outcome is near flat in N and only degrades once each region holds
    # too few cells to carry a priority. That claim cannot be tested on
    # a range that stops at eight, so the last doubling is run.
    "n_regions": [1, 2, 4, 8, 16],
}
# BOTH CONFIGURATIONS ARE RUN ONLY WHERE THE COMPARISON IS THE POINT.
# Fire load and resource level define the capacity balance, so the
# question there is whether adaptation earns more as the balance turns
# against it. Sensor coverage and the region count describe the
# decision layer's own inputs, and the static configuration adds
# nothing to those two beyond runtime.
ENV_TWO_ARM = ("n_ign", "pool")
TUNE_SWEEPS = {
    "j_threshold":     [0.15, 0.25, 0.35, 0.45, 0.60],
    "eta":             [0.30, 0.45, 0.60, 0.75, 0.90],
    # THE TWO EXTREMES ARE WHERE THE EXPECTATION LIVES. The threshold is
    # expected to be flat across its middle and to weaken only when it
    # is so low that everything claims full attention or so high that
    # nothing does. Sweeping 0.15 to 0.70 tests only the flat middle,
    # which cannot confirm or refute the expectation, so 0.05 and 0.95
    # are carried as well.
    # 1.00 closes the range: at that value only the leading
    # region is attended, which is the corner the derivation
    # predicts under a scarce pool. Without it the sweep can
    # report an optimum at the edge of its own grid and has no
    # way to say whether the true one lies beyond it.
    "attention_thr":   [0.05, 0.15, 0.25, 0.35, 0.50, 0.70,
                        0.95, 1.00],
    "horizon_min":     [8.0, 16.0, 24.0, 32.0, 48.0],
    "cycle_min":       [2.0, 4.0, 8.0, 12.0, 20.0],
    "revision_budget": [1, 2, 3, 4, 6],
}
W_SWEEPS = {
    "w_burn":  [0.5, 1.0, 2.0],
    "w_asset": [0.5, 1.0, 2.0],
    "w_pop":   [0.5, 1.0, 2.0],
}

# A FLAT LINE AT ONE OPERATING POINT IS NOT EVIDENCE THAT A GUARDRAIL
# DOES NOTHING. The satisficing bound and the no-harm horizon are
# expected to be inert while the fire is large and the pool is scarce,
# because a harder constraint binds first, and to move only where the
# fire is small enough and the pool ample enough for the guardrail
# itself to become the tightest thing in the loop. The sweep therefore
# runs a second time at the opposite corner of the calibration grid.
# Two dials are enough: these are the two the expectation names.
MARGINAL_POINT = dict(n_ign=1, pool=1.00)
#: EVERY DIAL IS SWEPT IN BOTH REGIMES, NOT ONLY THE TWO GUARDRAILS. A
#: flat line at the operating point can mean that the dial does nothing or
#: that the operating point leaves it no room to act. The comparison is
#: only informative when the same dial is also swept where resources are
#: not the binding constraint, so the marginal regime carries the whole
#: set rather than the two bounds it was first written for.
MARGINAL_SWEEPS = ("j_threshold", "horizon_min", "cycle_min", "eta",
                   "attention_thr")
MARGINAL_ENV_SWEEPS = ("n_sensors", "n_regions")

FIELDS = ["block", "param", "value", "arm", "seed",
          "n_ign", "pool", "n_sensors", "n_regions",
          "j_threshold", "eta", "attention_thr", "cycle_min",
          "horizon_min", "revision_budget",
          "w_burn", "w_asset", "w_pop",
          "j_phys", "j_total", "burned", "burned_ha", "evacuated",
          "affected", "out_min", "fs_frac", "tried_3", "acc_3",
          "rules_final",
          # THE ATTENTION THRESHOLD ACTS THROUGH THE COUNT IT PRODUCES,
          # so the count is what the study has to record. k_mean is the
          # attended regions averaged over the cycles of the run,
          # focus_share the fraction of funded capacity that went to the
          # busiest region, and demand_ratio how far the orders exceeded
          # the budget, which says whether anything was being rationed
          # at all.
          "k_mean", "n_fire_mean", "focus_share", "demand_ratio",
          "seconds"]


# ------------------------------------------------------------- the world
def build_world(seed: int):
    """One landscape family for the whole study.

    A sensitivity sweep varies one thing at a time, so the map has to
    be the constant. It is a mixed landscape with settlements on it,
    under critical fire weather, which is the condition the decision
    layer is meant for.
    """
    cfg = SimConfig(nx=80, ny=60, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=seed, preset="Rolling hills", n_settlements=5,
        population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.08
    w.meteo.wws[:] = 8.0
    return w


def pick_ignitions(w, base, seed: int, n: int):
    """The n burnable spots farthest from the resource bases.

    A fire that starts next to a base is out in minutes whatever the
    parameters say, so the ignition points are chosen where response is
    slowest. The spacing relaxes as n grows, because twelve mutually
    distant points do not exist on a map this size.
    """
    ok = ((w.fuel.fload > 0.4) & (w.fuel.ftype != 0)
          & (w.fuel.ftype != 5) & (w.fuel.ftype != 6))
    ys, xs = np.where(ok)
    order = np.argsort(-base.rtime[ys, xs])
    for gap in (20, 14, 10, 6, 3):
        spots = []
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            if all((x - a) ** 2 + (y - b) ** 2 > gap ** 2
                   for a, b in spots):
                spots.append((x, y))
            if len(spots) == n:
                return spots
    return spots


# --------------------------------------------------------------- one run
def run_point(seed, arm, env, tune, weights):
    t0 = time.time()
    w = build_world(seed)
    base, _ = dss.resource_suggestion(w)
    # RESOURCE LEVEL IS A SCALE ON THE SUGGESTED POOL, so 1.0 is the
    # pool the planner would actually deploy on this map and the sweep
    # reads as "what if a fraction of it were available".
    if env["pool"] != 1.0:
        base.ravail = base.ravail * float(env["pool"])
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    for key, val in weights.items():
        setattr(w.config.cost, key, float(val))
    for x, y in pick_ignitions(w, base, seed, int(env["n_ign"])):
        w.add_ignition(x, y, step=0, radius=1)

    sim = Simulator(w)
    sim.record_states = False
    max_steps = int(round(MAX_MIN / w.config.step_minutes))

    if arm == "freeburn":
        for i in range(max_steps):
            sim.step()
            if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
                break
        rep = compute_costs(sim)
        return dict(j_phys=round(float(rep.j_physical), 5),
                    j_total=round(float(rep.j_total), 5),
                    burned=int(sim.ever_burned.sum()),
                    burned_ha=round(float(rep.burned_area_ha), 2),
                    evacuated=round(float(rep.population_evacuated), 0),
                    affected=round(float(rep.population_exposed), 0),
                    out_min=-1, fs_frac="", tried_3=0, acc_3=0,
                    rules_final=0, seconds=round(time.time() - t0, 1))

    # OBSERVATION IS A DECISION-LAYER INPUT, not a physical one:
    # suppression is aimed at the fire the network reports, so a map
    # the network cannot see is a map the system cannot fight. The
    # dial is the NUMBER OF ASSETS DEPLOYED, not a coverage target:
    # the planner places assets in decreasing order of the risk-
    # weighted coverage they add, so keeping the first k of them is a
    # deployment of size k. A coverage target does not work as a dial,
    # because one satellite and one aerial pass already carry it past
    # any target the sweep could ask for.
    # n_sensors of None means no network, which the decision layer
    # reads as full observation; that is the setting every other block
    # runs under, matching the campaign of the previous section.
    net = None
    if env["n_sensors"] is not None:
        placements, _log = dss.suggest_network(w)
        keep = placements[:int(env["n_sensors"])]
        net = dss.SensorNetwork(
            [dss.Sensor(kind=q["kind"], x=int(q["x"]), y=int(q["y"]))
             for q in keep],
            ny=w.config.ny, nx=w.config.nx,
            cell_m=w.config.cell_size_m, seed=seed)

    adaptive = (arm == "adaptive")
    from dss import adapt as _adapt
    _orig = _adapt._genai_propose
    if adaptive:
        # the same offline proposer the campaign uses, so the gate
        # chain is priced rather than a particular language model
        from campaign5 import _make_template_proposer
        _adapt._genai_propose = _make_template_proposer(seed)
    try:
        eng = dss.DecisionEngine(
            dss.partition_n(w.config.nx, w.config.ny,
                            int(env["n_regions"])),
            base_pool=base, network=net, seed_profile="minimal",
            state_path=dss.isolated_store_path("sens"),
            j_threshold=float(tune["j_threshold"]),
            eta=float(tune["eta"]),
            attention_thr=float(tune["attention_thr"]),
            cycle_min=float(tune["cycle_min"]),
            horizon_min=float(tune["horizon_min"]),
            revision_budget=int(tune["revision_budget"]),
            adapt_on=adaptive, evfis_on=adaptive, genai_on=adaptive)
        if adaptive:
            eng.adapt_cooldown_min = max(
                24.0, float(getattr(eng, "adapt_cooldown_min", 5.0)))
        out_at = None
        for i in range(max_steps):
            # THE NETWORK HAS TO BE STEPPED. Its fused observation is
            # zero until it is updated, and a decision layer reading a
            # zero fire map does nothing at all, so an unstepped
            # network does not measure poor coverage, it measures a
            # broken harness. The application steps it the same way.
            if net is not None:
                net.update(sim, float(w.config.step_minutes))
            sim.step(resource_override=eng.maybe_decide(sim))
            if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
                out_at = (i + 1) * w.config.step_minutes
                break
    finally:
        _adapt._genai_propose = _orig

    ks, nfs, focus, dratio = [], [], [], []
    for c in eng.cycles:
        g = c.get("global_dss") or {}
        if g.get("k") is not None:
            ks.append(float(g["k"]))
            nfs.append(float(g.get("n_fire") or 0))
        pl = c.get("pool") or {}
        if pl.get("focus_share") is not None:
            focus.append(float(pl["focus_share"]))
        if pl.get("demand") and pl.get("budget"):
            dratio.append(float(pl["demand"]) / float(pl["budget"]))
    fs_hits = fs_all = tried3 = acc3 = 0
    for c in eng.cycles:
        ad = c.get("adaptation") or {}
        if int(ad.get("tried") or 0) == 3:
            tried3 += 1
            if ad.get("accepted"):
                acc3 += 1
        for rd in (c.get("regions") or {}).values():
            fs_all += 1
            fs_hits += 1 if rd.get("failsafe") else 0
    rep = compute_costs(sim)
    return dict(
        j_phys=round(float(rep.j_physical), 5),
        j_total=round(float(rep.j_total), 5),
        burned=int(sim.ever_burned.sum()),
        burned_ha=round(float(rep.burned_area_ha), 2),
        evacuated=round(float(rep.population_evacuated), 0),
        affected=round(float(rep.population_exposed), 0),
        out_min=(out_at if out_at is not None else -1),
        fs_frac=round(fs_hits / fs_all, 4) if fs_all else "",
        tried_3=tried3, acc_3=acc3,
        rules_final=len([r for r in eng.rules if r.active]),
        k_mean=round(float(np.mean(ks)), 3) if ks else "",
        n_fire_mean=round(float(np.mean(nfs)), 3) if nfs else "",
        focus_share=round(float(np.mean(focus)), 4) if focus else "",
        demand_ratio=round(float(np.mean(dratio)), 3) if dratio else "",
        seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------- jobs
def _row(block, param, value, arm, seed, env, tune, weights):
    r = dict(block=block, param=param, value=value, arm=arm, seed=seed)
    r.update(env)
    r.update(tune)
    r.update(weights)
    return r


def calibration_jobs():
    jobs = []
    for n in CAL_IGN:
        for seed in CAL_SEEDS:
            env = dict(ENV_BASE, n_ign=n, pool=1.0)
            jobs.append(_row("calibration", "freeburn", n, "freeburn",
                             seed, env, TUNE_BASE, W_BASE))
        for p in CAL_POOL:
            for seed in CAL_SEEDS:
                env = dict(ENV_BASE, n_ign=n, pool=p)
                jobs.append(_row("calibration", "grid", f"{n}|{p}",
                                 "adaptive", seed, env, TUNE_BASE,
                                 W_BASE))
    return jobs


def sweep_jobs(point):
    env0 = dict(ENV_BASE, **point)
    jobs = []
    for param, values in ENV_SWEEPS.items():
        arms = (("static", "adaptive") if param in ENV_TWO_ARM
                else ("adaptive",))
        for v in values:
            for arm in arms:
                for seed in SEEDS:
                    env = dict(env0)
                    env[param] = v
                    jobs.append(_row("environment", param, v, arm,
                                     seed, env, TUNE_BASE, W_BASE))
    for param, values in TUNE_SWEEPS.items():
        for v in values:
            for seed in SEEDS:
                tune = dict(TUNE_BASE)
                tune[param] = v
                jobs.append(_row("tuning", param, v, "adaptive", seed,
                                 env0, tune, W_BASE))
    for param, values in W_SWEEPS.items():
        for v in values:
            for seed in SEEDS:
                wts = dict(W_BASE)
                wts[param] = v
                jobs.append(_row("weights", param, v, "adaptive", seed,
                                 env0, TUNE_BASE, wts))
    envm = dict(ENV_BASE, **MARGINAL_POINT)
    for param in MARGINAL_SWEEPS:
        for v in TUNE_SWEEPS[param]:
            for seed in SEEDS:
                tune = dict(TUNE_BASE)
                tune[param] = v
                jobs.append(_row("marginal", param, v, "adaptive", seed,
                                 envm, tune, W_BASE))
    for param in MARGINAL_ENV_SWEEPS:
        for v in ENV_SWEEPS[param]:
            for seed in SEEDS:
                env = dict(envm)
                env[param] = v
                jobs.append(_row("marginal", param, v, "adaptive", seed,
                                 env, TUNE_BASE, W_BASE))
    return jobs


def _key(r):
    return (r["block"], r["param"], str(r["value"]), r["arm"],
            int(r["seed"]))


def _worker(job):
    try:
        env = {k: job[k] for k in ENV_BASE}
        tune = {k: job[k] for k in TUNE_BASE}
        wts = {k: job[k] for k in W_BASE}
        res = run_point(job["seed"], job["arm"], env, tune, wts)
        return dict(job, **res), None
    except Exception as exc:                       # pragma: no cover
        return job, f"{type(exc).__name__}: {exc}"


def run_jobs(jobs, workers, label):
    done = set()
    if os.path.exists(RUNS):
        with open(RUNS, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(_key(r))
    todo = [j for j in jobs if _key(j) not in done]
    print(f"{label}: {len(todo)} runs to do ({len(jobs) - len(todo)} "
          f"already on disk)")
    if not todo:
        return
    new = not os.path.exists(RUNS)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with open(RUNS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        if workers > 1:
            import multiprocessing as mp
            with mp.Pool(workers) as pool:
                for i, (row, err) in enumerate(
                        pool.imap_unordered(_worker, todo), 1):
                    _emit(w, f, row, err, i, len(todo), t0)
        else:
            for i, job in enumerate(todo, 1):
                row, err = _worker(job)
                _emit(w, f, row, err, i, len(todo), t0)


def _emit(w, f, row, err, i, n, t0):
    if err:
        print(f"  [{i}/{n}] FAILED {row['block']}/{row['param']}="
              f"{row['value']}/{row['arm']}/{row['seed']}: {err}")
        return
    w.writerow(row)
    f.flush()
    el = time.time() - t0
    print(f"  [{i}/{n}] {row['block']}/{row['param']}={row['value']} "
          f"{row['arm']} s{row['seed']}  J={row['j_phys']:.4f} "
          f"burned={row['burned']}  {el / i:.0f}s/run", flush=True)


# --------------------------------------------------- the operating point
BAND = (40.0, 60.0)        # where a parameter can be seen at all
SPREAD_FLAG = 10.0         # the smallest disagreement worth reporting


def choose_point(repoint=False):
    """The cell where the decision layer is neither winning nor losing.

    Read as a share of the free burn of the SAME map, ignition count AND
    SEED: near 0 the fire is beaten whatever the settings, near 100 it
    is lost whatever the settings, and in neither place can a parameter
    be seen. The band between 40 and 60 per cent is where the decision
    actually decides, and among the settings inside it the LEAST SEVERE
    one is taken, that is the smallest fire load and then the largest
    pool. Sitting at the edge of the design space would make the result
    a statement about an extreme rather than about the system.

    THE DENOMINATOR IS THE SAME SEED, not the mean of the seeds. Any
    other choice measures one world against another world's reference.

    THE SPREAD OVER SEEDS IS PRINTED for every cell, and a cell whose
    worlds fall on different sides of the band is marked. A mean of 54
    built from 28 and 80 is not a contested world; it is one lost world
    and one held world, and whoever reads the operating point off this
    table has to be told that.
    """
    rows = [r for r in csv.DictReader(open(RUNS, encoding="utf-8"))
            if r["block"] == "calibration"]
    free = {}
    for r in rows:
        if r["arm"] == "freeburn":
            free[(int(r["n_ign"]), int(r["seed"]))] = float(r["j_phys"])
    cells = {}
    for r in rows:
        if r["arm"] != "adaptive":
            continue
        n = int(r["n_ign"])
        base = free.get((n, int(r["seed"])))
        if not base or base <= 0:
            continue
        cells.setdefault((n, float(r["pool"])), []).append(
            100.0 * float(r["j_phys"]) / base)
    if not cells:
        raise SystemExit("no calibration rows with a free-burn reference")
    table = [(n, p, float(np.mean(v)), min(v), max(v), len(v))
             for (n, p), v in sorted(cells.items())]

    def regime(s):
        return 0 if s < BAND[0] else (1 if s <= BAND[1] else 2)

    band = [t for t in table if BAND[0] <= t[2] <= BAND[1]]
    if band:
        near = min(abs(t[2] - 50.0) for t in band)
        # settings whose severity is equivalent within one point are
        # not distinguishable by this grid, so the more ordinary
        # incident is preferred: fewer simultaneous starts first, then
        # the larger pool
        tied = [t for t in band if abs(t[2] - 50.0) <= near + 1.0]
        tied.sort(key=lambda t: (t[0], -t[1]))
        best = (tied[0][0], tied[0][1])
    else:
        best = min(table, key=lambda t: abs(t[2] - 50.0))[:2]
    print("\ncalibration grid, as a share of the free burn of the same "
          "world and seed:")
    for n, p, share, lo, hi, k in table:
        mark = "  <-- operating point" if (n, p) == best else ""
        warn = ("  WORLDS DISAGREE ON THE REGIME"
                if regime(lo) != regime(hi) and hi - lo > SPREAD_FLAG
                else "")
        print(f"  ignitions {n:2d}  pool {p:.2f}  {share:5.1f}%  "
              f"[{lo:5.1f}, {hi:5.1f}] over {k} worlds{mark}{warn}")

    point = dict(n_ign=best[0], pool=best[1])
    old = None
    if os.path.exists(POINT):
        with open(POINT, encoding="utf-8") as f:
            old = json.load(f)
    swept = False
    if os.path.exists(RUNS):
        with open(RUNS, encoding="utf-8") as f:
            swept = any(r["block"] != "calibration"
                        for r in csv.DictReader(f))
    moved = (old is not None
             and (int(old["n_ign"]), float(old["pool"]))
             != (point["n_ign"], point["pool"]))
    if moved and swept and not repoint:
        # THE RESUME KEY DOES NOT RECORD THE OPERATING POINT. Moving the
        # point while sweep rows are already on disk would leave one
        # file holding two studies, with no column in it to tell them
        # apart, and every later figure would average across both. The
        # stored point is therefore kept and the disagreement reported.
        print(f"\n*** the grid now prefers {point}, but "
              f"{os.path.basename(POINT)} holds {old} and "
              f"{os.path.basename(RUNS)} already contains sweep rows "
              "taken at the stored point.")
        print("*** KEEPING THE STORED POINT. To move it, delete the "
              "non-calibration rows of sens_runs.csv and rerun with "
              "--repoint.")
        return old
    with open(POINT, "w", encoding="utf-8") as f:
        json.dump(point, f, indent=1)
    if moved:
        print(f"\n*** the operating point has moved from {old} "
              f"to {point}.")
    print(f"\noperating point -> {point}")
    return point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=("calibrate", "sweep", "all"))
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = all cores minus one")
    ap.add_argument("--repoint", action="store_true",
                    help="allow the calibration to move the operating "
                         "point even though sweep rows already exist; "
                         "delete those rows first, or the file will hold "
                         "two studies at once")
    a = ap.parse_args()
    workers = a.workers or max(1, (os.cpu_count() or 2) - 1)

    if a.phase in ("calibrate", "all"):
        run_jobs(calibration_jobs(), workers, "calibration")
        point = choose_point(a.repoint)
    else:
        with open(POINT, encoding="utf-8") as f:
            point = json.load(f)
        print(f"operating point (from {os.path.basename(POINT)}): "
              f"{point}")

    if a.phase in ("sweep", "all"):
        run_jobs(sweep_jobs(point), workers, "sweeps")
    print(f"\ndone -> {RUNS}")


if __name__ == "__main__":
    main()
