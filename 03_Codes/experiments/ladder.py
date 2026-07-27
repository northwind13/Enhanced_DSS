"""Monte Carlo evaluation ladder for Chapter 5, Section 5.4.

One campaign fills Tables I-IV and Figures A-E of the thesis:

  arms      A0 no-DSS | A1 minimal static (5 rules) | A2 +evFIS
            A2g +GenAI only | A3 minimal + full adaptation
            A4 full static (40 rules) | A5 deployed (full + full)
  scenarios S-A nominal (resources sufficient)
            S-B capacity-limited (2 remote ignitions, pool x0.6)
            S-C S-B + degraded observation (outage + noise)

Every (scenario, seed) pair generates ONE world (map, ignitions,
weather); all arms replay the identical world, so differences are
attributable to the decision layer alone.  A0 additionally serves as
the no-action counterfactual of the same history.

Outputs (experiments/out/):
  ladder_runs.csv    one row per run: J components at checkpoints and
                     at the end, burned, time-to-out, success,
                     fail-safe fraction, coverage, adaptation counters
  ladder_curves.csv  burned area + J_phys time series (Figure A)
  ladder_funnel.csv  stage-3 gate funnel events (Table IV / Figure D)
  ladder_table1.csv  scenario x arm aggregate (mean +/- 95% CI)
  ladder_table2.csv  gap-closure summary (Table II)
  ladder_table3.csv  adaptation mechanism summary (Table III)

The campaign is RESUMABLE: finished (scenario, arm, seed) rows are
skipped on restart.  Default N = 50 paired seeds per scenario
(3 x 7 x 50 = 1050 runs).  Use --seeds / --scenarios / --arms
to split the campaign across sessions, e.g.:

  python experiments/ladder.py --seeds 50
  python experiments/ladder.py --scenarios SB --arms A1,A3,A4
  python experiments/ladder.py --quick        # smoke test (N=3)

Stage 3 uses the Claude API when ANTHROPIC_API_KEY is set and the
deterministic template proposer otherwise; the proposer actually used
is recorded per run in the 'genai_source' column.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

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
CHECKPOINTS_MIN = (120.0, 360.0)      # thesis Table I checkpoints
COVER_THR = 0.45                      # gap threshold (Chapter 4)

ARMS = {
    # name: (seed_profile, engine kwargs or None for no DSS)
    # thesis Table 5.6 (merged 5.4): seven-arm ablation ladder
    "A0":  ("full",    None),                                          # no DSS
    "A1":  ("minimal", dict(adapt_on=False)),                          # 5 rules, static
    "A2":  ("minimal", dict(adapt_on=True, evfis_on=True,  genai_on=False)),
    "A2g": ("minimal", dict(adapt_on=True, evfis_on=False, genai_on=True)),
    "A3":  ("minimal", dict(adapt_on=True, evfis_on=True,  genai_on=True)),
    "A4":  ("full",    dict(adapt_on=False)),                          # 40 rules, static
    "A5":  ("full",    dict(adapt_on=True, evfis_on=True,  genai_on=True)),
}
SCENARIOS = ("SA", "SB", "SC")
PLAN = [(s, a) for s in SCENARIOS for a in ARMS]


# ----------------------------------------------------------- worlds
def build_world(seed: int, scenario: str):
    cfg = SimConfig(nx=80, ny=60, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=seed, preset="Rolling hills", n_settlements=5,
        population_per_settlement=15000)
    # scenario severity (same map family, different fire weather)
    if scenario == "SA":
        w.fuel.fmoist[:] = 0.12
        w.meteo.wws[:] = 4.0
    else:                       # SB / SC: dry, windy, capacity-limited
        w.fuel.fmoist[:] = 0.08
        w.meteo.wws[:] = 8.0
    return w


def pick_ignitions(w, base, seed: int, scenario: str):
    """SA: one moderately reachable spot.  SB/SC: the two burnable
    spots FARTHEST from the bases (capacity-limited by construction,
    as in experiments/sensitivity.py)."""
    ok = (w.fuel.fload > 0.4) & (w.fuel.ftype != 0) \
        & (w.fuel.ftype != 5) & (w.fuel.ftype != 6)
    ys, xs = np.where(ok)
    order = np.argsort(-base.rtime[ys, xs])
    if scenario == "SA":
        mid = order[len(order) // 2]        # median travel time
        return [(int(xs[mid]), int(ys[mid]))]
    spots = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all((x - a) ** 2 + (y - b) ** 2 > 20 ** 2 for a, b in spots):
            spots.append((x, y))
        if len(spots) == 2:
            break
    return spots


# --------------------------------------- degraded observation (S-C)
_ORIG_FEATURES = dss_loop.ten_features
_ORIG_CONF = dss_loop.feature_confidence


def _install_outage(rng, outage=0.25, noise=0.10):
    """S-C shim: per decision cycle each region's features may be
    stale/noisy; confidences are scaled down accordingly, so the
    confidence gate of Layer 3 sees the degradation it was built for."""
    def feats(sim, r, network=None, pool=None):
        f = _ORIG_FEATURES(sim, r, network=network, pool=pool)
        if rng.random() < outage:
            return {k: float(np.clip(v + rng.normal(0.0, noise), 0, 1))
                    for k, v in f.items()}
        return f

    def conf(network, r):
        c = _ORIG_CONF(network, r)
        scale = 1.0 - outage * 0.8
        return {k: float(v) * scale for k, v in c.items()}

    dss_loop.ten_features = feats
    dss_loop.feature_confidence = conf


def _remove_outage():
    dss_loop.ten_features = _ORIG_FEATURES
    dss_loop.feature_confidence = _ORIG_CONF


# ----------------------------------------------------------- one run
def run_once(scenario: str, arm: str, seed: int, max_hours: float,
             curves_writer=None, funnel_writer=None):
    profile, ekw = ARMS[arm]
    w = build_world(seed, scenario)
    base, _ = dss.resource_suggestion(w)
    if scenario in ("SB", "SC"):
        base.rcap *= 0.6                     # capacity-limited pool
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    for x, y in pick_ignitions(w, base, seed, scenario):
        w.add_ignition(x, y, step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False

    rng = np.random.default_rng(10_000 + seed)
    if scenario == "SC":
        _install_outage(rng)

    eng = None
    if ekw is not None:
        eng = dss.DecisionEngine(
            dss.partition_n(w.config.nx, w.config.ny, 4),
            base_pool=base, seed_profile=profile,
            # one store per run: the rungs of a ladder must not inherit
            # each other's learning (dss.isolated_store_path)
            state_path=dss.isolated_store_path("ladder"), **ekw)

    max_steps = int(round(max_hours * 60.0 / w.config.step_minutes))
    out_at = None
    cp = {}                                   # checkpoint cost reports
    try:
        for i in range(max_steps):
            ov = eng.maybe_decide(sim) if eng is not None else None
            sim.step(resource_override=ov)
            t_min = (i + 1) * w.config.step_minutes
            if curves_writer is not None and (i % 5 == 0):
                rep_i = compute_costs(sim)
                curves_writer.writerow([scenario, arm, seed,
                                        round(t_min, 1),
                                        int(sim.ever_burned.sum()),
                                        round(float(getattr(
                                            rep_i, "j_physical",
                                            rep_i.j_total)), 5)])
            for c in CHECKPOINTS_MIN:
                if c not in cp and t_min >= c:
                    cp[c] = compute_costs(sim)
            if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
                out_at = t_min
                break
    finally:
        if scenario == "SC":
            _remove_outage()

    rep = compute_costs(sim)

    def costrow(r, tag):
        if r is None:
            return {f"{tag}_{k}": "" for k in
                    ("j_burn", "j_asset", "j_pop", "j_resp",
                     "j_delay", "j_total", "j_phys")}
        return {
            f"{tag}_j_burn":  round(float(r.j_burn), 5),
            f"{tag}_j_asset": round(float(r.j_asset), 5),
            f"{tag}_j_pop":   round(float(r.j_pop), 5),
            f"{tag}_j_resp":  round(float(r.j_resp), 5),
            f"{tag}_j_delay": round(float(getattr(r, "j_delay",
                                                  float("nan"))), 5),
            f"{tag}_j_total": round(float(r.j_total), 5),
            f"{tag}_j_phys":  round(float(getattr(r, "j_physical",
                                                  float("nan"))), 5),
        }

    row = dict(scenario=scenario, arm=arm, seed=seed,
               profile=profile,
               burned_cells=int(sim.ever_burned.sum()),
               burned_ha=round(int(sim.ever_burned.sum())
                               * (w.config.cell_size_m ** 2) / 1e4, 2),
               out_min=(out_at if out_at is not None else -1),
               success=int(out_at is not None))
    row.update(costrow(cp.get(CHECKPOINTS_MIN[0]), "t2h"))
    row.update(costrow(cp.get(CHECKPOINTS_MIN[1]), "t6h"))
    row.update(costrow(rep, "end"))

    # ---- mechanism metrics from the cycle chronicle ----
    tried = {1: 0, 2: 0, 3: 0}
    accepted = {1: 0, 2: 0, 3: 0}
    dj_sum = {1: 0.0, 2: 0.0, 3: 0.0}
    fs_hits = fs_all = 0
    cov_hits = cov_all = 0
    share_num = share_den = 0.0
    genai_source = ""
    if eng is not None:
        for c in eng.cycles:
            ad = c.get("adaptation") or {}
            st = int(ad.get("tried") or 0)
            if st in tried:
                tried[st] += 1
                if ad.get("accepted"):
                    accepted[st] += 1
                    dj_sum[st] += float(ad.get("dJ") or 0.0)
                info = ad.get("info") or {}
                if st == 3:
                    genai_source = info.get("source", genai_source)
                    if funnel_writer is not None:
                        funnel_writer.writerow([
                            scenario, arm, seed,
                            info.get("source", ""),
                            bool(info.get("package")),
                            info.get("gate", ""),
                            bool(ad.get("accepted")),
                            round(float(ad.get("dJ") or 0.0), 5)])
            for rd in (c.get("regions") or {}).values():
                fs_all += 1
                if rd.get("failsafe"):
                    fs_hits += 1
                fired = rd.get("fired") or []
                wmax = max((fw for _n, fw in fired), default=0.0)
                cov_all += 1
                if wmax >= COVER_THR:
                    cov_hits += 1
                for name, fw in fired:
                    share_den += fw
                    if name and name[0] in ("A", "G"):
                        share_num += fw
    row.update(
        rules_final=(len([r for r in eng.rules if r.active])
                     if eng is not None else 0),
        coverage=round(cov_hits / cov_all, 3) if cov_all else "",
        fs_frac=round(fs_hits / fs_all, 3) if fs_all else "",
        adapt_share=round(share_num / share_den, 4) if share_den else 0.0,
        tried_1=tried[1], tried_2=tried[2], tried_3=tried[3],
        acc_1=accepted[1], acc_2=accepted[2], acc_3=accepted[3],
        dj_1=round(dj_sum[1], 4), dj_2=round(dj_sum[2], 4),
        dj_3=round(dj_sum[3], 4),
        genai_source=genai_source)
    return row


# --------------------------------------------------------- aggregate
def _ci(vals):
    v = [x for x in vals if isinstance(x, (int, float))
         and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return "", ""
    m = float(np.mean(v))
    h = 1.96 * float(np.std(v, ddof=1)) / math.sqrt(len(v)) \
        if len(v) > 1 else 0.0
    return round(m, 4), round(h, 4)


def aggregate(rows):
    keys = ["end_j_burn", "end_j_asset", "end_j_pop", "end_j_resp",
            "end_j_delay", "end_j_total", "end_j_phys",
            "t2h_j_phys", "t6h_j_phys", "burned_ha", "out_min",
            "success", "coverage", "fs_frac", "adapt_share",
            "rules_final"]
    t1 = []
    for s in SCENARIOS:
        for a in ARMS:
            sub = [r for r in rows if r["scenario"] == s
                   and r["arm"] == a]
            if not sub:
                continue
            rec = dict(scenario=s, arm=a, n=len(sub))
            for k in keys:
                m, h = _ci([r.get(k) for r in sub])
                rec[k] = m
                rec[k + "_ci"] = h
            t1.append(rec)
    # gap closure (Table II): SB panel
    def _mean(arm, key="end_j_phys"):
        v = [r[key] for r in rows if r["scenario"] == "SB"
             and r["arm"] == arm and isinstance(r.get(key), (int, float))]
        return float(np.mean(v)) if v else float("nan")
    a4, a1, a3 = _mean("A4"), _mean("A1"), _mean("A3")
    gap = ((a1 - a3) / (a1 - a4) * 100.0
           if a1 == a1 and a4 == a4 and abs(a1 - a4) > 1e-9
           else float("nan"))
    t2 = [dict(config="A4 full static (upper ref)", j_phys=round(a4, 4)),
          dict(config="A1 minimal static (lower ref)", j_phys=round(a1, 4)),
          dict(config="A3 minimal + full adaptation", j_phys=round(a3, 4)),
          dict(config="gap to A4 closed (%)", j_phys=round(gap, 1))]
    # mechanism (Table III)
    t3 = []
    for a in ("A2", "A2g", "A3", "A5"):
        sub = [r for r in rows if r["arm"] == a]
        if not sub:
            continue
        t3.append(dict(
            arm=a, n=len(sub),
            tried=f"{sum(r['tried_1'] for r in sub)}/"
                  f"{sum(r['tried_2'] for r in sub)}/"
                  f"{sum(r['tried_3'] for r in sub)}",
            accepted=f"{sum(r['acc_1'] for r in sub)}/"
                     f"{sum(r['acc_2'] for r in sub)}/"
                     f"{sum(r['acc_3'] for r in sub)}",
            dj=f"{sum(r['dj_1'] for r in sub):.3f}/"
               f"{sum(r['dj_2'] for r in sub):.3f}/"
               f"{sum(r['dj_3'] for r in sub):.3f}",
            mean_rules_final=round(float(np.mean(
                [r["rules_final"] for r in sub])), 1),
            mean_adapt_share=round(float(np.mean(
                [r["adapt_share"] for r in sub])), 4)))
    return t1, t2, t3


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                              delimiter=";")
        wcsv.writeheader()
        wcsv.writerows(rows)
    print("wrote", path)


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50,
                    help="paired seeds per scenario (default 50)")
    ap.add_argument("--seed0", type=int, default=100,
                    help="first seed (worlds are seed-generated)")
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--arms", default="")
    ap.add_argument("--quick", action="store_true",
                    help="N=3 smoke test")
    args = ap.parse_args()
    n = 3 if args.quick else args.seeds
    scen = [s.strip().upper().replace("-", "")
            for s in args.scenarios.split(",") if s.strip()]
    armf = [a.strip() for a in args.arms.split(",") if a.strip()]
    os.makedirs(OUT, exist_ok=True)

    runs_path = os.path.join(OUT, "ladder_runs.csv")
    done = set()
    rows = []
    if os.path.exists(runs_path):
        with open(runs_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter=";"):
                for k, v in list(r.items()):
                    try:
                        r[k] = float(v) if "." in str(v) else int(v)
                    except (ValueError, TypeError):
                        pass
                rows.append(r)
                done.add((r["scenario"], r["arm"], int(r["seed"])))
        print(f"resuming: {len(done)} runs already logged")

    curves_f = open(os.path.join(OUT, "ladder_curves.csv"), "a",
                    newline="", encoding="utf-8-sig")
    funnel_f = open(os.path.join(OUT, "ladder_funnel.csv"), "a",
                    newline="", encoding="utf-8-sig")
    cw = csv.writer(curves_f, delimiter=";")
    fw = csv.writer(funnel_f, delimiter=";")
    if curves_f.tell() == 0:
        cw.writerow(["scenario", "arm", "seed", "t_min",
                     "burned_cells", "j_phys"])
    if funnel_f.tell() == 0:
        fw.writerow(["scenario", "arm", "seed", "source", "package",
                     "gate_rejected_at", "accepted", "dJ"])

    todo = [(s, a) for s, a in PLAN if s in scen
            and (not armf or a in armf)]
    total = len(todo) * n
    k = 0
    for s, a in todo:
        for i in range(n):
            seed = args.seed0 + i
            k += 1
            if (s, a, seed) in done:
                continue
            row = run_once(s, a, seed, args.max_hours,
                           curves_writer=cw, funnel_writer=fw)
            rows.append(row)
            new_file = not os.path.exists(runs_path) \
                or os.path.getsize(runs_path) == 0
            with open(runs_path, "a", newline="",
                      encoding="utf-8-sig") as f:
                wcsv = csv.DictWriter(f, fieldnames=list(row.keys()),
                                      delimiter=";")
                if new_file:
                    wcsv.writeheader()
                wcsv.writerow(row)
            curves_f.flush()
            funnel_f.flush()
            print(f"[{k}/{total}] {s} {a} seed={seed} "
                  f"Jphys={row['end_j_phys']} burned={row['burned_ha']}ha "
                  f"out={row['out_min']} adapt={row['adapt_share']}")
    curves_f.close()
    funnel_f.close()

    t1, t2, t3 = aggregate(rows)
    _write_csv(os.path.join(OUT, "ladder_table1.csv"), t1)
    _write_csv(os.path.join(OUT, "ladder_table2.csv"), t2)
    _write_csv(os.path.join(OUT, "ladder_table3.csv"), t3)
    print("done:", len(rows), "runs logged")


if __name__ == "__main__":
    main()
