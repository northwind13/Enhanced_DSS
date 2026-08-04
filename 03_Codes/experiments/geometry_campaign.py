"""Does the geometry diagnosis reach the generative stage, or stop at it?

The decision layer now measures one thing and puts it in front of stage
3: the pool is spent, the fire is growing through it, so what remains
open is WHERE the committed capacity is placed rather than how much of
it there is. Nothing selects what the stage proposes. The open question
is therefore empirical: does that evidence change what gets proposed?

Before the diagnosis existed, 83 recorded proposals produced ZERO clause
actuators. Every admitted intervention was a composite over channels
that already existed. The capability was implemented, prompted, gated
and documented, and never once exercised. This campaign is what turns
that from an anecdote into a measurement.

Two arms over the same worlds, the same seeds and the same gates:

  on    the diagnosis is raised when the loop measures it
  off   the diagnosis is suppressed, everything else identical

WHY THE OFF ARM IS NOT OPTIONAL. Without it a clause actuator appearing
in the on arm proves nothing: the model is stochastic and the earlier
zero came from a different prompt, a different rule base and a different
set of runs. The two arms differ in one line, so a difference between
them belongs to that line.

WHAT IS COUNTED. Proposals are classified by what they carry, not by
what they say: a package with non-empty `clauses` is a clause actuator,
a package with a `composition` is a composite, a package with a
`new_concept` is a concept, and anything else is a plain rule. Gate
rejections are counted by gate, because a campaign in which the stage
proposes clause actuators and loses all of them at G3 is a different
result from one in which it never proposes them, and the two would look
identical in a table that counted admissions alone.

Stage 3 needs a reachable model, so the default transport is the Claude
Code CLI. `--offline` swaps in the deterministic template proposer of
offline_proposer.py; that arm cannot produce a clause actuator by
construction and exists only to prove the harness runs.

    python experiments/geometry_campaign.py --seeds 5
    python experiments/geometry_campaign.py --seeds 5 --arms on
    python experiments/geometry_campaign.py --offline --seeds 2

Resumable: a finished (arm, seed) row is skipped.
Output: experiments/out/geometry_campaign.csv
        experiments/out/geometry_proposals.jsonl
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
from dss import adapt as _adapt                       # noqa: E402
from disaster_phyengine.core import Simulator         # noqa: E402
from disaster_phyengine.costs import compute_costs    # noqa: E402
from scenario import build_world, pick_ignitions      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
CSV_PATH = os.path.join(OUT, "geometry_campaign.csv")
JSONL_PATH = os.path.join(OUT, "geometry_proposals.jsonl")

#: THE POOL IS DELIBERATELY SHORT. The diagnosis is defined on a
#: saturated pool, so a campaign run at full strength would measure a
#: condition that never arises: at pool 1.0 with two ignitions it was
#: raised in 0 of 15 cycles, against 12 of 15 at 0.5. Four ignitions on
#: a quarter pool is the operating point of Section 5.5, and it
#:  saturates.
POOL = 0.25
N_IGN = 4
N_REGIONS = 4
CYCLE_MIN = 12.0
# THE NO-HARM HORIZON IS A PARAMETER OF THIS EXPERIMENT, not a constant.
# Every clause actuator of the first slice was a PREVENTIVE tactic, a
# coated belt or a cleared band around the assets, and every one of them
# died at G5 because neither reseeded rollout showed a gain. A coating
# does not pay inside twenty-four minutes: the fire has not reached the
# band yet. Whether the gate is rejecting bad tactics or rejecting slow
# ones is therefore a question about the horizon, and it is answered by
# sweeping it rather than by loosening the gate.
HORIZON_MIN = 24.0
J_TH = 0.35
ETA = 0.60
TAU_ATT = 0.35
MAX_MIN = 240.0

FIELDS = ["arm", "seed", "horizon_min", "patience", "earliest",
          "cycles", "diagnosed", "diag_share",
          "stage3_attempts", "accepted", "rule", "concept", "composite",
          "clause", "clause_accepted", "rej_G1", "rej_G2", "rej_G3",
          "rej_G4G5", "rej_G5b", "rej_other", "genai_retired", "burned_ha",
          "j_total", "seconds"]


def _classify(prop):
    """What kind of object a proposal carries.

    Read from the payload rather than from the prose: a proposal that
    talks about a new tactic and ships a re-weighting of two existing
    channels is a composite, whatever it calls itself.
    """
    if not isinstance(prop, dict):
        return "rule"
    if prop.get("new_concept"):
        return "concept"
    ni = prop.get("new_intervention") or {}
    if ni.get("clauses"):
        return "clause"
    if ni.get("composition") or ni:
        return "composite"
    return "rule"


def _gate_bucket(gate):
    g = str(gate or "")
    if g.startswith("G1"):
        return "rej_G1"
    if g.startswith("G2"):
        return "rej_G2"
    if g.startswith("G3"):
        return "rej_G3"
    if g.startswith("G5b"):
        return "rej_G5b"
    if g.startswith("G4") or g.startswith("G5"):
        return "rej_G4G5"
    return "rej_other"


#: HOW MANY CONSECUTIVE REJECTIONS RETIRE STAGE 3, and from which step
#: it may be picked at all. Both slices so far put every clause actuator
#: at step 0, because the bandit tries the untried stage first and three
#: early rejections then retire it for the rest of the run. At step 0 the
#: fire is nineteen cells and no asset lies within the fifteen-cell reach
#: of the assets sector, so a preventive shield is measured exactly where
#: it can do least. Raising the patience and holding stage 3 back moves
#: the same proposal to a moment when the fire is near what it defends.
PATIENCE = 3
EARLIEST_STEP = 0


def run_one(arm: str, seed: int, offline: bool,
            horizon_min: float = HORIZON_MIN,
            patience: int = PATIENCE,
            earliest_step: int = EARLIEST_STEP):
    w = build_world(seed)
    base, _ = dss.resource_suggestion(w)
    base.ravail = base.ravail * POOL
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    for x, y in pick_ignitions(w, base, seed, N_IGN):
        w.add_ignition(x, y, step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    eng = dss.DecisionEngine(
        dss.partition_n(w.config.nx, w.config.ny, N_REGIONS),
        base_pool=base,
        state_path=dss.isolated_store_path(f"geom_{arm}_{seed}"),
        cycle_min=CYCLE_MIN, horizon_min=float(horizon_min),
        j_threshold=J_TH, eta=ETA, attention_thr=TAU_ATT,
        adapt_on=True, evfis_on=True, genai_on=True,
        genai_patience=int(patience),
        seed_profile="full")

    # STAGE 3 IS HELD BACK, NOT DISABLED. The stage returns to the menu
    # the moment the step count passes, so the run still spends the same
    # budget on it and only the moment it is spent moves. The switch is
    # flipped on the INSTANCE inside the step loop below; a descriptor on
    # the class would have leaked into every other engine built in the
    # same process, which is how an experiment quietly contaminates the
    # run that follows it.
    _full_menu = eng.stages_allowed
    if earliest_step > 0:
        eng.stages_allowed = (tuple(x for x in _full_menu if x != 3)
                              or _full_menu)

    if arm == "off":
        # ONE LINE IS THE WHOLE DIFFERENCE between the arms. Retiring the
        # method rather than raising the threshold keeps the change
        # visible to anyone reading this file: a threshold set to a
        # large number reads like tuning, and this is not tuning.
        eng._diagnose_geometry = lambda *a, **k: None

    _orig_prop = _adapt._genai_propose
    if offline:
        from offline_proposer import make_template_proposer
        _adapt._genai_propose = make_template_proposer(seed)

    t0 = time.time()
    steps = int(round(MAX_MIN / w.config.step_minutes))
    try:
        for step_i in range(steps):
            if earliest_step and step_i >= earliest_step:
                eng.stages_allowed = _full_menu
            sim.step(resource_override=eng.maybe_decide(sim))
            if int((sim.state.burning > 0.5).sum()) == 0:
                break
    finally:
        _adapt._genai_propose = _orig_prop

    rep = compute_costs(sim)
    cell_ha = (w.config.cell_size_m ** 2) / 10000.0
    n_cyc = len(eng.cycles)
    diagnosed = int(eng.run_stats.get("geometry_diagnosis", 0))

    row = {f: 0 for f in FIELDS}
    row.update(arm=arm, seed=seed, horizon_min=float(horizon_min),
               patience=int(patience), earliest=int(earliest_step),
               cycles=n_cyc, diagnosed=diagnosed,
               diag_share=round(diagnosed / max(n_cyc, 1), 3),
               genai_retired=int(bool(getattr(eng, "_genai_dead", False))),
               burned_ha=round(float(sim.ever_burned.sum()) * cell_ha, 2),
               j_total=round(float(rep.j_total), 6),
               seconds=round(time.time() - t0, 1))

    # THE PROPOSAL LEDGER IS THE EVIDENCE, not the admitted store: a
    # clause actuator that was proposed and rejected still answers the
    # question this campaign asks.
    proposals = []
    gst = getattr(eng, "gstate", None)
    for p in (getattr(gst, "data", {}) or {}).get("genai_proposals", []):
        raw = p.get("repaired") or p.get("raw")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = None
        kind = _classify(raw)
        row["stage3_attempts"] += 1
        row[kind] += 1
        if p.get("accepted"):
            row["accepted"] += 1
            if kind == "clause":
                row["clause_accepted"] += 1
        else:
            row[_gate_bucket(p.get("gate"))] += 1
        # THE MEASURED BLOCK IS THE POINT, not the verdict. A package
        # rejected at G5 failed because neither reseeded rollout beat the
        # baseline, and only these numbers say whether it missed by a
        # hair or by a mile. The first slice did not carry them and the
        # question could not be answered afterwards.
        proposals.append(dict(arm=arm, seed=seed, kind=kind,
                              horizon_min=float(horizon_min),
                              accepted=bool(p.get("accepted")),
                              gate=p.get("gate"),
                              detail=str(p.get("detail") or "")[:200],
                              step=p.get("step"),
                              measured=p.get("measured"),
                              payload=raw))
    return row, proposals


def _done():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, encoding="utf-8") as f:
        return {(r["arm"], r["seed"]) for r in csv.DictReader(f)}


def _append(row, proposals):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            wtr.writeheader()
        wtr.writerow(row)
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        for p in proposals:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--arms", default="on,off")
    ap.add_argument("--offline", action="store_true",
                    help="deterministic template proposer instead of the "
                         "model; proves the harness, cannot produce a "
                         "clause actuator")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=PATIENCE,
                    help="consecutive rejections before stage 3 retires")
    ap.add_argument("--earliest", type=int, default=EARLIEST_STEP,
                    help="simulation step before which stage 3 is held "
                         "back (30 steps = 60 minutes)")
    ap.add_argument("--fresh", action="store_true",
                    help="start this tag from nothing. The existing "
                         "files are renamed with a timestamp rather "
                         "than deleted, because a finished slice is "
                         "evidence for what the system did before a "
                         "change and cannot be produced again")
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many seeds at once. The campaign "
                         "waits on model calls rather than on the "
                         "processor, so four workers finish in close to "
                         "a quarter of the time")
    ap.add_argument("--csv", default="",
                    help="write to this file instead of the default")
    ap.add_argument("--seed-list", default="", dest="seed_list",
                    help="internal: the exact seeds this worker owns")
    ap.add_argument("--horizon", type=float, default=HORIZON_MIN,
                    help="no-harm horizon in minutes (24 is the "
                         "operating point of Section 5.5)")
    a = ap.parse_args()

    arms = [x for x in a.arms.split(",") if x]
    global CSV_PATH, JSONL_PATH
    if a.csv:
        # a worker writes both of its files beside each other
        CSV_PATH = a.csv
        JSONL_PATH = a.csv.replace(".csv", ".jsonl")
    # THE GATE CHAIN IS PART OF THE SLICE IDENTITY. G5b, the attribution
    # gate, changes which packages are admitted, so a slice measured with
    # it must never be appended to or averaged with one measured without
    # it. The tag carries the gate so the two cannot be confused on disk.
    tagbits = ["g5b"] if getattr(_adapt, "G5B_MARGIN", None) is not None \
        else []
    if abs(a.horizon - HORIZON_MIN) > 1e-9:
        tagbits.append(f"h{int(a.horizon)}")
    if a.patience != PATIENCE:
        tagbits.append(f"p{a.patience}")
    if a.earliest != EARLIEST_STEP:
        tagbits.append(f"s{a.earliest}")
    # A WORKER IS HANDED ITS EXACT PATH and must not tag it again: the
    # parent already applied the tag, and a second pass produced shard
    # names like geometry_campaign_h99.w1_h99.csv that no merge step
    # would ever look for.
    if tagbits and not a.csv:
        tag = "_" + "".join(tagbits)
        CSV_PATH = CSV_PATH.replace(".csv", tag + ".csv")
        JSONL_PATH = JSONL_PATH.replace(".jsonl", tag + ".jsonl")

    seeds = ([int(x) for x in a.seed_list.split(",") if x]
             or [201 + i for i in range(a.seeds)])

    if a.fresh:
        # ARCHIVED, NOT DELETED. A finished slice records what the
        # system did under one state of the code and cannot be produced
        # again once that state is gone, so starting over renames it.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for p in (CSV_PATH, JSONL_PATH):
            if os.path.exists(p):
                root, ext = os.path.splitext(p)
                keep = f"{root}.archived_{stamp}{ext}"
                os.rename(p, keep)
                print("archived:", os.path.basename(keep))

    # SAY WHAT IS ABOUT TO HAPPEN. A resumable campaign that finds its
    # work already done exits at once, which reads like a failure to
    # start unless it says why.
    _pre = _done()
    _want = [(arm, s_) for s_ in seeds for arm in arms]
    _left = [(arm, s_) for arm, s_ in _want if (arm, str(s_)) not in _pre]
    print("=" * 62)
    print("campaign :", os.path.basename(CSV_PATH))
    print("proposals:", os.path.basename(JSONL_PATH))
    print(f"settings : horizon {a.horizon:.0f} min, patience "
          f"{a.patience}, earliest step {a.earliest}, "
          f"workers {a.workers}")
    print(f"planned  : {len(_want)} run(s) over arms {arms} "
          f"and seeds {seeds}")
    print(f"on file  : {len(_pre)} finished run(s)")
    print(f"to run   : {len(_left)}"
          + ("" if _left else "   (nothing left; pass --fresh to "
                              "start this tag over)"))
    for arm, s_ in _left:
        print(f"           {arm} seed {s_}")
    print("=" * 62, flush=True)
    if False:
        # a different horizon is a different experiment, so it gets its
        # own files instead of appending rows that look comparable
        tag = f"_h{int(a.horizon)}"
        CSV_PATH = CSV_PATH.replace(".csv", tag + ".csv")
        JSONL_PATH = JSONL_PATH.replace(".jsonl", tag + ".jsonl")
    # ---- FAN OUT OVER SEEDS. Each worker owns whole seeds and writes
    # its own shard, so two processes never append to one file and a
    # crashed worker cannot leave a half-written row behind. The shards
    # are merged when they all return.
    if a.workers > 1 and not a.csv:
        import subprocess
        # RESUME BEFORE FANNING OUT. A worker checks its own shard for
        # finished rows, and a shard starts empty, so distributing the
        # full seed list would make every worker redo work the main file
        # already holds. The parent owns the resume decision and hands
        # out only the seeds that still have something missing.
        _have = _done()
        _wanted = [(arm, s_) for s_ in seeds for arm in arms]
        _todo = sorted({s_ for arm, s_ in _wanted
                        if (arm, str(s_)) not in _have})
        if not _todo:
            print("nothing left to run:", CSV_PATH)
            return
        seeds = _todo
        print(f"{len(_have)} row(s) already on file, "
              f"{len(_todo)} seed(s) still to run")
        shards = []
        procs = []
        for w in range(a.workers):
            mine = seeds[w::a.workers]
            if not mine:
                continue
            shard = CSV_PATH.replace(".csv", f".w{w}.csv")
            shards.append((shard, mine))
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--seeds", str(a.seeds), "--horizon", str(a.horizon),
                   "--patience", str(a.patience),
                   "--earliest", str(a.earliest),
                   "--arms", a.arms, "--csv", shard,
                   "--seed-list", ",".join(str(x) for x in mine)]
            if a.offline:
                cmd.append("--offline")
            procs.append(subprocess.Popen(cmd))
        for p in procs:
            p.wait()
        merged = 0
        for shard, _mine in shards:
            if not os.path.exists(shard):
                continue
            with open(shard, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    _append(r, [])
                    merged += 1
            os.remove(shard)
            js = shard.replace(".csv", ".jsonl")
            if os.path.exists(js):
                with open(js, encoding="utf-8") as fh, \
                        open(JSONL_PATH, "a", encoding="utf-8") as out:
                    out.write(fh.read())
                os.remove(js)
        print(f"merged {merged} rows from {len(shards)} workers")
        print("written:", CSV_PATH)
        return

    done = _done()
    t0 = time.time()
    for seed in seeds:
        for arm in arms:
            if (arm, str(seed)) in done:
                continue
            if a.budget and time.time() - t0 > a.budget:
                print("budget reached, stopping cleanly")
                return
            row, props = run_one(arm, seed, a.offline, a.horizon,
                                 a.patience, a.earliest)
            _append(row, props)
            print(f"{arm:>4} seed={seed}  cycles={row['cycles']:>3} "
                  f"diag={row['diagnosed']:>3} "
                  f"({row['diag_share']:.0%})  attempts="
                  f"{row['stage3_attempts']:>3}  "
                  f"clause={row['clause']} "
                  f"(admitted {row['clause_accepted']})  "
                  f"composite={row['composite']} concept={row['concept']} "
                  f"rule={row['rule']}  retired={row['genai_retired']}  "
                  f"({row['seconds']}s)", flush=True)
    print("written:", CSV_PATH)
    print("        ", JSONL_PATH)


if __name__ == "__main__":
    main()
