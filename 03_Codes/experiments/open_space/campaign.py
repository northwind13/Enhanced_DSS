"""Open intervention space campaign (IEEE Systems Journal article).

Builds on the thesis ladder campaign (experiments/campaign5.py: worlds,
ignitions, arms, cost rows) and adds what the article needs:

  scenario families  S6 lakeside (drafting is the only extra capacity),
                     S7 fast WUI (only coating stops ignition in time),
                     S8 long incident on a spent pool
  arms               F5Ev (closed), Open/Evo, Open/Random, Open/LLM,
                     each with or without the withheld doctrine and with
                     or without the composition prior
  measurements       every stage-3 attempt with its proposer, gates and
                     forecasts; every admitted object with its clauses and
                     the minute it was admitted; rediscovery of withheld
                     tactics; forecast budget per proposer

Experiments (--exp): E1 rediscovery, E2 ladder over S1-S8, E3 proposer
comparison and prior ablation, E4 gate margin sensitivity. Resumable: a
finished (exp, scenario, arm, seed) row is skipped on restart.

  python experiments/open_space/campaign.py --exp E1 --seeds 201-210 --workers 2
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
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import dss                                              # noqa: E402
from dss import rules as dss_rules                      # noqa: E402
from dss import proposers as P                          # noqa: E402
from dss import actions as A                            # noqa: E402
from disaster_phyengine.core import Simulator           # noqa: E402
from disaster_phyengine.costs import compute_costs      # noqa: E402
import campaign5 as C5                                  # noqa: E402

OUT = os.path.join(HERE, "out")
WITHHELD_ALL = ("tactical_burn", "water_drafting", "retardant_drop")

# ------------------------------------------------------------ scenarios
NEW_SCENARIOS = {
    # id: (preset, settlements, pop, pool scale, n_ign, ignition mode,
    #      wind, fmoist, hours, water)
    "S6": dict(preset="River valley", n_set=4, pop=4000, pool=0.4,
               n_ign=2, mode="remote", wind=9.0, fmoist=0.08, hours=12,
               water="lake_near_front"),
    "S7": dict(preset="Mediterranean coast", n_set=12, pop=8000, pool=1.0,
               n_ign=2, mode="near_town", wind=12.0, fmoist=0.06, hours=12,
               water="lake_near_front"),
    "S8": dict(preset="Rolling hills", n_set=6, pop=5000, pool=0.4,
               n_ign=3, mode="remote", wind=8.0, fmoist=0.08, hours=18,
               water="preset"),
}
ALL_SCENARIOS = tuple(C5.SCENARIOS) + tuple(NEW_SCENARIOS)


def _carve_lake(w, cx, cy, r=3):
    """A small lake near (cx, cy): water cells carry no fuel."""
    ny, nx = w.config.ny, w.config.nx
    yy, xx = np.ogrid[0:ny, 0:nx]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    w.fuel.ftype[m] = 5
    w.fuel.fload[m] = 0.0
    try:
        w.fuel.fload0[m] = 0.0
    except Exception:
        pass
    try:
        w.value.vpop[m] = 0.0
        w.value.vbld[m] = 0.0
    except Exception:
        pass


def build_world(seed: int, scenario: str):
    if scenario in C5.SCENARIOS:
        return C5.build_world(seed, scenario)
    spec = NEW_SCENARIOS[scenario]
    from disaster_phyengine import terrain
    from disaster_phyengine.config import SimConfig
    cfg = SimConfig(nx=72, ny=50, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(cfg, seed=seed, preset=spec["preset"],
                                   n_settlements=spec["n_set"],
                                   population_per_settlement=spec["pop"])
    w.fuel.fmoist[:] = spec["fmoist"]
    w.meteo.wws[:] = spec["wind"]
    return w


def pick_ignitions(w, base, scenario: str):
    if scenario in C5.SCENARIOS:
        return C5.pick_ignitions(w, base, scenario)
    spec = NEW_SCENARIOS[scenario]
    ok = (w.fuel.fload > 0.4) & (w.fuel.ftype >= 1) & (w.fuel.ftype <= 4)
    ys, xs = np.where(ok)
    if spec["mode"] == "near_town":
        urban = (w.fuel.ftype == 6)
        if urban.any():
            uy, ux = np.where(urban)
            d2 = ((xs[:, None] - ux[None, :]) ** 2
                  + (ys[:, None] - uy[None, :]) ** 2).min(axis=1)
            near = (d2 >= 4 ** 2) & (d2 <= 12 ** 2)
            if near.any():
                ys, xs = ys[near], xs[near]
    order = np.argsort(-base.rtime[ys, xs])
    spots = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all((x - a) ** 2 + (y - b) ** 2 > 16 ** 2 for a, b in spots):
            spots.append((x, y))
        if len(spots) == spec["n_ign"]:
            break
    return spots or [(int(xs[0]), int(ys[0]))]


def place_water(w, spots, scenario: str, seed: int):
    spec = NEW_SCENARIOS.get(scenario)
    if not spec or spec["water"] != "lake_near_front":
        return
    rng = np.random.default_rng(777 + seed)
    ny, nx = w.config.ny, w.config.nx
    for (x, y) in spots[:1]:
        # 8 to 11 cells downwind of the first ignition
        wd = float(np.mean(w.meteo.wwd))
        d = float(rng.uniform(8, 11))
        cx = int(np.clip(round(x + d * np.cos(wd)), 3, nx - 4))
        cy = int(np.clip(round(y + d * np.sin(wd)), 3, ny - 4))
        _carve_lake(w, cx, cy, r=3)


# ----------------------------------------------------------------- arms
ARMS = {
    # name: (profile, engine kwargs, proposer kind)
    "T0":     ("minimal", None, None),
    "F5":     ("minimal", dict(adapt_on=False), None),
    "F5Ev":   ("minimal", dict(adapt_on=True, evfis_on=True,
                               genai_on=False), None),
    # closed layer on the WHOLE written doctrine (40 seed rules, Appendix E)
    # with the same self-tuning; the strong closed reference of the revision
    "F40Ev":  ("full",    dict(adapt_on=True, evfis_on=True,
                               genai_on=False), None),
    "Evo":    ("minimal", dict(adapt_on=True, evfis_on=True,
                               genai_on=True), "evo"),
    "Random": ("minimal", dict(adapt_on=True, evfis_on=True,
                               genai_on=True), "random"),
    "LLM":    ("minimal", dict(adapt_on=True, evfis_on=True,
                               genai_on=True), "llm"),
    "Tmpl":   ("minimal", dict(adapt_on=True, evfis_on=True,
                               genai_on=True), "template"),
    # the model as the candidate source of a screened search (same forecast
    # budget as Random); the VODA configuration of article v52
    "LLMScreen": ("minimal", dict(adapt_on=True, evfis_on=True,
                                  genai_on=True), "llmscreen"),
    # one unpriced random candidate per call, the counterpart of "LLM"
    "RandBlind": ("minimal", dict(adapt_on=True, evfis_on=True,
                                  genai_on=True), "randomblind"),
    # closed layer with the coordinator switched off (run_once sets
    # DSS_NO_COORD=1 for an arm ending in -nocoord)
    "F5Ev-nocoord": ("minimal", dict(adapt_on=True, evfis_on=True,
                                     genai_on=False), None),
    "F5Ev-r1": ("minimal", dict(adapt_on=True, evfis_on=True,
                                genai_on=False), None),
    "F5Ev-r16": ("minimal", dict(adapt_on=True, evfis_on=True,
                                 genai_on=False), None),
    # generative stage alone: no evFIS, every adaptation (coverage gaps
    # included) must come from the language-model proposer
    "LLMonly": ("minimal", dict(adapt_on=True, evfis_on=False,
                                genai_on=True), "llm"),
}

EXPERIMENTS = {
    "E1": dict(scenarios=("S6", "S7", "S8"),
               arms=("F5Ev", "Evo", "Random"),
               withhold=True, prior=True),
    "E2": dict(scenarios=ALL_SCENARIOS,
               arms=("T0", "F5", "F5Ev", "Evo", "Random"),
               withhold=False, prior=True),
    "E3": dict(scenarios=("S6", "S7", "S8"),
               arms=("Evo", "Random", "Evo-noprior", "Random-noprior"),
               withhold=True, prior=True),
    "E4": dict(scenarios=("S6", "S7", "S8"),
               arms=("Evo",), withhold=True, prior=True,
               margins=(1e-4, 1e-3, 1e-2)),
    # E5: the article ladder (S1-S5) with the LLM proposer and the
    # actuator library WITHHELD. The generative stage sees only the six
    # doctrine families and the seven clause effects, so every aerial or
    # drafting tactic it fields is one it composed from effects itself.
    # Reference arms T0 / F5 / F5Ev are unchanged from E2 (they never
    # see the library), so E2 rows of those arms pair with E5 LLM rows.
    "E5": dict(scenarios=ALL_SCENARIOS,
               arms=("LLM",),
               withhold=True, prior=True),
    # E6: bandit ablation. The closed layer alone with the stage selector
    # replaced by a fixed escalation order (DSS_STAGE_POLICY=fixed); pairs
    # with the E2 F5Ev rows, which use the bandit.
    "E6": dict(scenarios=ALL_SCENARIOS,
               arms=("F5Ev-fixed",),
               withhold=True, prior=True),
    # E7: withheld-knowledge test. One doctrine family is removed from the
    # seed rules (DSS_WITHHOLD_FAMILY_NAME, default containment_line) and
    # both layers run without it; the family stays in the vocabulary.
    "E7": dict(scenarios=ALL_SCENARIOS,
               arms=("F5Ev-nofam", "LLM-nofam"),
               withhold=True, prior=True),
    # E8: proposer control. The E5 setting (library withheld, same seeds,
    # same gates) with the language model replaced by a random search over
    # the same grammar: candidates are sampled, priced on shadow forecasts
    # within the forecast budget, and the best three go to the gate chain.
    # Pairs with the E5 LLM rows; prop_forecasts records the search cost.
    "E8": dict(scenarios=ALL_SCENARIOS,
               arms=("Random",),
               withhold=True, prior=True),
    # E9: VODA of article v52. The model writes several candidates per call,
    # they are screened on the shadow simulator with the forecast budget of
    # E8, and the best goes to the gates. E7-style withheld-family runs use
    # the arm LLMScreen-nofam with --exp E7.
    # E2P: the reference arms (T0, F5, F5Ev) under the same withheld
    # library as every other experiment, so no reference differs from the
    # open-layer runs in anything but the layer itself.
    "E2P": dict(scenarios=ALL_SCENARIOS,
                arms=("T0", "F5", "F5Ev"),
                withhold=True, prior=True),
    "E9": dict(scenarios=ALL_SCENARIOS,
               arms=("LLMScreen",),
               withhold=True, prior=True),
    # E10: one unpriced random candidate per call; pairs with E5 (LLM).
    "E10": dict(scenarios=ALL_SCENARIOS,
                arms=("RandBlind",),
                withhold=True, prior=True),
    # E11: coordination ablation, closed layer with every region attended;
    # pairs with the E2 F5Ev rows.
    "E11": dict(scenarios=ALL_SCENARIOS,
                arms=("F5Ev-nocoord",),
                withhold=True, prior=True),
    # LOO: leave-one-out replay of the E5 worlds (see experiments/benchmark/loo.py)
    "LOO": dict(scenarios=ALL_SCENARIOS,
                arms=("F5Ev",),
                withhold=True, prior=True),
}


def _make_proposer(kind: str, seed: int, budget: int, prior):
    if kind == "evo":
        return P.EvoProposer(seed=seed, forecast_budget=budget, prior=prior)
    if kind == "random":
        return P.RandomProposer(seed=seed, forecast_budget=budget,
                                prior=prior)
    if kind == "llmscreen":
        return P.LLMScreenProposer(seed=seed, forecast_budget=budget,
                                   prior=prior)
    if kind == "randomblind":
        return P.RandomBlindProposer(seed=seed)
    if kind == "llm":
        return P.LLMProposer()
    if kind == "template":
        return P.TemplateProposer(seed)
    raise ValueError(kind)


# ---------------------------------------------------------- rediscovery
TARGETS = {
    # withheld actuator: (effect, sectors, max rin)
    "water_drafting": ("draft", {"at_fire", "head", "ring", "flank",
                                 "rear"}, 4),
    "retardant_drop": ("coat", {"head", "ring"}, 6),
    "tactical_burn": ("ignite", {"head", "ring", "flank"}, 8),
}


def rediscovered(macros: dict) -> dict:
    """Which withheld tactics an admitted clause actuator re-composes."""
    out = {}
    for name, md in (macros or {}).items():
        for c in md.get("clauses") or []:
            eff = str(c.get("effect"))
            sec = str(c.get("sector", "ring"))
            rin = int((c.get("range") or [2, 6])[0])
            for tgt, (teff, tsecs, rmax) in TARGETS.items():
                if eff == teff and sec in tsecs and rin <= rmax:
                    out.setdefault(tgt, name)
    return out


# ------------------------------------------------- scripted re-installs
def _parse_rule_text(text: str):
    """'G1: IF a b is H AND c is >=M THEN x y 1.0, z 0.8 (adaptation-born ...'
    -> antecedents [(a_b, H), (c, >=M)], consequents [(x_y, 1.0), (z, 0.8)]."""
    import re as _re
    body = _re.sub(r"^G\d+:\s*", "", text.strip())
    body = body.split(" (adaptation-born")[0]
    m = _re.match(r"IF (.*?) THEN (.*)$", body)
    if not m:
        raise ValueError(f"unparsable rule text: {text[:80]}")
    ants = []
    for part in m.group(1).split(" AND "):
        v, t = part.rsplit(" is ", 1)
        ants.append((v.strip().replace(" ", "_"), t.strip()))
    cons = []
    for part in m.group(2).split(";")[0].split(","):
        part = part.strip()
        if not part:
            continue
        nm, val = part.rsplit(" ", 1)
        cons.append((nm.strip().replace(" ", "_"), float(val)))
    return ants, cons


def _install_scripted(eng, rec: dict) -> None:
    """Re-install one recorded admitted proposal without the gates: the
    object was admitted in the recorded run, the replay asks what the
    outcome is with and without it."""
    from dss.adapt import _install_package, _next_rule_name
    from dss.rules import Rule
    ants, cons = _parse_rule_text(rec["rule"])
    prop = {"antecedents": ants, "consequents": cons}
    kind = rec.get("kind")
    if kind in ("actuator", "macro"):
        name = next((i for i, _x in cons if i not in dss_rules.INTERVENTIONS
                     and i not in (eng.macros or {})), None)
        if name is None:
            name = cons[0][0]
        if kind == "actuator":
            prop["new_intervention"] = dict(name=name,
                                            clauses=list(rec["clauses"]))
        else:
            prop["new_intervention"] = dict(
                name=name, composition=[(a, b) for a, b in rec["composition"]])
    if kind == "concept" and rec.get("concept"):
        prop["new_concept"] = dict(rec["concept"])
    # the engine derives its active set from the store every cycle, so the
    # object goes in exactly the way an admission does: install, append the
    # rule, record the package and the rule change, re-sync
    b_rules = eng._snap_rules(eng.rules); b_parts = eng._snap_parts()
    b_vocab = (set(eng.hierarchy), set(eng.macros))
    _install_package(prop, eng)
    eng.rules.append(Rule(_next_rule_name("G", eng.rules, eng), ants, cons,
                          note=f"replay: re-installed at minute {rec['t_min']}"))
    trig = dict(step=-1, minute=float(rec["t_min"]), deficit=0.0, stage=3)
    nc = {c: eng.hierarchy[c] for c in set(eng.hierarchy) - b_vocab[0]}
    nm = {m: eng.macros[m] for m in set(eng.macros) - b_vocab[1]}
    if nc or nm:
        eng.record_package(concepts=nc, macros=nm, trigger=trig)
    eng._record_changes(3, b_rules, b_parts, eng.rules, trigger=trig)
    eng._sync_active_set()


# --------------------------------------------------------------- one run
def run_once(exp: str, scenario: str, arm: str, seed: int,
             budget: int = 36, margin: float | None = None,
             script: list | None = None):
    """script: recorded admitted proposals to re-install at their t_min
    (leave-one-out replay); each record carries rule text, kind, clauses,
    composition and concept, as proposals.jsonl stores them."""
    base_arm = arm.replace("-noprior", "").replace("-fixed", "") \
        .replace("-nofam", "")
    if base_arm == "F5Ev-nocoord":
        base_arm = "F5Ev"
    # LOCAL AGENTS: an arm ending in -rN partitions the grid into N regional
    # agents instead of 4 (F5Ev-r1 is one global agent, F5Ev-r16 sixteen)
    import re as _re_n
    _mreg = _re_n.search(r"-r(\d+)$", base_arm)
    n_regions = 4
    if _mreg:
        n_regions = int(_mreg.group(1))
        base_arm = base_arm[: _mreg.start()]
    os.environ["DSS_NO_COORD"] = "1" if arm.endswith("-nocoord") else ""
    os.environ["DSS_STAGE_POLICY"] = "fixed" if arm.endswith("-fixed") \
        else "bandit"
    # withheld-knowledge test: the family named by DSS_WITHHOLD_FAMILY_NAME
    # (default containment_line) is removed from the seed rules of a -nofam arm
    fam = os.environ.get("DSS_WITHHOLD_FAMILY_NAME", "containment_line")
    os.environ["DSS_WITHHOLD_FAMILY"] = fam if arm.endswith("-nofam") else ""
    profile, ekw, pkind = ARMS[base_arm]
    cfg_e = EXPERIMENTS[exp]
    withhold = bool(cfg_e.get("withhold"))
    use_prior = bool(cfg_e.get("prior")) and not arm.endswith("-noprior")
    hours = NEW_SCENARIOS.get(scenario, {}).get("hours", C5.MAX_HOURS)

    dss_rules.withhold(WITHHELD_ALL if withhold else ())
    os.environ["DSS_TRIAL_OBJ"] = "phys"
    from dss import adapt as _adapt
    if margin is not None:
        _adapt.G5_MARGIN = float(margin)
        _adapt.G5B_MARGIN = float(margin)

    w = build_world(seed, scenario)
    base, _ = dss.resource_suggestion(w)
    if scenario in C5.LIMITED:
        base.rcap *= 0.6
    elif scenario in NEW_SCENARIOS:
        base.rcap *= float(NEW_SCENARIOS[scenario]["pool"])
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    spots = pick_ignitions(w, base, scenario)
    place_water(w, spots, scenario, seed)
    for x, y in spots:
        w.add_ignition(x, y, step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    rng = np.random.default_rng(10_000 + seed)
    if scenario == "S5":
        C5._install_outage(rng)

    prev_hook = None
    proposer = None
    prior = P.CompositionPrior() if use_prior else None
    eng = None
    if ekw is not None:
        eng = dss.DecisionEngine(
            dss.partition_n(w.config.nx, w.config.ny, n_regions),
            base_pool=base, seed_profile=profile,
            state_path=dss.isolated_store_path(f"open_space_{exp}"),
            cycle_min=12.0, horizon_min=24.0, eta=C5.ETA, **ekw)
        # one adaptation window per decision cycle for EVERY arm (the
        # thesis ladder used 24 min); the open space is exercised only
        # when the loop is allowed to try, and the closed arm gets the
        # same number of windows
        eng.adapt_cooldown_min = 12.0
        # the doctrine-free coordinator order is off in every arm, so the
        # only way drafting reaches the field is through a rule
        eng.drafting_directive = False
        eng.revision_budget = 2
        if pkind:
            # a search proposer never "goes dead" on rejections; the LLM
            # keeps the thesis patience
            eng.genai_patience = 10 ** 6 if pkind in ("evo", "random",
                                                    "llmscreen") \
                else 3
            proposer = _make_proposer(pkind, seed, budget, prior)
            prev_hook = P.install(eng, proposer)
    max_steps = int(round(hours * 60.0 / w.config.step_minutes))
    out_at = None
    curves = []
    attempts = []
    admitted = []
    seen_macros = set()
    t0 = time.time()
    _cyc_s = []     # wall time of every decision cycle, live proposer included
    _coord = []     # (regions with priority, regions attended) per cycle
    try:
        pending = sorted(script or [], key=lambda r: float(r["t_min"]))
        for i in range(max_steps):
            t_now = (i + 1) * w.config.step_minutes
            while pending and float(pending[0]["t_min"]) <= t_now \
                    and eng is not None:
                _install_scripted(eng, pending.pop(0))
            _nc0 = len(eng.cycles) if eng is not None else 0
            _td0 = time.perf_counter()
            ov = eng.maybe_decide(sim) if eng is not None else None
            if eng is not None and len(eng.cycles) > _nc0:
                _cyc_s.append(time.perf_counter() - _td0)
                # COORDINATOR DIAGNOSTIC: a region with a non-zero priority
                # left outside the attended set was derated this cycle
                _nf = int(getattr(eng, "last_n_fire", 0) or 0)
                _na = int(getattr(eng, "last_k", 0) or 0)
                _coord.append((_nf, _na))
            sim.step(resource_override=ov)
            t_min = (i + 1) * w.config.step_minutes
            if i % 6 == 0:
                rep_i = compute_costs(sim)
                curves.append([exp, scenario, arm, seed, round(t_min, 1),
                               int(sim.ever_burned.sum()),
                               round(float(getattr(rep_i, "j_physical",
                                                   rep_i.j_total)), 5)])
            if eng is not None and eng.cycles:
                c = eng.cycles[-1]
                ad = c.get("adaptation") or {}
                if ad.get("tried") == 3 and not ad.get("_logged"):
                    ad["_logged"] = True
                    info = ad.get("info") or {}
                    prop = info.get("proposal") or {}
                    ni = prop.get("new_intervention") or {}
                    attempts.append(dict(
                        exp=exp, scenario=scenario, arm=arm, seed=seed,
                        t_min=round(t_min, 1),
                        source=info.get("source", ""),
                        accepted=bool(ad.get("accepted")),
                        detail=str(ad.get("detail") or "")[:200],
                        gate=(info.get("gate") or
                              (info.get("gates") or {}).get("verdict")),
                        package=bool(info.get("package")),
                        kind=("actuator" if ni.get("clauses") else
                              "macro" if ni.get("composition") else
                              "concept" if prop.get("new_concept") else
                              "rule"),
                        clauses=ni.get("clauses"),
                        composition=ni.get("composition"),
                        concept=prop.get("new_concept"),
                        rule=info.get("rule"),
                        dJ=float(ad.get("dJ") or 0.0),
                        search=prop.get("_search") or info.get("search"),
                        gates={k: v for k, v in
                               (info.get("gates") or {}).items()
                               if k in ("g3", "g4", "g5b")}))
                for mn, md in (eng.macros or {}).items():
                    if mn not in seen_macros:
                        seen_macros.add(mn)
                        admitted.append(dict(
                            exp=exp, scenario=scenario, arm=arm,
                            seed=seed, t_min=round(t_min, 1), name=mn,
                            clauses=md.get("clauses"),
                            composition=md.get("composition")))
            if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
                out_at = t_min
                break
    finally:
        if scenario == "S5":
            C5._remove_outage()
        if prev_hook is not None:
            P.restore(prev_hook)
        dss_rules.withhold(())
    rep = compute_costs(sim)
    ha = (w.config.cell_size_m ** 2) / 1e4
    vp0 = np.asarray(getattr(sim, "_vpop0", w.value.vpop))
    ft0 = np.asarray(w.fuel.ftype)
    row = dict(exp=exp, scenario=scenario, arm=arm, seed=seed,
               withhold=int(withhold), prior=int(use_prior),
               margin=(margin if margin is not None else ""),
               burned_ha=round(int(sim.ever_burned.sum()) * ha, 2),
               forest_ha=round(int((sim.ever_burned & ((ft0 == 3) |
                                                       (ft0 == 4))).sum())
                               * ha, 2),
               pop_affected=round(float(vp0[sim.ever_burned].sum()), 0),
               evacuated=round(float(sim.population_evacuated), 0),
               out_min=(out_at if out_at is not None else -1),
               success=int(out_at is not None),
               j_burn=round(float(rep.j_burn), 5),
               j_asset=round(float(rep.j_asset), 5),
               j_pop=round(float(rep.j_pop), 5),
               j_resp=round(float(rep.j_resp), 5),
               j_delay=round(float(getattr(rep, "j_delay", float("nan"))),
                             5),
               j_total=round(float(rep.j_total), 5),
               retard_units=round(float(getattr(sim, "retardant_applied", 0.0)), 2),
               resp_units=round(float(getattr(sim, "response_capacity_steps", 0.0)), 1),
               n_cycles=len(_cyc_s),
               n_regions=n_regions,
               coord_derate_cycles=sum(1 for a, b in _coord if a > b),
               coord_multi_fire_cycles=sum(1 for a, b in _coord if a > 1),
               cyc_med_s=round(float(np.median(_cyc_s)), 3) if _cyc_s else "",
               cyc_p90_s=round(float(np.percentile(_cyc_s, 90)), 3) if _cyc_s else "",
               cyc_max_s=round(float(max(_cyc_s)), 3) if _cyc_s else "",
               coated_ha=round(float(np.asarray(getattr(sim, "coated_cells", np.zeros(1))).sum()) * ha, 2),
               ignited_ha=round(float(getattr(sim, "ignition_applied", 0.0)) * ha, 2),
               j_phys=round(float(getattr(rep, "j_physical",
                                          float("nan"))), 5))
    tried = {1: 0, 2: 0, 3: 0}
    acc = {1: 0, 2: 0, 3: 0}
    n_sited = n_evac = 0
    if eng is not None:
        for c in eng.cycles:
            ad = c.get("adaptation") or {}
            st = int(ad.get("tried") or 0)
            if st in tried:
                tried[st] += 1
                acc[st] += 1 if ad.get("accepted") else 0
            for rd in (c.get("regions") or {}).values():
                u = rd.get("orders_final") or rd.get("u") or {}
                if float(u.get("evacuation", 0.0)) > 0.3:
                    n_evac += 1
                for mn in (eng.macros or {}):
                    if float(u.get(mn, 0.0)) > 0.3 and \
                            (eng.macros[mn].get("clauses")):
                        n_sited += 1
    redis = rediscovered(eng.macros if eng is not None else {})
    first_t = {}
    for a in admitted:
        for tgt, mname in redis.items():
            if a["name"] == mname and tgt not in first_t:
                first_t[tgt] = a["t_min"]
    row.update(
        rules_final=(len([r for r in eng.rules if r.active])
                     if eng is not None else 0),
        tried_1=tried[1], tried_2=tried[2], tried_3=tried[3],
        acc_1=acc[1], acc_2=acc[2], acc_3=acc[3],
        n_macros=len(eng.macros) if eng is not None else 0,
        n_actuators=(len([m for m in (eng.macros or {}).values()
                          if m.get("clauses")]) if eng is not None else 0),
        n_concepts=(len([c for c in (eng.hierarchy or {})
                         if c not in dss.concepts.HIERARCHY])
                    if eng is not None else 0),
        sited_orders=n_sited, evac_orders=n_evac,
        redis_draft=int("water_drafting" in redis),
        redis_coat=int("retardant_drop" in redis),
        redis_ignite=int("tactical_burn" in redis),
        t_draft=first_t.get("water_drafting", ""),
        t_coat=first_t.get("retardant_drop", ""),
        t_ignite=first_t.get("tactical_burn", ""),
        fam_withheld=os.environ.get("DSS_WITHHOLD_FAMILY", ""),
        fam_rules_active=(len([r for r in eng.rules if r.active and any(
            i == fam for i, _v in r.consequents)]) if eng is not None else 0),
        fam_rules_gen=(len([r for r in eng.rules if r.active and r.name.startswith("G")
                            and any(i == fam for i, _v in r.consequents)])
                       if eng is not None else 0),
        fam_first_min=(min([a["t_min"] for a in attempts if a.get("accepted")
                            and fam.replace("_", " ") in str(a.get("rule") or "")
                            .split(" (adaptation-born")[0].split("THEN")[-1]]
                           or [""]) if eng is not None else ""),
        menu_total=int((eng.run_stats or {}).get("menu_total", 0))
        if eng is not None else 0,
        menu_multi=int((eng.run_stats or {}).get("menu_multi", 0))
        if eng is not None else 0,
        proposer=(proposer.name if proposer else ""),
        prop_calls=(proposer.calls if proposer else 0),
        prop_forecasts=(getattr(proposer, "forecasts", 0)
                        if proposer else 0),
        prop_candidates=(getattr(proposer, "candidates_tried", 0)
                         if proposer else 0),
        prior_rows=(len(prior.table) if prior is not None else 0),
        wall_s=round(time.time() - t0, 1))
    return row, curves, attempts, admitted


# --------------------------------------------------------------- driver
def _worker(job):
    exp, sc, arm, seed, budget, margin = job
    try:
        return run_once(exp, sc, arm, seed, budget, margin) + (None,)
    except Exception as exc:
        import traceback
        return (None, [], [], [],
                f"{exp}/{sc}/{arm}/{seed}: {type(exc).__name__}: {exc}\n"
                + traceback.format_exc()[-800:])


def _done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["exp"], r["scenario"], r["arm"], int(r["seed"]),
                 str(r.get("margin", "")))
                for r in csv.DictReader(f)}


def _append_csv(path, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            wri.writeheader()
        wri.writerows(rows)


def _parse_seeds(s: str):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E1")
    ap.add_argument("--seeds", default="201-205")
    ap.add_argument("--arms", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budget", type=int, default=36)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = EXPERIMENTS[args.exp]
    arms = tuple(a for a in args.arms.split(",") if a) or cfg["arms"]
    scs = tuple(s for s in args.scenarios.split(",") if s) \
        or cfg["scenarios"]
    seeds = _parse_seeds(args.seeds)
    margins = cfg.get("margins", (None,))
    runs_p = os.path.join(args.out, "runs.csv")
    done = _done_keys(runs_p)
    jobs = [(args.exp, sc, arm, sd, args.budget, mg)
            for mg in margins for sc in scs for arm in arms for sd in seeds
            if (args.exp, sc, arm, sd, str(mg if mg is not None else ""))
            not in done]
    print(f"{args.exp}: {len(jobs)} runs to do ({len(done)} done)",
          flush=True)
    import multiprocessing as mp
    t0 = time.time()
    n = 0

    def _sink(res):
        nonlocal n
        row, curves, attempts, admitted, err = res
        n += 1
        if err:
            print("ERR", err, flush=True)
            with open(os.path.join(args.out, "errors.log"), "a") as f:
                f.write(err + "\n")
            return
        _append_csv(runs_p, [row])
        with open(os.path.join(args.out, "curves.csv"), "a",
                  newline="") as f:
            csv.writer(f).writerows(curves)
        with open(os.path.join(args.out, "proposals.jsonl"), "a",
                  encoding="utf-8") as f:
            for a in attempts:
                a.setdefault("margin", row.get("margin", ""))
                f.write(json.dumps(a, default=str) + "\n")
        with open(os.path.join(args.out, "admitted.jsonl"), "a",
                  encoding="utf-8") as f:
            for a in admitted:
                f.write(json.dumps(a, default=str) + "\n")
        print(f"[{n}/{len(jobs)}] {row['scenario']} {row['arm']} "
              f"s{row['seed']}: burned {row['burned_ha']} ha, "
              f"success {row['success']}, st3 {row['acc_3']}/"
              f"{row['tried_3']}, act {row['n_actuators']}, "
              f"redis d/c/i {row['redis_draft']}{row['redis_coat']}"
              f"{row['redis_ignite']}, {row['wall_s']}s "
              f"(elapsed {time.time() - t0:.0f}s)", flush=True)

    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for res in pool.imap_unordered(_worker, jobs):
                _sink(res)
    else:
        for job in jobs:
            _sink(_worker(job))


if __name__ == "__main__":
    main()
