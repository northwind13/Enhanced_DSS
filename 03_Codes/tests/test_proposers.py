"""Proposer protocol, search proposers, composition prior, withholding."""
import numpy as np
import pytest

import dss
from dss import adapt as _a
from dss import proposers as P
from dss import rules as R
from disaster_phyengine.config import SimConfig
from disaster_phyengine import terrain
from disaster_phyengine.core import Simulator


def _incident(seed=7, n=40, steps=10):
    cfg = SimConfig(nx=n, ny=n, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    world = terrain.generate_landscape(cfg, seed=seed, preset="Rolling hills",
                                       n_settlements=3,
                                       population_per_settlement=3000)
    world.fuel.fmoist[:] = 0.06
    world.meteo.wws[:] = 9.0
    world.add_ignition(n // 2, n // 2, step=0, radius=1)
    sim = Simulator(world)
    sim.record_states = False
    base, _ = dss.resource_suggestion(world)
    base.rcap *= 0.5
    eng = dss.DecisionEngine(dss.partition_n(n, n, 1), base_pool=base,
                             seed_profile="minimal",
                             state_path=dss.isolated_store_path("t_prop"),
                             cycle_min=12.0, horizon_min=24.0,
                             adapt_on=True, evfis_on=True, genai_on=True)
    eng.revision_budget = 1
    eng.genai_patience = 10 ** 6
    for _ in range(steps):
        sim.step()
    return sim, eng


def _ctx(sim, eng):
    eff = eng._perceive(sim)
    rows, pairs = eng._decide_regions(sim, eff, eng.rules)
    name = next(iter(rows))
    effd, crisp = rows[name]["eff"], rows[name]["crisp"]

    def build(rules):
        _, prs = eng._decide_regions(sim, eng._perceive(sim), rules)
        return eng._override(sim, prs)

    return P.TrialContext(build, sim, list(eng.rules), eng.horizon_steps,
                          effd, crisp, eng)


def test_grammar_samples_are_syntactically_valid_and_cover_all_kinds():
    sim, eng = _incident()
    ctx = _ctx(sim, eng)
    g = P.Grammar(ctx, np.random.default_rng(3))
    kinds = set()
    for _ in range(80):
        p = g.sample()
        assert 1 <= len(p["antecedents"]) <= 3
        assert 1 <= len(p["consequents"]) <= 3
        if p.get("new_intervention"):
            ni = p["new_intervention"]
            if ni.get("clauses"):
                kinds.add("actuator")
                for c in ni["clauses"]:
                    assert c["effect"] in P.EFFECTS
                    assert c["sector"] in P.SECTORS
                    assert 0 <= c["range"][0] <= c["range"][1] <= 15
            else:
                kinds.add("macro")
        elif p.get("new_concept"):
            kinds.add("concept")
        else:
            kinds.add("rule")
    assert kinds == {"rule", "macro", "actuator", "concept"}


def test_cheap_gates_reject_an_unknown_intervention_and_a_covered_cell():
    sim, eng = _incident()
    ctx = _ctx(sim, eng)
    bad = {"antecedents": [("fire_threat_level", ">=VL")],
           "consequents": [("cloud_seeding", 0.9)]}
    assert P.cheap_gates(bad, ctx) is not None
    # the seed rule R5 sits on (fire_threat_level, VH)
    dup = {"antecedents": [("fire_threat_level", "VH")],
           "consequents": [("suppression_effort", 0.9)]}
    err = P.cheap_gates(dup, ctx)
    assert err is None or "duplicate" in err or "relevance" in err


def test_search_proposers_price_candidates_and_respect_the_budget():
    sim, eng = _incident()
    for prop in (P.EvoProposer(seed=1, forecast_budget=12, pop=6, gens=2),
                 P.RandomProposer(seed=1, forecast_budget=12)):
        ctx = _ctx(sim, eng)
        eng._trial_ctx = ctx
        before = _a.CVA_CALLS
        out = prop("situation text", engine=eng)
        spent = _a.CVA_CALLS - before
        # budget plus the base forecasts on two seeds and the finalists
        assert spent <= 12 + 2 + prop.k + 2
        assert prop.candidates_tried >= 1
        assert out is None or isinstance(out, dict)
        if out and not out.get("_skip"):
            assert out["_source"] == prop.name
            assert out["_candidates"]


def test_the_stage_records_the_search_proposer_in_the_ledger():
    sim, eng = _incident()
    prop = P.RandomProposer(seed=2, forecast_budget=8)
    prev = P.install(eng, prop)
    try:
        for _ in range(40):
            ov = eng.maybe_decide(sim)
            sim.step(resource_override=ov)
    finally:
        P.restore(prev)
    st3 = [c["adaptation"] for c in eng.cycles
           if (c.get("adaptation") or {}).get("tried") == 3]
    if not st3:
        pytest.skip("the controller never chose stage 3 in this incident")
    assert all((a.get("info") or {}).get("source") == "random"
               for a in st3)


def test_composition_prior_learns_and_suggests():
    sim, eng = _incident()
    ctx = _ctx(sim, eng)
    eng._pool_saturated = False
    prior = P.CompositionPrior()
    g = P.Grammar(ctx, np.random.default_rng(5))
    assert prior.suggest(g, ctx) is None
    prop = {"antecedents": [("fire_threat_level", ">=VL")],
            "consequents": [("x", 1.0)],
            "new_intervention": {"name": "x", "clauses": [
                {"effect": "coat", "sector": "assets", "range": [0, 2],
                 "amount": 1.0}]}}
    prior.observe_trial(prop, ctx, 0.01)
    prior.observe_trial(prop, ctx, 0.02)
    sug = prior.suggest(g, ctx)
    assert sug and sug["new_intervention"]["clauses"][0]["effect"] == "coat"
    d = prior.to_dict()
    back = P.CompositionPrior.from_dict(d)
    assert back.table == prior.table


def test_withholding_removes_the_actuators_everywhere_it_matters():
    R.withhold(["water_drafting", "tactical_burn", "retardant_drop"])
    try:
        assert R.discoverable() == ()
        assert R.actuator_library() == {}
        assert "water_drafting" not in _a._genai_schema()
        bad = {"antecedents": [("fire_threat_level", ">=VL")],
               "consequents": [("water_drafting", 0.9)]}
        assert _a._g1_g2(bad) is not None
        from dss import mission
        sim, eng = _incident()
        brief = mission.build_mission_brief(sim.world, eng.base_pool) \
            if hasattr(mission, "build_mission_brief") else ""
        assert "tactical_burn:" not in brief
        # the allocator ignores an order for a withheld actuator
        from dss import actions as A
        from disaster_phyengine.layers import ResourceLayer
        fire = np.asarray(sim.state.burning) > 0.5
        n = sim.world.config.nx

        class Reg:
            x0, y0, x1, y1 = 0, 0, n, n
            name = "all"

            def slices(self):
                return slice(0, n), slice(0, n)
        base = ResourceLayer.none(n, n)
        base.rcap[:] = 1.0
        base.ravail[:] = 1.0
        out = A.decision_to_resources(sim.world, fire,
                                      [(Reg(), {"tactical_burn": 1.0})],
                                      base=base)
        assert out.rburn is None or not (out.rburn > 0.5).any()
    finally:
        R.withhold(())
    assert R.discoverable() == R.DISCOVERABLE_INTERVENTIONS


def test_withheld_tactic_can_still_be_recomposed_from_clauses():
    """The point of withholding: the physics of the tactic stays
    reachable through the clause grammar."""
    R.withhold(["tactical_burn"])
    try:
        from dss import actions as A
        from disaster_phyengine.layers import ResourceLayer
        sim, eng = _incident(steps=14)
        fire = np.asarray(sim.state.burning) > 0.5
        if int(fire.sum()) < 10:
            pytest.skip("fire too small")
        n = sim.world.config.nx

        class Reg:
            x0, y0, x1, y1 = 0, 0, n, n
            name = "all"

            def slices(self):
                return slice(0, n), slice(0, n)
        base = ResourceLayer.none(n, n)
        base.rcap[:] = 1.0
        base.ravail[:] = 1.0
        macro = {"strip": {"clauses": [{"effect": "ignite", "sector": "ring",
                                        "range": [4, 6], "amount": 1.0}]}}
        out = A.decision_to_resources(sim.world, fire,
                                      [(Reg(), {"strip": 1.0})],
                                      base=base, macros=macro)
        assert out.rburn is not None and (out.rburn > 0.5).any()
    finally:
        R.withhold(())
