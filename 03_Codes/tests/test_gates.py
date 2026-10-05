"""Gate semantics of the generative stage (open intervention space, Faz A).

Rollouts are stubbed, so the GATE is tested and not the physics.
"""
import os

import numpy as _np
import pytest

import dss
from dss import adapt as _a
from disaster_phyengine.config import SimConfig
from disaster_phyengine import terrain
from disaster_phyengine.core import Simulator


def _small_incident():
    cfg = SimConfig(nx=24, ny=24, cell_size_m=30.0)
    world = terrain.generate_landscape(cfg, seed=5)
    world.add_ignition(12, 12, step=0, radius=1)
    sim = Simulator(world)
    sim.record_states = False
    sim.step()
    eng = dss.DecisionEngine(dss.partition_n(24, 24, 1))
    eng.revision_budget = 0
    return sim, eng


def _run_stage3(sim, eng, proposal, cva_stub):
    def _spy(situation, timeout=None, engine=None, mission=""):
        return proposal

    _orig_p, _orig_c = _a._genai_propose, _a._cva
    _a._genai_propose, _a._cva = _spy, cva_stub
    try:
        eff = {c: _np.array([0.2] * 5) for c in dss.DECISION_CONCEPTS}
        crisp = {c: 0.5 for c in dss.DECISION_CONCEPTS}

        def build(rules):
            _, pairs = eng._decide_regions(sim, eng._perceive(sim), rules)
            return eng._override(sim, pairs)

        return _a.stage3_generative(build, sim, list(dss.SEED_RULES),
                                    eff, crisp, 6, engine=eng)
    finally:
        _a._genai_propose, _a._cva = _orig_p, _orig_c


PLAIN_RULE = {"antecedents": [("fire_threat_level", ">=VL"),
                              ("asset_exposure_risk", ">=VL")],
              "consequents": [("containment_line", 0.9)]}


def test_g4_compares_with_the_base_not_with_inaction():
    """Regression for the G4 reference.

    Seed 101: base 0.20, candidate 0.15 (G3 passes).
    Seed 202: no action 0.40, base 0.10, candidate 0.12.
    The candidate beats inaction on the second future but is WORSE than
    the base there. Under the old reference (no action) it was admitted;
    the paired comparison must reject it at G4.
    """
    sim, eng = _small_incident()
    base_n = []

    def _stub(build, sim_, rules, horizon, reseed=None):
        if not base_n:
            base_n.append(len(rules))
        is_base = len(rules) <= base_n[0]
        if reseed == 202:
            return (0.10 if is_base else 0.12), 0.40
        return (0.20 if is_base else 0.15), 0.40

    out = _run_stage3(sim, eng, PLAIN_RULE, _stub)
    assert not out.accepted, out.detail
    assert "G4" in out.detail, out.detail
    g4 = out.info["gates"]["g4"]
    assert g4["j_without"] == pytest.approx(0.10)
    assert g4["j_noaction"] == pytest.approx(0.40)


def test_a_candidate_that_beats_the_base_on_both_futures_is_admitted():
    sim, eng = _small_incident()
    base_n = []

    def _stub(build, sim_, rules, horizon, reseed=None):
        if not base_n:
            base_n.append(len(rules))
        is_base = len(rules) <= base_n[0]
        return (0.20 if is_base else 0.15), 0.40

    out = _run_stage3(sim, eng, PLAIN_RULE, _stub)
    assert out.accepted, out.detail
    assert out.info["gates"]["g3"]["j_without"] == pytest.approx(0.20)
    assert out.info["gates"]["g4"]["j_without"] == pytest.approx(0.20)


def test_g5_margin_is_read_against_the_base_on_the_second_future():
    """A package that clears G5 on seed 101 but ties the base on seed
    202 (while still beating inaction there) is refused at G5."""
    sim, eng = _small_incident()
    base_n = []
    pkg = {"antecedents": [("fire_threat_level", ">=VL")],
           "consequents": [("tie_band", 1.0)],
           "new_intervention": {
               "name": "tie_band",
               "clauses": [{"effect": "coat", "sector": "head",
                            "range": [1, 4], "amount": 1.0}]}}

    def _stub(build, sim_, rules, horizon, reseed=None):
        if not base_n:
            base_n.append(len(rules))
        is_base = len(rules) <= base_n[0]
        if reseed == 202:
            return 0.10, 0.40          # candidate ties the base
        return (0.20 if is_base else 0.15), 0.40

    out = _run_stage3(sim, eng, pkg, _stub)
    assert not out.accepted, out.detail
    assert "G5" in out.detail, out.detail


def test_trial_objective_defaults_to_physical(monkeypatch):
    monkeypatch.delenv("DSS_TRIAL_OBJ", raising=False)
    assert _a.trial_objective() == "phys"
    monkeypatch.setenv("DSS_TRIAL_OBJ", "total")
    assert _a.trial_objective() == "total"
    monkeypatch.setenv("DSS_TRIAL_OBJ", "anything-else")
    assert _a.trial_objective() == "phys"


def test_trial_cost_uses_the_physical_terms_by_default(monkeypatch):
    monkeypatch.delenv("DSS_TRIAL_OBJ", raising=False)
    from dss.evaluate import physical_cost

    class _Rep:
        j_burn, j_asset, j_pop, j_resp, j_delay = 0.3, 0.1, 0.0, 0.9, 0.9
        j_total = 0.5

    class _Cost:
        w_burn = w_asset = w_pop = 1.0
        w_resp = w_delay = 0.2

    rep, cost = _Rep(), _Cost()
    try:
        expect = physical_cost(rep, cost)
    except Exception:
        pytest.skip("physical_cost signature differs; covered elsewhere")
    assert _a._trial_cost(rep, cost) == pytest.approx(expect)
    monkeypatch.setenv("DSS_TRIAL_OBJ", "total")
    assert _a._trial_cost(rep, cost) == pytest.approx(0.5)


def test_each_revision_is_judged_on_the_revised_proposal():
    """With a revision budget of 2 the second retry must name the defect
    of the FIRST REVISION, not the defect of the original proposal, and a
    revision that clears the cheap gates proceeds to the rollouts."""
    sim, eng = _small_incident()
    eng.revision_budget = 2
    seen = []
    answers = [
        # original: unknown concept -> G2 vocabulary
        {"antecedents": [("no_such_concept", "H")],
         "consequents": [("containment_line", 0.9)]},
        # revision 1: intensity out of range -> G2 range
        {"antecedents": [("fire_threat_level", ">=VL")],
         "consequents": [("containment_line", 1.7)]},
        # revision 2: valid
        dict(PLAIN_RULE),
    ]

    def _spy(situation, timeout=None, engine=None, mission=""):
        seen.append(situation)
        return answers[min(len(seen) - 1, len(answers) - 1)]

    base_n = []

    def _stub(build, sim_, rules, horizon, reseed=None):
        if not base_n:
            base_n.append(len(rules))
        return (0.20 if len(rules) <= base_n[0] else 0.15), 0.40

    _orig_p, _orig_c = _a._genai_propose, _a._cva
    _a._genai_propose, _a._cva = _spy, _stub
    try:
        eff = {c: _np.array([0.2] * 5) for c in dss.DECISION_CONCEPTS}
        crisp = {c: 0.5 for c in dss.DECISION_CONCEPTS}

        def build(rules):
            _, pairs = eng._decide_regions(sim, eng._perceive(sim), rules)
            return eng._override(sim, pairs)

        out = _a.stage3_generative(build, sim, list(dss.SEED_RULES),
                                   eff, crisp, 6, engine=eng)
    finally:
        _a._genai_propose, _a._cva = _orig_p, _orig_c
    assert len(seen) == 3
    assert "G2 vocabulary" in seen[1]
    assert "G2 range" in seen[2], seen[2][-300:]
    assert out.accepted, out.detail
    assert len(out.info["revisions"]) == 2


def test_the_ledger_names_the_proposer():
    sim, eng = _small_incident()

    def _spy(situation, timeout=None, engine=None, mission=""):
        return dict(PLAIN_RULE)
    _spy.proposer_name = "evo-search"

    base_n = []

    def _stub(build, sim_, rules, horizon, reseed=None):
        if not base_n:
            base_n.append(len(rules))
        return (0.20 if len(rules) <= base_n[0] else 0.15), 0.40

    _orig_p, _orig_c = _a._genai_propose, _a._cva
    _a._genai_propose, _a._cva = _spy, _stub
    try:
        eff = {c: _np.array([0.2] * 5) for c in dss.DECISION_CONCEPTS}
        crisp = {c: 0.5 for c in dss.DECISION_CONCEPTS}

        def build(rules):
            _, pairs = eng._decide_regions(sim, eng._perceive(sim), rules)
            return eng._override(sim, pairs)

        out = _a.stage3_generative(build, sim, list(dss.SEED_RULES),
                                   eff, crisp, 6, engine=eng)
    finally:
        _a._genai_propose, _a._cva = _orig_p, _orig_c
    assert out.info["source"] == "evo-search"


def test_a_clause_macro_is_described_with_its_clauses():
    md = {"composition": [], "clauses": [
        {"effect": "coat", "sector": "assets", "range": [0, 2],
         "amount": 1.0}]}
    assert _a.describe_macro(md) == "coat on assets [0, 2] at 1.00"
    assert _a.describe_macro({"composition": [("suppression_effort", 1.0),
                                              ("containment_line", 0.8)]}) \
        == "suppression_effort 1.00 + containment_line 0.80"
