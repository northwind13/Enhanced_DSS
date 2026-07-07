"""Unit and integration tests for the DSS package (sensing + units design)."""

import numpy as np
import pytest

from disasteraware.config import SimConfig
from disasteraware.core import Simulator
from disasteraware.observation import observe
from disasteraware.world import World, Asset

from dss.fuzzy import FivePartition
from dss.features import extract_features, FEATURE_NAMES, observation_confidence
from dss.concepts import compute_concepts, gate_concepts, ALL_CONCEPTS
from dss.rules import default_rule_base, Rule, INTERVENTION_TYPES
from dss.inference import fire_rules
from dss.quality import decision_quality
from dss.sensing import Sensor, SensorNetwork
from dss.units import ResourceUnit, assign_units
from dss.agent import RegionalAgent
from dss.loop import DSSRunner


# ------------------------------------------------------------------ fixtures
def small_world(ny=24, nx=24):
    # calibrated for the literature-based fuel models: drier fuel and
    # stronger wind are needed for a sustained spreading fire
    cfg = SimConfig(ny=ny, nx=nx, max_steps=60)
    cfg.suppression.alpha_s = 0.30   # top of the Table B calibration range
    world = World.blank(cfg, default_fuel="grass", default_load=1.0,
                       default_moisture=0.03)
    world.add_forest_patch(4, 4, 19, 19, fuel_type="pine_litter",
                           load=1.0, moisture=0.05)
    world.add_asset(Asset("hospital", "critical", 20, 20, radius=1, value=1.0))
    world.add_asset(Asset("village", "population", 20, 18, radius=1,
                          population=500))
    world.add_ignition(6, 6, step=0)
    world.set_uniform_wind(10.0, 0.8)
    return world


def uniform_intervention(shape, **levels):
    base = {j: np.zeros(shape) for j in INTERVENTION_TYPES}
    for k, v in levels.items():
        base[k] = np.full(shape, float(v))
    return base


# ---------------------------------------------------------------- fuzzy layer
def test_partition_of_unity():
    part = FivePartition()
    x = np.linspace(0, 1, 2001)
    mu = part.fuzzify(x)
    assert mu.shape == (2001, 5)
    assert np.allclose(mu.sum(axis=-1), 1.0, atol=1e-9)
    assert (mu >= 0).all() and (mu <= 1).all()


def test_term_centers_are_prototypes():
    part = FivePartition()
    for i, c in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        mu = part.fuzzify(c)
        assert mu[i] == pytest.approx(1.0)


# ------------------------------------------------------------------- sensing
def test_uncovered_cells_have_zero_confidence():
    world = small_world()
    sim = Simulator(world)
    sim.step()
    net = SensorNetwork([Sensor.preset("in_situ", x=6, y=6)], world.shape)
    net.sample(sim, sim.state.step)
    obs, kappa = net.composite(sim.state.step)
    assert kappa[6, 6] > 0.9
    assert kappa[20, 20] == 0.0
    assert obs.burning[20, 20] == 0.0


def test_staleness_decays_confidence():
    world = small_world()
    sim = Simulator(world)
    sim.step()
    sat = Sensor.preset("satellite")               # period 12
    net = SensorNetwork([sat], world.shape)
    net.sample(sim, 0)
    c0 = float(net.conf["burning"].max())
    for k in range(1, 6):
        net.sample(sim, k)                         # sat not due again
    c5 = float(net.conf["burning"].max())
    assert 0 < c5 < c0                             # age term decays


def test_density_bonus_and_weakest_channel():
    world = small_world()
    sim = Simulator(world)
    sim.step()
    one = SensorNetwork([Sensor.preset("aerial", x=12, y=12)], world.shape)
    two = SensorNetwork([Sensor.preset("aerial", x=12, y=12),
                         Sensor.preset("in_situ", x=12, y=12)], world.shape)
    one.sample(sim, 1); two.sample(sim, 1)
    _, k1 = one.composite(1)
    _, k2 = two.composite(1)
    assert k2[12, 12] >= k1[12, 12]
    # field_report senses only burning: fload/intensity unsensed -> kappa 0
    rep = SensorNetwork([Sensor.preset("field_report", x=12, y=12)],
                        world.shape)
    rep.sample(sim, 0)
    _, kr = rep.composite(0)
    assert kr[12, 12] == 0.0


def test_ideal_network_full_confidence():
    world = small_world()
    sim = Simulator(world)
    sim.step()
    net = SensorNetwork.ideal(world.shape)
    net.sample(sim, sim.state.step)
    obs, kappa = net.composite(sim.state.step)
    assert np.allclose(kappa, 1.0)
    assert np.allclose(obs.intensity, sim.state.intensity)


# --------------------------------------------------------------------- units
def test_unit_tasked_within_radius_only():
    shape = (24, 24)
    demand = np.zeros(shape)
    demand[2, 2] = 1.0
    unit = ResourceUnit.preset("crew", x=20, y=20)  # radius 6
    layer, log = assign_units([unit], demand)
    assert len(log) == 0
    assert layer.rcap.sum() == 0.0


def test_unit_rasterizes_u_res_fields():
    shape = (24, 24)
    demand = np.zeros(shape)
    demand[10, 10] = 1.0
    unit = ResourceUnit.preset("engine", x=4, y=10, unit_id="e1")
    layer, log = assign_units([unit], demand)
    assert len(log) == 1 and log[0].unit_id == "e1"
    assert log[0].target == (10, 10)
    assert layer.rcap[10, 10] > 0
    assert layer.reff[10, 10] == pytest.approx(0.7)
    assert layer.ravail[10, 10] == pytest.approx(1.0)
    assert layer.rtime[10, 10] == pytest.approx(6 / 3.0)


def test_fleet_prefers_high_priority_demand():
    shape = (24, 24)
    demand = np.zeros(shape)
    demand[5, 5] = 1.0
    demand[18, 18] = 1.0
    priority = np.zeros(shape)
    priority[18, 18] = 1.0
    unit = ResourceUnit.preset("helicopter", x=12, y=12)
    _, log = assign_units([unit], demand, priority)
    assert log[0].target == (18, 18)


# ------------------------------------------------------------------- features
def test_features_bounded_and_complete():
    world = small_world()
    sim = Simulator(world)
    sim.step()
    obs = observe(sim)
    feats = extract_features(obs, world)
    assert set(feats.values.keys()) == set(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        v = feats[name]
        assert v.shape == world.shape
        assert (v >= 0).all() and (v <= 1).all(), name


def test_confidence_fallback_channels():
    kappa = observation_confidence((8, 8), epsilon=0.0)
    assert np.allclose(kappa, 1.0)
    kappa = observation_confidence((8, 8), epsilon=0.25)
    assert np.allclose(kappa, 0.5)
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4] = True
    kappa = observation_confidence((8, 8), epsilon=0.0, region_mask=mask)
    assert np.allclose(kappa[:4], 1.0) and np.allclose(kappa[4:], 0.0)


# ------------------------------------------------------------------- concepts
def test_concepts_bounded_and_complete():
    world = small_world()
    sim = Simulator(world)
    sim.step()
    feats = extract_features(observe(sim), world)
    concepts = compute_concepts(feats)
    assert set(concepts.keys()) == set(ALL_CONCEPTS)
    for name, c in concepts.items():
        assert (c >= 0).all() and (c <= 1).all(), name


def test_gating_identity_and_fallback():
    raw = {"c": np.full((4, 4), 0.8)}
    prior = {"c": np.full((4, 4), 0.2)}
    g = gate_concepts(raw, prior, np.ones((4, 4)))
    assert np.allclose(g["c"], 0.8)
    g = gate_concepts(raw, prior, np.zeros((4, 4)))
    assert np.allclose(g["c"], 0.2)
    g = gate_concepts(raw, None, np.zeros((4, 4)))
    assert np.allclose(g["c"], 0.8)


# ------------------------------------------------------------------ inference
def test_inference_convex_and_traceable():
    rb = default_rule_base()
    shape = (6, 6)
    signals = {name: np.full(shape, 0.75) for name in
               ("fire_threat_level", "asset_risk", "suppression_feasibility",
                "intervention_urgency", "evacuation_pressure",
                "access_road_status")}
    intervention, firings = fire_rules(rb, signals)
    assert len(firings) == len(rb.rules)
    for itype, u in intervention.items():
        assert (u >= 0).all() and (u <= 1).all(), itype
    assert intervention["suppression_effort"].mean() > 0.4


def test_rule_validation():
    with pytest.raises(ValueError):
        Rule("bad", {"fire_threat_level": "Sideways"}, {"evacuation": "High"})
    with pytest.raises(ValueError):
        Rule("bad2", {"fire_threat_level": "High"}, {"teleport": "High"})


# -------------------------------------------------------------------- quality
def test_quality_bounds():
    shape = (5, 5)
    concepts = {name: np.full(shape, 0.6) for name in ALL_CONCEPTS}
    perfect = {j: np.full(shape, 0.6) for j in INTERVENTION_TYPES}
    assert decision_quality(perfect, concepts) == pytest.approx(1.0)
    worst = {j: np.zeros(shape) if j != "evacuation" else np.ones(shape)
             for j in perfect}
    q = decision_quality(worst, concepts)
    assert 0.0 <= q < 1.0


# ---------------------------------------------------------------- closed loop
def test_agent_decides_in_region_only():
    world = small_world()
    sim = Simulator(world)
    sim.step()
    agent = RegionalAgent("A00", region=(0, 0, 11, 11),
                          grid_shape=world.shape)
    obs = observe(sim, region=(0, 0, 11, 11))
    dec = agent.decide(obs, world)
    outside = ~agent.region_mask
    for j, u in dec.intervention.items():
        assert np.allclose(u[outside], 0.0), j


def test_closed_loop_runs_and_traces():
    world = small_world()
    sim = Simulator(world)
    runner = DSSRunner(sim, n_regions=(2, 2), quality_threshold=0.5)
    results = runner.run(n_steps=20)
    assert len(results) > 0
    assert len(runner.audit.records) == len(results) * len(runner.agents)
    rec = runner.audit.records[-1]
    assert 0.0 <= rec.quality <= 1.0
    assert rec.concept_means and rec.intervention_means


def test_closed_loop_with_sensors_and_units():
    world = small_world()
    sim = Simulator(world)
    sensors = [Sensor.preset("satellite"),
               Sensor.preset("aerial", x=8, y=8),
               Sensor.preset("in_situ", x=20, y=19)]
    units = [ResourceUnit.preset("engine", x=2, y=21, unit_id="e1"),
             ResourceUnit.preset("helicopter", x=12, y=22, unit_id="h1"),
             ResourceUnit.preset("crew", x=21, y=21, unit_id="c1")]
    runner = DSSRunner(sim, n_regions=(2, 2), sensors=sensors, units=units,
                       quality_threshold=0.5, seed=7)
    results = runner.run(n_steps=25)
    assert len(results) > 0
    tasked = [a for r in results for a in r.global_decision.assignments]
    assert len(tasked) > 0
    for a in tasked:
        assert a.unit_id in ("e1", "h1", "c1")
    rec = runner.audit.records[-1]
    assert rec.confidence_mean < 1.0


def test_dss_mitigates_fire_impact():
    """DSS suppression must cut fuel consumption without enlarging the fire."""
    world_a = small_world()
    sim_a = Simulator(world_a)
    sim_a.run(n_steps=40)
    burned_no_dss = float(sim_a.ever_burned.sum())
    consumed_no_dss = float(sim_a.fuel_consumed_total.sum())

    world_b = small_world()
    sim_b = Simulator(world_b)
    runner = DSSRunner(sim_b, n_regions=(2, 2), quality_threshold=0.5)
    runner.run(n_steps=40)
    burned_dss = float(sim_b.ever_burned.sum())
    consumed_dss = float(sim_b.fuel_consumed_total.sum())
    suppressed_dss = float(sim_b.fuel_suppressed_total.sum())

    assert suppressed_dss > 0.0
    assert consumed_dss < 0.95 * consumed_no_dss
    assert burned_dss <= burned_no_dss


def test_resource_budget_scaling():
    world = small_world()
    sim = Simulator(world)
    runner = DSSRunner(sim, n_regions=(2, 2), suppression_budget=5.0,
                       quality_threshold=0.5)
    result = runner.step()
    total_rcap = float(result.global_decision.resource_layer.rcap.sum())
    assert total_rcap <= 5.0 + 1e-6


# ------------------------------------------------------------ effect model
def test_mitigation_tracker_records_at_ignition_only():
    from dss.mitigation import MitigationTracker
    shape = (4, 4)
    tr = MitigationTracker(shape=shape)
    interv = {j: np.zeros(shape) for j in
              ("evacuation", "asset_protection", "public_warning")}
    interv["evacuation"][1, 1] = 0.8

    burned = np.zeros(shape, dtype=bool)
    tr.update(interv, burned)
    burned[1, 1] = True
    tr.update({j: np.zeros(shape) for j in interv}, burned)
    assert tr.at_burn["evacuation"][1, 1] == pytest.approx(0.8)

    late = {j: np.zeros(shape) for j in interv}
    late["evacuation"][1, 1] = 1.0
    tr.update(late, burned)
    assert tr.at_burn["evacuation"][1, 1] == pytest.approx(0.8)


def test_mitigated_costs_discount_losses():
    from dss.mitigation import MitigationTracker, mitigated_costs, RHO_EVAC
    from disasteraware.costs import compute_costs
    world = small_world()
    sim = Simulator(world)
    tr = MitigationTracker(shape=world.shape)
    full = {"evacuation": np.ones(world.shape),
            "asset_protection": np.ones(world.shape),
            "public_warning": np.ones(world.shape)}
    for _ in range(30):
        tr.update(full, sim.ever_burned)
        sim.step()
    tr.update(full, sim.ever_burned)

    base = compute_costs(sim)
    mit = mitigated_costs(sim, tr)
    assert mit.burned_area_ha == pytest.approx(base.burned_area_ha)
    assert mit.suppression_cost == pytest.approx(base.suppression_cost)
    if base.population_exposed > 0:
        assert mit.population_exposed == pytest.approx(
            base.population_exposed * (1 - RHO_EVAC))
    assert mit.total_economic_cost <= base.total_economic_cost


# ------------------------------------------------------------ decision cost
def test_decision_cost_bounded_and_deterministic():
    from dss.evaluate import evaluate_intervention
    world = small_world()
    sim = Simulator(world)
    for _ in range(5):
        sim.step()
    u = uniform_intervention(world.shape, suppression_effort=0.5,
                             resource_deployment=0.5)
    a = evaluate_intervention(sim, u, horizon=8)
    b = evaluate_intervention(sim, u, horizon=8)
    assert 0.0 <= a.total <= 1.0
    for v in a.terms.values():
        assert 0.0 <= v <= 1.0
    assert a.total == b.total
    assert sim.state.step == 5


def test_do_nothing_costs_more_than_response():
    from dss.evaluate import evaluate_intervention
    world = small_world()
    sim = Simulator(world)
    for _ in range(5):
        sim.step()
    nothing = uniform_intervention(world.shape)
    strong = uniform_intervention(world.shape, suppression_effort=0.9,
                                  resource_deployment=0.9)
    j0 = evaluate_intervention(sim, nothing, horizon=12)
    j1 = evaluate_intervention(sim, strong, horizon=12)
    assert j0.terms["del"] > j1.terms["del"]
    assert j0.terms["burn"] >= j1.terms["burn"]
    assert j0.total > j1.total


def test_protective_action_is_not_free():
    from dss.evaluate import evaluate_intervention
    world = small_world()
    sim = Simulator(world)
    for _ in range(5):
        sim.step()
    targeted = uniform_intervention(world.shape, suppression_effort=0.6)
    blanket = uniform_intervention(world.shape, suppression_effort=0.6,
                                   evacuation=1.0, public_warning=1.0,
                                   asset_protection=1.0)
    j_t = evaluate_intervention(sim, targeted, horizon=8)
    j_b = evaluate_intervention(sim, blanket, horizon=8)
    assert j_b.terms["sup"] > j_t.terms["sup"]


def test_delay_term_rewards_coverage():
    from dss.evaluate import evaluate_intervention
    world = small_world()
    sim = Simulator(world)
    for _ in range(5):
        sim.step()
    uncovered = uniform_intervention(world.shape)
    covered = uniform_intervention(world.shape, suppression_effort=0.8,
                                   resource_deployment=0.8)
    j_u = evaluate_intervention(sim, uncovered, horizon=8)
    j_c = evaluate_intervention(sim, covered, horizon=8)
    assert j_u.terms["del"] == 1.0
    assert j_c.terms["del"] == 0.0
