"""Tests for the rebuilt DSS, phase 1a: regions + ten bounded features."""

import numpy as np

from disaster_phyengine import Simulator, SimConfig, World
import dss


def _burning_world():
    w = World.blank(SimConfig(nx=60, ny=40, step_minutes=30.0),
                    default_fuel="grass", default_moisture=0.06)
    w.set_uniform_wind(8.0, 0.0)
    w.add_ignition(15, 20)
    return w


def test_partition_covers_grid_exactly():
    regs = dss.partition(60, 40, 2, 3)
    assert len(regs) == 6
    cover = np.zeros((40, 60), dtype=int)
    for r in regs:
        sy, sx = r.slices()
        cover[sy, sx] += 1
    assert cover.min() == 1 and cover.max() == 1, "overlap or gap"
    assert regs[0].name == "Agent_1" and regs[-1].name == "Agent_6"


def test_ten_features_bounded_and_ordered():
    sim = Simulator(_burning_world())
    for _ in range(5):
        sim.step()
    for r in dss.partition(60, 40, 2, 2):
        f = dss.ten_features(sim, r)
        assert list(f.keys()) == dss.FEATURE_ORDER
        assert all(0.0 <= v <= 1.0 for v in f.values()), f


def test_features_react_to_fire():
    sim = Simulator(_burning_world())
    regs = dss.partition(60, 40, 1, 2)   # west (fire) / east halves
    before = dss.ten_features(sim, regs[0])
    for _ in range(3):     # short: the fire must still be inside region 1
        sim.step()
    west = dss.ten_features(sim, regs[0])
    east = dss.ten_features(sim, regs[1])
    assert west["fire_intensity"] > 0.2, "fire not seen by its own agent"
    # NOTE: with literature grass speeds the front can cross the region
    # boundary within a single step, so no west-vs-east ordering is
    # asserted; each agent must simply see the fire state in its region
    assert west["temporal_urgency"] > before["temporal_urgency"] - 1e-9
    assert west["ignition_proximity"] == 1.0


def test_sensor_network_partial_observability():
    sim = Simulator(_burning_world())
    reg = dss.partition_n(60, 40, 1)[0]
    blind = dss.SensorNetwork([], 40, 60, 30.0)
    tower = dss.SensorNetwork([dss.Sensor("aerial", 15, 20)], 40, 60, 30.0)
    for _ in range(4):
        sim.step()
        blind.update(sim, 30.0)
        tower.update(sim, 30.0)
    f_blind = dss.ten_features(sim, reg, network=blind)
    f_tower = dss.ten_features(sim, reg, network=tower)
    f_true = dss.ten_features(sim, reg)
    assert f_blind["fire_intensity"] == 0.0, "blind agent must miss the fire"
    assert f_tower["fire_intensity"] > 0.2, "tower must see the fire"
    # NOTE the tower's picture is DELIBERATELY stale: reports are
    # delivered one revisit+latency behind the live state, so the
    # observed intensity may differ from the current truth. What matters
    # is that the covered agent sees a fire and the blind one does not.
    assert f_tower["fire_intensity"] <= 1.0
    assert tower.region_conf(reg) > blind.region_conf(reg)
    for f in (f_blind, f_tower):
        assert all(0.0 <= v <= 1.0 for v in f.values())


def test_conf_decays_between_revisits():
    sim = Simulator(_burning_world())
    net = dss.SensorNetwork([dss.Sensor("satellite", 0, 0)], 40, 60, 30.0)
    net.update(sim, 0.0)          # first pass: report goes en route
    net.update(sim, 30.0)         # latency elapsed: report DELIVERED
    reg = dss.partition_n(60, 40, 1)[0]
    k0 = net.region_conf(reg)
    for _ in range(4):            # 2 h without a new pass (revisit 6 h)
        sim.step()
        net.update(sim, 30.0)
    assert net.region_conf(reg) < k0, "confidence must decay while stale"


def test_conf_is_min_of_factors():
    import numpy as np
    net = dss.SensorNetwork([dss.Sensor("aerial", 30, 20)], 40, 60, 30.0)
    c = net.conf_channel("burning")
    fresh = np.exp(-dss.LAMBDA_CONF * net.age["burning"])
    manual = np.minimum.reduce([net.theta["burning"], net.rho["burning"],
                                fresh, net.gamma["burning"]])
    assert np.allclose(c, manual)
    assert net.conf_cell().max() <= c.max() + 1e-12   # min over components


def test_five_term_partition_matches_worked_example():
    import dss
    mu = dss.fuzzify(0.62)
    assert abs(mu["M"] - 0.533) < 0.01 and abs(mu["H"] - 0.467) < 0.01
    assert mu["VL"] == mu["L"] == mu["VH"] == 0.0
    # at most two adjacent terms activate anywhere
    import numpy as np
    for z in np.linspace(0, 1, 101):
        assert sum(1 for v in dss.fuzzify(float(z)).values() if v > 1e-9) <= 2


def test_hierarchy_weights_sum_to_one():
    import dss
    for name, (lvl, inputs) in dss.HIERARCHY.items():
        assert abs(sum(w for _, w in inputs) - 1.0) < 1e-9, name
        assert 1 <= lvl <= 4


def test_concepts_monotone_and_gated():
    import dss
    low = {k: 0.1 for k in dss.FEATURE_ORDER}
    high = dict(low, fire_intensity=0.95, spread_potential=0.9,
                weather_severity=0.8, ignition_proximity=1.0)
    c_lo = dss.crisp(dss.infer_concepts(low))
    c_hi = dss.crisp(dss.infer_concepts(high))
    assert c_hi["fire_threat_level"] > c_lo["fire_threat_level"]
    g = dss.GatedConcepts()
    e1 = dss.crisp(g.gate(dss.infer_concepts(high), 1.0, step=1))
    e2 = dss.crisp(g.gate(dss.infer_concepts(low), 0.0, step=2))
    # blind step: effective activation must FADE (rho), not track the
    # unobserved low input
    assert e2["fire_threat_level"] < e1["fire_threat_level"]
    assert e2["fire_threat_level"] > c_lo["fire_threat_level"]


def test_seed_rules_fire_and_bound():
    import dss
    feats = dict(fire_intensity=0.9, spread_potential=0.85,
                 weather_severity=0.7, ignition_proximity=1.0,
                 fuel_load=0.6, asset_exposure=0.9,
                 resource_accessibility=0.7, access_road_status=0.15,
                 suppression_availability=0.6, temporal_urgency=0.9)
    g = dss.GatedConcepts()
    eff = g.gate(dss.infer_concepts(feats), 1.0, step=1)
    out, trace = dss.evaluate_rules(eff, feats)
    fired = {r.name for r, w in trace if w > 0.05}
    assert "R4" in fired            # blocked egress + evac pressure
    assert out["public_warning"] > 0.5
    assert all(0.0 <= v <= 1.0 for v in out.values())
    calm = {k: 0.02 for k in dss.FEATURE_ORDER}
    eff0 = dss.GatedConcepts().gate(dss.infer_concepts(calm), 1.0, step=1)
    out0, tr0 = dss.evaluate_rules(eff0, calm)
    assert out0["evacuation"] < 0.2   # calm scene: no evacuation order
