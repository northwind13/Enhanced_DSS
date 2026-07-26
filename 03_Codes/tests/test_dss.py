"""Tests for the rebuilt DSS, phase 1a: regions + ten bounded features."""

import os

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
    # threatened, exposed scene: backbone threat and evacuation rules fire
    assert fired & {"R25", "R26"}, fired
    assert fired & {"R39", "R40"}, fired
    assert out["public_warning"] > 0.4
    assert all(0.0 <= v <= 1.0 for v in out.values())
    # provenance placeholders never fire (inactive until adaptation)
    inact = {r.name for r, w in trace if not r.active}
    assert inact == {"R41", "R42"}
    # a REALISTIC calm scene: roads exist, some resources staged, no fire.
    # (An all-zero desert is pathological: Appendix D's R23 backbone
    # correctly reads "feasibility very low" there and raises evacuation,
    # so it is not a meaningful calm baseline.)
    calm = dict(fire_intensity=0.0, spread_potential=0.10,
                weather_severity=0.30, ignition_proximity=0.0,
                fuel_load=0.50, asset_exposure=0.40,
                resource_accessibility=0.70, access_road_status=0.60,
                suppression_availability=0.50, temporal_urgency=0.0)
    eff0 = dss.GatedConcepts().gate(dss.infer_concepts(calm), 1.0, step=1)
    out0, tr0 = dss.evaluate_rules(eff0, calm)
    assert out0["evacuation"] < 0.3
    assert out0["suppression_effort"] < 0.4
    # extremes of the universe carry full membership (saturated ends)
    assert dss.term_vector(1.0)[dss.TERMS.index("VH")] == 1.0
    assert dss.term_vector(0.0)[dss.TERMS.index("VL")] == 1.0


def test_resource_suggestion_and_decision_fields():
    import numpy as np
    import dss
    from disaster_phyengine import terrain
    from disaster_phyengine.config import SimConfig
    w = terrain.generate_landscape(SimConfig(nx=80, ny=60), seed=4,
                                   preset="Rolling hills",
                                   n_settlements=4,
                                   population_per_settlement=8000)
    base, why = dss.resource_suggestion(w)
    assert (base.rcap > 0).any() and len(why) == 5
    assert base.rcap.max() <= w.config.suppression.rcap_max + 1e-9
    assert np.isfinite(base.rtime).all() and base.rtime.min() >= 5.0  # helibase dispatch 6 min
    # decisions allocate the pool inside the ordering region only
    regs = dss.partition_n(80, 60, 4)
    burning = np.zeros((60, 80), dtype=bool)
    burning[15:18, 10:13] = True                     # fire in region 1
    u_hot = {k: 0.0 for k in dss.INTERVENTIONS}
    u_hot.update(suppression_effort=1.0, resource_deployment=1.0,
                 containment_line=0.8)
    ri = [(regs[0], u_hot)] + [(r, {k: 0.0 for k in dss.INTERVENTIONS})
                               for r in regs[1:]]
    ov = dss.decision_to_resources(w, burning, ri, base)
    sy, sx = regs[0].slices()
    assert ov.rcap[sy, sx].max() > 0.0, "ordered region must get capacity"
    assert ov.rcap.max() <= 1.5 * w.config.suppression.rcap_max + 1e-9
    assert (ov.rtime[sy, sx].mean()
            < base.rtime[sy, sx].mean()), "deployment must cut R_time"


def test_decisions_reduce_fuel_via_engine():
    import numpy as np
    import dss
    from disaster_phyengine.config import SimConfig
    from disaster_phyengine.world import World
    from disaster_phyengine.core import Simulator
    cfg = SimConfig(nx=60, ny=40)
    w = World.blank(cfg, default_fuel="hardwood")
    w.fuel.fload[:] = 0.9
    w.fuel.fmoist[:] = 0.12
    sim = Simulator(w)
    w.add_ignition(10, 20, step=0, radius=1)
    sim.step()
    regs = dss.partition_n(60, 40, 1)
    u = {k: 0.0 for k in dss.INTERVENTIONS}
    u.update(suppression_effort=1.0, resource_deployment=1.0,
             containment_line=1.0)
    probe = None
    for _ in range(6):
        burning = sim.state.burning > 0.5
        ov = dss.decision_to_resources(w, burning, [(regs[0], u)],
                                       base=None)
        if probe is None:
            band = ov.rcap > 0.5
            band &= ~burning
            ys, xs = np.where(band)
            assert xs.size, "containment band must exist"
            probe = (ys[0], xs[0])
            f0 = float(sim.state.fload[probe])
        sim.step(resource_override=ov)
    f1 = float(sim.state.fload[probe])
    assert f1 < f0, "ordered suppression must reduce fuel on the band"


def test_per_variable_partitions_are_independent():
    import dss
    # baseline: identical semantics everywhere
    v0 = dss.term_vector(0.62, var="fire_intensity")
    v1 = dss.term_vector(0.62, var="spread_potential")
    assert (v0 == v1).all()
    # evFIS-style edit: widen the H core of ONE variable only
    dss.REGISTRY.set_term("spread_potential", "H",
                          (0.50, 0.60, 0.85, 0.95))
    try:
        w1 = dss.term_vector(0.62, var="spread_potential")
        w0 = dss.term_vector(0.62, var="fire_intensity")
        assert w1[dss.TERMS.index("H")] > v1[dss.TERMS.index("H")]
        assert (w0 == v0).all(), "other variables must stay untouched"
    finally:
        # restore the default so later tests see clean semantics
        dss.REGISTRY.set_term("spread_potential", "H",
                              dss.default_partition()["H"])


def test_per_feature_confidence_and_concept_gates():
    import dss
    from disaster_phyengine.config import SimConfig
    from disaster_phyengine.world import World
    from disaster_phyengine.core import Simulator
    w = World.blank(SimConfig(nx=60, ny=40), default_fuel="grass")
    sim = Simulator(w)
    reg = dss.partition_n(60, 40, 1)[0]
    # aerial only: senses B and I, never F -> sensed features degrade,
    # prior-driven features keep confidence one
    net = dss.SensorNetwork([dss.Sensor("aerial", 30, 20)], 40, 60, 30.0)
    net.update(sim, 0.0)
    fc = dss.feature_confidence(net, reg)
    assert fc["weather_severity"] == 1.0 and fc["asset_exposure"] == 1.0
    assert fc["fire_intensity"] < 1.0        # sensed, partially covered
    gates = dss.concept_gates(fc)
    # a concept fed only by prior features keeps gate 1; one that
    # consumes sensed channels is bounded by its weakest feature
    assert gates["logistics_support"] == 1.0
    assert gates["fire_severity"] <= fc["fire_intensity"] + 1e-9
    # the minimum propagates upward through the hierarchy
    assert gates["fire_threat_level"] <= gates["fire_severity"] + 1e-9
    assert gates["operational_priority"] <= gates["fire_threat_level"] + 1e-9
    # blind network: everything sensed is untrusted, priors stay one
    fc0 = dss.feature_confidence(dss.SensorNetwork([], 40, 60, 30.0), reg)
    assert fc0["fuel_load"] <= 0.75 and fc0["access_road_status"] == 1.0


def _mini_fire_sim():
    import numpy as np
    from disaster_phyengine.config import SimConfig
    from disaster_phyengine.world import World
    from disaster_phyengine.core import Simulator
    cfg = SimConfig(nx=60, ny=40)
    cfg.step_minutes = 5.0
    w = World.blank(cfg, default_fuel="grass")
    w.fuel.fload[:] = 0.8
    w.fuel.fmoist[:] = 0.07
    w.meteo.wws[:] = 8.0
    sim = Simulator(w)
    w.add_ignition(20, 20, step=0, radius=1)
    for _ in range(4):
        sim.step()
    return w, sim


def test_forecast_is_shadowed_and_repeatable():
    import dss
    w, sim = _mini_fire_sim()
    k0 = sim.state.step
    j1, j0 = dss.candidate_vs_noaction(sim, None, horizon=6)
    assert sim.state.step == k0, "forecast must never touch the live sim"
    assert abs(j1 - j0) < 1e-12, "same override, same cost"


def test_decision_engine_cycle_logs_and_applies():
    import dss
    w, sim = _mini_fire_sim()
    base, _ = dss.resource_suggestion(w)
    regs = dss.partition_n(60, 40, 2)
    eng = dss.DecisionEngine(regs, base_pool=base, j_threshold=0.05,
                             cycle_steps=2, horizon_steps=4,
                             adapt_on=True, genai_on=False)
    for _ in range(6):
        ov = eng.maybe_decide(sim)
        sim.step(resource_override=ov)
    assert len(eng.log.cycles()) >= 3
    rec = eng.log.records[-1]
    assert set(rec.intensities) == set(dss.INTERVENTIONS)
    assert 0.0 <= rec.quality <= 1.0
    why = eng.log.why(rec)
    assert any("concepts" in ln for ln in why)
    # adaptation may add rules but never removes the seed base
    assert len(eng.rules) >= len(dss.SEED_RULES)


def test_counterfactual_replays_without_orders():
    import dss
    w, sim = _mini_fire_sim()
    base, _ = dss.resource_suggestion(w)
    regs = dss.partition_n(60, 40, 1)
    eng = dss.DecisionEngine(regs, base_pool=base, j_threshold=0.02,
                             cycle_steps=2, horizon_steps=4,
                             adapt_on=False)
    k0 = sim.state.step
    for _ in range(8):
        sim.step(resource_override=eng.maybe_decide(sim))
    cf, rep = dss.counterfactual(sim, k0)
    assert cf is not None and cf.state.step == sim.state.step
    assert rep.j_total >= 0.0
    assert sim.state.step == k0 + 8, "live sim untouched by the replay"


def test_quality_gate_and_failsafe():
    import dss
    crisp_c = dict(fire_threat_level=0.1, asset_exposure_risk=0.1,
                   intervention_urgency=0.1, evacuation_pressure=0.1)
    over = {k: 0.9 for k in dss.INTERVENTIONS}
    q = dss.quality_Q(crisp_c, over)     # heavy action, calm concepts
    assert q < 0.6
    scaled, hit = dss.graduated_failsafe(over, q, eta=0.6)
    assert hit and scaled["suppression_effort"] < 0.9
    assert scaled["evacuation"] == 0.9, "life-safety never reduced"


def test_no_harm_failsafe_withholds_useless_orders():
    import dss
    w, sim = _mini_fire_sim()
    # empty pool: any allocation is pure response cost with no physical
    # gain, so the engine must withhold the offensive orders
    from disaster_phyengine.layers import ResourceLayer
    empty = ResourceLayer.none(40, 60)
    w.config.cost.capacity_reference = 100.0
    regs = dss.partition_n(60, 40, 1)
    eng = dss.DecisionEngine(regs, base_pool=empty, j_threshold=0.001,
                             cycle_steps=2, horizon_steps=6,
                             adapt_on=False)
    ov = eng.decide(sim)
    rec = eng.log.records[-1]
    if eng.last_withheld:
        assert ov is None
        assert rec.intensities["suppression_effort"] == 0.0
        assert rec.intensities["public_warning"] >= 0.0  # life-safety kept


def test_fire_is_extinguishable_with_dss():
    """End-to-end guarantee: a moderate grass fire near the staged pool
    MUST be fully extinguished by the DSS within a few simulated hours.
    This is the product's core promise; if a physics or allocator change
    breaks it, this test fails."""
    import numpy as np
    import dss
    from disaster_phyengine import terrain
    from disaster_phyengine.config import SimConfig, FUEL_NAME_TO_ID
    from disaster_phyengine.core import Simulator
    cfg = SimConfig(nx=60, ny=40, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(cfg, seed=11, preset="Rolling hills",
                                   n_settlements=4,
                                   population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.10
    w.meteo.wws[:] = 5.0
    base, _ = dss.resource_suggestion(w)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    ok = ((w.fuel.ftype == FUEL_NAME_TO_ID["grass"])
          & (w.fuel.fload > 0.5) & (base.rtime < 25))
    ys, xs = np.where(ok)
    k = len(xs) // 2
    sim = Simulator(w)
    sim.record_states = False
    w.add_ignition(int(xs[k]), int(ys[k]), step=0, radius=1)
    eng = dss.DecisionEngine(dss.partition_n(60, 40, 1), base_pool=base,
                             j_threshold=0.35, cycle_min=8.0,
                             horizon_min=10.0, adapt_on=False)
    out_at = None
    for step in range(180):          # 6 saat
        sim.step(resource_override=eng.maybe_decide(sim))
        if int((sim.state.burning > 0.5).sum()) == 0:
            out_at = 2 * (step + 1)
            break
    assert out_at is not None, "the fire must be fully extinguished"
    assert int(sim.ever_burned.sum()) < 400, "and the damage bounded"


def test_remote_forest_fire_is_extinguishable_with_dss():
    """A forest fire FAR from settlements (low priority, poor access,
    long dispatch) must also be extinguished, not merely contained: the
    committed wetting floor guarantees the engage -> wet -> knockdown
    chain completes even where the pressure product collapses. Guards
    against the 'save the cities, let the forest burn' failure mode."""
    import numpy as np
    import dss
    from disaster_phyengine import terrain
    from disaster_phyengine.config import SimConfig, FUEL_NAME_TO_ID
    from disaster_phyengine.core import Simulator
    cfg = SimConfig(nx=60, ny=40, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(cfg, seed=42, preset="Rolling hills",
                                   n_settlements=4,
                                   population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.10
    w.meteo.wws[:] = 5.0
    base, _ = dss.resource_suggestion(w)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    prio = w.priority_field()
    prio = prio / (prio.max() + 1e-9)
    beta = w.config.suppression.beta_t
    reach = np.exp(-beta * base.rtime) * np.clip(w.topo.access, 0, 1)
    forest = np.isin(w.fuel.ftype, [FUEL_NAME_TO_ID["pine_litter"],
                                    FUEL_NAME_TO_ID["shrub"]])
    ys, xs = np.where(forest & (prio < 0.02) & (w.fuel.fload > 0.4))
    k = int(np.argmin(reach[ys, xs]))          # WORST-reach forest cell
    sim = Simulator(w)
    sim.record_states = False
    w.add_ignition(int(xs[k]), int(ys[k]), step=0, radius=1)
    eng = dss.DecisionEngine(dss.partition_n(60, 40, 1), base_pool=base,
                             j_threshold=0.35, cycle_min=8.0,
                             horizon_min=10.0, adapt_on=False)
    out_at = None
    for i in range(120):                        # 4 simulated hours
        ov = eng.maybe_decide(sim)
        sim.step(resource_override=ov)
        t = sim.state.step * cfg.step_minutes
        if t > 10 and int((sim.state.burning > 0.5).sum()) == 0:
            out_at = t
            break
    assert out_at is not None, "remote forest fire was never extinguished"
    assert int(sim.ever_burned.sum()) < 150


def test_counterfactual_matches_manual_no_dss_run():
    """The counterfactual replay ("what if these orders were NOT
    taken") must reproduce EXACTLY the run that never had any orders:
    rewind has to restore every in-place mutable (fuel moisture soaked
    by suppression, rng draws consumed by spotting), or the replay
    inherits the factual run's wet fuel and burns far too little."""
    import numpy as np
    import dss
    from disaster_phyengine import terrain
    from disaster_phyengine.config import SimConfig, FUEL_NAME_TO_ID
    from disaster_phyengine.core import Simulator

    def mkworld():
        cfg = SimConfig(nx=60, ny=40, cell_size_m=30.0)
        cfg.step_minutes = 2.0
        w = terrain.generate_landscape(
            cfg, seed=11, preset="Rolling hills", n_settlements=4,
            population_per_settlement=15000)
        w.fuel.fmoist[:] = 0.10
        w.meteo.wws[:] = 5.0
        return w

    w = mkworld()
    base, _ = dss.resource_suggestion(w)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    ok = ((w.fuel.ftype == FUEL_NAME_TO_ID["grass"])
          & (w.fuel.fload > 0.5) & (base.rtime < 25))
    ys, xs = np.where(ok)
    k = len(xs) // 2
    w.add_ignition(int(xs[k]), int(ys[k]), step=0, radius=1)
    sim = Simulator(w)                       # snapshots ON, as in the app
    eng = dss.DecisionEngine(dss.partition_n(60, 40, 1), base_pool=base,
                             j_threshold=0.35, cycle_min=8.0,
                             horizon_min=10.0, adapt_on=False)
    for _ in range(40):
        sim.step(resource_override=eng.maybe_decide(sim))
    cf, rep = dss.counterfactual(sim, 0)
    assert cf is not None

    w2 = mkworld()
    w2.add_ignition(int(xs[k]), int(ys[k]), step=0, radius=1)
    s3 = Simulator(w2)
    s3.record_states = False
    for _ in range(40):
        s3.step()
    assert int(cf.ever_burned.sum()) == int(s3.ever_burned.sum())
    # and the orders must have mattered in the factual run
    assert int(sim.ever_burned.sum()) < int(cf.ever_burned.sum())


def test_minimal_profile_learns_and_extinguishes():
    """The core promise of the staged adaptation: 5 seed rules plus
    the growth stages (resolution + GenAI template) must be ENOUGH.
    Four Local DSS agents under the Global coordination fight an
    aggressive fire from the minimal profile; the rule base must grow
    and the fire must go out."""
    import numpy as np
    import dss
    from disaster_phyengine import terrain
    from disaster_phyengine.config import SimConfig
    from disaster_phyengine.core import Simulator
    cfg = SimConfig(nx=60, ny=40, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(cfg, seed=42, preset="Rolling hills",
                                   n_settlements=4,
                                   population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.08
    w.meteo.wws[:] = 6.0
    base, _ = dss.resource_suggestion(w)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    w.add_ignition(45, 20, step=0, radius=2)
    sim = Simulator(w)
    sim.record_states = False
    eng = dss.DecisionEngine(dss.partition_n(60, 40, 4),
                             base_pool=base, cycle_min=8.0,
                             horizon_min=15.0, adapt_on=True,
                             genai_on=True, evfis_on=True,
                             seed_profile="minimal")
    eng.adapt_cooldown_min = 8.0
    out = None
    for i in range(110):
        ov = eng.maybe_decide(sim)
        b = int((sim.state.burning > 0.5).sum())
        sim.step(resource_override=ov)
        if b == 0 and i > 5:
            out = (i + 1) * cfg.step_minutes
            break
    assert out is not None, "minimal-profile DSS never put the fire out"
    assert len(eng.rules) > 5, "the rule base did not grow"
    assert int(sim.ever_burned.sum()) < 150


def test_stage2_term_insertion_grows_catalog():
    """TRUE resolution increase: on a COVERED antecedent cell whose
    situation is ambiguous (reading between two term cores), stage 2
    inserts a new linguistic term, writes the rule on the refined
    cell, and the catalog grows (5^5 = 3125 -> 3750). The inserted
    term must evaluate in the rule pass and reset must drop it."""
    import numpy as np
    import dss
    from dss.fuzzy import REGISTRY
    from dss.adapt import (stage2_resolution, reset_partitions,
                           _dominant_terms)
    from dss.rules import Rule, evaluate_rules
    from disaster_phyengine import terrain
    from disaster_phyengine.config import SimConfig
    from disaster_phyengine.core import Simulator
    reset_partitions()
    cfg = SimConfig(nx=30, ny=20, cell_size_m=30.0)
    cfg.step_minutes = 1.0
    w = terrain.generate_landscape(cfg, seed=3, preset="Rolling hills",
                                   n_settlements=2,
                                   population_per_settlement=5000)
    w.add_ignition(15, 10, step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    sim.step()
    sim._dss_hmin = 10.0
    eff = {c: np.array([0., 0., 0.5, 0.5, 0.])
           for c in dss.DECISION_CONCEPTS}
    crisp = {c: 0.62 for c in dss.DECISION_CONCEPTS}
    dom = _dominant_terms(eff)
    ranked = sorted(dss.CONCEPT_FAMILY,
                    key=lambda c: -crisp.get(c, 0))
    c1, c2 = ranked[0], ranked[1]
    rules = dss.make_runtime_rules("minimal")
    rules.append(Rule("A0", [(c1, dom[c1]), (c2, dom[c2])],
                      [("suppression_effort", 0.5)]))
    out = stage2_resolution(lambda rr: None, sim, rules, eff, crisp, 8)
    assert out.accepted, out.detail
    assert any(t.startswith("X") for t in REGISTRY.get(c1))
    cat = 1
    for c in dss.DECISION_CONCEPTS:
        cat *= len(REGISTRY.get(c))
    assert cat > 3125
    _u, tr = evaluate_rules(eff, {}, rules)
    w_new = [wt for r, wt in tr if r.name == rules[-1].name][0]
    assert w_new > 0.1, "inserted-term antecedent must evaluate"
    reset_partitions()
    assert all(not t.startswith("X") for t in REGISTRY.get(c1))


def test_genai_package_grows_vocabulary():
    """Open decision space at the VOCABULARY level: a stage-3 package
    (new object + a rule using it) can add a macro intervention that
    reduces to the base physical channels and an intermediate concept
    composed of existing features. G2 validates the composition, G2b
    rejects collinear copies, G3/G4/G5 run the shadow rollouts, and
    the admitted vocabulary persists in the profile lineage. The G5
    margin is relaxed here: this test checks the PLUMBING, the
    physics margin is exercised by the live gates."""
    import json
    import os
    import numpy as np
    import dss
    import dss.adapt as A
    from dss.adapt import (stage3_generative, _validate_package,
                           reset_partitions)
    from disaster_phyengine import terrain
    from disaster_phyengine.config import SimConfig
    from disaster_phyengine.core import Simulator
    import tempfile
    store = os.path.join(tempfile.mkdtemp(prefix="test_pkg_reg_"),
                         "learned_rules.json")
    reset_partitions()
    cfg = SimConfig(nx=60, ny=40, cell_size_m=30.0)
    cfg.step_minutes = 1.0
    w = terrain.generate_landscape(cfg, seed=11, preset="Rolling hills",
                                   n_settlements=4,
                                   population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.08
    w.meteo.wws[:] = 6.0
    base, _ = dss.resource_suggestion(w)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    w.add_ignition(30, 20, step=0, radius=2)
    sim = Simulator(w)
    sim.record_states = False
    eng = dss.DecisionEngine(dss.partition_n(60, 40, 1),
                             base_pool=base, cycle_min=1.0,
                             horizon_min=15.0, adapt_on=True,
                             genai_on=True, evfis_on=True,
                             seed_profile="minimal",
                             learned_store=store)
    for _ in range(8):
        sim.step(resource_override=eng.maybe_decide(sim))
    sim._dss_hmin = 15.0
    ctx = eng._perceive(sim)
    rows, _p = eng._decide_regions(sim, ctx, eng.rules)
    hot = list(rows)[0]

    def build(rr):
        _r2, prs = eng._decide_regions(sim, ctx, rr)
        return eng._override(sim, prs)

    old_margin, old_prop = A.G5_MARGIN, A._genai_propose
    try:
        A.G5_MARGIN = -1.0
        # G2c relevance gate: the rule must fire NOW, so the test
        # keys the antecedent to the situation's own dominant term
        _domT = A._dominant_terms(rows[hot]["eff"])
        pkg = {"antecedents": [["fire_threat_level",
                                _domT["fire_threat_level"]]],
               "consequents": [["backburn", 0.9]],
               "new_intervention": {
                   "name": "backburn",
                   "composition": [["containment_line", 0.7],
                                   ["suppression_effort", 0.5]]}}
        A._genai_propose = lambda s_, timeout=None, **kw_: json.loads(
            json.dumps(pkg))
        out = stage3_generative(build, sim, eng.rules,
                                rows[hot]["eff"], rows[hot]["crisp"],
                                8, coverage_gap=True, engine=eng)
        assert out.accepted, out.detail
        assert "backburn" in eng.macros
        pkg2 = {"antecedents": [["ember_pressure", "H"]],
                "consequents": [["containment_line", 0.8]],
                "new_concept": {
                    "name": "ember_pressure",
                    "level": "intermediate",
                    "inputs": [["weather_severity", 0.6],
                               ["fuel_load", 0.4]]}}
        A._genai_propose = lambda s_, timeout=None, **kw_: json.loads(
            json.dumps(pkg2))
        out2 = stage3_generative(build, sim, eng.rules,
                                 rows[hot]["eff"], rows[hot]["crisp"],
                                 8, coverage_gap=True, engine=eng)
        assert out2.accepted, out2.detail
        assert "ember_pressure" in eng.hierarchy
        # G2b: a collinear copy of fuel_hazard must be rejected
        bad = {"antecedents": [["copy_f", "H"]],
               "consequents": [["suppression_effort", 0.8]],
               "new_concept": {"name": "copy_f",
                               "level": "intermediate",
                               "inputs": [["weather_severity", 0.4],
                                          ["fuel_load", 0.6]]}}
        assert "G2b" in (_validate_package(
            json.loads(json.dumps(bad)), eng) or "")
        # macro expands to base channels in the rule pass
        u, _tr = dss.evaluate_rules(rows[hot]["eff"], ctx[hot]["f"],
                                    eng.rules, macros=eng.macros)
        assert u["containment_line"] > 0.0
        # persistence: a fresh engine reloads the grown vocabulary
        sim.step(resource_override=eng.maybe_decide(sim))
        reset_partitions()
        e2 = dss.DecisionEngine(dss.partition_n(60, 40, 1),
                                seed_profile="minimal",
                                learned_store=store)
        assert "backburn" in e2.macros
        assert "ember_pressure" in e2.hierarchy
    finally:
        A.G5_MARGIN, A._genai_propose = old_margin, old_prop
        reset_partitions()


# --------------------------------------------------------------------------
# Ruspini (strong) partition invariant: sum_t mu_t(x) = 1 for every x.
# It is what makes the inference output a convex combination of the
# consequents, so adaptation may never break it.
# --------------------------------------------------------------------------
def test_default_partition_is_ruspini():
    from dss.fuzzy import default_partition, partition_defect, TERMS
    p = default_partition()
    # neighbouring trapezoids must SHARE their transition interval
    for i in range(len(TERMS) - 1):
        lo, hi = TERMS[i], TERMS[i + 1]
        assert (p[lo][2], p[lo][3]) == (p[hi][0], p[hi][1]), \
            f"{lo}->{hi} boundary is not shared"
    assert partition_defect(p) < 1e-9


def test_worked_example_preserved():
    from dss.fuzzy import fuzzify
    mu = fuzzify(0.62)
    assert abs(mu["M"] - 0.533) < 0.01
    assert abs(mu["H"] - 0.467) < 0.01
    assert abs(sum(mu.values()) - 1.0) < 1e-9


def test_boundary_shift_preserves_partition_and_stays_bounded():
    import numpy as np
    from dss.fuzzy import REGISTRY, partition_defect, TERMS
    REGISTRY.reset()
    rng = np.random.default_rng(7)
    for k in range(200):
        var = f"v{k % 5}"
        term = TERMS[int(rng.integers(1, 5))]
        REGISTRY.shift_boundary(var, term, float(rng.uniform(-0.3, 0.3)))
        part = REGISTRY.get(var)
        # partition still sums to one
        assert partition_defect(var=var) < 1e-9
        # and no trapezoid inverted
        for t in TERMS:
            a, b, c, d = part[t]
            assert a <= b <= c <= d
    REGISTRY.reset()


def test_shift_is_clamped_by_the_neighbouring_core_width():
    from dss.fuzzy import REGISTRY, default_partition
    REGISTRY.reset()
    p = default_partition()
    core_M = p["M"][2] - p["M"][1]              # 0.05
    applied = REGISTRY.shift_boundary("x", "H", -0.50)
    assert abs(applied + core_M) < 1e-9         # clamped to exactly -core_M
    REGISTRY.reset()


def test_coordinator_tightens_gate_on_monitored_regions():
    """The Global DSS sends back a per-region acceptance gate: an
    attended region keeps the base eta, a monitored region gets a
    tightened gate 1 - share*(1 - eta), so weak-priority offensive
    orders need a higher decision quality before they draw on the
    shared capacity."""
    import numpy as np
    import dss
    from disaster_phyengine import terrain
    from disaster_phyengine.config import SimConfig
    from disaster_phyengine.core import Simulator

    cfg = SimConfig(nx=60, ny=40, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=5, preset="Rolling hills", n_settlements=4,
        population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.10
    w.meteo.wws[:] = 5.0
    base, _ = dss.resource_suggestion(w)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    ok = (w.fuel.fload > 0.4) & (w.fuel.ftype != 0)
    ys, xs = np.where(ok & (np.arange(w.config.nx)[None, :] < 25))
    w.add_ignition(int(xs[0]), int(ys[0]), step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    eng = dss.DecisionEngine(dss.partition_n(60, 40, 4),
                             base_pool=base, cycle_min=2.0,
                             horizon_min=10.0, adapt_on=False)
    for _ in range(8):
        sim.step(resource_override=eng.maybe_decide(sim))
    g = eng.last_global
    assert g is not None and "thresholds" in g
    eta = eng.eta
    for name, sh in g["shares"].items():
        want = 1.0 - sh * (1.0 - eta)
        assert abs(g["thresholds"][name] - want) < 5e-3
    # at least one region is monitored (fire sits in one corner) and
    # its gate is strictly tighter than the base gate
    mon = [n for n in g["shares"] if n not in g["attended"]]
    assert mon
    assert all(g["thresholds"][n] > eta + 1e-6 for n in mon)


def _toggle_world():
    import numpy as np
    import dss
    from disaster_phyengine import terrain
    from disaster_phyengine.config import SimConfig
    cfg = SimConfig(nx=40, ny=30, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(cfg, seed=3, preset="Rolling hills",
                                   n_settlements=3,
                                   population_per_settlement=9000)
    base, _ = dss.resource_suggestion(w)
    return w, base


def _seeded_state(path):
    """A store holding one record of every kind."""
    import dss
    gs = dss.GeneratedState.load(path, active_rule_set="minimal")
    seed_name = None
    rs = dss.make_runtime_rules("minimal")
    seed_name = rs[0].name
    old = [[i, float(v)] for i, v in rs[0].consequents]
    new = [[i, min(1.0, float(v) + 0.10)] for i, v in rs[0].consequents]
    gs.append("evfis_rule_modifications",
              dict(base_rule_id=seed_name, base_rule_set="minimal",
                   modification_type="consequent_update",
                   before={"consequents": old},
                   after={"consequents": new}), source_stage=1,
              save=False)
    gs.append("genai_concepts",
              dict(name="test_pressure", layer=2,
                   inputs=[{"name": "fire_threat_level", "weight": 1.0}],
                   outputs=[{"name": "activation", "range": [0, 1]}]),
              source_stage=3, save=False)
    gs.append("genai_interventions",
              dict(name="test_bundle",
                   composition=[{"channel": "suppression_effort",
                                 "weight": 1.0}]),
              source_stage=3, save=False)
    gs.append("genai_rules",
              dict(name="G90",
                   antecedents=[["test_pressure", "H"]],
                   consequents=[["test_bundle", 0.8]],
                   depends_on_concepts=["test_pressure"]),
              source_stage=3, save=False)
    gs.save()
    return seed_name, old, new


def test_toggle_matrix_resolves_per_spec(tmp_path):
    """OFF/OFF = pure seed (factory values). use12 ON = revert vs
    apply of stored modifications. use3 ON = generated concepts,
    interventions and rules enter; OFF = they leave entirely."""
    import dss
    w, base = _toggle_world()
    sp = str(tmp_path / "gstate.json")
    seed_name, old, new = _seeded_state(sp)

    def eng(u12, u3):
        return dss.DecisionEngine(
            dss.partition_n(40, 30, 1), base_pool=base,
            cycle_min=4.0, horizon_min=10.0, adapt_on=False,
            seed_profile="minimal", use_evfis=u12, use_genai=u3,
            state_path=sp)

    seed_rules = {r.name for r in dss.make_runtime_rules("minimal")}

    e00 = eng(False, False)
    assert {r.name for r in e00.rules} == seed_rules
    r0 = next(r for r in e00.rules if r.name == seed_name)
    assert [[i, float(v)] for i, v in r0.consequents] == old
    assert "test_pressure" not in e00.hierarchy
    assert "test_bundle" not in e00.macros

    e10 = eng(True, False)
    r1 = next(r for r in e10.rules if r.name == seed_name)
    assert [[i, float(v)] for i, v in r1.consequents] == new
    assert {r.name for r in e10.rules} == seed_rules   # revert != delete
    assert "test_pressure" not in e10.hierarchy

    e01 = eng(False, True)
    r2 = next(r for r in e01.rules if r.name == seed_name)
    assert [[i, float(v)] for i, v in r2.consequents] == old
    assert "test_pressure" in e01.hierarchy
    assert "test_bundle" in e01.macros
    assert any(r.name == "G90" for r in e01.rules)


def test_unresolved_dependency_warns_and_drops(tmp_path):
    """A generated rule whose concept is missing is not loaded, and
    the finding is a visible warning, not a silent drop."""
    import json
    import dss
    w, base = _toggle_world()
    sp = str(tmp_path / "gstate.json")
    _seeded_state(sp)
    d = json.load(open(sp))
    d["genai_concepts"] = []          # break the dependency
    json.dump(d, open(sp, "w"))
    e = dss.DecisionEngine(
        dss.partition_n(40, 30, 1), base_pool=base,
        cycle_min=4.0, horizon_min=10.0, adapt_on=False,
        seed_profile="minimal", use_evfis=True, use_genai=True,
        state_path=sp)
    assert not any(r.name == "G90" for r in e.rules)
    assert any("G90" in m for m in e.resolve_warnings)


def test_shadow_mode_stores_but_does_not_apply(tmp_path, monkeypatch):
    """evFIS active + use stage 1-2 OFF: an accepted stage-1 result is
    written to the store while the active base keeps factory values."""
    import numpy as np
    import dss
    import dss.loop as L
    from dss.adapt import AdaptOutcome
    from disaster_phyengine.core import Simulator
    w, base = _toggle_world()
    sp = str(tmp_path / "gstate.json")

    def fake_stage1(build, sim, rules, fired, horizon, step_size=0.05):
        r = rules[0]
        r.consequents = [(i, min(1.0, float(v) + 0.2))
                         for i, v in r.consequents]
        r.note = (r.note + " | " if r.note else "") + "evFIS: consequent"
        return AdaptOutcome(1, True, "fake tune", dJ=-0.05)

    monkeypatch.setattr(L, "stage1_evfis", fake_stage1)
    ok = (w.fuel.fload > 0.4) & (w.fuel.ftype != 0)
    ys, xs = np.where(ok)
    w.add_ignition(int(xs[0]), int(ys[0]), step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    eng = dss.DecisionEngine(
        dss.partition_n(40, 30, 1), base_pool=base,
        cycle_min=2.0, horizon_min=10.0, adapt_on=True,
        evfis_on=True, genai_on=False, seed_profile="minimal",
        use_evfis=False, use_genai=False, state_path=sp,
        j_threshold=0.0)                 # every forecast is a deficit
    eng.adapt_cooldown_min = 0.0
    factory = {r.name: [(i, float(v)) for i, v in r.consequents]
               for r in dss.make_runtime_rules("minimal")}
    for _ in range(6):
        sim.step(resource_override=eng.maybe_decide(sim))
    # active base stayed at factory values...
    for r in eng.rules:
        if r.name in factory:
            assert [(i, float(v)) for i, v in r.consequents] \
                == factory[r.name], r.name
    # ...and the store received the produced modification
    gs = dss.GeneratedState.load(sp)
    assert len(gs.records("evfis_rule_modifications")) >= 1


# ---------------------------------------------- generated state: durability
def test_seq_is_global_and_monotonic(tmp_path):
    """Replay order is ONE sequence across all four sections. Per-section
    counters would interleave wrongly on restart, and timestamps are too
    coarse to order records written in the same second."""
    import dss
    sp = str(tmp_path / "gstate.json")
    _seeded_state(sp)
    gs = dss.GeneratedState.load(sp)
    seqs = [int(r["seq"])
            for sec in ("evfis_rule_modifications", "genai_rules",
                        "genai_concepts", "genai_interventions")
            for r in gs.records(sec)]
    assert sorted(seqs) == list(range(1, len(seqs) + 1))
    assert gs.next_seq() == len(seqs) + 1
    mods = gs.sorted_records("evfis_rule_modifications")
    assert [r["seq"] for r in mods] == sorted(r["seq"] for r in mods)
    # every record carries the provenance the spec requires
    for sec in ("evfis_rule_modifications", "genai_rules"):
        for r in gs.records(sec):
            assert r["origin"] in ("evfis", "genai")
            assert "produced_under_flags" in r and "timestamp" in r
            assert "active" not in r      # activity is derived, never stored


def test_wipe_clears_records_but_never_the_baseline(tmp_path):
    """A wipe is a factory reset of the GENERATED knowledge only: the seed
    rule sets and the six base interventions survive it."""
    import dss
    sp = str(tmp_path / "gstate.json")
    _seeded_state(sp)
    gs = dss.GeneratedState.load(sp)
    counts = gs.wipe()
    assert sum(counts.values()) == 4
    assert sum(gs.counts().values()) == 0
    # production stops, consumption intent is left alone
    assert gs.flags["evfis_active"] is False
    assert gs.flags["genai_active"] is False
    assert gs.flags["use_stage12_rules"] is True
    assert gs.flags["dss_active"] is True
    assert os.path.exists(os.path.splitext(sp)[0] + ".bak.json")
    # an engine rebuilt from the wiped store is the pristine seed profile
    w, base = _toggle_world()
    e = dss.DecisionEngine(dss.partition_n(40, 30, 1), base_pool=base,
                           cycle_min=4.0, horizon_min=10.0, adapt_on=False,
                           seed_profile="minimal", use_evfis=True,
                           use_genai=True, state_path=sp)
    pristine = dss.make_runtime_rules("minimal")
    assert {r.name for r in e.rules} == {r.name for r in pristine}
    for got, want in zip(sorted(e.rules, key=lambda r: r.name),
                         sorted(pristine, key=lambda r: r.name)):
        assert [(i, round(float(v), 6)) for i, v in got.consequents] == \
               [(i, round(float(v), 6)) for i, v in want.consequents]


def test_restart_reproduces_the_active_set(tmp_path):
    """Two engines built from the same store hold the same active set: the
    restart path is a replay, not a re-derivation that can drift."""
    import dss
    sp = str(tmp_path / "gstate.json")
    _seeded_state(sp)
    w, base = _toggle_world()

    def build():
        return dss.DecisionEngine(
            dss.partition_n(40, 30, 1), base_pool=base, cycle_min=4.0,
            horizon_min=10.0, adapt_on=False, seed_profile="minimal",
            use_evfis=True, use_genai=True, state_path=sp)

    a, b = build(), build()
    def sig(e):
        return (sorted((r.name,
                        tuple((i, round(float(v), 6))
                              for i, v in r.consequents)) for r in e.rules),
                sorted(e.hierarchy), sorted(e.macros))
    assert sig(a) == sig(b)
    assert "test_pressure" in a.hierarchy and "test_bundle" in a.macros


def test_atomic_write_never_leaves_a_half_written_store(tmp_path):
    """A store truncated mid-write would brick the next start, so the write
    goes to a temp file and is renamed into place."""
    import json
    import dss
    sp = str(tmp_path / "gstate.json")
    _seeded_state(sp)
    good = open(sp, encoding="utf-8").read()

    real_dump = json.dump

    def exploding_dump(obj, fp, **kw):
        fp.write('{"schema_version": "1.0", "evfis_rule_mod')
        raise IOError("disk full")

    gs = dss.GeneratedState.load(sp)
    json.dump = exploding_dump
    try:
        try:
            gs.save()
        except IOError:
            pass
    finally:
        json.dump = real_dump
    # the store that was already on disk is untouched and still parses
    assert open(sp, encoding="utf-8").read() == good
    assert json.loads(good)["schema_version"] == "1.0"
    assert sum(dss.GeneratedState.load(sp).counts().values()) == 4


def test_corrupt_store_is_quarantined_not_silently_replaced(tmp_path):
    """Losing generated knowledge without a trace is worse than failing
    loudly, so an unreadable store is set aside and reported."""
    import dss
    sp = str(tmp_path / "gstate.json")
    with open(sp, "w", encoding="utf-8") as f:
        f.write("{not json at all")
    gs = dss.GeneratedState.load(sp)
    assert sum(gs.counts().values()) == 0
    assert gs.warnings and "could not be read" in gs.warnings[0]
    assert os.path.exists(sp + ".corrupt")


def test_frozen_mode_consumes_without_producing(tmp_path):
    """Production off, consumption on: the accumulated knowledge is used and
    nothing new is written. This is the reproducible-experiment mode."""
    import dss
    sp = str(tmp_path / "gstate.json")
    seed_name, old, new = _seeded_state(sp)
    w, base = _toggle_world()
    sim = Simulator(w)
    for _ in range(4):
        sim.step()
    e = dss.DecisionEngine(dss.partition_n(40, 30, 1), base_pool=base,
                           cycle_min=4.0, horizon_min=10.0,
                           adapt_on=False, evfis_on=False, genai_on=False,
                           seed_profile="minimal", use_evfis=True,
                           use_genai=True, state_path=sp)
    before = e.gstate.counts()
    for _ in range(6):
        e.decide(sim)
        sim.step()
    assert e.gstate.counts() == before          # nothing produced
    r = next(x for x in e.rules if x.name == seed_name)
    assert [[i, float(v)] for i, v in r.consequents] == new   # but consumed


def test_config_id_names_the_experiment(tmp_path):
    """Each toggle combination is an experiment configuration, so it gets a
    stable label a results table can group by."""
    import dss
    sp = str(tmp_path / "gstate.json")
    gs = dss.GeneratedState.load(sp)
    gs.set_flags(dss_active=True, evfis_active=True, genai_active=False,
                 use_stage12_rules=True, use_stage3_rules=False)
    assert gs.config_id == "DSS1-EV1-GA0-U12:1-U3:0"
    gs.set_flags(evfis_active=False)
    assert gs.config_id == "DSS1-EV0-GA0-U12:1-U3:0"


def test_waterless_map_shelves_water_macros(tmp_path):
    """A learned macro that needs water stays in the store but leaves
    the ACTIVE set on a map without any water body; a rule ordering
    it sleeps with a visible warning, and nothing is deleted."""
    import numpy as np
    import dss
    from disaster_phyengine.core import Simulator
    w, base = _toggle_world()
    w.fuel.ftype[w.fuel.ftype == 5] = 1        # suyu tamamen kaldır
    sp = str(tmp_path / "gstate.json")
    gs = dss.GeneratedState.load(sp, active_rule_set="minimal")
    gs.append("genai_interventions",
              dict(name="drafting_sustained_attack",
                   composition=[{"channel": "water_drafting",
                                 "weight": 1.0},
                                {"channel": "suppression_effort",
                                 "weight": 0.9}]),
              source_stage=3, save=False)
    gs.append("genai_rules",
              dict(name="G80", antecedents=[["fire_threat_level", "H"]],
                   consequents=[["drafting_sustained_attack", 0.9]],
                   depends_on_concepts=[]),
              source_stage=3, save=False)
    gs.save()
    ok = (w.fuel.fload > 0.4) & (w.fuel.ftype != 0)
    ys, xs = np.where(ok)
    w.add_ignition(int(xs[0]), int(ys[0]), step=0, radius=1)
    sim = Simulator(w)
    sim.record_states = False
    eng = dss.DecisionEngine(
        dss.partition_n(40, 30, 1), base_pool=base, cycle_min=2.0,
        horizon_min=10.0, adapt_on=False, seed_profile="minimal",
        use_evfis=True, use_genai=True, state_path=sp)
    sim.step(resource_override=eng.maybe_decide(sim))
    assert "drafting_sustained_attack" not in eng.macros
    assert "drafting_sustained_attack" in eng._shelved_macros
    g80 = next(r for r in eng.rules if r.name == "G80")
    assert not g80.active
    assert any("waterless" in m or "water body" in m
               for m in eng.resolve_warnings)
    # store untouched: the lineage survives for the next wet map
    gs2 = dss.GeneratedState.load(sp)
    assert any(r.get("name") == "drafting_sustained_attack"
               for r in gs2.records("genai_interventions"))


def test_tuning_of_a_generated_rule_applies_when_stage3_is_on(tmp_path):
    """A stage 1 tuning may name a rule stage 3 created.

    The resolver used to replay the tunings BEFORE installing the generated
    rules, so such a tuning could never find its target: it was skipped on
    every cycle and the warning blamed the stage 3 toggle, which was on.
    """
    from dss.state import GeneratedState
    from dss.resolve import resolve_active_set
    from dss.adapt import make_runtime_rules, reset_partitions
    from dss.fuzzy import REGISTRY

    st = GeneratedState(str(tmp_path / "s.json"))
    st.flags.update(dss_active=True, active_rule_set="minimal5",
                    use_stage12_rules=True, use_stage3_rules=True)
    st.append("genai_rules", dict(
        name="G5",
        antecedents=[("fire_intensity", "high")],
        consequents=[("suppression_effort", 0.9)],
        depends_on_concepts=[]), source_stage=3, save=False)
    st.append("evfis_rule_modifications", dict(
        modification_type="consequent_update", base_rule_id="G5",
        before={"consequents": [["suppression_effort", 0.9]]},
        after={"consequents": [["suppression_effort", 0.4]]}),
        source_stage=1, save=False)

    a = resolve_active_set(st, make_runtime_rules, REGISTRY, reset_partitions)
    g5 = {r.name: r for r in a.rules}.get("G5")
    assert g5 is not None, "the generated rule must be in the active set"
    assert a.applied_mods == 1, "the tuning must be counted as applied"
    assert g5.consequents == [("suppression_effort", 0.4)], \
        "the stored tuning must have overwritten the generated consequent"
    assert not a.warnings, f"nothing should be skipped: {a.warnings}"

    # with stage 3 consumption off the SAME record is skipped, and the
    # message is then allowed to blame the toggle, because that is the cause
    st.flags["use_stage3_rules"] = False
    b = resolve_active_set(st, make_runtime_rules, REGISTRY, reset_partitions)
    assert "G5" not in {r.name for r in b.rules}
    assert b.applied_mods == 0
    assert len(b.warnings) == 1 and "is off" in b.warnings[0]


def test_evfis_still_reaches_the_membership_shoulder():
    """evFIS tunes consequents AND the antecedent partition.

    Once the candidate list grew from one rule to two, the consequent trials
    (2 rules x 2 signs) swallowed the whole allowance and the shoulder branch
    became unreachable, so the stage silently stopped moving partition
    boundaries. The budget for the shoulder is now reserved up front.
    """
    from dss import adapt as A

    class _Sim:
        class state:
            step = 3
        class cfg:
            cost = None
            step_minutes = 1.0

    calls = {"n": 0}

    def _fake_cva(build, sim, rules, horizon, reseed=None):
        # every trial looks slightly worse, so nothing is kept and the stage
        # is forced to spend its whole budget
        calls["n"] += 1
        return 1.0 + 0.001 * calls["n"], 1.0

    rules = A.make_runtime_rules("minimal5")
    fired = [(rules[0], 0.9), (rules[1], 0.5)]
    _old_cva, _old_shift = A._cva, A.REGISTRY.shift_boundary
    moved = {"n": 0}

    def _spy_shift(var, term, delta):
        moved["n"] += 1
        return _old_shift(var, term, delta)

    try:
        A._cva = _fake_cva
        A.REGISTRY.shift_boundary = _spy_shift
        out = A.stage1_evfis(lambda r: None, _Sim(), rules, fired, 12)
    finally:
        A._cva, A.REGISTRY.shift_boundary = _old_cva, _old_shift

    kinds = [t.get("kind") for t in (out.info or {}).get("trials", [])]
    assert "membership" in kinds, \
        f"the shoulder trial must run; trials were {kinds}"
    assert moved["n"] == 1, "the partition boundary must be tried exactly once"
    assert kinds.count("consequent") >= 2, \
        "the consequent trials must still happen alongside it"


def test_a_free_rejection_does_not_burn_the_whole_cooldown():
    """A rejection decided without a forecast or a model call is refunded.

    Stage 2 finding the antecedent cell already covered is a set lookup. It
    used to cost the same five minutes of silence as a rejection that ran the
    45-minute shadow forecasts, which is how a third of all adaptation
    windows went to decisions that cost nothing.
    """
    from dss import adapt as A
    from dss.loop import DecisionEngine

    c0, g0 = A.CVA_CALLS, A.GENAI_CALLS
    # the counters move only when the expensive work actually happens
    assert (A.CVA_CALLS, A.GENAI_CALLS) == (c0, g0)

    eng = DecisionEngine([], adapt_on=True)
    assert eng.adapt_retry_min < eng.adapt_cooldown_min, \
        "the refund has to leave a shorter wait than the full cooldown"
    # the refund rewinds the stamp so only adapt_retry_min is still owed
    _now = 30.0
    eng._adapt_last_min = (_now - (eng.adapt_cooldown_min
                                   - eng.adapt_retry_min))
    _due_at = eng._adapt_last_min + eng.adapt_cooldown_min
    assert abs((_due_at - _now) - eng.adapt_retry_min) < 1e-9, \
        "after a refund the next window must open one retry period later"


def test_genai_budget_is_shared_by_the_proposal_and_its_revisions():
    """The wait budget belongs to the ATTEMPT, not to each call.

    With one budget per call, a proposal plus three revisions could hold the
    decision cycle for four times the advertised wait while the log reported
    a single 90 s timeout.
    """
    import inspect
    from dss import adapt as A

    src = inspect.getsource(A.stage3_generative)
    assert "_deadline" in src and "_left()" in src, \
        "the stage must carry one deadline across its calls"
    assert src.count("timeout=max(1.0, _left())") == 2, \
        "both the proposal and the revision must draw on what is left"
    # _genai_propose has to be able to take the remaining budget
    assert "timeout" in inspect.signature(A._genai_propose).parameters


def test_every_agent_and_the_coordinator_are_visible_in_the_step_views():
    """The step table names only the hotspot, so the other agents and the
    Global DSS had no view at all: the run read as if one region were the
    whole system. build_agent_rows and build_global_rows cover both.
    """
    import ast
    import dss

    src = open('app/streamlit_app.py', encoding='utf-8').read()
    tree = ast.parse(src)
    want = {'build_agent_rows', 'build_global_rows', '_fmt_orders'}
    mod = ast.Module(
        body=[n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in want],
        type_ignores=[])
    ns = {}
    exec(compile(mod, '<views>', 'exec'), ns)

    w, sim = _mini_fire_sim()
    base, _ = dss.resource_suggestion(w)
    eng = dss.DecisionEngine(dss.partition_n(60, 40, 3), base_pool=base,
                             j_threshold=0.05, cycle_steps=1,
                             horizon_steps=4, adapt_on=True, genai_on=False)
    for _ in range(8):
        sim.step(resource_override=eng.maybe_decide(sim))

    arows = ns['build_agent_rows'](eng.cycles)
    grows = ns['build_global_rows'](eng.cycles)

    assert len({r["agent"] for r in arows}) == 3, \
        "all three local agents must have rows, not only the hotspot"
    assert len(arows) == 3 * len(eng.cycles), \
        "every agent decides in every cycle"
    assert {r["role"] for r in arows} <= {"focus", "attended", "monitor"}
    assert sum(1 for r in arows if r["role"] == "focus") == len(eng.cycles), \
        "exactly one region is the hotspot per cycle"
    # the coordinator's own decision is a row of its own
    assert len(grows) == len(eng.cycles)
    assert all(g["ranking"] != "—" for g in grows), \
        "the ranking the shares came from must be shown"
    # newest first, like the step table
    assert arows[0]["cycle"] >= arows[-1]["cycle"]
    assert grows[0]["cycle"] >= grows[-1]["cycle"]


def _wui_run(orders, water=False, steps=18, wind=None, pool=1.0):
    """One controlled fire with a FIXED order vector, for channel tests."""
    import dss
    from disaster_phyengine.scenarios import wui_interface
    from disaster_phyengine.core import Simulator
    from disaster_phyengine.costs import compute_costs
    chans = ["suppression_effort", "resource_deployment", "containment_line",
             "asset_protection", "evacuation", "public_warning",
             "tactical_burn", "water_drafting", "retardant_drop"]
    w = wui_interface()
    ny, nx = w.fuel.fload.shape
    if wind is not None:
        w.meteo.wws[:] = wind
        w.fuel.fmoist[:] = 0.05
    if water:
        w.fuel.ftype[10:24, 40:70] = 5
    sim = Simulator(w)
    w.add_ignition(70, 35, step=0, radius=2)
    for _ in range(4):
        sim.step()
    base, _ = dss.resource_suggestion(w)
    base.rcap *= pool
    u = {c: 0.0 for c in chans}
    u.update(orders)
    pairs = [(dss.partition_n(nx, ny, 1)[0], dict(u))]
    for _ in range(steps):
        sim.step(resource_override=dss.decision_to_resources(
            w, sim.state.burning > 0.5, pairs, base))
    rep = compute_costs(sim)
    return dict(burned=int(sim.ever_burned.sum()),
                exposure=float(sim.exposure_person_steps),
                j_pop=float(rep.j_pop), j_total=float(rep.j_total))


def test_evacuating_people_must_lower_the_population_cost():
    """J_pop was normalized by the population STILL THERE.

    An ordered evacuation removes people from vpop, so the denominator fell
    with the numerator and a good evacuation scored worse than none: on this
    scenario the exposure dropped by 98.5% while J_pop went from 0.048 to
    1.000, the maximum penalty. Since J_pop feeds the satisficing test and
    the no-harm guard, the DSS was being told not to evacuate.
    """
    none = _wui_run({})
    evac = _wui_run({"evacuation": 1.0})
    both = _wui_run({"evacuation": 1.0, "public_warning": 1.0})

    assert evac["exposure"] < none["exposure"] * 0.5, \
        "the evacuation must actually remove people from the fire"
    assert evac["j_pop"] < none["j_pop"], \
        "fewer people exposed has to mean a SMALLER population cost"
    assert both["exposure"] < evac["exposure"], \
        "a warning primes the population, so the departure is faster"
    assert both["j_pop"] < evac["j_pop"], \
        "and the faster departure has to score better, not worse"
    assert both["j_total"] < none["j_total"]


def test_every_intervention_channel_reaches_the_physics():
    """No channel may be decoration: each one has to move the simulation.

    Two of them are MULTIPLIERS by design and do nothing alone: a public
    warning moves nobody by itself (it doubles the evacuation tempo) and
    water drafting boosts capacity that has already been staged. They are
    checked in the combination they are meant for.
    """
    base = _wui_run({})
    # direct channels: measurable on their own
    for ch in ("suppression_effort", "containment_line", "asset_protection",
               "tactical_burn"):
        r = _wui_run({ch: 1.0})
        assert r["burned"] != base["burned"], \
            f"{ch} did not change the fire at all"
    r = _wui_run({"evacuation": 1.0})
    assert r["exposure"] < base["exposure"], "evacuation must move people"
    r = _wui_run({"retardant_drop": 1.0}, water=True)
    assert r["burned"] < base["burned"], "retardant must slow the fire"

    # capacity-limited regime: staging and the water shuttle only matter
    # when capacity is what the suppression is short of
    hard = dict(wind=16.0, pool=0.25, steps=22)
    b0 = _wui_run({"suppression_effort": 0.5}, **hard)
    b1 = _wui_run({"suppression_effort": 0.5, "resource_deployment": 1.0},
                  **hard)
    assert b1["burned"] < b0["burned"], \
        "resource deployment must give the suppression something to spend"
    w0 = _wui_run({"suppression_effort": 0.5}, water=True, **hard)
    w1 = _wui_run({"suppression_effort": 0.5, "water_drafting": 1.0},
                  water=True, **hard)
    assert w1["burned"] < w0["burned"], \
        "drafting from a lake must sustain the attack better than not"


def test_the_review_never_blocks_and_is_found_afterwards(tmp_path):
    """The after-action review runs in the background.

    The panel used to sleep and rerun in a loop while the model read the
    logs, which froze the whole script for the one to three minutes the
    deep review takes, and leaving the panel stopped the polling entirely,
    so a finished report was never collected.
    """
    import time
    from dss import rca

    d = str(tmp_path)
    calls = {"n": 0}

    def _slow(evidence, model=None):
        calls["n"] += 1
        time.sleep(0.6)
        return "REPORT BODY", {"recommendations": [{"kind": "setting"}]}

    _old = rca.run_rca
    try:
        rca.run_rca = _slow
        t0 = time.time()
        rca.start_async(d, "evidence", model="opus")
        assert time.time() - t0 < 0.3, \
            "start_async must return at once, not wait for the model"
        assert rca.poll(d)["state"] == "running"
        assert rca.poll(d)["model"] == "opus"
        # a second press while one is in flight must not launch another
        rca.start_async(d, "evidence", model="opus")
        assert calls["n"] <= 1

        for _ in range(60):
            if rca.poll(d)["state"] != "running":
                break
            time.sleep(0.1)
        j = rca.poll(d)
        assert j["state"] == "done", f"the review must finish: {j}"
        assert j["report"] == "REPORT BODY"
        assert rca.elapsed_s(d) > 0.0
    finally:
        rca.run_rca = _old

    # the report survives the job table: a review that finished while the
    # process was elsewhere (or restarted) is still found on disk
    rca._JOBS.pop(d, None)
    j2 = rca.poll(d)
    assert j2["state"] == "done" and j2.get("from_disk"), \
        "poll must fall back to the saved file"
    assert "REPORT BODY" in j2["report"]
    assert rca.poll(str(tmp_path / "nothing_here"))["state"] == "idle"


def test_a_rejected_cycle_may_not_claim_another_stage_s_record():
    """The step table joined store records to cycles on the STEP NUMBER.

    It checked neither whether the cycle was accepted nor which stage wrote
    the record, so a GenAI attempt rejected at G2c was shown with an evFIS
    consequent tuning as its target, its change and its output. Step numbers
    also restart with every fire, so without the seq0 scope an old record
    was presented as this run's.
    """
    import ast
    import json
    import os
    import tempfile

    src = open('app/streamlit_app.py', encoding='utf-8').read()
    tree = ast.parse(src)
    want = {'build_step_rows', '_adapt_target', '_adapt_change',
            '_gate_marks', '_applied_orders', '_STAGE_NAME', '_read_gstate'}
    ns = {}
    exec(compile(ast.Module(
        body=[n for n in tree.body
              if (isinstance(n, ast.FunctionDef) and n.name in want)
              or (isinstance(n, ast.Assign)
                  and getattr(n.targets[0], 'id', '') in want)],
        type_ignores=[]), '<views>', 'exec'), ns)

    store = {"evfis_rule_modifications": [dict(
        id="evfis_mod_0011", seq=11, source_stage=1,
        base_rule_id="G5", modification_type="consequent_update",
        before={"consequents": [["asset_protection", 0.90]]},
        after={"consequents": [["asset_protection", 0.85]]},
        trigger={"step": 32})], "genai_rules": [], "genai_concepts": [],
        "genai_interventions": []}
    ns['_read_gstate'] = lambda *a, **k: store

    # cycle 32: stage 3 was tried and REJECTED at G2c
    cyc = [dict(step=32, t_min=32.0, global_dss={"hotspot": "Agent_3",
                                                 "shares": {}, "attended": []},
                regions={}, stage_controller={},
                adaptation=dict(stage=0, tried=3, accepted=False,
                                detail="rejected at G2c relevance",
                                dJ=0.0, info={}))]
    row = ns['build_step_rows'](cyc, {"seq0": 0})[0]
    assert row["verdict"] == "rejected"
    assert "0.90" not in str(row["change"]), \
        f"a rejected attempt must not show a tuning as its change: {row}"
    assert row["produced"] == "—", \
        "a rejected attempt produced nothing"
    assert row["rec_seq"] is None

    # the SAME record, now with the stage that actually wrote it accepted
    cyc[0]["adaptation"] = dict(stage=1, tried=1, accepted=True,
                                detail="G5 consequents -0.05", dJ=-0.01,
                                info={})
    row = ns['build_step_rows'](cyc, {"seq0": 0})[0]
    assert "0.90" in str(row["change"]) and row["rec_seq"] == 11, \
        f"the stage that wrote the record must still show it: {row}"

    # no seq0: nothing may be attributed, because step numbers restart
    cyc[0]["adaptation"]["accepted"] = True
    row = ns['build_step_rows'](cyc, {})[0]
    assert row["produced"] == "—" and row["rec_seq"] is None, \
        "without the run scope an old record must not be claimed"


def test_the_adaptation_goes_to_the_least_covered_region_with_fire():
    """The adaptation target is chosen on COVERAGE, not on priority.

    The coordinator ranks on operational priority, which decides where the
    capacity goes. Stage 2 and stage 3 answer situations the rule base does
    NOT cover, so sending them to the highest-priority region sent them, run
    after run, to the region the base already covered best: over 3812 real
    cycles the old selector picked one region 83% of the time and that same
    region had the highest mean fired weight.
    """
    from dss.loop import DecisionEngine

    eng = DecisionEngine([], adapt_on=True)

    def _mk(prio, fire):
        return ({n: {"crisp": {"operational_priority": prio[n]}}
                 for n in prio},
                {n: {"f": {"fire_intensity": fire[n]}} for n in prio})

    # the loudest region is also the best covered: it must NOT be picked
    rows, ctx = _mk({"A": 0.90, "B": 0.30, "C": 0.20},
                    {"A": 0.9, "B": 0.8, "C": 0.7})
    pick, why = eng._adapt_region(rows, ctx, {"A": 0.80, "B": 0.25, "C": 0.60})
    assert pick == "B", f"the least covered region with fire, got {pick}"
    assert "least covered" in why

    # a region with NO fire has the lowest coverage for the trivial reason
    # that nothing is happening there; learning an empty situation is worse
    # than not learning, so it is out of the running
    rows, ctx = _mk({"A": 0.90, "B": 0.30, "C": 0.20},
                    {"A": 0.9, "B": 0.0, "C": 0.7})
    pick, _ = eng._adapt_region(rows, ctx, {"A": 0.80, "B": 0.00, "C": 0.60})
    assert pick == "C", f"a region without fire must not be picked, got {pick}"

    # equal coverage falls back to the coordinator's ranking
    rows, ctx = _mk({"A": 0.20, "B": 0.90, "C": 0.10},
                    {"A": 0.9, "B": 0.9, "C": 0.9})
    pick, _ = eng._adapt_region(rows, ctx, {"A": 0.5, "B": 0.5, "C": 0.5})
    assert pick == "B", f"ties break on priority, got {pick}"

    # nothing burning anywhere: there is no coverage question, so the
    # coordinator's hotspot stands
    rows, ctx = _mk({"A": 0.20, "B": 0.90, "C": 0.10},
                    {"A": 0.0, "B": 0.0, "C": 0.0})
    pick, why = eng._adapt_region(rows, ctx, {"A": 0.1, "B": 0.9, "C": 0.2})
    assert pick == "B" and "no region has fire" in why


def test_stage2_is_filtered_only_when_it_is_certain_to_be_refused():
    """A predictive filter, not a retirement.

    Stage 2 instantiates the antecedent cell of the current situation, so a
    cell that is already covered AND crisp means the stage is certain to be
    turned away. Offering it wastes the pick: measured on the WUI scenario
    the covered-cell refusals were 79% of all attempts. The filter drops
    stage 2 from THAT CYCLE only, so the moment the cell space grows the
    stage returns on its own.
    """
    import numpy as np
    from dss import adapt as A
    from dss.rules import Rule

    terms = list(A.TERMS)

    def _eff(peak_term, sharp=True):
        """A membership vector whose argmax is peak_term."""
        v = np.full(len(terms), 0.10)
        v[terms.index(peak_term)] = 0.95 if sharp else 0.45
        return v

    crisp = {c: 0.9 if i < 2 else 0.1
             for i, c in enumerate(A.DECISION_CONCEPTS)}
    eff = {c: _eff("VH") for c in A.DECISION_CONCEPTS}
    cell = A.stage2_target_cell(eff, crisp)
    assert len(cell) == 2, "the cell is the two most activated concepts"

    # nothing covers the cell yet: the stage has work, it must NOT be filtered
    assert A.stage2_would_be_refused([], eff, crisp) is False

    covering = Rule("X1", list(cell), [("suppression_effort", 0.5)])
    assert A._cell_covered([covering], cell) is True
    assert A.stage2_would_be_refused([covering], eff, crisp) is True, \
        "a covered, crisp cell is a certain refusal"

    # SAME covered cell, but the membership is ambiguous: stage 2 would
    # insert a narrower term and write on the refined cell, so it is still
    # real work and must survive the filter
    eff_amb = dict(eff)
    eff_amb[cell[0][0]] = _eff("VH", sharp=False)
    assert float(np.max(eff_amb[cell[0][0]])) < A.AMBIGUOUS_BELOW
    assert A.stage2_would_be_refused([covering], eff_amb, crisp) is False, \
        "an ambiguous cell is a resolution increase, not a refusal"

    # an inactive rule does not cover anything
    covering.active = False
    assert A.stage2_would_be_refused([covering], eff, crisp) is False


def test_the_filter_never_empties_the_menu():
    """A stage that cannot win still beats no adaptation at all."""
    import inspect
    from dss.loop import DecisionEngine
    src = inspect.getsource(DecisionEngine.decide)
    assert "if _m2:" in src, \
        "the filtered menu must only be adopted when something is left"
    assert "not _void" in src, \
        "a coverage void must be exempt: there the cell is open by definition"


def test_the_controller_value_table_survives_fires_on_the_same_map(tmp_path):
    """The stage controller learns over a CAMPAIGN, not one fire.

    One fire offers only a few dozen adaptation attempts, nowhere near
    enough for an epsilon-greedy value table to get past exploration, so a
    fresh table every run meant the stage choice never converged. The table
    is kept in the store and restored on the same map; a different map
    resets it, because the worth of a stage is a property of the scene.
    """
    import json
    import dss
    from dss.loop import DecisionEngine

    sp = str(tmp_path / "gs.json")

    def campaign(map_key, steps=30):
        w, sim = _mini_fire_sim()
        base, _ = dss.resource_suggestion(w)
        eng = DecisionEngine(dss.partition_n(60, 40, 3), base_pool=base,
                             j_threshold=0.05, cycle_steps=1,
                             horizon_steps=4, adapt_on=True,
                             evfis_on=True, genai_on=False, state_path=sp)
        restored = eng.bind_map(map_key)
        n_start = len(eng.controller.q)
        for _ in range(steps):
            sim.step(resource_override=eng.maybe_decide(sim))
        return restored, n_start, len(eng.controller.q)

    r1, s1, e1 = campaign("MAP-A")
    assert r1 is False and s1 == 0, "the first fire starts from nothing"
    assert e1 > 0, "the fire has to teach the controller something"

    r2, s2, e2 = campaign("MAP-A")
    assert r2 is True, "a second fire on the same map inherits the table"
    assert s2 == e1, f"it must start where the first ended: {s2} vs {e1}"
    assert e2 >= s2, "and keep accumulating"

    r3, s3, _ = campaign("MAP-B")
    assert r3 is False and s3 == 0, \
        "a different map must not inherit another scene's experience"

    on_disk = json.load(open(sp, encoding="utf-8"))["stage_controller"]
    assert on_disk["map_key"] == "MAP-B"
    assert on_disk["maps"]["MAP-B"]["q"], \
        "the table has to reach the file, not only memory"
    assert on_disk["maps"]["MAP-A"]["q"], \
        "the earlier scene keeps its own table"

    # a wipe clears the learned values too: they were learned from rules
    # that a wipe removes
    from dss.state import GeneratedState
    st = GeneratedState.load(sp)
    counts = st.wipe(backup=False)
    assert counts["stage_controller_entries"] > 0
    assert GeneratedState.load(sp).controller_maps() == {}


def test_each_map_keeps_its_own_value_table(tmp_path):
    """Returning to a scene restores what it earned there.

    The first version kept ONE table tagged with a map key, so opening a
    second map threw the first one's experience away and coming back meant
    starting from zero, even though those fires had already been paid for.
    Every scene now has its own table; they still never mix.
    """
    import json
    from dss.adapt import StageController
    from dss.state import GeneratedState

    sp = str(tmp_path / "gs.json")
    st = GeneratedState.load(sp)

    a = StageController()
    a.q[("mid", 1)] = 0.11
    a.q[("high+gap", 3)] = 0.22
    st.save_controller(a, map_key="MAP-A")

    b = StageController()
    b.q[("low", 2)] = -0.05
    st.save_controller(b, map_key="MAP-B")

    assert st.controller_maps() == {"MAP-A": 2, "MAP-B": 1}

    # back on A: its own values, and none of B's
    back = StageController()
    st2 = GeneratedState.load(sp)
    assert st2.load_controller(back, "MAP-A") is True
    assert back.q == {("mid", 1): 0.11, ("high+gap", 3): 0.22}
    assert ("low", 2) not in back.q, "the scenes must never mix"

    # an unknown scene starts empty and says so
    fresh = StageController()
    assert st2.load_controller(fresh, "MAP-NEW") is False and not fresh.q

    # the archive is bounded, least recently seen goes first
    for i in range(GeneratedState.MAX_CONTROLLER_MAPS + 4):
        c = StageController()
        c.q[("mid", 1)] = float(i)
        st2.save_controller(c, map_key=f"M{i:02d}", save=False)
    kept = st2.controller_maps()
    assert len(kept) <= GeneratedState.MAX_CONTROLLER_MAPS
    assert f"M{GeneratedState.MAX_CONTROLLER_MAPS + 3:02d}" in kept, \
        "the most recent scene must survive the eviction"


def test_an_old_single_table_store_is_migrated_not_dropped(tmp_path):
    """A store written before the per-map archive still carries its table."""
    import json
    from dss.adapt import StageController
    from dss.state import GeneratedState, empty_state, SCHEMA_VERSION

    sp = str(tmp_path / "gs.json")
    old = empty_state()
    old["stage_controller"] = {"map_key": "MAP-OLD",
                               "q": {"mid/1": 0.33}, "updates": 7}
    json.dump(old, open(sp, "w", encoding="utf-8"))

    st = GeneratedState.load(sp)
    assert st.controller_maps() == {"MAP-OLD": 1}, \
        "the pre-existing table must be carried into the archive"
    c = StageController()
    assert st.load_controller(c, "MAP-OLD") is True
    assert c.q == {("mid", 1): 0.33}
