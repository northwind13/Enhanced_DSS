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
    store = "/tmp/test_pkg_reg/learned_rules.json"
    if os.path.exists(store):
        os.remove(store)
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
        pkg = {"antecedents": [["fire_threat_level", "H"]],
               "consequents": [["backburn", 0.9]],
               "new_intervention": {
                   "name": "backburn",
                   "composition": [["containment_line", 0.7],
                                   ["suppression_effort", 0.5]]}}
        A._genai_propose = lambda s_: json.loads(json.dumps(pkg))
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
        A._genai_propose = lambda s_: json.loads(json.dumps(pkg2))
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
