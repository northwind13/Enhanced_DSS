"""Unit tests for the DisasterAware simulation core.

Run with:  pytest -q   (from the 03_Codes directory)
"""

import numpy as np
import pytest

from disaster_phyengine import (SimConfig, World, Asset, Simulator, compute_costs,
                           scenarios, io_utils)
from disaster_phyengine.spread import rate_of_spread, propagation_influence


def _simple_world(nx=20, ny=20, wind_dir=0.0, wind=8.0):
    cfg = SimConfig(nx=nx, ny=ny, cell_size_m=30.0, max_steps=200,
                    step_minutes=30.0)   # reference calibration step
    w = World.blank(cfg, default_fuel="grass", default_load=1.0, default_moisture=0.05)
    w.set_uniform_wind(speed=wind, direction_rad=wind_dir)
    w.add_ignition(x=nx // 2, y=ny // 2, step=0)
    return w


def test_fuel_is_non_negative_and_non_increasing():
    sim = Simulator(_simple_world())
    f_prev = sim.state.fload.copy()
    for _ in range(60):
        sim.step()
        assert np.all(sim.state.fload >= -1e-9), "fuel went negative"
        assert np.all(sim.state.fload <= f_prev + 1e-9), "fuel increased"
        f_prev = sim.state.fload.copy()


def test_burning_status_is_binary_and_intensity_bounded():
    sim = Simulator(_simple_world())
    sim.run(n_steps=80)
    b = sim.state.burning
    assert set(np.unique(b)).issubset({0.0, 1.0})
    assert np.all(sim.state.intensity >= 0.0) and np.all(sim.state.intensity <= 1.0)


def test_fire_actually_spreads():
    sim = Simulator(_simple_world())
    sim.step()  # ignition takes hold
    n0 = sim.ever_burned.sum()
    sim.run(n_steps=40)
    assert sim.ever_burned.sum() > n0, "fire did not spread from the ignition cell"


def test_intensity_zero_where_not_burning():
    sim = Simulator(_simple_world())
    sim.run(n_steps=50)
    not_burning = sim.state.burning < 0.5
    assert np.all(sim.state.intensity[not_burning] == 0.0)


def test_wind_drives_directional_spread():
    # wind blowing toward +x should push the fire east of the ignition more than west
    sim = Simulator(_simple_world(nx=41, ny=41, wind_dir=0.0, wind=12.0))
    for _ in range(2):   # realistic grass speeds cross the domain in a few steps
        sim.step()
    cx = 20
    burned = sim.ever_burned
    east = burned[:, cx + 1:].sum()
    west = burned[:, :cx].sum()
    assert east > west, f"expected more spread east (got east={east}, west={west})"


def test_ros_increases_with_wind():
    w_low = _simple_world(wind=2.0)
    w_high = _simple_world(wind=15.0)
    r_low = rate_of_spread(w_low.fuel, w_low.topo, w_low.meteo, w_low.config.spread)
    r_high = rate_of_spread(w_high.fuel, w_high.topo, w_high.meteo, w_high.config.spread)
    assert r_high.mean() > r_low.mean()


def test_moisture_above_extinction_stops_spread():
    w = _simple_world()
    w.fuel.fmoist[:] = 0.9  # well above any extinction threshold
    sim = Simulator(w)
    sim.run(n_steps=40)
    # only the directly ignited cell may burn; no propagation
    assert sim.ever_burned.sum() <= 5


def test_suppression_reduces_burned_area():
    # strong, sustained suppression depletes fuel ahead of the front and should
    # contain the fire to a smaller footprint than the unsuppressed baseline
    base = Simulator(_simple_world(nx=61, ny=61, wind=8.0))
    base.run(n_steps=50, stop_when_quiescent=False)
    burned_no_supp = base.ever_burned.sum()

    w = _simple_world(nx=61, ny=61, wind=8.0)
    w.config.suppression.alpha_s = 0.9
    w.set_resource_field(rcap=1.0, ravail=1.0, reff=1.0, rtime=0.2)
    supp = Simulator(w)
    supp.run(n_steps=50, stop_when_quiescent=False)
    burned_supp = supp.ever_burned.sum()
    assert supp.fuel_suppressed_total.sum() > 0
    assert burned_supp < burned_no_supp


def test_firebreak_blocks_spread():
    # tests the SURFACE spread barrier; ember spotting is disabled here
    # because embers legitimately fly over firebreaks (that is their point)
    w = _simple_world(nx=41, ny=41, wind=10.0, wind_dir=0.0)
    w.config.spread.spotting = False
    w.clear_fuel(25, 0, 26, 40)  # vertical firebreak east of ignition
    sim = Simulator(w)
    sim.run(n_steps=80)
    assert sim.ever_burned[:, 30:].sum() == 0, "fire crossed the firebreak"


def test_spotting_can_cross_firebreak_with_wind():
    w = _simple_world(nx=41, ny=41, wind=12.0, wind_dir=0.0)
    w.config.spread.spotting = True
    w.config.spread.spot_prob = 0.8      # make the stochastic jump near-certain
    w.clear_fuel(25, 0, 26, 40)
    sim = Simulator(w)
    sim.run(n_steps=40, stop_when_quiescent=False)
    assert sim.ever_burned[:, 30:].sum() > 0, "embers never crossed the break"


def test_costs_are_consistent():
    sim = Simulator(scenarios.wui_interface())
    sim.run()
    rep = compute_costs(sim)
    assert rep.burned_area_ha >= rep.burned_forest_ha >= 0
    assert 0.0 <= rep.j_total <= 1.0
    for term in (rep.j_burn, rep.j_asset, rep.j_pop, rep.j_resp, rep.j_delay):
        assert 0.0 <= term <= 1.0
    assert rep.population_person_steps >= 0


def test_scenario_serialization_roundtrip(tmp_path):
    w = scenarios.grassland_run()
    path = str(tmp_path / "scenario.json")
    io_utils.save_scenario(w, path)
    w2 = io_utils.load_scenario(path)
    assert w2.shape == w.shape
    assert np.array_equal(w2.fuel.ftype, w.fuel.ftype)
    assert len(w2.ignitions) == len(w.ignitions)


def test_all_builtin_scenarios_run():
    for name, builder in scenarios.SCENARIOS.items():
        sim = Simulator(builder())
        sim.run(n_steps=40)
        assert sim.ever_burned.sum() > 0, f"{name} produced no fire"


def test_observation_does_not_mutate_state():
    from disaster_phyengine import observe
    sim = Simulator(_simple_world())
    sim.run(n_steps=20)
    before = (sim.state.burning.copy(), sim.state.fload.copy(),
              sim.state.intensity.copy(), sim.state.tau.copy())
    o = observe(sim, epsilon=0.0)
    # faithful read equals state
    assert np.array_equal(o.burning, before[0])
    assert np.array_equal(o.fload, before[1])
    # noisy read must not touch the underlying state
    o2 = observe(sim, epsilon=0.2, seed=1)
    assert np.array_equal(sim.state.fload, before[1])
    assert np.all(o2.fload >= 0.0) and np.all(o2.fload <= 1.0)


def test_observation_region_window():
    from disaster_phyengine import observe
    sim = Simulator(_simple_world(nx=30, ny=30))
    sim.run(n_steps=15)
    o = observe(sim, region=(0, 0, 10, 10))
    assert o.burning[:, 20:].sum() == 0  # outside the window is masked to zero


def test_suppression_capped_at_available_fuel():
    # F_red must never exceed available fuel (Eq. 135 / REQ-SUP-06)
    w = _simple_world(nx=25, ny=25)
    w.config.suppression.alpha_s = 1.0
    w.set_resource_field(rcap=10.0, ravail=1.0, reff=1.0, rtime=0.0)
    sim = Simulator(w)
    for _ in range(30):
        sim.step()
        assert np.all(sim.state.fload >= -1e-9)


def test_default_realism_modes_on():
    # literature-realistic defaults: elliptical kernel and ember spotting on
    cfg = SimConfig()
    assert cfg.spread.elliptical is True and cfg.spread.spotting is True


def test_elliptical_mode_runs_and_spreads():
    w = _simple_world(nx=41, ny=41, wind=12.0)
    w.config.spread.elliptical = True
    sim = Simulator(w); sim.run(n_steps=30)
    assert sim.ever_burned.sum() > 1


def test_spotting_ignites_ahead():
    w = _simple_world(nx=61, ny=61, wind=15.0)
    w.config.spread.spotting = True
    w.config.spread.spot_prob = 0.3
    w.config.spread.spot_intensity_min = 0.0
    sim = Simulator(w); sim.run(n_steps=25, stop_when_quiescent=False)
    base = _simple_world(nx=61, ny=61, wind=15.0)
    b = Simulator(base); b.run(n_steps=25, stop_when_quiescent=False)
    assert sim.ever_burned.sum() >= b.ever_burned.sum()


def test_behavior_fields():
    from disaster_phyengine import behavior
    sim = Simulator(scenarios.wui_interface()); sim.run(n_steps=25, stop_when_quiescent=False)
    fli = behavior.fireline_intensity(sim)
    fl = behavior.flame_length_field(sim)
    assert fli.min() >= 0 and fl.min() >= 0
    assert behavior.perimeter_mask(sim).sum() >= 0


def test_emc_moisture_bounds():
    from disaster_phyengine.fuel_moisture import equilibrium_moisture
    m = equilibrium_moisture(np.array([30.0, 15.0]), np.array([20.0, 90.0]))
    assert np.all(m >= 0.01) and np.all(m <= 0.6)


def test_standard_fuel_resolve():
    from disaster_phyengine import fuels_standard
    fid, load, moist = fuels_standard.resolve("Anderson 13", 3)
    assert fid in range(0, 6) and 0 <= load <= 1


def test_ignition_at_corner_and_edges_no_crash():
    for (ix, iy) in [(0, 0), (40, 0), (0, 40), (40, 40), (20, 0), (0, 20)]:
        w = _simple_world(nx=41, ny=41, wind=6.0)
        w.ignitions.clear()
        w.add_ignition(ix, iy, step=0, radius=1)
        sim = Simulator(w)
        sim.run(n_steps=30)
        assert sim.state.burning.shape == (41, 41)
        assert sim.ever_burned.sum() >= 1


def test_zero_wind_is_isotropic():
    w = _simple_world(nx=41, ny=41, wind=0.0)
    w.ignitions.clear(); w.add_ignition(20, 20, step=0)
    sim = Simulator(w)
    for _ in range(16):
        sim.step()
    b = sim.ever_burned
    east, west = b[:, 21:].sum(), b[:, :20].sum()
    north, south = b[:20, :].sum(), b[21:, :].sum()
    assert abs(int(east) - int(west)) <= 3
    assert abs(int(north) - int(south)) <= 3


def test_strong_wind_is_directional():
    w = _simple_world(nx=41, ny=41, wind=14.0, wind_dir=0.0)
    w.ignitions.clear(); w.add_ignition(20, 20, step=0)
    sim = Simulator(w)
    for _ in range(2):   # realistic grass speeds cross the domain in a few steps
        sim.step()
    b = sim.ever_burned
    assert b[:, 21:].sum() > b[:, :20].sum()


def test_city_scenario_burns_structures():
    sim = Simulator(scenarios.city_wui())
    sim.run()
    rep = compute_costs(sim)
    assert rep.asset_value_lost > 0
    assert rep.j_asset > 0




def test_validation_metrics_bounds():
    from disaster_phyengine import validation
    a = np.zeros((12, 12), dtype=bool); a[2:7, 2:7] = True
    b = np.zeros((12, 12), dtype=bool); b[3:8, 3:8] = True
    m = validation.compare_masks(a, b)
    assert 0.0 <= m["jaccard"] <= m["dice"] <= 1.0
    assert validation.compare_masks(a, a)["jaccard"] == 1.0
    d = validation.front_distance_errors(a, b, cell_size_m=30.0)
    assert d["mean_m"] >= 0.0 and d["p90_m"] >= d["mean_m"] * 0.5
