"""Unit tests for the DisasterAware simulation core.

Run with:  pytest -q   (from the 03_Codes directory)
"""

import numpy as np
import pytest

from disasteraware import (SimConfig, World, Asset, Simulator, compute_costs,
                           scenarios, io_utils)
from disasteraware.spread import rate_of_spread, propagation_influence


def _simple_world(nx=20, ny=20, wind_dir=0.0, wind=8.0):
    cfg = SimConfig(nx=nx, ny=ny, cell_size_m=30.0, max_steps=200)
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
    for _ in range(18):
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
    w = _simple_world(nx=41, ny=41, wind=10.0, wind_dir=0.0)
    w.clear_fuel(25, 0, 26, 40)  # vertical firebreak east of ignition
    sim = Simulator(w)
    sim.run(n_steps=80)
    assert sim.ever_burned[:, 30:].sum() == 0, "fire crossed the firebreak"


def test_costs_are_consistent():
    sim = Simulator(scenarios.wui_interface())
    sim.run()
    rep = compute_costs(sim)
    assert rep.burned_area_ha >= rep.burned_forest_ha >= 0
    assert rep.total_economic_cost >= 0
    assert rep.expected_casualties >= 0


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
