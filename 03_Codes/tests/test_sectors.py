"""Sector geometry and clause/library physics parity (open intervention
space, Faz A3/A4)."""
import numpy as np
import pytest

from disaster_phyengine.config import SimConfig
from disaster_phyengine import terrain
from disaster_phyengine.core import Simulator
from disaster_phyengine.layers import ResourceLayer
from dss import actions as A


def _world(seed=5, n=40, ignite=(20, 20), steps=4, dry=False):
    cfg = SimConfig(nx=n, ny=n, cell_size_m=30.0)
    world = terrain.generate_landscape(cfg, seed=seed)
    if dry:
        # dry fuel and a steady wind give a fire large enough to have a
        # head, a flank and a rear within a few steps
        world.fuel.fmoist[:] = 0.04
        world.meteo.wws[:] = 10.0
    world.add_ignition(ignite[0], ignite[1], step=0, radius=1)
    sim = Simulator(world)
    sim.record_states = False
    for _ in range(steps):
        sim.step()
    fire = np.asarray(sim.state.burning, dtype=bool)
    return world, sim, fire


def _cheb(core):
    """Four-neighbour distance field to `core` (brute force, small)."""
    ny, nx = core.shape
    d = np.full((ny, nx), 10 ** 6)
    ys, xs = np.where(core)
    for y in range(ny):
        for x in range(nx):
            d[y, x] = int(np.min(np.abs(ys - y) + np.abs(xs - x))) \
                if len(ys) else 10 ** 6
    return d


def test_band_is_an_inclusive_distance_interval_and_rin_zero_includes_core():
    core = np.zeros((15, 15), dtype=bool)
    core[7, 7] = True
    d = _cheb(core)
    assert np.array_equal(A._band(core, 0, 3), d <= 3)
    assert np.array_equal(A._band(core, 2, 4), (d >= 2) & (d <= 4))
    assert np.array_equal(A._band(core, 3, 3), d == 3)
    assert not A._band(core, 4, 2).any()


def test_range_applies_to_assets_populated_and_at_fire():
    world, sim, fire = _world()
    assert fire.any()
    pop = np.asarray(world.value.vpop) > 1e-6
    if not pop.any():
        pytest.skip("world without population")
    p0 = A._sector_mask("populated", fire, world, 0, 0, None)
    p2 = A._sector_mask("populated", fire, world, 0, 2, None)
    p_ring = A._sector_mask("populated", fire, world, 1, 2, None)
    assert np.array_equal(p0, pop)
    assert p2.sum() > p0.sum()
    assert not (p_ring & pop).any()
    f0 = A._sector_mask("at_fire", fire, world, 0, 0, None)
    f3 = A._sector_mask("at_fire", fire, world, 0, 3, None)
    assert np.array_equal(f0, fire)
    assert f3.sum() > f0.sum()
    cores = A._asset_cores(world, fire)
    if cores.any():
        a0 = A._sector_mask("assets", fire, world, 0, 0, None)
        a3 = A._sector_mask("assets", fire, world, 0, 3, None)
        assert np.array_equal(a0, cores)
        assert a3.sum() > a0.sum()


def test_two_clauses_that_differ_in_range_never_compile_to_the_same_cells():
    world, sim, fire = _world()
    for sec in A.SECTORS:
        m1 = A._sector_mask(sec, fire, world, 0, 4, None)
        m2 = A._sector_mask(sec, fire, world, 0, 6, None)
        if not m1.any() and not m2.any():
            continue
        assert not np.array_equal(m1, m2), sec


def test_head_flank_rear_partition_the_band_for_a_tiny_fire():
    world, sim, fire = _world(steps=1)
    assert 0 < int(fire.sum()) < 50
    band = A._sector_mask("ring", fire, world, 2, 5, None)
    head = A._sector_mask("head", fire, world, 2, 5, None)
    flank = A._sector_mask("flank", fire, world, 2, 5, None)
    rear = A._sector_mask("rear", fire, world, 2, 5, None)
    assert np.array_equal(head | flank | rear, band)
    assert not (head & rear).any()
    assert head.any() and rear.any()


def _region_all(n):
    class R:
        x0, y0, x1, y1 = 0, 0, n, n
        name = "all"

        def slices(self):
            return slice(0, n), slice(0, n)
    return R()


def test_clause_ignite_and_tactical_burn_write_the_same_field():
    world, sim, fire = _world(seed=7, steps=10, dry=True)
    assert int(fire.sum()) >= 50
    n = world.config.nx
    base = ResourceLayer.none(n, n)
    base.rcap[:] = 1.0
    base.ravail[:] = 1.0
    reg = _region_all(n)
    out_lib = A.decision_to_resources(
        world, fire, [(reg, {"tactical_burn": 1.0})], base=base)
    macro = {"firing_strip": {"clauses": [
        {"effect": "ignite", "sector": "head", "range": [4, 6],
         "amount": 1.0}]}}
    out_cl = A.decision_to_resources(
        world, fire, [(reg, {"firing_strip": 1.0})], base=base,
        macros=macro)
    assert out_lib.rburn is not None and out_cl.rburn is not None
    lit_lib = out_lib.rburn > 0.5
    lit_cl = out_cl.rburn > 0.5
    # the clause strip is the head part of the library strip: wherever
    # both light, they light at the same strength
    both = lit_lib & lit_cl
    assert both.any()
    assert np.allclose(out_lib.rburn[both], out_cl.rburn[both])
    assert not (lit_cl & ~lit_lib).any()


def test_clause_draft_equals_library_water_drafting_on_its_cells():
    world, sim, fire = _world(seed=7, steps=10, dry=True)
    assert np.asarray(world.fuel.ftype == 5).any()
    n = world.config.nx
    base = ResourceLayer.none(n, n)
    base.rcap[:] = 1.0
    base.ravail[:] = 1.0
    reg = _region_all(n)
    out_lib = A.decision_to_resources(
        world, fire, [(reg, {"suppression_effort": 1.0,
                             "water_drafting": 1.0})], base=base)
    out_none = A.decision_to_resources(
        world, fire, [(reg, {"suppression_effort": 1.0})], base=base)
    macro = {"shore_draft": {"clauses": [
        {"effect": "draft", "sector": "at_fire", "range": [0, 15],
         "amount": 1.0}]}}
    out_cl = A.decision_to_resources(
        world, fire, [(reg, {"suppression_effort": 1.0,
                             "shore_draft": 1.0})], base=base,
        macros=macro)
    m = A._sector_mask("at_fire", fire, world, 0, 15, None) \
        & (out_none.rcap > 1e-6)
    assert m.any()
    boosted = (out_lib.rcap > out_none.rcap + 1e-9) & m
    assert boosted.any(), "drafting had no effect near the fire"
    assert np.allclose(out_cl.rcap[m], out_lib.rcap[m])


def test_clause_coat_matches_retardant_drop_on_the_shared_band():
    world, sim, fire = _world(seed=7, steps=10, dry=True)
    assert int(fire.sum()) >= 50
    assert np.asarray(world.fuel.ftype == 5).any()
    n = world.config.nx
    base = ResourceLayer.none(n, n)
    base.rcap[:] = 1.0
    base.ravail[:] = 1.0
    reg = _region_all(n)
    out_lib = A.decision_to_resources(
        world, fire, [(reg, {"retardant_drop": 1.0})], base=base)
    macro = {"head_coat": {"clauses": [
        {"effect": "coat", "sector": "head", "range": [3, 5],
         "amount": 1.0}]}}
    out_cl = A.decision_to_resources(
        world, fire, [(reg, {"head_coat": 1.0})], base=base, macros=macro)
    both = (out_lib.rret > 0) & (out_cl.rret > 0)
    assert both.any()
    assert np.allclose(out_lib.rret[both], out_cl.rret[both])
    assert not ((out_cl.rret > 0) & ~(out_lib.rret > 0)).any()
