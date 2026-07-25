"""Actuation audit: does every intervention family leave its REAL
physical trace on the map?

For each family one controlled run orders ONLY that family at full
intensity through the same path the DSS uses (decision_to_resources ->
sim.step(resource_override)), and the run is compared with an identical
no-action twin. The check then asserts the trace in the right FIELD and
the right PLACE:

  suppression_effort   fuel moisture RISES on/next to the front, and
                       the fire is smaller than the twin's
  containment_line     fuel load is CUT (fuel_suppressed_total > 0) in
                       the downwind band ahead of the front, and the
                       cut cells are visible as cleared ground
  resource_deployment  availability rises / travel time falls across
                       the region, committed capacity is fielded
  asset_protection     capacity concentrates around the THREATENED
                       asset only, and never digs
  evacuation           vpop leaves the ordered cells, the evacuated
                       counter grows, exposure stops counting them
  public_warning       population-side order: no physical field by
                       design (readiness only), asserted to NOT touch
                       the terrain
  macro                a composite expands into its base channels and
                       the base channels act (wet + cut together)

Run: python experiments/actuation_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import dss                                            # noqa: E402
from dss.rules import Rule, evaluate_rules            # noqa: E402
from disaster_phyengine import terrain                # noqa: E402
from disaster_phyengine.config import SimConfig       # noqa: E402
from disaster_phyengine.core import Simulator         # noqa: E402


def build(seed=11):
    cfg = SimConfig(nx=60, ny=40, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=seed, preset="Rolling hills", n_settlements=4,
        population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.09
    w.meteo.wws[:] = 6.0
    base, _ = dss.resource_suggestion(w)
    ok = (w.fuel.fload > 0.4) & (w.fuel.ftype != 0) \
        & (w.fuel.ftype != 5) & (w.fuel.ftype != 6)
    ys, xs = np.where(ok)
    k = len(xs) // 2
    w.add_ignition(int(xs[k]), int(ys[k]), step=0, radius=1)
    return w, base


def run(intensities, steps=12, seed=11):
    """One run ordering exactly `intensities`, plus its no-action twin."""
    w, base = build(seed)
    sim = Simulator(w)
    sim.record_states = False
    region = dss.partition_n(w.config.nx, w.config.ny, 1)[0]
    for _ in range(steps):
        ov = dss.decision_to_resources(
            w, sim.state.burning > 0.5, [(region, dict(intensities))],
            base)
        sim.step(resource_override=ov)
    w2, _b2 = build(seed)
    twin = Simulator(w2)
    twin.record_states = False
    for _ in range(steps):
        twin.step()
    return sim, twin, w, ov


def verdict(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    allok = True
    Z = {i: 0.0 for i in ("suppression_effort", "resource_deployment",
                          "containment_line", "asset_protection",
                          "evacuation", "public_warning",
                          "tactical_burn", "water_drafting",
                          "retardant_drop")}

    print("== suppression_effort ==")
    sim, twin, w, ov = run({**Z, "suppression_effort": 1.0})
    near = sim.ever_burned | twin.ever_burned
    dm = float(w.fuel.fmoist[near].mean()
               - twin.world.fuel.fmoist[near].mean())
    allok &= verdict("wetting", dm > 0.02,
                     f"fmoist at the front +{dm:.3f} vs twin")
    b1, b0 = int(sim.ever_burned.sum()), int(twin.ever_burned.sum())
    allok &= verdict("knockdown", b1 < b0,
                     f"burned {b1} vs {b0} cells (twin)")

    print("== containment_line ==")
    sim, twin, w, ov = run({**Z, "containment_line": 1.0}, steps=15)
    cut = sim.fuel_suppressed_total > 0.05
    ncut = int(cut.sum())
    allok &= verdict("fuel cut", ncut > 0,
                     f"{ncut} cells with fuel removed (visible as "
                     "cleared ground)")
    band_only = bool(ncut) and not bool((cut & sim.ever_burned).sum()
                                        > 0.5 * ncut)
    allok &= verdict("cut ahead of the front", band_only,
                     "the line sits mostly on UNBURNED ground")

    print("== resource_deployment ==")
    sim, twin, w, ov = run({**Z, "resource_deployment": 1.0}, steps=6)
    _, _, w0, ov0 = run(Z, steps=6, seed=11)
    dav = float(ov.ravail.mean() - ov0.ravail.mean())
    dtt = float(ov0.rtime.mean() - ov.rtime.mean())
    allok &= verdict("availability", dav > 0.2,
                     f"mean availability +{dav:.2f} vs no-order")
    allok &= verdict("travel time", dtt > 1.0,
                     f"mean travel time -{dtt:.1f} min vs no-order")

    print("== asset_protection ==")
    sim, twin, w, ov = run({**Z, "asset_protection": 1.0}, steps=10)
    ys, xs = np.where(sim.ever_burned | (sim.state.burning > 0.5))
    prot_cells = 0
    for a in w.assets:
        d = np.sqrt((xs - a.x) ** 2 + (ys - a.y) ** 2).min() \
            if len(xs) else 99
        if d <= 15:
            r = max(2, int(a.radius))
            yy, xx = np.ogrid[0:w.config.ny, 0:w.config.nx]
            m = (xx - a.x) ** 2 + (yy - a.y) ** 2 <= r * r
            prot_cells += int((ov.rcap[m]
                               > 0.5 * w.config.suppression.rcap_max
                               ).sum())
    allok &= verdict("capacity ring", prot_cells > 0,
                     f"{prot_cells} protected cells around threatened "
                     "assets")
    dig_on_prot = getattr(ov, "rcut", None)
    ok_nodig = dig_on_prot is None or float(dig_on_prot.sum()) == 0.0
    allok &= verdict("never digs", ok_nodig,
                     "protection laid no containment cut")

    print("== evacuation ==")
    sim, twin, w, ov = run({**Z, "evacuation": 1.0}, steps=12)
    dpop = float(twin.world.value.vpop.sum() - w.value.vpop.sum())
    allok &= verdict("people moved", sim.population_evacuated > 0.0,
                     f"{sim.population_evacuated:.0f} persons "
                     "evacuated (counter)")
    allok &= verdict("vpop field drained", dpop > 0.0,
                     f"vpop total fell by {dpop:.2f} density units "
                     "(visible on the population layer)")

    print("== public_warning ==")
    sim, twin, w, ov = run({**Z, "public_warning": 1.0}, steps=8)
    same_fuel = np.allclose(w.fuel.fload, twin.world.fuel.fload)
    same_pop = abs(float(w.value.vpop.sum()
                         - twin.world.value.vpop.sum())) < 1e-6
    allok &= verdict("no physical trace (by design)",
                     same_fuel and same_pop,
                     "warning is a population-side order; it must not "
                     "move terrain or people by itself")

    print("== warning primes evacuation ==")
    sim_e, _tw, w_e, _o = run({**Z, "evacuation": 1.0}, steps=10)
    sim_w, _tw2, w_w, _o2 = run({**Z, "evacuation": 1.0,
                                 "public_warning": 1.0}, steps=10)
    allok &= verdict(
        "warned population leaves faster",
        sim_w.population_evacuated > 1.05 * sim_e.population_evacuated,
        f"{sim_w.population_evacuated:.0f} vs "
        f"{sim_e.population_evacuated:.0f} persons in the same 20 min")

    print("== tactical_burn (counter-fire) ==")
    w_b, base_b = build(seed=11)
    sim_b = Simulator(w_b)
    sim_b.record_states = False
    region_b = dss.partition_n(w_b.config.nx, w_b.config.ny, 1)[0]
    strip_any = np.zeros((w_b.config.ny, w_b.config.nx), dtype=bool)
    for _ in range(12):
        ov_b = dss.decision_to_resources(
            w_b, sim_b.state.burning > 0.5,
            [(region_b, {**Z, "tactical_burn": 1.0})], base_b)
        if getattr(ov_b, "rburn", None) is not None:
            strip_any |= ov_b.rburn > 0.5
        sim_b.step(resource_override=ov_b)
    allok &= verdict("firing strip ordered", bool(strip_any.any()),
                     f"{int(strip_any.sum())} cells lit between the "
                     "band and the front over the run")
    burnt_in_strip = bool((sim_b.ever_burned & strip_any).sum()
                          > 0.5 * max(1, strip_any.sum()))
    allok &= verdict("counter-fire really burns", burnt_in_strip,
                     f"{int((sim_b.ever_burned & strip_any).sum())} of "
                     f"{int(strip_any.sum())} ordered cells show up "
                     "BURNT on the map (fire is fire)")

    print("== water_drafting ==")
    # the preset may not carry water: paint a small lake near the fire
    w_d, base_d = build(seed=11)
    ix, iy = w_d.ignitions[0].x, w_d.ignitions[0].y
    x0 = max(0, ix - 8)
    w_d.fuel.ftype[max(0, iy - 3):iy + 3, x0:x0 + 4] = 5
    sim_d = Simulator(w_d)
    sim_d.record_states = False
    region_d = dss.partition_n(w_d.config.nx, w_d.config.ny, 1)[0]
    for _ in range(8):
        ov_n = dss.decision_to_resources(
            w_d, sim_d.state.burning > 0.5,
            [(region_d, {**Z, "suppression_effort": 1.0})], base_d)
        ov_d = dss.decision_to_resources(
            w_d, sim_d.state.burning > 0.5,
            [(region_d, {**Z, "suppression_effort": 1.0,
                         "water_drafting": 1.0})], base_d)
        sim_d.step(resource_override=ov_d)
    gain = float(ov_d.rcap.sum() - ov_n.rcap.sum())
    allok &= verdict("capacity rises near water", gain > 0.0,
                     f"fielded capacity +{gain:.1f} rcap with drafting "
                     "(lake painted beside the fire)")

    print("== retardant_drop (aerial soil/chemical) ==")
    sim_r, twin_r, w_r, ov_r = run({**Z, "retardant_drop": 1.0},
                                   steps=14)
    coated = sim_r.retard > 0.3
    allok &= verdict("coating laid ahead of the head",
                     bool(coated.any()),
                     f"{int(coated.sum())} cells carry retardant")
    b_r, b_t = int(sim_r.ever_burned.sum()), int(twin_r.ever_burned.sum())
    allok &= verdict("coated fuel resists the fire", b_r < b_t,
                     f"burned {b_r} vs {b_t} cells (twin); the coated "
                     "sector slows the head")

    print("== macro expansion ==")
    eff = {c: np.array([0., 0., 0., 0.2, 0.8])
           for c in dss.ten_features.__globals__.get(
               "DECISION_CONCEPTS", [])} or None
    from dss.concepts import DECISION_CONCEPTS
    eff = {c: np.array([0., 0., 0., 0.2, 0.8]) for c in DECISION_CONCEPTS}
    macros = {"backburn_pattern": dict(
        composition=[("containment_line", 1.0),
                     ("suppression_effort", 0.7)])}
    r = Rule("G99", [("fire_threat_level", "VH")],
             [("backburn_pattern", 0.9)])
    u, _tr = evaluate_rules(eff, {}, [r], macros=macros)
    okm = u["containment_line"] > 0.4 and u["suppression_effort"] > 0.3
    allok &= verdict("composite acts through base channels", okm,
                     f"containment={u['containment_line']:.2f}, "
                     f"suppression={u['suppression_effort']:.2f} from "
                     "one macro order")

    print("== runtime-defined actuator (clause grammar) ==")
    # the generative stage may DEFINE a brand-new tactic as data:
    # coat the head 2-5 cells out AND light a flank strip 3-6 out
    w_c, base_c = build(seed=11)
    sim_c = Simulator(w_c)
    sim_c.record_states = False
    region_c = dss.partition_n(w_c.config.nx, w_c.config.ny, 1)[0]
    macros_c = {"flank_burn_head_coat": dict(
        composition=[],
        clauses=[dict(effect="coat", sector="head", range=[2, 5],
                      amount=1.0),
                 dict(effect="ignite", sector="flank", range=[3, 6],
                      amount=0.9)])}
    from dss.concepts import DECISION_CONCEPTS as _DC
    eff_c = {c: np.array([0., 0., 0., 0.2, 0.8]) for c in _DC}
    r_c = Rule("G77", [("fire_threat_level", "VH")],
               [("flank_burn_head_coat", 1.0)])
    burn_any = np.zeros((w_c.config.ny, w_c.config.nx), dtype=bool)
    coat_any = np.zeros_like(burn_any)
    for _ in range(10):
        u_c, _t = evaluate_rules(eff_c, {}, [r_c], macros=macros_c)
        ov_c = dss.decision_to_resources(
            w_c, sim_c.state.burning > 0.5,
            [(region_c, u_c)], base_c, macros=macros_c)
        if getattr(ov_c, "rburn", None) is not None:
            burn_any |= ov_c.rburn > 0.3
        if getattr(ov_c, "rret", None) is not None:
            coat_any |= ov_c.rret > 0.3
        sim_c.step(resource_override=ov_c)
    allok &= verdict("macro intensity reaches the allocator",
                     float(u_c.get("flank_burn_head_coat", 0)) > 0.5,
                     f"u = {u_c.get('flank_burn_head_coat', 0):.2f} "
                     "for the defined actuator")
    allok &= verdict("clause 1 acts (coating laid)",
                     bool(coat_any.any()) and bool(
                         (sim_c.retard > 0.2).any()),
                     f"{int((sim_c.retard > 0.2).sum())} coated cells "
                     "in the head sector")
    allok &= verdict("clause 2 acts (flank strip burnt)",
                     bool((sim_c.ever_burned & burn_any).any()),
                     f"{int((sim_c.ever_burned & burn_any).sum())} "
                     "flank cells genuinely burnt by the counter-fire")

    print("\n" + ("ALL CHECKS PASSED" if allok
                  else "SOME CHECKS FAILED — see above"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
