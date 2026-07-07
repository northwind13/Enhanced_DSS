"""DSS demonstration: the same wildfire with and without decision support.

Site setup: sensor assets (satellite + UAV + in-situ + field report) and a
resource fleet (engines, crews, helicopter, dozer) are deployed on the map.
The DSS run senses the fire through the network only, tasks the units, and
the report shows what it knew (confidence map), what it decided (concept and
assignment views), and what it changed (fuel and cost curves).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from disasteraware.config import SimConfig
from disasteraware.core import Simulator
from disasteraware.costs import compute_costs
from disasteraware.world import World, Asset

from dss.loop import DSSRunner
from dss.sensing import Sensor
from dss.units import ResourceUnit
from dss.features import extract_features
from dss.concepts import compute_concepts
from dss.evaluate import evaluate_intervention

N_STEPS = 80


def build_world():
    cfg = SimConfig(ny=40, nx=40, max_steps=N_STEPS + 20)
    cfg.suppression.alpha_s = 0.30
    world = World.blank(cfg, default_fuel="grass", default_load=1.0,
                        default_moisture=0.03)
    world.add_forest_patch(6, 6, 33, 33, fuel_type="pine_litter",
                           load=1.0, moisture=0.05)
    world.add_asset(Asset("hospital", "critical", 34, 34, radius=2, value=1.0))
    world.add_asset(Asset("village", "population", 34, 30, radius=2,
                          population=1200))
    world.add_asset(Asset("houses", "building", 30, 34, radius=2, value=0.8))
    world.add_road_segment(0, 36, 39, 36, width=1)
    world.add_ignition(8, 8, step=0)
    world.set_uniform_wind(10.0, 0.8)
    return world


def site_setup():
    sensors = [
        Sensor.preset("satellite", sensor_id="sat_1"),
        Sensor.preset("aerial", x=12, y=12, sensor_id="uav_1"),
        Sensor.preset("aerial", x=28, y=28, sensor_id="uav_2"),
        Sensor.preset("in_situ", x=34, y=33, sensor_id="ground_1"),
        Sensor.preset("field_report", x=8, y=8, sensor_id="report_1"),
    ]
    units = [
        ResourceUnit.preset("engine", x=2, y=37, unit_id="engine_1"),
        ResourceUnit.preset("engine", x=37, y=37, unit_id="engine_2"),
        ResourceUnit.preset("crew", x=2, y=37, unit_id="crew_1"),
        ResourceUnit.preset("crew", x=37, y=37, unit_id="crew_2"),
        ResourceUnit.preset("helicopter", x=20, y=38, unit_id="heli_1"),
        ResourceUnit.preset("dozer", x=20, y=37, unit_id="dozer_1"),
    ]
    return sensors, units


def main():
    print("DisasterAware DSS demo | sensor agi + kaynak filosu\n")
    sensors, units = site_setup()

    sim_ref = Simulator(build_world())
    consumed_ref = []
    for _ in range(N_STEPS):
        sim_ref.step()
        consumed_ref.append(float(sim_ref.fuel_consumed_total.sum()))
    ref = compute_costs(sim_ref)

    sim_dss = Simulator(build_world())
    runner = DSSRunner(sim_dss, n_regions=(2, 2), sensors=sensors,
                       units=units, quality_threshold=0.6, seed=42)
    consumed_dss, j_series = [], []
    for _ in range(N_STEPS):
        res = runner.step()
        consumed_dss.append(float(sim_dss.fuel_consumed_total.sum()))
        j = evaluate_intervention(sim_dss, res.global_decision.intervention,
                                  horizon=1, units=units)
        j_series.append(j.total)
        if res.diag.n_burning == 0 and sim_dss.ever_burned.any():
            break
    dss_phys = compute_costs(sim_dss)
    dss_mit = runner.history[-1].mitigated

    rows = [
        ("yanan alan (ha)", ref.burned_area_ha, dss_phys.burned_area_ha),
        ("tuketilen yakit", ref.fuel_consumed_total, dss_phys.fuel_consumed_total),
        ("bina kaybi", ref.building_loss, dss_mit.building_loss),
        ("kritik altyapi kaybi", ref.critical_infrastructure_loss,
         dss_mit.critical_infrastructure_loss),
        ("maruz nufus", ref.population_exposed, dss_mit.population_exposed),
        ("bastirma maliyeti", ref.suppression_cost, dss_phys.suppression_cost),
    ]
    print(f"{'metrik':28s} {'DSS yok':>12s} {'DSS var':>12s}")
    for name, a, b in rows:
        print(f"{name:28s} {a:12.2f} {b:12.2f}")

    last = runner.audit.records[-1]
    print(f"\nson adim: Q = {last.quality}  ort. guven = {last.confidence_mean}"
          f"  fail-safe = {last.fail_safe_applied}")
    print("en guclu kurallar:", ", ".join(r["rule_id"] for r in last.top_rules))
    gdec = runner.history[-1].global_decision
    if gdec.assignments:
        print("birim atamalari:", ", ".join(
            f"{a.unit_id}->{a.target} ({a.travel_time:.1f} adim)"
            for a in gdec.assignments))
    print(f"karar maliyeti J (son adim) = {j_series[-1]:.3f}")

    obs_full, kappa = runner.network.composite(sim_dss.state.step)
    feats = extract_features(obs_full, sim_dss.world, kappa=kappa)
    concepts = compute_concepts(feats)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("DisasterAware DSS demo raporu (sensor agi + kaynak filosu)",
                 fontsize=13)

    def burnmap(ax, sim, title):
        img = np.zeros((*sim.world.shape, 3))
        img[..., 1] = 0.5 * (sim.world.fuel.fload0 > 0)
        img[sim.ever_burned] = (0.25, 0.25, 0.25)
        img[sim.state.burning > 0.5] = (1.0, 0.3, 0.0)
        vals = sim.world.value
        assets = (vals.vbld + vals.vcrit) > 0.3
        img[assets] = (0.2, 0.4, 1.0)
        ax.imshow(img, origin="lower")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    burnmap(axes[0, 0], sim_ref, f"DSS yok | yanan {ref.burned_area_ha:.1f} ha")
    burnmap(axes[0, 1], sim_dss,
            f"DSS var | yanan {dss_phys.burned_area_ha:.1f} ha")
    for u in units:
        axes[0, 1].plot(u.x, u.y, "ws", ms=6, mec="black")
    for a in gdec.assignments:
        axes[0, 1].plot(a.target[0], a.target[1], "y*", ms=12, mec="black")

    im = axes[0, 2].imshow(kappa, origin="lower", cmap="viridis",
                           vmin=0, vmax=1)
    axes[0, 2].set_title("gozlem guveni (kappa)")
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])
    for s in sensors:
        if s.radius is not None:
            axes[0, 2].plot(s.x, s.y, "w^", ms=7, mec="black")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.8)

    im = axes[1, 0].imshow(concepts["fire_threat_level"], origin="lower",
                           cmap="inferno", vmin=0, vmax=1)
    axes[1, 0].set_title("konsept: fire threat level (algilanan)")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

    axes[1, 1].plot(consumed_ref, label="DSS yok", color="tab:red")
    axes[1, 1].plot(consumed_dss, label="DSS var", color="tab:blue")
    axes[1, 1].set_title("kumulatif tuketilen yakit")
    axes[1, 1].set_xlabel("adim"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(j_series, color="tab:purple")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title("karar maliyeti J (Eq. 9)")
    axes[1, 2].set_xlabel("adim"); axes[1, 2].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).with_name("dss_demo_report.png")
    fig.savefig(out, dpi=110)
    print(f"\ngorsel rapor: {out.name}")
    runner.audit.to_json(str(Path(__file__).with_name("dss_audit_log.json")))
    print("iz kaydi   : dss_audit_log.json")


if __name__ == "__main__":
    main()
