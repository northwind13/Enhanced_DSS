"""DisasterAware interactive simulator dashboard.

Run with:
    streamlit run app/streamlit_app.py

The dashboard is the supervisory UI of Section 4.4: it visualizes the wildfire
state, lets the operator edit the landscape (forests, assets, firebreaks),
schedule ignitions, tune every model parameter and read the cost report and time
series. All operator actions modify the external input set only; the simulation
state is never written directly, preserving the architecture of Section 4.1.
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disasteraware import (Simulator, World, SimConfig, Asset, compute_costs,
                           scenarios, io_utils, FUEL_NAME_TO_ID)
from disasteraware import viz

st.set_page_config(page_title="DisasterAware Simulator", layout="wide")

FUEL_TYPES = ["grass", "shrub", "pine_litter", "hardwood"]
ASSET_KINDS = ["building", "critical", "population", "evac_route"]


# --------------------------------------------------------------------- state
def _new_simulator(world: World) -> None:
    st.session_state.world = world
    st.session_state.sim = Simulator(world)
    st.session_state.cost_series = []
    st.session_state.playing = False


def _ensure_state() -> None:
    if "sim" not in st.session_state:
        _new_simulator(scenarios.wui_interface())


def _record_costs() -> None:
    rep = compute_costs(st.session_state.sim)
    st.session_state.cost_series.append(rep.to_dict())


_ensure_state()
sim: Simulator = st.session_state.sim
world: World = st.session_state.world
cfg = world.config


# -------------------------------------------------------------------- sidebar
st.sidebar.title("DisasterAware")
st.sidebar.caption("Grid based wildfire simulator (thesis Chapter 4)")

with st.sidebar.expander("Scenario", expanded=True):
    scen_name = st.selectbox("Built in scenario", list(scenarios.SCENARIOS.keys()))
    col_a, col_b = st.columns(2)
    if col_a.button("Load scenario", use_container_width=True):
        _new_simulator(scenarios.SCENARIOS[scen_name]())
        st.rerun()
    if col_b.button("Blank grid", use_container_width=True):
        _new_simulator(World.blank(SimConfig(nx=100, ny=70)))
        st.rerun()

    up = st.file_uploader("Load scenario file", type=["json", "yaml", "yml"])
    if up is not None:
        tmp = os.path.join("/tmp", up.name)
        with open(tmp, "wb") as fh:
            fh.write(up.getbuffer())
        _new_simulator(io_utils.load_scenario(tmp))
        st.success(f"Loaded {up.name}")
        st.rerun()

with st.sidebar.expander("Weather and environment", expanded=True):
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 30.0,
                           float(world.meteo.wws.mean()), 0.5)
    wind_dir_deg = st.slider("Wind direction (deg, 0 = +x east)", 0, 360,
                             int(np.degrees(world.meteo.wwd.mean())) % 360, 5)
    moisture = st.slider("Fuel moisture (fraction)", 0.0, 0.6,
                         float(world.fuel.fmoist.mean()), 0.01)
    if st.button("Apply weather", use_container_width=True):
        world.set_uniform_wind(wind_speed, np.radians(wind_dir_deg))
        world.fuel.fmoist[:] = moisture
        st.rerun()

with st.sidebar.expander("Model parameters"):
    cfg.spread.theta_ign = st.slider("Ignition threshold theta_ign",
                                     0.01, 0.5, float(cfg.spread.theta_ign), 0.01)
    cfg.spread.w0 = st.slider("Wind reference w0", 1.0, 15.0,
                              float(cfg.spread.w0), 0.5)
    cfg.suppression.alpha_s = st.slider("Suppression gain alpha_s",
                                        0.0, 1.0, float(cfg.suppression.alpha_s), 0.05)
    cfg.suppression.gamma_I = st.slider("Intensity resistance gamma_I",
                                        0.0, 5.0, float(cfg.suppression.gamma_I), 0.5)
    cfg.intensity.beta = st.slider("Intensity gain beta", 0.5, 3.0,
                                   float(cfg.intensity.beta), 0.1)

with st.sidebar.expander("Run controls", expanded=True):
    c1, c2 = st.columns(2)
    if c1.button("Step", use_container_width=True):
        sim.step(); _record_costs(); st.rerun()
    if c2.button("Run 10", use_container_width=True):
        for _ in range(10):
            sim.step()
        _record_costs(); st.rerun()
    c3, c4 = st.columns(2)
    if c3.button("Run to end", use_container_width=True):
        sim.run(); _record_costs(); st.rerun()
    if c4.button("Reset", use_container_width=True):
        sim.reset(); st.session_state.cost_series = []; st.rerun()
    st.session_state.playing = st.toggle("Auto play", st.session_state.playing)


# ----------------------------------------------------------------------- main
st.title("Wildfire Simulation")

tab_sim, tab_edit, tab_layers, tab_gis, tab_about = st.tabs(
    ["Simulation", "Map editor", "Data layers", "GIS import", "About"])

with tab_sim:
    left, right = st.columns([3, 2])
    with left:
        show_value = st.checkbox("Overlay protection priority", value=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        if show_value:
            base = viz.value_overlay_rgb(world)
            img = base.copy()
            img[sim.ever_burned] = [0.18, 0.15, 0.13]
            act = sim.state.burning > 0.5
            img[act] = [0.95, 0.30, 0.08]
            ax.imshow(img, origin="upper")
        else:
            ax.imshow(viz.fire_state_rgb(sim), origin="upper")
        ax.set_title(f"Step {sim.state.step}   "
                     f"active cells: {int((sim.state.burning > 0.5).sum())}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        rep = compute_costs(sim)
        st.subheader("Cost report")
        m1, m2 = st.columns(2)
        m1.metric("Burned area (ha)", f"{rep.burned_area_ha:,.1f}")
        m2.metric("Burned forest (ha)", f"{rep.burned_forest_ha:,.1f}")
        m1.metric("Active fire cells", f"{rep.active_fire_cells:,}")
        m2.metric("Fuel consumed", f"{rep.fuel_consumed_total:,.0f}")
        st.divider()
        m3, m4 = st.columns(2)
        m3.metric("Building loss", f"{rep.building_loss:,.0f}")
        m4.metric("Critical infra loss", f"{rep.critical_infrastructure_loss:,.0f}")
        m3.metric("Population exposed", f"{rep.population_exposed:,.0f}")
        m4.metric("Expected casualties", f"{rep.expected_casualties:,.1f}")
        m3.metric("Suppression cost", f"{rep.suppression_cost:,.0f}")
        m4.metric("Total economic cost", f"{rep.total_economic_cost:,.0f}")

    series = st.session_state.cost_series
    if len(series) > 1:
        st.subheader("Time series")
        steps = [r["step"] for r in series]
        cc1, cc2 = st.columns(2)
        with cc1:
            f1, a1 = plt.subplots(figsize=(5, 2.6))
            a1.plot(steps, [r["burned_area_ha"] for r in series], color="#c0392b")
            a1.set_xlabel("step"); a1.set_ylabel("burned area (ha)")
            st.pyplot(f1, use_container_width=True); plt.close(f1)
        with cc2:
            f2, a2 = plt.subplots(figsize=(5, 2.6))
            a2.plot(steps, [r["total_economic_cost"] for r in series], color="#2c3e50")
            a2.set_xlabel("step"); a2.set_ylabel("total cost")
            st.pyplot(f2, use_container_width=True); plt.close(f2)


with tab_edit:
    st.subheader("Edit the landscape")
    st.caption("All edits apply to the current grid. Press Reset in the sidebar "
               "after editing to restart the fire from the new landscape.")
    ec1, ec2 = st.columns(2)

    with ec1:
        st.markdown("**Add forest / fuel patch**")
        fx0 = st.number_input("x0", 0, cfg.nx - 1, 5, key="fx0")
        fy0 = st.number_input("y0", 0, cfg.ny - 1, 5, key="fy0")
        fx1 = st.number_input("x1", 0, cfg.nx - 1, min(40, cfg.nx - 1), key="fx1")
        fy1 = st.number_input("y1", 0, cfg.ny - 1, min(40, cfg.ny - 1), key="fy1")
        ftype = st.selectbox("Fuel type", FUEL_TYPES, index=2)
        fload = st.slider("Fuel load", 0.0, 1.0, 1.0, 0.05)
        if st.button("Paint forest patch"):
            world.add_forest_patch(fx0, fy0, fx1, fy1, fuel_type=ftype,
                                   load=fload, moisture=moisture)
            sim.reset(); st.session_state.cost_series = []
            st.success("Forest patch added"); st.rerun()

        st.markdown("**Firebreak / clear fuel**")
        bx0 = st.number_input("bx0", 0, cfg.nx - 1, 50, key="bx0")
        by0 = st.number_input("by0", 0, cfg.ny - 1, 0, key="by0")
        bx1 = st.number_input("bx1", 0, cfg.nx - 1, 52, key="bx1")
        by1 = st.number_input("by1", 0, cfg.ny - 1, cfg.ny - 1, key="by1")
        if st.button("Clear fuel (firebreak)"):
            world.clear_fuel(bx0, by0, bx1, by1)
            sim.reset(); st.session_state.cost_series = []
            st.success("Firebreak added"); st.rerun()

    with ec2:
        st.markdown("**Add asset**")
        aname = st.text_input("Name", "New asset")
        akind = st.selectbox("Kind", ASSET_KINDS)
        ax_ = st.number_input("asset x", 0, cfg.nx - 1, cfg.nx // 2, key="ax")
        ay_ = st.number_input("asset y", 0, cfg.ny - 1, cfg.ny // 2, key="ay")
        ar_ = st.number_input("radius", 0, 30, 3, key="ar")
        aval = st.slider("value / intensity", 0.0, 1.0, 1.0, 0.05)
        apop = st.number_input("population (if population asset)", 0, 1_000_000, 0)
        if st.button("Place asset"):
            world.add_asset(Asset(aname, akind, int(ax_), int(ay_),
                                  int(ar_), float(aval), float(apop)))
            st.success(f"Asset '{aname}' placed"); st.rerun()

        st.markdown("**Schedule ignition**")
        ix = st.number_input("ignition x", 0, cfg.nx - 1, cfg.nx // 3, key="ix")
        iy = st.number_input("ignition y", 0, cfg.ny - 1, cfg.ny // 2, key="iy")
        istep = st.number_input("at step", 0, 1000, 0, key="istep")
        irad = st.number_input("ignition radius", 0, 20, 1, key="irad")
        if st.button("Add ignition"):
            world.add_ignition(int(ix), int(iy), int(istep), int(irad))
            st.success("Ignition scheduled"); st.rerun()

    st.divider()
    if st.button("Save scenario to file"):
        out = os.path.join(os.getcwd(), "scenario_export.json")
        io_utils.save_scenario(world, out)
        st.success(f"Saved to {out}")


with tab_layers:
    st.subheader("External data layers")
    layer = st.selectbox("Layer", ["Fuel type", "Fuel load", "Fuel moisture",
                                   "Elevation", "Slope", "Aspect",
                                   "Wind speed", "Protection priority",
                                   "Population density"])
    field = {
        "Fuel type": world.fuel.ftype.astype(float),
        "Fuel load": world.fuel.fload,
        "Fuel moisture": world.fuel.fmoist,
        "Elevation": world.topo.elev,
        "Slope": world.topo.slope,
        "Aspect": world.topo.aspect,
        "Wind speed": world.meteo.wws,
        "Protection priority": world.priority_field(),
        "Population density": world.value.vpop,
    }[layer]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(field, origin="upper", cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(layer)
    st.pyplot(fig, use_container_width=True); plt.close(fig)


with tab_gis:
    st.subheader("GIS raster import")
    st.caption("Import a DEM and an optional fuel class raster. Requires the "
               "optional 'rasterio' package. Slope and aspect are derived from "
               "the DEM and everything is resampled onto the simulation grid.")
    gnx = st.number_input("grid nx", 10, 400, 100)
    gny = st.number_input("grid ny", 10, 400, 80)
    gcell = st.number_input("cell size (m)", 1.0, 1000.0, 30.0)
    dem_file = st.file_uploader("DEM raster (GeoTIFF)", type=["tif", "tiff"], key="dem")
    fuel_file = st.file_uploader("Fuel class raster (GeoTIFF)",
                                 type=["tif", "tiff"], key="fuelr")
    if st.button("Import rasters"):
        try:
            from disasteraware import gis
            paths = {}
            for tag, f in [("dem", dem_file), ("fuel", fuel_file)]:
                if f is not None:
                    p = os.path.join("/tmp", f.name)
                    with open(p, "wb") as fh:
                        fh.write(f.getbuffer())
                    paths[tag] = p
            new_cfg = SimConfig(nx=int(gnx), ny=int(gny), cell_size_m=float(gcell))
            new_world = gis.world_from_rasters(
                new_cfg, dem_path=paths.get("dem"), fuel_path=paths.get("fuel"))
            _new_simulator(new_world)
            st.success("Rasters imported"); st.rerun()
        except ImportError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover
            st.error(f"Import failed: {exc}")


with tab_about:
    st.markdown(
        """
### About this simulator

This is the DisasterAware Simulation Core implemented from Chapter 4 and the
appendices of the thesis. Wildfire evolution is a discrete time, grid based
state space system with the per cell state `s = (B, Fload, I, tau)`.

**Core equations**

- Burning status update: `B_{k+1} = max(B_pers, B_prop, I_ign)` (Eq. 43)
- Propagation influence: wind aligned weighted sum over the 8 neighbourhood (Eq. 46, 48)
- Rate of spread: Rothermel type `r_base * g_moist * g_wind * g_slope * g_aspect` (Eq. 123)
- Fuel mass: combustion plus suppression depletion with non negativity (Eq. 68)
- Suppression to fuel reduction mapping (Eq. 130 to 135)
- Fire intensity proxy `I = B * tanh(beta F + gamma_w W + gamma_s S)` (Eq. 137)

The decision support system is intentionally not part of this core; it will be
built on top of the observable state and the external input interface.
        """
    )


# auto play loop
if st.session_state.playing and not sim.is_quiescent():
    sim.step()
    _record_costs()
    time.sleep(0.15)
    st.rerun()
elif st.session_state.playing and sim.is_quiescent() and sim.ever_burned.any():
    st.session_state.playing = False
