"""DisasterAware interactive simulator dashboard.

Run with:
    streamlit run app/streamlit_app.py

The Map Editor is a mouse driven, live workspace (Section 4.4). On a relief
shaded map that shows the wildfire as it evolves you can:

* brush forests with the selected thesis fuel class (drag to paint)
* drag rectangles for block fuel or firebreaks
* click to drop assets and ignition points, before, at the start of and during
  the simulation

Every action modifies the external input set only; the simulation state is never
written directly (Section 4.1).
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disasteraware import (Simulator, World, SimConfig, Asset, compute_costs,
                           scenarios, io_utils, terrain, viz, FUEL_MODELS,
                           FUEL_NAME_TO_ID)


# --------------------------------------------------- streamlit canvas compat
def _install_canvas_compat():
    """streamlit-drawable-canvas calls streamlit.elements.image.image_to_url,
    which moved to streamlit.elements.lib.image_utils and changed signature in
    recent Streamlit. Bridge the old call onto the new API so the canvas works
    across versions."""
    try:
        import streamlit.elements.image as old_img
        if hasattr(old_img, "image_to_url"):
            return
        from streamlit.elements.lib import image_utils as IU
        from streamlit.elements.lib.layout_utils import LayoutConfig

        def _shim(image, width, clamp, channels, output_format, image_id):
            return IU.image_to_url(image, LayoutConfig(width=int(width)),
                                   clamp, channels, output_format, image_id)

        old_img.image_to_url = _shim
    except Exception:
        pass


_install_canvas_compat()

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

st.set_page_config(page_title="DisasterAware Simulator", layout="wide")

FUEL_TYPES = ["grass", "shrub", "pine_litter", "hardwood"]
ASSET_KINDS = ["building", "critical", "population", "evac_route"]


# --------------------------------------------------------------------- state
def _new_simulator(world: World) -> None:
    st.session_state.world = world
    st.session_state.sim = Simulator(world)
    st.session_state.cost_series = []
    st.session_state.playing = False
    st.session_state.canvas_key = st.session_state.get("canvas_key", 0) + 1
    st.session_state.applied_count = 0


def _ensure_state() -> None:
    if "sim" not in st.session_state:
        _new_simulator(terrain.generate_landscape(
            SimConfig(nx=120, ny=80, cell_size_m=30.0), seed=42))
        st.session_state.world.add_ignition(20, 40, step=0, radius=1)
    st.session_state.setdefault("applied_count", 0)


def _record_costs() -> None:
    st.session_state.cost_series.append(compute_costs(st.session_state.sim).to_dict())


def _display_scale(nx: int) -> int:
    return int(max(4, min(14, 900 // max(nx, 1))))


_ensure_state()
sim: Simulator = st.session_state.sim
world: World = st.session_state.world
cfg = world.config


# ------------------------------------------------------------ canvas parsing
def _clip_xy(gx, gy):
    return (int(np.clip(gx, 0, cfg.nx - 1)), int(np.clip(gy, 0, cfg.ny - 1)))


def _path_points(obj):
    """Extract grid coordinate samples from a fabric freedraw path object."""
    pts = []
    for cmd in obj.get("path", []):
        nums = [v for v in cmd[1:] if isinstance(v, (int, float))]
        for i in range(0, len(nums) - 1, 2):
            pts.append((nums[i], nums[i + 1]))
    return pts


def _apply_canvas_objects(objects, tool, scale, **kw):
    """Translate fabric.js objects from the canvas into world edits."""
    n = 0
    for obj in objects:
        otype = obj.get("type")
        if otype == "rect":
            left, top = obj.get("left", 0), obj.get("top", 0)
            w = obj.get("width", 0) * obj.get("scaleX", 1)
            h = obj.get("height", 0) * obj.get("scaleY", 1)
            x0, y0 = _clip_xy(left / scale, top / scale)
            x1, y1 = _clip_xy((left + w) / scale, (top + h) / scale)
            if tool == "Rectangle fuel":
                world.add_forest_patch(x0, y0, x1, y1, fuel_type=kw["fuel"],
                                       load=kw["load"], moisture=kw["moisture"])
            elif tool == "Firebreak":
                world.clear_fuel(x0, y0, x1, y1)
            n += 1
        elif otype == "path" and tool == "Brush fuel":
            seen = set()
            for px, py in _path_points(obj):
                gx, gy = _clip_xy(px / scale, py / scale)
                if (gx, gy) in seen:
                    continue
                seen.add((gx, gy))
                world.add_forest_disk(gx, gy, kw.get("brush", 2),
                                      fuel_type=kw["fuel"], load=kw["load"],
                                      moisture=kw["moisture"])
            n += 1
        elif otype in ("circle", "path"):
            left, top = obj.get("left", 0), obj.get("top", 0)
            rad = obj.get("radius", obj.get("width", 0) / 2)
            gx, gy = _clip_xy((left + rad) / scale, (top + rad) / scale)
            if tool == "Place asset":
                world.add_asset(Asset(kw["aname"], kw["akind"], gx, gy,
                                      kw["aradius"], kw["avalue"], kw["apop"]))
            elif tool == "Add ignition":
                world.add_ignition(gx, gy, step=kw["istep"], radius=kw["iradius"])
            n += 1
    return n


# -------------------------------------------------------------------- sidebar
st.sidebar.title("DisasterAware")
st.sidebar.caption("Grid based wildfire simulator (thesis Chapter 4)")

with st.sidebar.expander("New map / scenario", expanded=True):
    src = st.radio("Source", ["Realistic landscape", "Built in scenario",
                              "Blank grid"], index=0)
    nx = st.number_input("Resolution X (nx)", 20, 400, int(cfg.nx), 10)
    ny = st.number_input("Resolution Y (ny)", 20, 400, int(cfg.ny), 10)
    cell = st.number_input("Cell size (m)", 1.0, 1000.0, float(cfg.cell_size_m), 5.0)

    if src == "Realistic landscape":
        seed = st.number_input("Seed", 0, 99999, 42)
        relief = st.slider("Relief (m)", 0.0, 1200.0, 450.0, 50.0)
        forest = st.slider("Forest density", 0.0, 0.95, 0.45, 0.05)
        moist = st.slider("Base moisture", 0.02, 0.4, 0.08, 0.01)
        water = st.slider("Water fraction", 0.0, 0.3, 0.06, 0.02)
        if st.button("Generate map", use_container_width=True, type="primary"):
            new_cfg = SimConfig(nx=int(nx), ny=int(ny), cell_size_m=float(cell))
            _new_simulator(terrain.generate_landscape(
                new_cfg, seed=int(seed), relief_m=relief, forest_density=forest,
                base_moisture=moist, water_level=water))
            st.rerun()
    elif src == "Built in scenario":
        scen = st.selectbox("Scenario", list(scenarios.SCENARIOS.keys()))
        if st.button("Load scenario", use_container_width=True, type="primary"):
            _new_simulator(scenarios.SCENARIOS[scen]())
            st.rerun()
    else:
        dfuel = st.selectbox("Default fuel", FUEL_TYPES, index=0)
        if st.button("Create blank grid", use_container_width=True, type="primary"):
            new_cfg = SimConfig(nx=int(nx), ny=int(ny), cell_size_m=float(cell))
            _new_simulator(World.blank(new_cfg, default_fuel=dfuel))
            st.rerun()

    up = st.file_uploader("Load scenario file", type=["json", "yaml", "yml"])
    if up is not None:
        tmp = os.path.join(os.path.dirname(__file__), "_upload_" + up.name)
        with open(tmp, "wb") as fh:
            fh.write(up.getbuffer())
        _new_simulator(io_utils.load_scenario(tmp))
        st.success(f"Loaded {up.name}")
        st.rerun()

with st.sidebar.expander("Weather", expanded=True):
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 30.0,
                           float(world.meteo.wws.mean()), 0.5)
    wind_dir_deg = st.slider("Wind direction (deg)", 0, 360,
                             int(np.degrees(world.meteo.wwd.mean())) % 360, 5)
    if st.button("Apply wind", use_container_width=True):
        world.set_uniform_wind(wind_speed, np.radians(wind_dir_deg))
        st.rerun()

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
    st.caption(f"Step {sim.state.step}   active cells "
               f"{int((sim.state.burning > 0.5).sum())}")

if not HAS_CANVAS:
    st.sidebar.warning("Install streamlit-drawable-canvas for mouse editing:\n\n"
                       "pip install streamlit-drawable-canvas")


# ----------------------------------------------------------------------- main
st.title("DisasterAware Wildfire Simulator")

tab_edit, tab_sim, tab_layers, tab_params, tab_gis, tab_about = st.tabs(
    ["Map editor", "Simulation", "Data layers", "Parameters",
     "GIS import", "About"])


# ============================================================== MAP EDITOR ===
with tab_edit:
    tools_col, view_col = st.columns([1, 3.4])

    with tools_col:
        st.subheader("Tools")
        tool = st.radio("Active tool",
                        ["Brush fuel", "Rectangle fuel", "Firebreak",
                         "Place asset", "Add ignition", "Inspect"],
                        help="Brush: drag to paint. Rectangle / Firebreak: drag "
                             "a box. Asset / Ignition: click a point.")

        kw = {}
        if tool in ("Brush fuel", "Rectangle fuel"):
            kw["fuel"] = st.selectbox("Fuel type (thesis classes)", FUEL_TYPES,
                                      index=2)
            m = FUEL_MODELS[FUEL_NAME_TO_ID[kw["fuel"]]]
            st.caption(f"r_base={m.r_base}  m_ext={m.m_ext}  "
                       f"a_w={m.a_w}  a_s={m.a_s}  b_base={m.b_base}")
            kw["load"] = st.slider("Fuel load", 0.0, 1.0, 1.0, 0.05)
            kw["moisture"] = st.slider("Moisture", 0.0, 0.6, 0.08, 0.01)
            if tool == "Brush fuel":
                kw["brush"] = st.slider("Brush size (cells)", 1, 12, 3)
        elif tool == "Place asset":
            kw["aname"] = st.text_input("Asset name", "Asset")
            kw["akind"] = st.selectbox("Kind", ASSET_KINDS)
            kw["aradius"] = st.number_input("Radius (cells)", 0, 40, 3)
            kw["avalue"] = st.slider("Value / intensity", 0.0, 1.0, 1.0, 0.05)
            kw["apop"] = st.number_input("Population", 0, 1_000_000, 0)
        elif tool == "Add ignition":
            kw["iradius"] = st.number_input("Ignition radius", 0, 20, 1)
            now = st.checkbox("Ignite at current step (live)", value=True)
            kw["istep"] = sim.state.step if now else st.number_input(
                "At step", 0, 5000, 0)
            st.caption("Tip: works during a running simulation too. The click "
                       "schedules an ignition that fires on the next step.")
        else:
            kw["fuel"] = "pine_litter"; kw["load"] = 1.0; kw["moisture"] = 0.08

        drawing_mode = {"Brush fuel": "freedraw", "Rectangle fuel": "rect",
                        "Firebreak": "rect", "Place asset": "point",
                        "Add ignition": "point", "Inspect": "transform"}[tool]

        st.divider()
        live = st.toggle("Live paint (apply on release)", value=True,
                         help="On: every stroke is baked into the map "
                              "immediately. Off: draw several shapes then press "
                              "Apply edits.")
        st.markdown("**View**")
        show_fire = st.checkbox("Live fire", value=True)
        show_value = st.checkbox("Protection priority", value=False)
        show_hs = st.checkbox("Relief shading", value=True)
        show_grid = st.checkbox("Grid", value=False)
        show_labels = st.checkbox("Asset labels", value=True)

        st.divider()
        st.markdown("**Precise entry (x, y)**")
        px = st.number_input("x", 0, cfg.nx - 1, cfg.nx // 2, key="px")
        py = st.number_input("y", 0, cfg.ny - 1, cfg.ny // 2, key="py")
        if st.button("Apply at (x, y)", use_container_width=True):
            if tool == "Add ignition":
                world.add_ignition(int(px), int(py), step=kw.get("istep", 0),
                                   radius=int(kw.get("iradius", 1)))
            elif tool == "Place asset":
                world.add_asset(Asset(kw.get("aname", "Asset"),
                                      kw.get("akind", "building"),
                                      int(px), int(py), int(kw.get("aradius", 3)),
                                      float(kw.get("avalue", 1.0)),
                                      float(kw.get("apop", 0))))
            elif tool in ("Brush fuel", "Rectangle fuel"):
                world.add_forest_disk(int(px), int(py), kw.get("brush", 5),
                                      fuel_type=kw["fuel"], load=kw["load"],
                                      moisture=kw["moisture"])
            elif tool == "Firebreak":
                world.clear_fuel(int(px) - 1, 0, int(px) + 1, cfg.ny - 1)
            st.rerun()

    with view_col:
        scale = _display_scale(cfg.nx)
        bg = viz.render_pil(world, sim=sim, scale=scale, show_fire=show_fire,
                            show_value=show_value, show_hillshade=show_hs,
                            show_grid=show_grid, show_labels=show_labels)
        st.caption(f"Grid {cfg.nx} x {cfg.ny} cells, {cfg.cell_size_m:.0f} m each "
                   f"({cfg.nx * cfg.cell_size_m / 1000:.1f} x "
                   f"{cfg.ny * cfg.cell_size_m / 1000:.1f} km).  "
                   f"Step {sim.state.step}, "
                   f"active fire {int((sim.state.burning > 0.5).sum())} cells.")

        if HAS_CANVAS:
            stroke = {"Brush fuel": "#1f7a1f", "Rectangle fuel": "#1f7a1f",
                      "Firebreak": "#3070b0", "Place asset": "#ffd000",
                      "Add ignition": "#ff5a00", "Inspect": "#888888"}[tool]
            sw = kw.get("brush", 2) * scale if tool == "Brush fuel" else 2
            result = st_canvas(
                fill_color="rgba(255, 160, 0, 0.20)",
                stroke_width=int(sw), stroke_color=stroke,
                background_image=bg, update_streamlit=True,
                height=cfg.ny * scale, width=cfg.nx * scale,
                drawing_mode=drawing_mode,
                point_display_radius=max(3, scale // 2),
                key=f"canvas_{st.session_state.canvas_key}",
            )

            objs = (result.json_data or {}).get("objects", []) if result else []
            new_objs = objs[st.session_state.applied_count:]

            if live and new_objs:
                _apply_canvas_objects(new_objs, tool, scale, **kw)
                st.session_state.canvas_key += 1
                st.session_state.applied_count = 0
                st.rerun()

            b1, b2, b3 = st.columns(3)
            if not live and b1.button("Apply edits", type="primary",
                                      use_container_width=True):
                cnt = _apply_canvas_objects(new_objs, tool, scale, **kw)
                st.session_state.canvas_key += 1
                st.session_state.applied_count = 0
                st.toast(f"Applied {cnt} edit(s)")
                st.rerun()
            if b2.button("Clear drawing", use_container_width=True):
                st.session_state.canvas_key += 1
                st.session_state.applied_count = 0
                st.rerun()
            if b3.button("Step fire", use_container_width=True):
                sim.step(); _record_costs(); st.rerun()
        else:
            st.image(bg, caption="Install streamlit-drawable-canvas to draw "
                                 "with the mouse. Use the (x, y) entry meanwhile.")

    st.divider()
    e1, e2, e3, e4 = st.columns(4)
    if e1.button("Reset fire (keep map)"):
        sim.reset(); st.session_state.cost_series = []; st.rerun()
    if e2.button("Clear all assets"):
        world.assets.clear()
        from disasteraware.layers import ValueLayer
        world.value = ValueLayer.empty(cfg.ny, cfg.nx)
        st.rerun()
    if e3.button("Clear ignitions"):
        world.ignitions.clear(); st.rerun()
    if e4.button("Save scenario"):
        out = os.path.join(os.path.dirname(__file__), "scenario_export.json")
        io_utils.save_scenario(world, out)
        st.success(f"Saved to {out}")


# ============================================================== SIMULATION ===
with tab_sim:
    left, right = st.columns([3, 2])
    with left:
        sv1, sv2, sv3 = st.columns(3)
        sfire = sv1.checkbox("Fire state", value=True, key="sim_fire")
        sval = sv2.checkbox("Priority overlay", value=False, key="sim_val")
        sgrid = sv3.checkbox("Grid", value=False, key="sim_grid")
        scale = _display_scale(cfg.nx)
        st.image(viz.render_pil(world, sim=sim, scale=scale, show_fire=sfire,
                                show_value=sval, show_grid=sgrid,
                                show_labels=True))
        st.caption(f"Step {sim.state.step}    "
                   f"active fire cells {int((sim.state.burning > 0.5).sum())}")

    with right:
        rep = compute_costs(sim)
        st.subheader("Cost report")
        m1, m2 = st.columns(2)
        m1.metric("Burned area (ha)", f"{rep.burned_area_ha:,.1f}")
        m2.metric("Burned forest (ha)", f"{rep.burned_forest_ha:,.1f}")
        m1.metric("Fuel consumed", f"{rep.fuel_consumed_total:,.0f}")
        m2.metric("Active cells", f"{rep.active_fire_cells:,}")
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


# ============================================================== DATA LAYERS ==
with tab_layers:
    st.subheader("External data layers")
    layer = st.selectbox("Layer", ["Fuel type", "Fuel load", "Fuel moisture",
                                   "Elevation", "Slope", "Aspect", "Accessibility",
                                   "Wind speed", "Protection priority",
                                   "Population density", "Building", "Critical"])
    field = {
        "Fuel type": world.fuel.ftype.astype(float),
        "Fuel load": world.fuel.fload,
        "Fuel moisture": world.fuel.fmoist,
        "Elevation": world.topo.elev,
        "Slope": world.topo.slope,
        "Aspect": world.topo.aspect,
        "Accessibility": world.topo.access,
        "Wind speed": world.meteo.wws,
        "Protection priority": world.priority_field(),
        "Population density": world.value.vpop,
        "Building": world.value.vbld,
        "Critical": world.value.vcrit,
    }[layer]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(field, origin="upper", cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(layer)
    st.pyplot(fig, use_container_width=True); plt.close(fig)


# ============================================================== PARAMETERS ===
with tab_params:
    st.subheader("All model parameters (thesis Chapter 4 and appendices)")
    st.caption("Changes take effect on the next step. Press Reset to restart "
               "the fire under new parameters.")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("**Spread (Appendix A)**")
        cfg.spread.theta_ign = st.number_input("theta_ign (Eq. 45)", 0.0, 1.0,
                                               float(cfg.spread.theta_ign), 0.01)
        cfg.spread.w0 = st.number_input("w0 wind ref (Eq. 126)", 0.5, 30.0,
                                        float(cfg.spread.w0), 0.5)
        cfg.spread.eps_fuel = st.number_input("eps_fuel (Eq. 44)", 1e-6, 0.1,
                                              float(cfg.spread.eps_fuel),
                                              format="%.5f")
        cfg.spread.slope_clip_rad = st.number_input("slope clip (rad)", 0.1, 1.5,
                                                    float(cfg.spread.slope_clip_rad), 0.05)
        cfg.spread.diagonal_distance_weighting = st.checkbox(
            "Diagonal 1/sqrt(2) weighting", value=cfg.spread.diagonal_distance_weighting)

    with p2:
        st.markdown("**Suppression (Appendix B)**")
        cfg.suppression.alpha_s = st.number_input("alpha_s (Eq. 130)", 0.0, 1.0,
                                                  float(cfg.suppression.alpha_s), 0.05)
        cfg.suppression.beta_t = st.number_input("beta_t (Eq. 133)", 0.0, 1.0,
                                                 float(cfg.suppression.beta_t), 0.05)
        cfg.suppression.gamma_I = st.number_input("gamma_I (Eq. 134)", 0.0, 5.0,
                                                  float(cfg.suppression.gamma_I), 0.1)
        cfg.suppression.rcap_max = st.number_input("R_cap_max (Eq. 131)", 0.1, 100.0,
                                                   float(cfg.suppression.rcap_max), 0.1)
        st.markdown("**Intensity (Appendix C)**")
        cfg.intensity.beta = st.number_input("beta (Eq. 137)", 0.1, 3.0,
                                             float(cfg.intensity.beta), 0.1)
        cfg.intensity.gamma_w = st.number_input("gamma_w", 0.0, 1.0,
                                                float(cfg.intensity.gamma_w), 0.05)
        cfg.intensity.gamma_s = st.number_input("gamma_s", 0.0, 1.0,
                                                float(cfg.intensity.gamma_s), 0.05)

    with p3:
        st.markdown("**Value weights (Eq. 55)**")
        cfg.value_weights.w_bld = st.number_input("w_bld", 0.0, 1.0,
                                                  float(cfg.value_weights.w_bld), 0.05)
        cfg.value_weights.w_crit = st.number_input("w_crit", 0.0, 1.0,
                                                   float(cfg.value_weights.w_crit), 0.05)
        cfg.value_weights.w_pop = st.number_input("w_pop", 0.0, 1.0,
                                                  float(cfg.value_weights.w_pop), 0.05)
        cfg.value_weights.w_evac = st.number_input("w_evac", 0.0, 1.0,
                                                   float(cfg.value_weights.w_evac), 0.05)
        st.markdown("**Time**")
        cfg.dt = st.number_input("dt", 0.1, 10.0, float(cfg.dt), 0.1)
        cfg.max_steps = int(st.number_input("max steps", 10, 5000,
                                            int(cfg.max_steps), 10))

    with st.expander("Cost model unit values"):
        c = cfg.cost
        q1, q2 = st.columns(2)
        c.cost_per_burned_ha = q1.number_input("Cost / burned ha",
                                               0.0, 1e7, float(c.cost_per_burned_ha))
        c.building_unit_value = q2.number_input("Building unit value",
                                                0.0, 1e9, float(c.building_unit_value))
        c.critical_unit_value = q1.number_input("Critical unit value",
                                                0.0, 1e10, float(c.critical_unit_value))
        c.statistical_life_value = q2.number_input("Statistical life value",
                                                   0.0, 1e9, float(c.statistical_life_value))
        c.population_at_risk_fraction = q1.number_input("Pop. at risk fraction",
                                                        0.0, 1.0,
                                                        float(c.population_at_risk_fraction), 0.01)
        c.suppression_unit_cost = q2.number_input("Suppression unit cost",
                                                  0.0, 1e6, float(c.suppression_unit_cost))


# ============================================================== GIS IMPORT ===
with tab_gis:
    st.subheader("GIS raster import")
    st.caption("Import a DEM and an optional fuel class raster (needs rasterio). "
               "Slope and aspect are derived from the DEM and everything is "
               "resampled onto the chosen grid.")
    gnx = st.number_input("grid nx", 10, 400, 120, key="gnx")
    gny = st.number_input("grid ny", 10, 400, 80, key="gny")
    gcell = st.number_input("cell size (m)", 1.0, 1000.0, 30.0, key="gcell")
    dem_file = st.file_uploader("DEM (GeoTIFF)", type=["tif", "tiff"], key="dem")
    fuel_file = st.file_uploader("Fuel class raster (GeoTIFF)",
                                 type=["tif", "tiff"], key="fuelr")
    if st.button("Import rasters"):
        try:
            from disasteraware import gis
            paths = {}
            for tag, f in [("dem", dem_file), ("fuel", fuel_file)]:
                if f is not None:
                    p = os.path.join(os.path.dirname(__file__), "_gis_" + f.name)
                    with open(p, "wb") as fh:
                        fh.write(f.getbuffer())
                    paths[tag] = p
            new_cfg = SimConfig(nx=int(gnx), ny=int(gny), cell_size_m=float(gcell))
            _new_simulator(gis.world_from_rasters(
                new_cfg, dem_path=paths.get("dem"), fuel_path=paths.get("fuel")))
            st.success("Rasters imported"); st.rerun()
        except ImportError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover
            st.error(f"Import failed: {exc}")


# ================================================================== ABOUT ====
with tab_about:
    st.markdown(
        """
### About

DisasterAware Simulation Core implemented from thesis Chapter 4 and appendices.
Per cell state `s = (B, Fload, I, tau)`; deterministic transition operator Phi.

- Burning status `B_{k+1} = max(B_pers, B_prop, I_ign)` (Eq. 43)
- Wind aligned propagation over the 8 neighbourhood (Eq. 46, 48)
- Rothermel rate of spread `r_base g_moist g_wind g_slope g_aspect` (Eq. 123)
- Fuel mass with combustion and suppression depletion (Eq. 68)
- Suppression to fuel reduction mapping (Eq. 130 to 135)
- Intensity proxy `I = B tanh(beta F + gamma_w W + gamma_s S)` (Eq. 137)

The DSS is not part of this core; it couples only through the external input
interface (`sim.step(resource_override=..., extra_ignition=...)`).
        """
    )
    st.markdown("**Fuel classes (thesis Table A.1 / A.2)**")
    st.table({
        "fuel": [m.name for m in FUEL_MODELS.values()],
        "r_base": [m.r_base for m in FUEL_MODELS.values()],
        "m_ext": [m.m_ext for m in FUEL_MODELS.values()],
        "b_base": [m.b_base for m in FUEL_MODELS.values()],
    })


# ---------------------------------------------------------------- auto play
if st.session_state.playing and not sim.is_quiescent():
    sim.step(); _record_costs(); time.sleep(0.12); st.rerun()
elif st.session_state.playing and sim.is_quiescent() and sim.ever_burned.any():
    st.session_state.playing = False
