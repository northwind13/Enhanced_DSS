"""DisasterAware interactive wildfire simulator dashboard.

Run with:  streamlit run app/streamlit_app.py
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


def _install_canvas_compat():
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
ASSET_LABELS = {"building": "Building", "critical": "Critical facility",
                "population": "Population", "evac_route": "Evacuation route"}
TOOLS = [("Brush fuel", "Brush"), ("Rectangle fuel", "Rect fuel"),
         ("Firebreak", "Firebreak"), ("Road / access", "Road"),
         ("Place asset", "Asset"), ("Inspect", "Inspect")]


def _new_simulator(world: World) -> None:
    st.session_state.world = world
    st.session_state.sim = Simulator(world)
    st.session_state.cost_series = []
    st.session_state.playing = False
    st.session_state.canvas_key = st.session_state.get("canvas_key", 0) + 1
    st.session_state.sim_canvas_key = st.session_state.get("sim_canvas_key", 0) + 1
    st.session_state.applied_count = 0
    st.session_state.sim_applied = 0


def _ensure_state() -> None:
    if "sim" not in st.session_state:
        _new_simulator(terrain.generate_landscape(
            SimConfig(nx=120, ny=80, cell_size_m=30.0), seed=42,
            preset="Mountain forest"))
        st.session_state.world.add_ignition(20, 40, step=0, radius=1)
    st.session_state.setdefault("applied_count", 0)
    st.session_state.setdefault("sim_applied", 0)
    st.session_state.setdefault("sim_canvas_key", 1)
    st.session_state.setdefault("tool", "Brush fuel")
    st.session_state.setdefault("zoom", 8)


def _record_costs() -> None:
    st.session_state.cost_series.append(compute_costs(st.session_state.sim).to_dict())


def legend_html() -> str:
    groups = {}
    for grp, lab, hexc in viz.legend_entries():
        groups.setdefault(grp, []).append((lab, hexc))
    html = "<div style='font-size:0.9em'>"
    for grp, items in groups.items():
        html += f"<div style='font-weight:600;margin:6px 0 2px'>{grp}</div>"
        for lab, hexc in items:
            html += ("<div style='display:flex;align-items:center;gap:6px;"
                     "margin:1px 0'><span style='width:14px;height:14px;"
                     f"background:{hexc};border:1px solid #555;display:inline-block;"
                     f"flex:none'></span><span>{lab}</span></div>")
    return html + "</div>"


def _fit_scale():
    return int(max(4, min(16, 900 // max(cfg.nx, 1))))


def layer_toggles(prefix: str):
    return dict(
        show_hillshade=st.checkbox("Relief", value=True, key=f"{prefix}_relief"),
        show_fire=st.checkbox("Fire", value=True, key=f"{prefix}_fire"),
        show_value=st.checkbox("Protection value", value=False, key=f"{prefix}_val"),
        show_roads=st.checkbox("Roads", value=True, key=f"{prefix}_roads"),
        show_grid=st.checkbox("Grid", value=False, key=f"{prefix}_grid"),
    )


_ensure_state()
sim: Simulator = st.session_state.sim
world: World = st.session_state.world
cfg = world.config


def _clip(gx, gy):
    return (int(np.clip(gx, 0, cfg.nx - 1)), int(np.clip(gy, 0, cfg.ny - 1)))


def _path_points(obj):
    pts = []
    for cmd in obj.get("path", []):
        nums = [v for v in cmd[1:] if isinstance(v, (int, float))]
        for i in range(0, len(nums) - 1, 2):
            pts.append((nums[i], nums[i + 1]))
    return pts


def _apply_edits(objects, tool, scale, **kw):
    n = 0
    for obj in objects:
        otype = obj.get("type")
        if otype == "rect":
            left, top = obj.get("left", 0), obj.get("top", 0)
            w = obj.get("width", 0) * obj.get("scaleX", 1)
            h = obj.get("height", 0) * obj.get("scaleY", 1)
            x0, y0 = _clip(left / scale, top / scale)
            x1, y1 = _clip((left + w) / scale, (top + h) / scale)
            if tool == "Rectangle fuel":
                world.add_forest_patch(x0, y0, x1, y1, fuel_type=kw["fuel"],
                                       load=kw["load"], moisture=kw["moisture"])
            elif tool == "Firebreak":
                world.clear_fuel(x0, y0, x1, y1)
            elif tool == "Road / access":
                world.add_road_rect(x0, y0, x1, y1)
            n += 1
        elif otype == "path":
            seen = set()
            for px, py in _path_points(obj):
                gx, gy = _clip(px / scale, py / scale)
                if (gx, gy) in seen:
                    continue
                seen.add((gx, gy))
                if tool == "Brush fuel":
                    world.add_forest_disk(gx, gy, kw.get("brush", 2),
                                          fuel_type=kw["fuel"], load=kw["load"],
                                          moisture=kw["moisture"])
                elif tool == "Firebreak":
                    world.clear_fuel_disk(gx, gy, kw.get("brush", 2))
                elif tool == "Road / access":
                    world.add_road_disk(gx, gy, kw.get("brush", 1))
            n += 1
        elif otype == "circle":
            left, top = obj.get("left", 0), obj.get("top", 0)
            rad = obj.get("radius", obj.get("width", 0) / 2)
            gx, gy = _clip((left + rad) / scale, (top + rad) / scale)
            if tool == "Place asset":
                name = kw["aname"] or ASSET_LABELS.get(kw["akind"], "Asset")
                world.add_asset(Asset(name, kw["akind"], gx, gy,
                                      kw["aradius"], kw["avalue"], kw["apop"]))
            elif tool == "Firebreak":
                world.clear_fuel_disk(gx, gy, kw.get("fb_point_r", 1))
            elif tool == "Road / access":
                world.add_road_disk(gx, gy, kw.get("brush", 1))
            n += 1
    return n


def _apply_ignitions(objects, scale, step, radius):
    n = 0
    for obj in objects:
        if obj.get("type") == "circle":
            left, top = obj.get("left", 0), obj.get("top", 0)
            rad = obj.get("radius", obj.get("width", 0) / 2)
            gx, gy = _clip((left + rad) / scale, (top + rad) / scale)
            world.add_ignition(gx, gy, step=step, radius=radius)
            n += 1
    return n


# -------------------------------------------------------------------- sidebar
st.sidebar.title("DisasterAware")
st.sidebar.caption("Grid based wildfire simulator")

with st.sidebar.expander("New map / scenario", expanded=True):
    src = st.radio("Source", ["Landscape type", "Built in scenario", "Blank grid"])
    nx = st.number_input("Resolution X (nx)", 20, 400, int(cfg.nx), 10)
    ny = st.number_input("Resolution Y (ny)", 20, 400, int(cfg.ny), 10)
    cell = st.number_input("Cell size (m)", 1.0, 1000.0, float(cfg.cell_size_m), 5.0)
    if src == "Landscape type":
        ltype = st.selectbox("Type", list(terrain.PRESETS.keys()))
        seed = st.number_input("Seed", 0, 99999, 42)
        gen_assets = st.checkbox("Add town, assets and roads", value=True)
        if st.button("Generate map", use_container_width=True, type="primary"):
            _new_simulator(terrain.generate_landscape(
                SimConfig(nx=int(nx), ny=int(ny), cell_size_m=float(cell)),
                seed=int(seed), preset=ltype, with_assets=gen_assets,
                with_roads=gen_assets))
            st.rerun()
    elif src == "Built in scenario":
        scen = st.selectbox("Scenario", list(scenarios.SCENARIOS.keys()))
        if st.button("Load scenario", use_container_width=True, type="primary"):
            _new_simulator(scenarios.SCENARIOS[scen]()); st.rerun()
    else:
        dfuel = st.selectbox("Default fuel", FUEL_TYPES)
        if st.button("Create blank grid", use_container_width=True, type="primary"):
            _new_simulator(World.blank(
                SimConfig(nx=int(nx), ny=int(ny), cell_size_m=float(cell)),
                default_fuel=dfuel)); st.rerun()
    up = st.file_uploader("Load scenario file", type=["json", "yaml", "yml"])
    if up is not None:
        tmp = os.path.join(os.path.dirname(__file__), "_upload_" + up.name)
        with open(tmp, "wb") as fh:
            fh.write(up.getbuffer())
        _new_simulator(io_utils.load_scenario(tmp)); st.success(f"Loaded {up.name}")
        st.rerun()

with st.sidebar.expander("Run controls", expanded=True):
    c1, c2 = st.columns(2)
    if c1.button("Step", use_container_width=True):
        sim.step(); _record_costs(); st.rerun()
    if c2.button("Run 10", use_container_width=True):
        [sim.step() for _ in range(10)]; _record_costs(); st.rerun()
    c3, c4 = st.columns(2)
    if c3.button("Run to end", use_container_width=True):
        sim.run(); _record_costs(); st.rerun()
    if c4.button("Reset", use_container_width=True):
        sim.reset(); st.session_state.cost_series = []; st.rerun()
    st.caption(f"Step {sim.state.step}   active {int((sim.state.burning>0.5).sum())}")

if not HAS_CANVAS:
    st.sidebar.warning("Install streamlit-drawable-canvas for mouse editing.")


st.title("DisasterAware Wildfire Simulator")
tab_sim, tab_edit, tab_layers, tab_params, tab_manual, tab_gis = st.tabs(
    ["Simulation", "Map editor", "Data layers", "Parameters", "Manual",
     "GIS import"])


# ============================================================== SIMULATION ===
with tab_sim:
    view_col, side_col = st.columns([3.4, 1.2])

    with side_col:
        rc1, rc2 = st.columns(2)
        if rc1.button("Step", use_container_width=True, key="s_step"):
            sim.step(); _record_costs(); st.rerun()
        if rc2.button("Step 10", use_container_width=True, key="s_step10"):
            [sim.step() for _ in range(10)]; _record_costs(); st.rerun()
        rc3, rc4 = st.columns(2)
        if rc3.button("Run to end", use_container_width=True, key="s_runend"):
            sim.run(); _record_costs(); st.rerun()
        if rc4.button("Reset fire", use_container_width=True, key="s_reset"):
            sim.reset(); st.session_state.cost_series = []; st.rerun()
        st.session_state.playing = st.toggle(
            "Run (animate step by step)", st.session_state.playing, key="s_play")

        with st.expander("Conditions (change often)", expanded=True):
            ws = st.slider("Wind speed (m/s)", 0.0, 30.0,
                           float(world.meteo.wws.mean()), 0.5)
            wd = st.slider("Wind direction (deg)", 0, 360,
                           int(np.degrees(world.meteo.wwd.mean())) % 360, 5)
            st.image(viz.render_compass(np.radians(wd), ws, size=120))
            mo = st.slider("Fuel moisture", 0.0, 0.6,
                           float(world.fuel.fmoist.mean()), 0.01)
            th = st.slider("Ignition threshold", 0.01, 0.5,
                           float(cfg.spread.theta_ign), 0.01)
            if st.button("Apply conditions", use_container_width=True):
                world.set_uniform_wind(ws, np.radians(wd))
                world.fuel.fmoist[:] = mo
                cfg.spread.theta_ign = th
                st.rerun()

        with st.expander("Ignition", expanded=True):
            ig_live = st.checkbox("At current step", value=True)
            ig_step = sim.state.step if ig_live else st.number_input(
                "Step", 0, 5000, 0, key="ig_step")
            ig_rad = st.number_input("Radius", 0, 20, 1, key="ig_rad")

        with st.expander("Layers", expanded=False):
            flags = layer_toggles("sim")
        with st.expander("Legend", expanded=False):
            st.markdown(legend_html(), unsafe_allow_html=True)

    with view_col:
        vmode = st.radio("View", ["2D map", "3D terrain"], horizontal=True,
                         label_visibility="collapsed")
        if vmode == "3D terrain":
            st.caption("Drag to rotate, scroll to zoom.")
            try:
                st.plotly_chart(viz.fire_surface_figure(world, sim=sim),
                                use_container_width=True)
            except Exception as exc:
                st.info(f"3D needs plotly. {exc}")
        else:
            place = st.checkbox("Click map to place ignition "
                                "(off = scroll to zoom / drag to pan)", value=False)
            scale = _fit_scale()
            if place and HAS_CANVAS:
                bg = viz.render_pil(world, sim=sim, scale=scale, show_labels=True,
                                    **flags)
                res = st_canvas(stroke_width=2, stroke_color="#ff5a00",
                                background_image=bg, update_streamlit=True,
                                height=cfg.ny * scale, width=cfg.nx * scale,
                                drawing_mode="point",
                                point_display_radius=max(3, scale // 2),
                                key=f"simcanvas_{st.session_state.sim_canvas_key}_{scale}")
                objs = (res.json_data or {}).get("objects", []) if res else []
                new = objs[st.session_state.sim_applied:]
                if new:
                    _apply_ignitions(new, scale, ig_step, int(ig_rad))
                    st.session_state.sim_canvas_key += 1
                    st.session_state.sim_applied = 0
                    st.rerun()
            else:
                st.plotly_chart(viz.map_figure_2d(world, sim=sim, scale=scale,
                                                  **flags),
                                use_container_width=True,
                                config={"scrollZoom": True,
                                        "displayModeBar": False})
            st.caption(f"Step {sim.state.step}    active fire "
                       f"{int((sim.state.burning>0.5).sum())} cells.")

    rep = compute_costs(sim)
    st.subheader("Cost report")
    m = st.columns(4)
    m[0].metric("Burned area (ha)", f"{rep.burned_area_ha:,.1f}")
    m[1].metric("Burned forest (ha)", f"{rep.burned_forest_ha:,.1f}")
    m[2].metric("Active cells", f"{rep.active_fire_cells:,}")
    m[3].metric("Fuel consumed", f"{rep.fuel_consumed_total:,.0f}")
    m[0].metric("Building loss", f"{rep.building_loss:,.0f}")
    m[1].metric("Critical infra loss", f"{rep.critical_infrastructure_loss:,.0f}")
    m[2].metric("Population exposed", f"{rep.population_exposed:,.0f}")
    m[3].metric("Expected casualties", f"{rep.expected_casualties:,.1f}")
    m[0].metric("Suppression cost", f"{rep.suppression_cost:,.0f}")
    m[1].metric("Total economic cost", f"{rep.total_economic_cost:,.0f}")

    series = st.session_state.cost_series
    if len(series) > 1:
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


# ============================================================== MAP EDITOR ===
with tab_edit:
    tools_col, view_col, legend_col = st.columns([1.2, 3.0, 0.9])

    with tools_col:
        st.markdown("**Tool palette**")
        pal = st.columns(2)
        for i, (tid, short) in enumerate(TOOLS):
            btn_type = "primary" if st.session_state.tool == tid else "secondary"
            if pal[i % 2].button(short, key=f"tool_{tid}", use_container_width=True,
                                 type=btn_type):
                st.session_state.tool = tid; st.rerun()
        tool = st.session_state.tool
        st.caption(f"Active: {tool}")
        st.divider()

        kw = {}
        if tool in ("Brush fuel", "Rectangle fuel"):
            kw["fuel"] = st.selectbox("Fuel type", FUEL_TYPES, index=2)
            mdl = FUEL_MODELS[FUEL_NAME_TO_ID[kw["fuel"]]]
            st.caption(f"r_base={mdl.r_base}  m_ext={mdl.m_ext}  b_base={mdl.b_base}")
            kw["load"] = st.slider("Fuel load", 0.0, 1.0, 1.0, 0.05)
            kw["moisture"] = st.slider("Moisture", 0.0, 0.6, 0.08, 0.01)
            if tool == "Brush fuel":
                kw["brush"] = st.slider("Brush size", 1, 12, 3)
            drawing_mode = "freedraw" if tool == "Brush fuel" else "rect"
        elif tool == "Firebreak":
            shape = st.selectbox("Shape", ["Brush", "Rectangle", "Point"])
            if shape == "Brush":
                kw["brush"] = st.slider("Brush size", 1, 12, 2); drawing_mode = "freedraw"
            elif shape == "Rectangle":
                drawing_mode = "rect"
            else:
                kw["fb_point_r"] = st.slider("Point radius", 0, 10, 1); drawing_mode = "point"
        elif tool == "Road / access":
            shape = st.selectbox("Shape", ["Brush", "Rectangle"])
            kw["brush"] = st.slider("Road width", 1, 8, 1)
            drawing_mode = "freedraw" if shape == "Brush" else "rect"
        elif tool == "Place asset":
            kw["akind"] = st.selectbox("Kind", ASSET_KINDS,
                                       format_func=lambda k: ASSET_LABELS[k])
            kw["aname"] = st.text_input("Name (blank = kind)", "")
            kw["aradius"] = st.number_input("Radius (cells)", 0, 40, 3)
            kw["avalue"] = st.slider("Value", 0.0, 1.0, 1.0, 0.05)
            kw["apop"] = st.number_input("Population", 0, 1_000_000, 0)
            drawing_mode = "point"
        else:
            drawing_mode = "transform"
        live = st.toggle("Live paint", value=True)

    with view_col:
        scale = _fit_scale()
        st.caption("Drawing canvas (fixed zoom). For scroll zoom use the "
                   "Simulation tab.")
        flags2 = dict(
            show_hillshade=st.session_state.get("edit_relief", True),
            show_fire=st.session_state.get("edit_fire", True),
            show_value=st.session_state.get("edit_val", False),
            show_roads=st.session_state.get("edit_roads", True),
            show_grid=st.session_state.get("edit_grid", False))
        bg = viz.render_pil(world, sim=sim, scale=scale, show_labels=True, **flags2)
        st.caption(f"Grid {cfg.nx} x {cfg.ny}, {cfg.cell_size_m:.0f} m/cell.")
        if HAS_CANVAS and drawing_mode != "transform":
            stroke = {"Brush fuel": "#1f7a1f", "Rectangle fuel": "#1f7a1f",
                      "Firebreak": "#3070b0", "Road / access": "#b08020",
                      "Place asset": "#ffd000"}.get(tool, "#888")
            sw = kw.get("brush", 2) * scale if drawing_mode == "freedraw" else 2
            result = st_canvas(fill_color="rgba(255,160,0,0.20)",
                               stroke_width=int(sw), stroke_color=stroke,
                               background_image=bg, update_streamlit=True,
                               height=cfg.ny * scale, width=cfg.nx * scale,
                               drawing_mode=drawing_mode,
                               point_display_radius=max(3, scale // 2),
                               key=f"canvas_{st.session_state.canvas_key}_{scale}")
            objs = (result.json_data or {}).get("objects", []) if result else []
            new_objs = objs[st.session_state.applied_count:]
            if live and new_objs:
                _apply_edits(new_objs, tool, scale, **kw)
                st.session_state.canvas_key += 1
                st.session_state.applied_count = 0
                st.rerun()
            b1, b2 = st.columns(2)
            if not live and b1.button("Apply edits", type="primary",
                                      use_container_width=True):
                _apply_edits(new_objs, tool, scale, **kw)
                st.session_state.canvas_key += 1
                st.session_state.applied_count = 0
                st.rerun()
            if b2.button("Clear drawing", use_container_width=True):
                st.session_state.canvas_key += 1
                st.session_state.applied_count = 0
                st.rerun()
        else:
            st.image(bg)

    with legend_col:
        with st.expander("Layers", expanded=True):
            layer_toggles("edit")
        with st.expander("Legend", expanded=True):
            st.markdown(legend_html(), unsafe_allow_html=True)

    st.divider()
    e1, e2, e3 = st.columns(3)
    if e1.button("Clear all assets"):
        world.assets.clear()
        from disasteraware.layers import ValueLayer
        world.value = ValueLayer.empty(cfg.ny, cfg.nx); st.rerun()
    if e2.button("Clear ignitions"):
        world.ignitions.clear(); st.rerun()
    if e3.button("Save scenario"):
        out = os.path.join(os.path.dirname(__file__), "scenario_export.json")
        io_utils.save_scenario(world, out); st.success(f"Saved to {out}")


# ============================================================== DATA LAYERS ==
LAYER_EQ = {
    "Fuel type": r"U_{Fuel,k}=[\,F_{type},\,F_{load,0},\,F_{moist,k}\,]^T",
    "Fuel load": r"F_{load,k+1}=\max(0,\,F_{load,k}-F_{load,k}B_kF_{burn,k}-F_{red,k})",
    "Fuel moisture": r"g_{moist}=\max\!\left(0,\,1-\frac{F_{moist}}{m_{ext}}\right)",
    "Elevation": r"U_{Geo}=[\,G_{elev},\,G_{slope},\,G_{aspect},\,G_{access}\,]^T",
    "Slope": r"g_{slope}=1+a_s\,\tan(G_{slope})",
    "Aspect": r"g_{aspect}=1+a_{asp}\cos(G_{aspect}-W_{wd})",
    "Accessibility": r"\eta_{reach}=e^{-\beta_t R_{time}}\,G_{access}",
    "Wind speed": r"g_{wind}=1+a_w\tanh\!\left(\frac{W_{ws}}{w_0}\right)",
    "Protection priority":
        r"V_{prio}=w_{bld}V_{bld}+w_{crit}V_{crit}+w_{pop}\tilde V_{pop}+w_{evac}\tilde V_{evac}",
    "Population density": r"\text{persons per km}^2",
    "Building": r"V_{bld}\in\{0,1\}",
    "Critical": r"V_{crit}\in[0,1]",
}
with tab_layers:
    st.subheader("Terrain")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**2D relief**")
        st.image(viz.terrain_pil(world, scale=max(4, 600 // max(cfg.nx, 1))))
    with tc2:
        st.markdown("**3D surface** (drag to rotate, scroll to zoom)")
        try:
            st.plotly_chart(viz.fire_surface_figure(world, sim=sim),
                            use_container_width=True)
        except Exception as exc:
            st.info(f"3D view needs plotly. {exc}")
    st.divider()
    st.subheader("Input field")
    layer = st.selectbox("Layer", list(LAYER_EQ.keys()))
    st.latex(LAYER_EQ[layer])
    field = {
        "Fuel type": world.fuel.ftype.astype(float),
        "Fuel load": world.fuel.fload, "Fuel moisture": world.fuel.fmoist,
        "Elevation": world.topo.elev, "Slope": world.topo.slope,
        "Aspect": world.topo.aspect, "Accessibility": world.topo.access,
        "Wind speed": world.meteo.wws,
        "Protection priority": world.priority_field(),
        "Population density": world.value.vpop, "Building": world.value.vbld,
        "Critical": world.value.vcrit,
    }[layer]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(field, origin="upper", cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title(layer)
    st.pyplot(fig, use_container_width=True); plt.close(fig)


# ============================================================== PARAMETERS ===
with tab_params:
    st.subheader("Model parameters")
    st.caption("These shape the fire behaviour model. Day to day conditions live "
               "in the Simulation tab. Hover the ? on any control.")
    with st.expander("Fire spread and propagation", expanded=True):
        a, b = st.columns(2)
        cfg.spread.w0 = a.number_input("Wind reference speed w0 (m/s)", 0.5, 30.0,
                                       float(cfg.spread.w0), 0.5,
                                       help="Wind where amplification saturates.")
        cfg.spread.eps_fuel = b.number_input("Extinction fuel threshold", 1e-6, 0.1,
                                             float(cfg.spread.eps_fuel), format="%.5f",
                                             help="Fuel below which a cell stops burning.")
        cfg.spread.slope_clip_rad = a.number_input("Slope clip (rad)", 0.1, 1.5,
                                                   float(cfg.spread.slope_clip_rad), 0.05,
                                                   help="Caps very steep slopes.")
        cfg.spread.diagonal_distance_weighting = b.checkbox(
            "Diagonal distance weighting", value=cfg.spread.diagonal_distance_weighting,
            help="Rounder, more realistic spread shape.")
    with st.expander("Suppression effectiveness"):
        a, b = st.columns(2)
        cfg.suppression.alpha_s = a.number_input("Global suppression gain", 0.0, 1.0,
                                                 float(cfg.suppression.alpha_s), 0.05,
                                                 help="Higher = contained faster.")
        cfg.suppression.beta_t = b.number_input("Travel-time decay", 0.0, 1.0,
                                                float(cfg.suppression.beta_t), 0.05,
                                                help="Higher = distant cells weaker.")
        cfg.suppression.gamma_I = a.number_input("Intensity resistance", 0.0, 5.0,
                                                 float(cfg.suppression.gamma_I), 0.1,
                                                 help="Higher = intense fire resists.")
        cfg.suppression.rcap_max = b.number_input("Reference capacity", 0.1, 100.0,
                                                  float(cfg.suppression.rcap_max), 0.1)
    with st.expander("Fire intensity"):
        a, b = st.columns(2)
        cfg.intensity.beta = a.number_input("Intensity gain", 0.1, 3.0,
                                            float(cfg.intensity.beta), 0.1)
        cfg.intensity.gamma_w = b.number_input("Wind weight", 0.0, 1.0,
                                               float(cfg.intensity.gamma_w), 0.05)
        cfg.intensity.gamma_s = a.number_input("Slope weight", 0.0, 1.0,
                                               float(cfg.intensity.gamma_s), 0.05)
        cfg.intensity.wws_max = b.number_input("Reference max wind (m/s)", 1.0, 60.0,
                                               float(cfg.intensity.wws_max), 1.0)
    with st.expander("Protection priority weights (sum to 1)"):
        a, b = st.columns(2)
        cfg.value_weights.w_crit = a.number_input("Critical facility weight", 0.0, 1.0,
                                                  float(cfg.value_weights.w_crit), 0.05)
        cfg.value_weights.w_pop = b.number_input("Population weight", 0.0, 1.0,
                                                 float(cfg.value_weights.w_pop), 0.05)
        cfg.value_weights.w_bld = a.number_input("Building weight", 0.0, 1.0,
                                                 float(cfg.value_weights.w_bld), 0.05)
        cfg.value_weights.w_evac = b.number_input("Evacuation weight", 0.0, 1.0,
                                                  float(cfg.value_weights.w_evac), 0.05)
    with st.expander("Simulation timing"):
        a, b = st.columns(2)
        cfg.dt = a.number_input("Time step", 0.1, 10.0, float(cfg.dt), 0.1)
        cfg.max_steps = int(b.number_input("Max steps", 10, 5000, int(cfg.max_steps), 10))
    with st.expander("Economic cost model"):
        c = cfg.cost; a, b = st.columns(2)
        c.cost_per_burned_ha = a.number_input("Cost per burned ha", 0.0, 1e7,
                                              float(c.cost_per_burned_ha))
        c.building_unit_value = b.number_input("Building unit value", 0.0, 1e9,
                                               float(c.building_unit_value))
        c.critical_unit_value = a.number_input("Critical facility unit value", 0.0,
                                               1e10, float(c.critical_unit_value))
        c.statistical_life_value = b.number_input("Statistical life value", 0.0, 1e9,
                                                  float(c.statistical_life_value))
        c.population_at_risk_fraction = a.number_input("Population at risk fraction",
                                                       0.0, 1.0,
                                                       float(c.population_at_risk_fraction), 0.01)
        c.suppression_unit_cost = b.number_input("Suppression unit cost", 0.0, 1e6,
                                                 float(c.suppression_unit_cost))


# ================================================================== MANUAL ===
def _eq(latex, defs):
    st.latex(latex)
    st.markdown("\n".join(f"- {d}" for d in defs))


with tab_manual:
    st.subheader("DisasterAware simulation model")
    st.markdown("A grid based, discrete time wildfire model. Each equation is "
                "followed by the definition of the symbols it introduces.")
    st.markdown("#### State and evolution")
    _eq(r"s_k=[\,B_k,\;F_{load,k},\;I_k,\;\tau_k\,]^T",
        [r"$s_k$ - state of one cell at step $k$",
         r"$B_k\in\{0,1\}$ - burning status", r"$F_{load,k}\in[0,1]$ - remaining fuel",
         r"$I_k\in[0,1]$ - intensity proxy", r"$\tau_k$ - time since ignition"])
    _eq(r"S_{k+1}(x,y)=\Phi\big(S_k(x,y),\,F_{in,k}\big)",
        [r"$S_k$ - the whole grid of states", r"$\Phi$ - transition operator",
         r"$F_{in,k}$ - external input set", r"$(x,y)$ - cell coordinates"])
    st.markdown("#### Burning status")
    _eq(r"B_{k+1}=\max\big(B_{pers},\,B_{prop},\,I_{Ign,k}\big)",
        [r"$B_{pers}$ - keeps burning if fuel remains",
         r"$B_{prop}$ - ignited by neighbours", r"$I_{Ign,k}$ - external ignition"])
    _eq(r"\Psi_k(x,y)=\tfrac{1}{8}\sum_{(i,j)\in N_8}B_k(i,j)R_{spread,k}(i,j)g_{dir}",
        [r"$\Psi_k$ - accumulated spread influence", r"$N_8$ - eight neighbours",
         r"$R_{spread}$ - neighbour spread rate",
         r"$g_{dir}=\max(0,\cos(W_{wd}-\theta))$ - wind alignment"])
    st.markdown("#### Rate of spread")
    _eq(r"R_{spread}=r_{base}\,g_{moist}\,g_{wind}\,g_{slope}\,g_{aspect}",
        [r"$r_{base}$ - fuel base spread", r"$g_{moist}=\max(0,1-F_{moist}/m_{ext})$",
         r"$g_{wind}=1+a_w\tanh(W_{ws}/w_0)$",
         r"$g_{slope}=1+a_s\tan G_{slope}$, $g_{aspect}=1+a_{asp}\cos(G_{aspect}-W_{wd})$"])
    st.markdown("#### Fuel, suppression, intensity, ignition time")
    _eq(r"F_{load,k+1}=\max(0,F_{load,k}-F_{load,k}B_kF_{burn,k}-F_{red,k})",
        [r"$F_{burn}=\mathrm{sat}(b_{base}(1-F_{moist}))$ - burn fraction",
         r"$F_{red}=\min(F_{load},\alpha_s\eta_{cap}\eta_{avail}\eta_{reach}\eta_{eff})$"])
    _eq(r"I_{k+1}=B_{k+1}\tanh\big(\beta(\tilde F+\gamma_W\tilde W+\gamma_S\tilde S)\big)",
        [r"$\tilde F,\tilde W,\tilde S$ - normalized fuel, wind, slope",
         r"$\beta,\gamma_W,\gamma_S$ - gains"])
    _eq(r"\tau_{k+1}=0\;/\;\tau_k+\Delta t\;/\;0\quad(\text{new / continuing / out})",
        [r"$\Delta t$ - time step; memory of burning duration"])
    st.markdown("#### Fuel class parameters")
    st.table({"fuel": [m.name for m in FUEL_MODELS.values()],
              "r_base": [m.r_base for m in FUEL_MODELS.values()],
              "m_ext": [m.m_ext for m in FUEL_MODELS.values()],
              "a_w": [m.a_w for m in FUEL_MODELS.values()],
              "a_s": [m.a_s for m in FUEL_MODELS.values()],
              "b_base": [m.b_base for m in FUEL_MODELS.values()]})


# ============================================================== GIS IMPORT ===
with tab_gis:
    st.subheader("GIS raster import")
    st.caption("Load a real elevation and optional fuel raster onto the grid. "
               "Needs the optional rasterio package.")
    g1, g2, g3 = st.columns(3)
    gnx = g1.number_input("grid nx", 10, 400, 120, key="gnx")
    gny = g2.number_input("grid ny", 10, 400, 80, key="gny")
    gcell = g3.number_input("cell size (m)", 1.0, 1000.0, 30.0, key="gcell")
    dem_file = st.file_uploader("Elevation raster (GeoTIFF)", type=["tif", "tiff"],
                                key="dem")
    fuel_file = st.file_uploader("Fuel raster (GeoTIFF)", type=["tif", "tiff"],
                                 key="fuelr")
    if st.button("Import rasters", type="primary"):
        try:
            from disasteraware import gis
            paths = {}
            for tag, f in [("dem", dem_file), ("fuel", fuel_file)]:
                if f is not None:
                    p = os.path.join(os.path.dirname(__file__), "_gis_" + f.name)
                    with open(p, "wb") as fh:
                        fh.write(f.getbuffer())
                    paths[tag] = p
            _new_simulator(gis.world_from_rasters(
                SimConfig(nx=int(gnx), ny=int(gny), cell_size_m=float(gcell)),
                dem_path=paths.get("dem"), fuel_path=paths.get("fuel")))
            st.success("Imported. Use the Layers toggles to inspect."); st.rerun()
        except ImportError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover
            st.error(f"Import failed: {exc}")


if st.session_state.playing and not sim.is_quiescent():
    sim.step(); _record_costs(); time.sleep(0.12); st.rerun()
elif st.session_state.playing and sim.is_quiescent() and sim.ever_burned.any():
    st.session_state.playing = False
