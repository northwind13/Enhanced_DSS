"""DisasterAware interactive wildfire simulator dashboard.

Run with:  streamlit run app/streamlit_app.py

Uses page based navigation (one page runs per interaction) so the app stays
fast enough to animate the simulation and to build a DSS on top.
"""

import os
import sys
import time

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
FIREBREAK_TYPES = {"Water": 5, "Bare ground": 0}


# --------------------------------------------------------------------- state
def _new_simulator(world: World) -> None:
    st.session_state.world = world
    st.session_state.sim = Simulator(world)
    st.session_state.cost_series = []
    st.session_state.anim_on = False
    st.session_state.canvas_key = st.session_state.get("canvas_key", 0) + 1
    st.session_state.map_version = st.session_state.get("map_version", 0) + 1
    st.session_state.sim_applied = 0


def _ensure_state() -> None:
    if "sim" not in st.session_state:
        _new_simulator(terrain.generate_landscape(
            SimConfig(nx=100, ny=70, cell_size_m=30.0), seed=42,
            preset="Mountain forest"))
        st.session_state.world.add_ignition(25, 35, step=0, radius=2)
    st.session_state.setdefault("tool", "Fuel")
    st.session_state.setdefault("cost_series", [])
    st.session_state.setdefault("anim_on", False)
    st.session_state.setdefault("canvas_key", 1)
    st.session_state.setdefault("map_version", 1)
    st.session_state.setdefault("edit_undo", [])


def _push_snapshot() -> None:
    """Save the editable layers so a map edit can be undone."""
    w = st.session_state.world
    snap = {
        "ftype": w.fuel.ftype.copy(), "fload": w.fuel.fload.copy(),
        "fload0": w.fuel.fload0.copy(), "fmoist": w.fuel.fmoist.copy(),
        "elev": w.topo.elev.copy(), "slope": w.topo.slope.copy(),
        "aspect": w.topo.aspect.copy(), "access": w.topo.access.copy(),
        "vbld": w.value.vbld.copy(), "vcrit": w.value.vcrit.copy(),
        "vpop": w.value.vpop.copy(), "vevac": w.value.vevac.copy(),
        "roads": None if w.roads is None else w.roads.copy(),
        "assets": list(w.assets), "ignitions": list(w.ignitions)}
    st.session_state.setdefault("edit_undo", []).append(snap)
    st.session_state.edit_undo = st.session_state.edit_undo[-12:]


def _restore_snapshot() -> None:
    stack = st.session_state.setdefault("edit_undo", [])
    if not stack:
        return
    s = stack.pop()
    w = st.session_state.world
    w.fuel.ftype[:] = s["ftype"]; w.fuel.fload[:] = s["fload"]
    w.fuel.fload0[:] = s["fload0"]; w.fuel.fmoist[:] = s["fmoist"]
    w.topo.elev[:] = s["elev"]; w.topo.slope[:] = s["slope"]
    w.topo.aspect[:] = s["aspect"]; w.topo.access[:] = s["access"]
    w.value.vbld[:] = s["vbld"]; w.value.vcrit[:] = s["vcrit"]
    w.value.vpop[:] = s["vpop"]; w.value.vevac[:] = s["vevac"]
    w.roads = None if s["roads"] is None else s["roads"].copy()
    w.assets[:] = s["assets"]; w.ignitions[:] = s["ignitions"]
    st.session_state.sim.reset(); st.session_state.cost_series = []


def _record_costs() -> None:
    st.session_state.cost_series.append(compute_costs(st.session_state.sim).to_dict())


def _fit_scale(nx) -> int:
    return int(max(4, min(16, 900 // max(nx, 1))))


def legend_html() -> str:
    groups = {}
    for grp, lab, hexc in viz.legend_entries():
        groups.setdefault(grp, []).append((lab, hexc))
    html = "<div style='font-size:0.9em'>"
    for grp, items in groups.items():
        html += f"<div style='font-weight:600;margin:6px 0 2px'>{grp}</div>"
        for lab, hexc in items:
            html += ("<div style='display:flex;align-items:center;gap:6px;margin:1px 0'>"
                     f"<span style='width:14px;height:14px;background:{hexc};"
                     "border:1px solid #555;display:inline-block;flex:none'></span>"
                     f"<span>{lab}</span></div>")
    return html + "</div>"


_ensure_state()
sim: Simulator = st.session_state.sim
world: World = st.session_state.world
cfg = world.config


# ------------------------------------------------------------ canvas parsing
def _clip(gx, gy):
    return (int(np.clip(gx, 0, cfg.nx - 1)), int(np.clip(gy, 0, cfg.ny - 1)))


def _path_points(obj):
    pts = []
    for cmd in obj.get("path", []):
        nums = [v for v in cmd[1:] if isinstance(v, (int, float))]
        for i in range(0, len(nums) - 1, 2):
            pts.append((nums[i], nums[i + 1]))
    return pts


def _do_point(gx, gy, kw):
    tool = kw["tool"]
    if tool == "Fuel":
        world.paint_disk(gx, gy, kw.get("brush", 0), FUEL_NAME_TO_ID[kw["ftype"]],
                         load=kw["load"], moisture=kw["moisture"])
    elif tool == "Firebreak":
        world.paint_disk(gx, gy, kw.get("brush", 0), kw["fbid"], load=0.0)
    elif tool == "Access":
        world.add_road_disk(gx, gy, kw.get("brush", 1))
    elif tool == "Asset":
        name = kw["aname"] or ASSET_LABELS.get(kw["akind"], "Asset")
        world.add_asset(Asset(name, kw["akind"], gx, gy, kw["aradius"],
                              kw["avalue"], kw["apop"]))
    elif tool == "Elevation":
        world.bump_terrain(gx, gy, kw.get("brush", 3),
                           kw.get("elev_delta", 40.0), recompute=False)


def _apply_edits(objects, scale, kw):
    tool = kw["tool"]
    n = 0
    for obj in objects:
        otype = obj.get("type")
        if otype == "rect":
            left, top = obj.get("left", 0), obj.get("top", 0)
            w = obj.get("width", 0) * obj.get("scaleX", 1)
            h = obj.get("height", 0) * obj.get("scaleY", 1)
            x0, y0 = _clip(left / scale, top / scale)
            x1, y1 = _clip((left + w) / scale, (top + h) / scale)
            if tool == "Fuel":
                world.paint_rect(x0, y0, x1, y1, FUEL_NAME_TO_ID[kw["ftype"]],
                                 load=kw["load"], moisture=kw["moisture"])
            elif tool == "Firebreak":
                world.paint_rect(x0, y0, x1, y1, kw["fbid"], load=0.0)
            elif tool == "Access":
                world.add_road_rect(x0, y0, x1, y1)
            n += 1
        elif otype == "path":
            # walk the recorded points and fill the gap between each pair so a
            # fast free-hand stroke stays continuous instead of dropping dots
            seen = set()
            prev = None
            for px, py in _path_points(obj):
                cur = (px / scale, py / scale)
                if prev is None:
                    seg = [cur]
                else:
                    d = max(abs(cur[0] - prev[0]), abs(cur[1] - prev[1]))
                    m = max(1, int(np.ceil(d)))
                    seg = [(prev[0] + (cur[0] - prev[0]) * t / m,
                            prev[1] + (cur[1] - prev[1]) * t / m)
                           for t in range(1, m + 1)]
                for sx, sy in seg:
                    gxy = _clip(sx, sy)
                    if gxy in seen:
                        continue
                    seen.add(gxy)
                    _do_point(gxy[0], gxy[1], kw)
                prev = cur
            n += 1
        elif otype == "circle":
            left, top = obj.get("left", 0), obj.get("top", 0)
            rad = obj.get("radius", obj.get("width", 0) / 2)
            gx, gy = _clip((left + rad) / scale, (top + rad) / scale)
            _do_point(gx, gy, kw)
            n += 1
    if kw.get("tool") == "Elevation" and n:
        world.recompute_slope_aspect()
    return n


# -------------------------------------------------------------------- sidebar
_PAGES = ["Simulation", "Map editor", "Data layers", "Parameters",
          "Risk", "Validation", "GIS import", "System Description"]
_PAGE_ICONS = {"Simulation": "\U0001F525", "Map editor": "✏️", "Data layers": "\U0001F5FA️",
               "Parameters": "⚙️", "Risk": "\U0001F4C8", "Validation": "✅",
               "GIS import": "\U0001F30D", "System Description": "\U0001F4D8"}

with st.sidebar:
    st.title("DisasterAware")
    st.caption("Grid based wildfire simulator")

    # --- run controls at the top ---
    st.markdown("**Run**")
    c1, c2 = st.columns(2)
    if c1.button("Step", use_container_width=True):
        sim.step(); _record_costs(); st.rerun()
    if c2.button("Step 10", use_container_width=True):
        [sim.step() for _ in range(10)]; _record_costs(); st.rerun()
    c3, c4 = st.columns(2)
    if c3.button("Run to end", use_container_width=True):
        sim.run(); _record_costs(); st.rerun()
    if c4.button("Reset fire", use_container_width=True):
        sim.reset(); st.session_state.cost_series = []; st.rerun()
    st.toggle("Animate step by step", key="anim_on")
    st.caption(f"Step {sim.state.step}    active fire "
               f"{int((sim.state.burning > 0.5).sum())} cells")

    st.divider()

    # --- page navigation (icon menu) ---
    page = st.radio("Navigation", _PAGES, key="nav_page",
                    format_func=lambda p: f"{_PAGE_ICONS[p]}   {p}",
                    label_visibility="collapsed")

    st.divider()

    # --- map generation at the bottom ---
    with st.expander("New map / scenario", expanded=False):
        src = st.radio("Source", ["Landscape type", "Built in scenario",
                                  "Blank grid"])
        nx = st.number_input("Resolution X (nx)", 20, 400, int(cfg.nx), 10)
        ny = st.number_input("Resolution Y (ny)", 20, 400, int(cfg.ny), 10)
        cell = st.number_input("Cell size (m)", 1.0, 1000.0,
                               float(cfg.cell_size_m), 5.0)
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
            if st.button("Create blank grid", use_container_width=True,
                         type="primary"):
                _new_simulator(World.blank(
                    SimConfig(nx=int(nx), ny=int(ny), cell_size_m=float(cell)),
                    default_fuel=dfuel)); st.rerun()
        up = st.file_uploader("Load scenario file", type=["json", "yaml", "yml"])
        if up is not None:
            tmp = os.path.join(os.path.dirname(__file__), "_upload_" + up.name)
            with open(tmp, "wb") as fh:
                fh.write(up.getbuffer())
            _new_simulator(io_utils.load_scenario(tmp)); st.rerun()

    if not HAS_CANVAS:
        st.warning("Install streamlit-drawable-canvas for mouse editing.")

st.title("DisasterAware Wildfire Simulator")


# ============================================================== SIMULATION ===
def page_simulation():
    view_col, side_col = st.columns([3.4, 1.2])
    with side_col:
        cur_wd = float(world.meteo.wwd.mean())
        with st.expander("Conditions (Simulation)", expanded=True):
            ws = st.slider("Wind speed (m/s)", 0.0, 30.0,
                           float(world.meteo.wws.mean()), 0.5)
            d0 = int(round((np.degrees(cur_wd) % 360) / 45.0)) * 45  # nearest 45
            d0 = min(d0, 360)
            wd_deg = st.select_slider(
                "Wind direction (deg)", options=list(range(0, 361, 45)), value=d0)
            st.image(viz.render_compass(np.radians(wd_deg), ws, size=140),
                     caption="wind blows toward the arrow")
            mo = st.slider("Fuel moisture", 0.0, 0.6,
                           float(world.fuel.fmoist.mean()), 0.01)
            th = st.slider("Ignition threshold", 0.01, 0.5,
                           float(cfg.spread.theta_ign), 0.01)
            emc = st.checkbox("Auto fuel moisture from weather (EMC)", value=False,
                              help="Equilibrium Moisture Content (Simard): compute "
                                   "dead fuel moisture from air temperature and "
                                   "relative humidity. Hotter/drier air -> drier "
                                   "fuel -> faster fire. Off = use the slider value.")
            # apply immediately whenever a condition changes (no Apply button)
            _sig = (round(ws, 3), int(wd_deg), round(mo, 3), round(th, 3), bool(emc))
            if st.session_state.get("cond_sig") != _sig:
                world.meteo.wws[:] = ws
                world.meteo.wwd[:] = np.radians(wd_deg)
                if emc:
                    from disasteraware.fuel_moisture import update_dead_fuel_moisture
                    update_dead_fuel_moisture(world)
                else:
                    world.fuel.fmoist[:] = mo
                cfg.spread.theta_ign = th
                st.session_state.cond_sig = _sig
        with st.expander("Ignition", expanded=False):
            ig_live = st.checkbox("At current step", value=True)
            ig_step = sim.state.step if ig_live else st.number_input(
                "Step", 0, 5000, 0, key="ig_step")
            ig_rad = st.number_input("Radius", 0, 20, 1, key="ig_rad")
        with st.expander("Layers", expanded=False):
            flags = dict(
                show_hillshade=st.checkbox("Relief", True, key="l_relief"),
                show_fire=st.checkbox("Fire", True, key="l_fire"),
                show_value=st.checkbox("Protection value", True, key="l_val"),
                show_roads=st.checkbox("Roads", True, key="l_roads"),
                show_grid=st.checkbox("Grid", False, key="l_grid"),
                show_perimeter=st.checkbox("Fire perimeter", True, key="l_per"),
                show_spread_arrows=st.checkbox("Spread arrows", False, key="l_arr"))
        with st.expander("Legend", expanded=True):
            st.markdown(legend_html(), unsafe_allow_html=True)

    with view_col:
        st.session_state.setdefault("sim_view", "2D map")
        vmode = st.radio("View", ["2D map", "3D terrain"], horizontal=True,
                         label_visibility="collapsed", key="sim_view")
        scale = _fit_scale(cfg.nx)
        playing = st.session_state.get("anim_on", False)
        if vmode == "3D terrain":
            ign3d = st.checkbox("Click terrain to place ignition "
                                "(off = free rotate / zoom)", value=False,
                                key="ign3d")
            st.caption("Drag to rotate, scroll to zoom. The view (rotation and "
                       "zoom) is kept between steps.")
            fig = viz.fire_surface_figure(world, sim=sim,
                                          pick=(ign3d and not playing))
            key3d = f"plot3d_{st.session_state.map_version}"
            if ign3d and not playing:
                ev = st.plotly_chart(fig, use_container_width=True,
                                     config={"scrollZoom": True},
                                     on_select="rerun", selection_mode="points",
                                     key=key3d)
                _place_from_selection(ev, ig_step, int(ig_rad))
            else:
                # pure viewer: no on_select so the camera is never reset on a step
                st.plotly_chart(fig, use_container_width=True,
                                config={"scrollZoom": True}, key=key3d)
        else:
            place = st.checkbox("Click map to place ignition "
                                "(off = scroll to zoom / pan)", value=False,
                                key="sim_place")
            if place and HAS_CANVAS and not playing:
                bg = viz.render_pil(world, sim=sim, scale=scale,
                                    show_labels=True, **flags)
                res = st_canvas(stroke_width=2, stroke_color="#a200de",
                                background_image=bg, update_streamlit=True,
                                height=cfg.ny * scale, width=cfg.nx * scale,
                                drawing_mode="point",
                                point_display_radius=max(3, scale // 2),
                                key=f"simc_{st.session_state.canvas_key}_{scale}")
                objs = (res.json_data or {}).get("objects", []) if res else []
                new = objs[st.session_state.get("sim_applied", 0):]
                if new:
                    for o in new:
                        if o.get("type") == "circle":
                            rad = o.get("radius", 0)
                            gx, gy = _clip((o["left"] + rad) / scale,
                                           (o["top"] + rad) / scale)
                            world.add_ignition(gx, gy, step=ig_step, radius=int(ig_rad))
                    st.session_state.canvas_key += 1
                    st.rerun()
            elif playing:
                # fast image frames while animating (keeps the loop responsive)
                st.image(viz.render_pil(world, sim=sim, scale=scale,
                                        show_labels=True, **flags))
            else:
                # paused: zoomable, pannable plotly with click to ignite; zoom is
                # preserved across steps via the figure uirevision
                ev2 = st.plotly_chart(
                    viz.map_figure_2d(world, sim=sim, scale=scale, **flags),
                    use_container_width=True, on_select="rerun",
                    selection_mode="points",
                    key=f"plot2d_{st.session_state.map_version}",
                    config={"scrollZoom": True,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
                _place_from_selection(ev2, ig_step, int(ig_rad), scale=scale)
        st.caption(f"State S_k at step k = {sim.state.step} "
                   f"(t = {sim.state.step * cfg.dt:.0f}).  Active fire "
                   f"{int((sim.state.burning > 0.5).sum())} cells.")

    _cost_panel()


def _selection_points(ev):
    sel = getattr(ev, "selection", None)
    if sel is None and isinstance(ev, dict):
        sel = ev.get("selection")
    if not sel:
        return []
    if isinstance(sel, dict):
        return sel.get("points", []) or []
    return getattr(sel, "points", []) or []


def _place_from_selection(ev, ig_step, ig_rad, scale=None):
    """Drop an ignition from a plotly click selection. De-duplicates so the same
    selection is not re-applied on every rerun."""
    pts = _selection_points(ev)
    if not pts:
        return
    px, py = pts[0].get("x", 0), pts[0].get("y", 0)
    if scale:
        px, py = px / scale, py / scale
    gx, gy = _clip(round(px), round(py))
    sig = (gx, gy)
    if st.session_state.get("last_sel_sig") == sig:
        return
    st.session_state["last_sel_sig"] = sig
    world.add_ignition(gx, gy, step=ig_step, radius=ig_rad)
    st.toast(f"Ignition at ({gx}, {gy})")
    st.rerun()


_COST_COMPONENTS = [
    ("Forest", "forest_value_loss", "#2e8b57"),
    ("Structures", "building_loss", "#d9822b"),
    ("Critical", "critical_infrastructure_loss", "#c0392b"),
    ("Human", "human_cost", "#8e44ad"),
    ("Suppression", "suppression_cost", "#2c3e50"),
]


def _cost_panel():
    import matplotlib.pyplot as plt
    rep = compute_costs(sim)
    st.divider()
    st.subheader("Cost function and impact")

    # the cost function: each term computed separately, then summed
    st.latex(r"C_{tot}=C_{forest}+C_{struct}+C_{crit}+C_{human}+C_{supp}")
    st.caption("Terms are computed independently over the burned footprint and "
               "summed. Forest = consumed fuel value plus land rehabilitation, "
               "Struct = buildings, Crit = critical facilities, "
               "Human = expected casualties times statistical life value, "
               "Supp = suppression effort.")

    # physical impact
    m = st.columns(4)
    m[0].metric("Burned area (ha)", f"{rep.burned_area_ha:,.1f}")
    m[1].metric("Burned forest (ha)", f"{rep.burned_forest_ha:,.1f}")
    m[2].metric("Population exposed", f"{rep.population_exposed:,.0f}")
    m[3].metric("Expected casualties", f"{rep.expected_casualties:,.1f}")

    # cost breakdown: each component and the total
    d = rep.to_dict()
    cc = st.columns(len(_COST_COMPONENTS) + 1)
    for i, (lab, key, _) in enumerate(_COST_COMPONENTS):
        cc[i].metric(lab, f"{d[key]:,.0f}")
    cc[-1].metric("TOTAL", f"{rep.total_economic_cost:,.0f}")

    series = st.session_state.cost_series
    if len(series) > 1:
        steps = [r["step"] for r in series]
        g1, g2, g3 = st.columns(3)
        # (1) each cost component over time
        with g1:
            f1, a1 = plt.subplots(figsize=(4.6, 2.6))
            for lab, key, col in _COST_COMPONENTS:
                a1.plot(steps, [r[key] for r in series], color=col, label=lab)
            a1.set_title("Cost components"); a1.set_xlabel("step")
            a1.set_ylabel("cost"); a1.legend(fontsize=6, loc="upper left")
            f1.tight_layout(); st.pyplot(f1, use_container_width=True); plt.close(f1)
        # (2) total cost = sum of components
        with g2:
            f2, a2 = plt.subplots(figsize=(4.6, 2.6))
            a2.plot(steps, [r["total_economic_cost"] for r in series],
                    color="#111111", lw=2)
            a2.set_title("Total cost"); a2.set_xlabel("step")
            a2.set_ylabel(r"$C_{tot}$")
            f2.tight_layout(); st.pyplot(f2, use_container_width=True); plt.close(f2)
        # (3) physical growth: burned area and casualties
        with g3:
            f3, a3 = plt.subplots(figsize=(4.6, 2.6))
            a3.plot(steps, [r["burned_area_ha"] for r in series],
                    color="#c0392b", label="burned area")
            a3.set_xlabel("step"); a3.set_ylabel("ha", color="#c0392b")
            a3b = a3.twinx()
            a3b.plot(steps, [r["expected_casualties"] for r in series],
                     color="#8e44ad", label="casualties")
            a3b.set_ylabel("casualties", color="#8e44ad")
            a3.set_title("Physical impact")
            f3.tight_layout(); st.pyplot(f3, use_container_width=True); plt.close(f3)
    else:
        st.caption("Step the simulation to build the cost curves over time.")


# ============================================================== MAP EDITOR ===
def page_editor():
    tools_col, view_col, legend_col = st.columns([1.2, 3.0, 0.9])
    with tools_col:
        st.markdown("**Tool palette**")
        pal = st.columns(2)
        tool_defs = [("Fuel", "Fuel"), ("Firebreak", "Firebreak"),
                     ("Access", "Access"), ("Asset", "Asset"),
                     ("Elevation", "Elevation")]
        for i, (tid, short) in enumerate(tool_defs):
            typ = "primary" if st.session_state.tool == tid else "secondary"
            if pal[i % 2].button(short, key=f"t_{tid}", use_container_width=True,
                                 type=typ):
                st.session_state.tool = tid; st.rerun()
        tool = st.session_state.tool
        st.caption(f"Active: {tool}")
        st.divider()
        kw = {"tool": tool}
        if tool == "Fuel":
            use_std = st.checkbox("Use standard fuel model", value=False,
                                  help="Anderson 13 / Scott and Burgan catalogue "
                                       "mapped to the internal classes.")
            if use_std:
                from disasteraware import fuels_standard
                cat = st.selectbox("Catalogue", ["Anderson 13", "Scott & Burgan"])
                code = st.selectbox("Model", list(fuels_standard.catalog(cat).keys()),
                                    format_func=lambda k:
                                    f"{k} - {fuels_standard.catalog(cat)[k][0]}")
                fid, ld, mo = fuels_standard.resolve(cat, code)
                inv = {v: k for k, v in FUEL_NAME_TO_ID.items()}
                kw["ftype"] = inv[fid]; kw["load"] = ld; kw["moisture"] = mo
                st.caption(f"-> {kw['ftype']}, load {ld}, moisture {mo}")
            else:
                kw["ftype"] = st.selectbox("Fuel type", FUEL_TYPES, index=2)
                mdl = FUEL_MODELS[FUEL_NAME_TO_ID[kw["ftype"]]]
                st.caption(f"r_base={mdl.r_base}  m_ext={mdl.m_ext}  b_base={mdl.b_base}")
                kw["load"] = st.slider("Fuel load", 0.0, 1.0, 1.0, 0.05)
                kw["moisture"] = st.slider("Moisture", 0.0, 0.6, 0.08, 0.01)
            shape = st.radio("Shape", ["Brush", "Rectangle", "Point"], horizontal=True)
        elif tool == "Firebreak":
            fbname = st.selectbox("Firebreak type", list(FIREBREAK_TYPES.keys()))
            kw["fbid"] = FIREBREAK_TYPES[fbname]
            st.caption("Non flammable barrier the fire cannot cross.")
            shape = st.radio("Shape", ["Brush", "Rectangle", "Point"], horizontal=True)
        elif tool == "Access":
            st.caption("Roads set accessibility so resources can reach; non flammable.")
            kw["brush"] = st.slider("Road width", 1, 8, 1)
            shape = st.radio("Shape", ["Brush", "Rectangle"], horizontal=True)
        elif tool == "Asset":
            kw["akind"] = st.selectbox("Asset kind", ASSET_KINDS,
                                       format_func=lambda k: ASSET_LABELS[k])
            kw["aname"] = st.text_input("Name (blank = kind)", "")
            kw["aradius"] = st.number_input("Radius (cells)", 0, 40, 3)
            kw["avalue"] = st.slider("Value", 0.0, 1.0, 1.0, 0.05)
            kw["apop"] = st.number_input("Population", 0, 1_000_000, 0)
            shape = "Point"
        else:  # Elevation
            st.caption("Raise or lower the ground. Uphill slope speeds the fire.")
            direction = st.radio("Action", ["Raise", "Lower"], horizontal=True)
            amt = st.slider("Amount (m)", 5, 200, 40, 5)
            kw["elev_delta"] = float(amt if direction == "Raise" else -amt)
            shape = st.radio("Shape", ["Brush", "Point"], horizontal=True)
        if shape == "Brush":
            kw["brush"] = st.slider("Brush size", 1, 12, 3, key="brushsz")
            drawing_mode = "freedraw"
        elif shape == "Rectangle":
            drawing_mode = "rect"
        else:
            if "brush" not in kw:
                kw["brush"] = st.number_input("Point radius", 0, 10, 1)
            drawing_mode = "point"
        live = st.toggle("Live paint", value=True)

    with legend_col:
        with st.expander("Layers", expanded=True):
            eflags = dict(
                show_hillshade=st.checkbox("Relief", True, key="e_relief"),
                show_fire=st.checkbox("Fire", True, key="e_fire"),
                show_value=st.checkbox("Protection value", False, key="e_val"),
                show_roads=st.checkbox("Roads", True, key="e_roads"),
                show_grid=st.checkbox("Grid", False, key="e_grid"))
        with st.expander("Legend", expanded=True):
            st.markdown(legend_html(), unsafe_allow_html=True)

    with view_col:
        from io import BytesIO
        scale = _fit_scale(cfg.nx)
        bg = viz.render_pil(world, sim=sim, scale=scale, show_labels=True, **eflags)
        # our own controls (the built in canvas toolbar is hidden below): a real
        # PNG download, an edit undo and a drawing clear that all work
        t1, t2, t3 = st.columns(3)
        _buf = BytesIO(); bg.save(_buf, format="PNG")
        t1.download_button("Download PNG", _buf.getvalue(), file_name="map.png",
                           mime="image/png", use_container_width=True)
        if t2.button("Undo edit", use_container_width=True,
                     disabled=not st.session_state.get("edit_undo")):
            _restore_snapshot(); st.session_state.canvas_key += 1; st.rerun()
        if t3.button("Clear drawing", use_container_width=True):
            st.session_state.canvas_key += 1; st.rerun()
        if HAS_CANVAS:
            stroke = {"Fuel": "#1f7a1f", "Firebreak": "#3070b0",
                      "Access": "#b08020", "Asset": "#ffd000",
                      "Elevation": "#7a5230"}[tool]
            sw = kw.get("brush", 2) * scale if drawing_mode == "freedraw" else 2
            # the key also depends on the layer flags so toggling a layer (grid,
            # relief, ...) rebuilds the canvas with the fresh background at once
            flagsig = abs(hash(tuple(sorted(eflags.items())))) % 100000
            ckey = f"canvas_{st.session_state.canvas_key}_{scale}_{flagsig}"
            result = st_canvas(fill_color="rgba(255,160,0,0.20)",
                               stroke_width=int(sw), stroke_color=stroke,
                               background_image=bg, update_streamlit=True,
                               height=cfg.ny * scale, width=cfg.nx * scale,
                               drawing_mode=drawing_mode, display_toolbar=False,
                               point_display_radius=max(3, scale // 2),
                               key=ckey)
            objs = (result.json_data or {}).get("objects", []) if result else []
            if live and objs:
                _push_snapshot(); _apply_edits(objs, scale, kw)
                st.session_state.canvas_key += 1; st.rerun()
            if not live and st.button("Apply edits", type="primary",
                                      use_container_width=True):
                _push_snapshot(); _apply_edits(objs, scale, kw)
                st.session_state.canvas_key += 1; st.rerun()
        else:
            st.image(bg)

    st.divider()
    e1, e2 = st.columns(2)
    if e1.button("Clear all assets"):
        world.assets.clear()
        from disasteraware.layers import ValueLayer
        world.value = ValueLayer.empty(cfg.ny, cfg.nx); st.rerun()
    if e2.button("Clear ignitions"):
        world.ignitions.clear(); st.rerun()


# ============================================================== DATA LAYERS ==
LAYER_EQ = {
    "Fuel type": r"U_{Fuel,k}=[\,F_{type},\,F_{load,0},\,F_{moist,k}\,]^T",
    "Fuel load": r"F_{load,k+1}=\max(0,\,F_{load,k}-F_{load,k}B_kF_{burn,k}-F_{red,k})",
    "Elevation": r"U_{Geo}=[\,G_{elev},\,G_{slope},\,G_{aspect},\,G_{access}\,]^T",
    "Slope": r"g_{slope}=1+a_s\,\tan(G_{slope})",
    "Accessibility": r"\eta_{reach}=e^{-\beta_t R_{time}}\,G_{access}",
    "Wind speed": r"g_{wind}=1+a_w\tanh(W_{ws}/w_0)",
    "Protection priority":
        r"V_{prio}=w_{bld}V_{bld}+w_{crit}V_{crit}+w_{pop}\tilde V_{pop}+w_{evac}\tilde V_{evac}",
    "Ignition time (time to burn)": r"t_{ign}(x,y)=\min\{k: B_k(x,y)=1\}",
}


def page_layers():
    import matplotlib.pyplot as plt
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
            st.info(f"3D needs plotly. {exc}")
    st.divider()
    st.subheader("Fire behaviour (FARSITE / FlamMap style)")
    bkind = st.selectbox("Behaviour output",
                         ["None", "Fireline intensity", "Flame length",
                          "Rate of spread", "Crown fire"])
    bmap = {"Fireline intensity": "fireline_intensity",
            "Flame length": "flame_length", "Rate of spread": "rate_of_spread",
            "Crown fire": "crown_fire"}
    if bkind != "None":
        st.caption("Byram fireline intensity I=H w R, flame length "
                   "L=0.0775 I^0.46; crown fire = burning forest above threshold.")
        st.image(viz.behavior_pil(sim, kind=bmap[bkind],
                                  scale=max(4, 700 // max(cfg.nx, 1))))

    st.divider()
    st.subheader("Input field")
    layer = st.selectbox("Layer", list(LAYER_EQ.keys()))
    st.latex(LAYER_EQ[layer])
    if layer == "Ignition time (time to burn)":
        st.caption("Red = ignites first, blue = ignites later "
                   "(Kose et al., 3D wildfire visualization).")
        st.image(viz.ignition_time_pil(sim, scale=max(4, 700 // max(cfg.nx, 1))))
    else:
        field = {
            "Fuel type": world.fuel.ftype.astype(float),
            "Fuel load": world.fuel.fload, "Elevation": world.topo.elev,
            "Slope": world.topo.slope, "Accessibility": world.topo.access,
            "Wind speed": world.meteo.wws,
            "Protection priority": world.priority_field(),
        }[layer]
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(field, origin="upper", cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title(layer)
        st.pyplot(fig, use_container_width=True); plt.close(fig)


# ============================================================== PARAMETERS ===
def page_params():
    st.subheader("Model parameters")
    st.caption("These shape the fire behaviour model. Day to day conditions live "
               "in the Simulation tab. Hover the ? on any control.")
    with st.expander("Realism modes (optional, off = thesis behaviour)"):
        a, b = st.columns(2)
        cfg.spread.elliptical = a.checkbox(
            "Elliptical spread (Cell2Fire/FARSITE)", value=cfg.spread.elliptical,
            help="Wind elongated elliptical spread instead of the cosine weight.")
        cfg.spread.lb_ratio_wind = b.number_input(
            "Ellipse wind elongation", 0.0, 0.5, float(cfg.spread.lb_ratio_wind), 0.01)
        cfg.spread.spotting = a.checkbox(
            "Ember spotting", value=cfg.spread.spotting,
            help="Intense cells throw embers downwind ahead of the front.")
        cfg.spread.spot_prob = b.number_input(
            "Spot probability", 0.0, 1.0, float(cfg.spread.spot_prob), 0.01)
        cfg.spread.spot_distance = int(a.number_input(
            "Spot distance (cells)", 1, 40, int(cfg.spread.spot_distance)))
        cfg.intensity.crown_fire_threshold = b.number_input(
            "Crown fire intensity threshold", 0.0, 1.0,
            float(cfg.intensity.crown_fire_threshold), 0.05)

    with st.expander("Fire spread and propagation", expanded=True):
        a, b = st.columns(2)
        cfg.spread.w0 = a.number_input("Wind reference speed w0 (m/s)", 0.5, 30.0,
                                       float(cfg.spread.w0), 0.5,
                                       help="Wind where amplification saturates.")
        cfg.spread.eps_fuel = b.number_input("Extinction fuel threshold", 1e-6, 0.1,
                                             float(cfg.spread.eps_fuel), format="%.5f",
                                             help="Fuel below which a cell stops burning.")
        cfg.spread.slope_clip_rad = a.number_input("Slope clip (rad)", 0.1, 1.5,
                                                   float(cfg.spread.slope_clip_rad), 0.05)
        cfg.spread.diagonal_distance_weighting = b.checkbox(
            "Diagonal distance weighting", value=cfg.spread.diagonal_distance_weighting)
    with st.expander("Suppression effectiveness"):
        a, b = st.columns(2)
        cfg.suppression.alpha_s = a.number_input("Global suppression gain", 0.0, 1.0,
                                                 float(cfg.suppression.alpha_s), 0.05)
        cfg.suppression.beta_t = b.number_input("Travel-time decay", 0.0, 1.0,
                                                float(cfg.suppression.beta_t), 0.05)
        cfg.suppression.gamma_I = a.number_input("Intensity resistance", 0.0, 5.0,
                                                 float(cfg.suppression.gamma_I), 0.1)
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


# ====================================================== SYSTEM DESCRIPTION ===
def page_system_description():
    """Full mathematical description of the model (thesis Chapter 4, App. A-C).

    The content lives in app/system_description.py to keep this file small."""
    from system_description import render
    render()


# ============================================================== GIS IMPORT ===
def page_gis():
    st.subheader("GIS raster import")
    st.caption("Load a real elevation and optional fuel raster. Needs rasterio.")
    g1, g2, g3 = st.columns(3)
    gnx = g1.number_input("grid nx", 10, 400, 120, key="gnx")
    gny = g2.number_input("grid ny", 10, 400, 80, key="gny")
    gcell = g3.number_input("cell size (m)", 1.0, 1000.0, 30.0, key="gcell")
    dem_file = st.file_uploader("Elevation raster (GeoTIFF)", type=["tif", "tiff"], key="dem")
    fuel_file = st.file_uploader("Fuel raster (GeoTIFF)", type=["tif", "tiff"], key="fuelr")
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
            st.success("Imported."); st.rerun()
        except ImportError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover
            st.error(f"Import failed: {exc}")


# ------------------------------------------------------------- page dispatch
def page_risk():
    st.subheader("Burn probability (Monte Carlo)")
    st.caption("Runs the simulator many times under perturbed wind and ignition "
               "to estimate, per cell, how often it burns (FlamMap style risk).")
    c1, c2, c3 = st.columns(3)
    n_runs = int(c1.number_input("Runs", 5, 200, 30, 5))
    n_steps = int(c2.number_input("Steps per run", 20, 1000, 150, 10))
    wind_j = c3.number_input("Wind speed jitter (m/s)", 0.0, 15.0, 2.0, 0.5)
    c4, c5 = st.columns(2)
    dir_j = c4.number_input("Wind direction jitter (deg)", 0.0, 90.0, 25.0, 5.0)
    ign_j = int(c5.number_input("Ignition jitter (cells)", 0, 20, 2))
    if not world.ignitions:
        st.warning("Add at least one ignition (Simulation tab) first.")
    if st.button("Run Monte Carlo", type="primary", disabled=not world.ignitions):
        from disasteraware import monte_carlo
        bar = st.progress(0.0)
        prob = monte_carlo.burn_probability(
            world, n_runs=n_runs, n_steps=n_steps, wind_speed_jitter=wind_j,
            wind_dir_jitter_deg=dir_j, ignition_jitter=ign_j,
            progress=lambda f: bar.progress(min(1.0, f)))
        st.session_state.burn_prob = prob
    if "burn_prob" in st.session_state:
        prob = st.session_state.burn_prob
        st.image(viz.probability_pil(world, prob,
                                     scale=max(4, 700 // max(cfg.nx, 1))))
        st.caption(f"Mean burn probability {float(prob.mean()):.2f}; "
                   f"cells with >50% risk: {int((prob > 0.5).sum())}.")


def page_validation():
    st.subheader("Validation against observed data")
    st.caption("Upload the observed burned area for the same landscape and time "
               "window, then score how well the simulation matches it "
               "(Jaccard/IoU, Dice, front position error).")
    st.markdown("Steps: import the real DEM/fuel on the GIS import page, set the "
                "observed wind and ignition, run the fire, then upload the "
                "observed burned area here. See VALIDATION_GUIDE.md for data "
                "sources (EFFIS, MTBS, NASA FIRMS, SRTM, CORINE, ERA5).")
    up = st.file_uploader("Observed burned area (GeoTIFF or PNG, burned = bright)",
                          type=["tif", "tiff", "png", "jpg"])
    thr = st.slider("Burned threshold", 0.0, 1.0, 0.5, 0.05)
    if up is not None:
        import numpy as _np
        ny, nx = cfg.ny, cfg.nx
        name = up.name.lower()
        try:
            if name.endswith((".tif", ".tiff")):
                path = os.path.join(os.path.dirname(__file__), "_obs_" + up.name)
                with open(path, "wb") as fh:
                    fh.write(up.getbuffer())
                from disasteraware import gis
                arr = gis._read_resampled(path, ny, nx, nearest=True)
                arr = (arr - arr.min()) / max(arr.ptp(), 1e-9)
            else:
                from PIL import Image
                im = Image.open(up).convert("L").resize((nx, ny))
                arr = _np.asarray(im, dtype=float) / 255.0
            obs = arr > thr
            from disasteraware import validation, viz as _viz
            metrics = validation.validate_run(sim, obs)
            m = st.columns(4)
            m[0].metric("Jaccard / IoU", f"{metrics['jaccard']:.2f}")
            m[1].metric("Dice", f"{metrics['dice']:.2f}")
            m[2].metric("Hit rate", f"{metrics['hit_rate']:.2f}")
            m[3].metric("False alarm", f"{metrics['false_alarm']:.2f}")
            m[0].metric("Front error mean (m)", f"{metrics['mean_m']:.0f}")
            m[1].metric("Front error p90 (m)", f"{metrics['p90_m']:.0f}")
            st.image(_viz.agreement_pil(sim.ever_burned, obs, world,
                                        scale=max(4, 700 // max(nx, 1))))
            st.caption("Green = correctly burned (hit), red = simulated only "
                       "(false alarm), blue = observed only (missed).")
        except ImportError as exc:
            st.error(f"GeoTIFF needs rasterio. {exc}")
        except Exception as exc:  # pragma: no cover
            st.error(f"Could not read the file: {exc}")
    else:
        st.info("No observed data uploaded yet. Run a fire first, then upload the "
                "observed burned area to score it.")


PAGES = {"Simulation": page_simulation, "Map editor": page_editor,
         "Data layers": page_layers, "Parameters": page_params,
         "Risk": page_risk, "Validation": page_validation,
         "GIS import": page_gis, "System Description": page_system_description}
PAGES[page]()

# animate: advance one step per rerun while playing (fast: only this page runs)
if st.session_state.get("anim_on", False):
    _pending = any(ev.step >= sim.state.step for ev in world.ignitions)
    _finished = sim.is_quiescent() and (
        sim.ever_burned.any() or (not _pending and sim.state.step > 1))
    if not _finished and sim.state.step < cfg.max_steps:
        sim.step(); _record_costs(); time.sleep(0.08); st.rerun()
    else:
        st.session_state.anim_on = False
