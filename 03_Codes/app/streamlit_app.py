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

import dataclasses

from disasteraware import (Simulator, World, SimConfig, Asset, compute_costs,
                           scenarios, io_utils, terrain, viz, FUEL_MODELS,
                           FUEL_NAME_TO_ID, SpreadParams, SuppressionParams,
                           IntensityParams, ValueWeights, CostParams)

# frozen copy of the default fuel table (A.1 + B.1) for the reset buttons
_THESIS_FUELS = {fid: dataclasses.replace(m) for fid, m in FUEL_MODELS.items()}


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
try:
    from streamlit_plotly_events import plotly_events as _plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

st.set_page_config(page_title="DisasterAware Simulator", layout="wide")

FUEL_TYPES = ["grass", "shrub", "pine_litter", "hardwood"]
ASSET_KINDS = ["building", "critical", "population", "evac_route"]
ASSET_LABELS = {"building": "Building", "critical": "Critical facility",
                "population": "Population", "evac_route": "Evacuation route"}
FIREBREAK_TYPES = {"Water": 5, "Bare ground": 0}


# --------------------------------------------------------------------- state
def _resize_world(w: World, nx2: int, ny2: int) -> World:
    """Resample the whole world to a new grid size (nearest neighbour). All
    layers, roads, assets and ignitions are carried over; slope and aspect are
    recomputed from the resampled elevation."""
    import dataclasses as _dc
    from disasteraware.layers import (MeteoLayer, TopoLayer, FuelLayer,
                                      ValueLayer, ResourceLayer)
    ny1, nx1 = w.shape
    yi = np.minimum((np.arange(ny2) * ny1 / ny2).astype(int), ny1 - 1)
    xi = np.minimum((np.arange(nx2) * nx1 / nx2).astype(int), nx1 - 1)

    def R(a):
        return np.asarray(a)[yi][:, xi].copy()

    cfg2 = SimConfig.from_dict({**w.config.to_dict(),
                                "nx": int(nx2), "ny": int(ny2)})
    w2 = World.blank(cfg2)
    w2.meteo = MeteoLayer(**{k: R(getattr(w.meteo, k))
                             for k in ("temp", "rh", "wws", "wwd",
                                       "gust", "prec")})
    w2.topo = TopoLayer(elev=R(w.topo.elev), slope=R(w.topo.slope),
                        aspect=R(w.topo.aspect), access=R(w.topo.access))
    w2.fuel = FuelLayer(ftype=R(w.fuel.ftype).astype(int),
                        fload=R(w.fuel.fload), fmoist=R(w.fuel.fmoist),
                        fload0=R(w.fuel.fload0))
    w2.value = ValueLayer(vbld=R(w.value.vbld), vcrit=R(w.value.vcrit),
                          vpop=R(w.value.vpop), vevac=R(w.value.vevac))
    w2.resource = ResourceLayer(rcap=R(w.resource.rcap),
                                ravail=R(w.resource.ravail),
                                reff=R(w.resource.reff),
                                rtime=R(w.resource.rtime))
    if getattr(w, "roads", None) is not None:
        w2.roads = R(w.roads)
    sx, sy = nx2 / nx1, ny2 / ny1
    w2.assets = [_dc.replace(a, x=int(a.x * sx), y=int(a.y * sy))
                 for a in w.assets]
    w2.ignitions = [_dc.replace(e, x=int(e.x * sx), y=int(e.y * sy))
                    for e in w.ignitions]
    w2.recompute_slope_aspect()
    return w2


def _new_simulator(world: World) -> None:
    st.session_state.world = world
    st.session_state.sim = Simulator(world)
    st.session_state.cost_series = []
    st.session_state["anim_stop"] = True   # applied before the toggle renders
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

# one-time migration: the elliptical kernel and ember spotting became
# default-ON (engine v0.2). Worlds created earlier carry the old flags in
# their stored config, so flip them once per session; afterwards the
# Parameters checkboxes stay authoritative.
if not st.session_state.get("realism_on_migrated"):
    cfg.spread.elliptical = True
    cfg.spread.spotting = True
    st.session_state["realism_on_migrated"] = True


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


def _click3d(fig, key):
    """Return the grid cell (gx, gy) clicked on the 3D pick-dot grid, or None.

    Streamlit's own chart selections only work on 2D charts, so 3D clicks
    come from the streamlit-plotly-events component. Callers must include a
    nonce (canvas_key) in the key and bump it after applying the click, so
    the remembered event is never re-applied."""
    if not HAS_PLOTLY_EVENTS:
        st.plotly_chart(fig, use_container_width=True,
                        config={"scrollZoom": True}, key=key)
        st.warning("3D clicking needs the 'streamlit-plotly-events' package: "
                   "run  pip install streamlit-plotly-events  in the app "
                   "environment and restart. Use the manual cell entry "
                   "below meanwhile.")
        return None
    place_idx, place = None, None
    for _i, _t in enumerate(fig.data):
        if getattr(_t, "name", "") == "place":
            place_idx, place = _i, _t
            break
    ev = _plotly_events(fig, click_event=True, override_height=470, key=key)
    if ev and place_idx is not None:
        e = ev[0]
        if int(e.get("curveNumber", -1)) == place_idx:
            n = int(e.get("pointNumber", -1))
            if 0 <= n < len(place.x):
                return int(place.x[n]), int(place.y[n])
    return None


def _manual_cell_apply(label, apply_fn, keyp):
    """Fallback x/y entry for 3D interaction without the events package."""
    mc1, mc2, mc3 = st.columns([1, 1, 1.2])
    mgx = mc1.number_input("x", 0, cfg.nx - 1, cfg.nx // 2, key=f"{keyp}_x")
    mgy = mc2.number_input("y", 0, cfg.ny - 1, cfg.ny // 2, key=f"{keyp}_y")
    if mc3.button(label, key=f"{keyp}_go", use_container_width=True):
        apply_fn(int(mgx), int(mgy))


def _edit_from_selection(ev, kw):
    """Apply the active editor tool at a 3D terrain click (point/brush edits).

    The chart key changes after every applied edit, so the selection cannot
    re-fire; repeated clicks on the same cell (e.g. raising terrain step by
    step) therefore work naturally."""
    pts = _selection_points(ev)
    if not pts:
        return
    px, py = pts[0].get("x", 0), pts[0].get("y", 0)
    gx, gy = _clip(round(px), round(py))
    _push_snapshot()
    _do_point(gx, gy, kw)
    if kw["tool"] == "Elevation":
        world.recompute_slope_aspect()
        z = float(world.topo.elev[gy, gx])
        st.toast(f"Elevation at ({gx}, {gy}) -> {z:.0f} m")
    else:
        st.toast(f"{kw['tool']} applied at ({gx}, {gy})")
    st.session_state.canvas_key += 1
    st.rerun()


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


def _wind_speed_label(ws: float) -> str:
    for lim, name in [(0.4, "calm"), (1.5, "light air"), (3.3, "light breeze"),
                      (5.4, "gentle breeze"), (7.9, "moderate breeze"),
                      (10.7, "fresh breeze"), (13.8, "strong breeze"),
                      (17.1, "near gale"), (20.7, "gale"), (24.4, "strong gale"),
                      (28.4, "storm"), (32.6, "violent storm")]:
        if ws <= lim:
            break
    else:
        name = "hurricane force"
    return f"{ws:.1f} m/s  \u2248 {ws*3.6:.0f} km/h  \u00b7  {name}"


def _fmt_sim_time(minutes: float) -> str:
    if minutes < 60:
        return f"{minutes:.0f} min"
    if minutes < 1440:
        h = minutes / 60.0
        return f"{h:.1f} h" if h % 1 else f"{h:.0f} h"
    d = minutes / 1440.0
    return f"{d:.1f} d" if d % 1 else f"{d:.0f} d"


def _wind_compass(cur_deg: float, key: str):
    """Clickable compass dial (cartesian scatter, so point clicks work).

    Returns the clicked direction in degrees (math convention: 0 = east,
    counter-clockwise) or None. The arrow points where the wind blows toward;
    the ring is labelled at 0/90/180/270 degrees."""
    import plotly.graph_objects as go
    opts = list(range(0, 360, 15))
    tho = np.radians(opts)
    near = min(opts, key=lambda o: min((o - cur_deg) % 360, (cur_deg - o) % 360))
    cols = ["#d62728" if o == near else "rgba(130,130,130,0.55)" for o in opts]
    fig = go.Figure()
    tt = np.linspace(0, 2 * np.pi, 91)
    fig.add_trace(go.Scatter(x=np.cos(tt), y=np.sin(tt), mode="lines",
                             line=dict(width=1, color="rgba(128,128,128,0.35)"),
                             hoverinfo="skip"))
    a = np.radians(cur_deg)
    fig.add_trace(go.Scatter(x=[0, 0.72 * np.cos(a)], y=[0, 0.72 * np.sin(a)],
                             mode="lines", line=dict(width=5, color="#1f77b4"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0.80 * np.cos(a)], y=[0.80 * np.sin(a)],
                             mode="markers",
                             marker=dict(symbol="arrow", size=16,
                                         color="#1f77b4",
                                         angle=(90 - cur_deg) % 360),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.cos(tho), y=np.sin(tho), mode="markers",
                             marker=dict(size=10, color=cols),
                             customdata=opts,
                             hovertemplate="%{customdata}\u00b0"
                                           "<extra>click to set</extra>"))
    for ang, (tx, ty) in [(0, (1.24, 0)), (90, (0, 1.24)),
                          (180, (-1.26, 0)), (270, (0, -1.26))]:
        fig.add_annotation(x=tx, y=ty, text=f"{ang}\u00b0", showarrow=False,
                           font=dict(size=11))
    fig.update_layout(height=205, margin=dict(l=6, r=6, t=6, b=6),
                      showlegend=False, dragmode=False,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False, range=[-1.4, 1.4],
                                 fixedrange=True),
                      yaxis=dict(visible=False, range=[-1.4, 1.4],
                                 fixedrange=True, scaleanchor="x"))
    ev = st.plotly_chart(fig, use_container_width=True,
                         config={"displayModeBar": False},
                         on_select="rerun", selection_mode="points", key=key)
    for pt in _selection_points(ev):
        cd = pt.get("customdata")
        if cd is not None:
            try:
                cd = cd[0] if isinstance(cd, (list, tuple)) else cd
                return float(cd) % 360.0
            except (TypeError, ValueError):
                pass
        x_, y_ = pt.get("x"), pt.get("y")
        if x_ is not None and y_ is not None and (x_ or y_):
            return float(np.degrees(np.arctan2(y_, x_))) % 360.0
    return None



# -------------------------------------------------------------------- sidebar
_PAGES = ["Simulation", "Map editor", "Data layers", "Parameters",
          "GIS import", "Validation", "System Description"]
_PAGE_ICONS = {"Simulation": "\U0001F525", "Map editor": "✏️", "Data layers": "\U0001F5FA️",
               "Parameters": "⚙️", "Validation": "✅",
               "GIS import": "\U0001F30D", "System Description": "\U0001F4D8"}

with st.sidebar:
    st.title("DisasterAware")
    st.caption("Enhanced Decision Support System for Wildfire "
               "Disaster Response and Management")
    import disasteraware as _da
    _engine_ok = hasattr(Simulator, "rewind") and hasattr(sim, "rewindable_steps")
    st.caption(f"engine v{getattr(_da, '__version__', '?')}"
               + ("" if _engine_ok else " \u00b7 \u26a0 outdated in memory"))
    if not _engine_ok:
        st.error("The simulation engine changed on disk but the running "
                 "process still uses the old version (Python caches "
                 "imported modules). Close the app completely and start "
                 "it again (run_dashboard.bat). Until then rewind and "
                 "time-step scaling stay inactive.")

    # --- simulation control panel ---
    if st.session_state.pop("anim_stop", False):
        st.session_state.anim_on = False
    with st.container(border=True):
        st.markdown("**Simulation**")
        xsteps = int(st.number_input(
            "Steps per click (X)", 1, 1000, 10, key="step_x",
            help="How many steps the 'Step X' button advances at once."))
        c1, c2 = st.columns(2)
        if c1.button("Step", use_container_width=True,
                     help="Advance the fire by one time step."):
            sim.step(); _record_costs(); st.rerun()
        if c2.button(f"Step {xsteps}", use_container_width=True,
                     help="Advance X steps at once."):
            [sim.step() for _ in range(xsteps)]; _record_costs(); st.rerun()
        st.toggle("Animate step by step", key="anim_on",
                  help="Advance automatically, one step per refresh, until "
                       "the fire is over.")
        c3, c4 = st.columns(2)
        if c3.button("Run to end", use_container_width=True,
                     help="Runs until the fire is out, the step cap "
                          "(max_steps) is reached, or a 30 s compute budget "
                          "is spent \u2014 press again to continue."):
            _bar = st.progress(0.0, text="Running \u2026")
            _t0 = time.time()
            _start = sim.state.step
            _limit = int(cfg.max_steps)
            _reason = f"step cap {_limit} reached"
            while sim.state.step < _limit:
                _d = sim.step(); _record_costs()
                if (_d.n_burning == 0 and sim.state.step > 1
                        and sim.ever_burned.any()
                        and not any(ev.step >= sim.state.step
                                    for ev in world.ignitions)):
                    _reason = "fire is out"
                    break
                if time.time() - _t0 > 30.0:
                    _reason = "30 s budget \u2014 press again to continue"
                    break
                _bar.progress(min(1.0, (sim.state.step - _start)
                                  / max(1, _limit - _start)),
                              text=f"step {sim.state.step} \u00b7 "
                                   f"active {_d.n_burning} cells")
            _bar.empty()
            st.toast(f"Stopped at step {sim.state.step}: {_reason}")
            st.rerun()
        if c4.button("Reset fire", use_container_width=True,
                     help="Clear the fire and the cost series; the map and "
                          "all edits stay."):
            sim.reset(); st.session_state.cost_series = []; st.rerun()
        st.caption(f"Step {sim.state.step} \u00b7 "
                   f"t = {_fmt_sim_time(sim.state.step * float(getattr(cfg, 'step_minutes', 30.0)))} \u00b7 "
                   f"active fire {int((sim.state.burning > 0.5).sum())} cells")
        avail = (sim.rewindable_steps
                 if hasattr(sim, "rewindable_steps") else [])
        lo = int(avail[0]) if avail else 0
        hi = int(sim.state.step)
        if avail and hi > lo:
            st.markdown("**Rewind** \u2014 go back, change conditions, replay")
            def _do_rewind(_k):
                if sim.rewind(int(_k)):
                    st.session_state.cost_series = [
                        r for r in st.session_state.cost_series
                        if r.get("step", 0) <= int(_k)]
                    st.rerun()
                else:
                    st.warning("Snapshot for that step is no longer stored.")

            rsl = st.slider("Rewind to step", lo, hi, hi, 1, key="rw_sl",
                            label_visibility="collapsed",
                            help="Dragging the slider rewinds immediately.")
            if int(rsl) < hi:
                _do_rewind(int(rsl))   # slider rewinds instantly
            rc1, rc2 = st.columns([1.4, 1])
            rnum = rc1.number_input("k", lo, hi, hi, 1, key="rw_num",
                                    label_visibility="collapsed",
                                    help="Type the exact step, then press Go.")
            if rc2.button("Go", use_container_width=True,
                          disabled=int(rnum) >= hi,
                          help="Restore the full state at step k. The "
                               "history after it is discarded; replay with "
                               "new wind, moisture or resources."):
                _do_rewind(int(rnum))

    st.divider()

    # --- page navigation (icon menu) ---
    if st.session_state.get("nav_page") not in _PAGES:
        st.session_state.pop("nav_page", None)
    page = st.radio("Navigation", _PAGES, key="nav_page",
                    format_func=lambda p: f"{_PAGE_ICONS[p]}   {p}",
                    label_visibility="collapsed")

    st.divider()


    if not HAS_CANVAS:
        st.warning("Install streamlit-drawable-canvas for mouse editing.")

st.title("Enhanced Decision Support System for Wildfire Disaster "
         "Response and Management (DisasterAware)")


# ============================================================== SIMULATION ===
def page_simulation():
    view_col, side_col = st.columns([3.0, 1.9])
    with side_col:
        sc1, sc2 = st.columns(2)

        # ---------- column 1: conditions + ignition ----------
        with sc1:
            with st.expander("Conditions", expanded=True):
                ws = st.slider(
                    "$W_{ws}$ — wind speed (m/s)", 0.0, 30.0,
                    float(world.meteo.wws.mean()), 0.5,
                    help="Uniform wind over the whole map. Spread saturates "
                         "toward w0 (Parameters); above ~21 m/s is storm force.")
                st.caption(_wind_speed_label(ws))
                cur_deg = float(np.degrees(world.meteo.wwd.mean())) % 360.0
                st.markdown("$W_{wd}$ — **wind direction**: click the "
                            "dial; the arrow shows where the wind blows "
                            "**toward**")
                sel = _wind_compass(cur_deg,
                                    key=f"wind_dial_{st.session_state.map_version}")
                if sel is not None and min((sel - cur_deg) % 360,
                                           (cur_deg - sel) % 360) > 0.5:
                    world.meteo.wwd[:] = np.radians(sel)
                    st.rerun()
                st.caption(f"Current: {cur_deg:.0f}\u00b0 "
                           "(0\u00b0 = east, 90\u00b0 = north)")
                emc = st.checkbox(
                    "Auto fuel moisture from weather (EMC)", value=True,
                    help="Equilibrium Moisture Content (Simard 1968): computes "
                         "dead surface fuel moisture from air temperature and "
                         "relative humidity every step. Hotter and drier air "
                         "gives drier fuel and a faster fire. When on, the "
                         "slider below is ignored.")
                mo = st.slider(
                    "$F_{moist}$ — fuel moisture, whole map (mass fraction)",
                    0.0, 0.6,
                    float(world.fuel.fmoist.mean()), 0.01, disabled=emc,
                    help="Surface fuel moisture as a mass fraction, applied "
                         "uniformly to every cell. It is a daily weather "
                         "condition, which is why it lives here; spatially "
                         "varying moisture can be painted in the Map editor "
                         "(moving this slider overwrites painted values). "
                         "At the extinction moisture m_ext of a fuel class "
                         "the spread stops entirely.")
                st.markdown("**Time per step** \u2014 what one step "
                            "$k \\to k{+}1$ represents")
                tc1, tc2 = st.columns([1.1, 1.1])
                _sm = float(getattr(cfg, "step_minutes", 30.0))
                if _sm % 1440 == 0 and _sm >= 1440:
                    _u0, _v0 = "days", _sm / 1440
                elif _sm % 60 == 0 and _sm >= 60:
                    _u0, _v0 = "hours", _sm / 60
                else:
                    _u0, _v0 = "minutes", _sm
                tval = tc1.number_input("Value", 1.0, 1440.0, float(_v0), 1.0,
                                        key="dt_val")
                tunit = tc2.selectbox("Unit", ["minutes", "hours", "days"],
                                      index=["minutes", "hours",
                                             "days"].index(_u0), key="dt_unit")
                cfg.step_minutes = tval * {"minutes": 1, "hours": 60,
                                           "days": 1440}[tunit]
                st.caption(f"Step {sim.state.step} = "
                           f"{_fmt_sim_time(sim.state.step * cfg.step_minutes)} "
                           "of fire time. Changing this rescales the "
                           "dynamics (System Description Sec. 9, note 8): a "
                           "longer step simply advances the clock further, "
                           "the physical speed in m/min stays the same.")
                # live head-fire speed so the m/min meaning is always visible
                from disasteraware.spread import rate_of_spread as _rosf
                _ros = _rosf(world.fuel, world.topo, world.meteo, cfg.spread)
                _pos = _ros[_ros > 0]
                if _pos.size:
                    _v = float(np.percentile(_pos, 95))
                    _cellm = cfg.cell_size_m / 30.0
                    _vm = _v * _cellm          # m/min at the reference
                    st.caption(f"Head-fire potential right now: \u2248 "
                               f"{_vm:.0f} m/min ({_vm*0.06:.1f} km/h) on "
                               "the fastest cells \u2014 independent of "
                               "the step length; per step the front moves "
                               f"\u2248 {_vm*cfg.step_minutes/cfg.cell_size_m:.2f} "
                               "cells.")
                # apply weather immediately, each control independently, so an
                # unrelated change never overwrites a painted field
                if abs(ws - float(world.meteo.wws.mean())) > 1e-9:
                    world.meteo.wws[:] = ws
                if emc:
                    from disasteraware.fuel_moisture import update_dead_fuel_moisture
                    update_dead_fuel_moisture(world)
                else:
                    _last_mo = st.session_state.get("last_mo")
                    if _last_mo is not None and abs(mo - _last_mo) > 1e-9:
                        world.fuel.fmoist[:] = mo
                    st.session_state["last_mo"] = mo
            with st.expander("Ignition", expanded=True):
                ig_live = st.checkbox(
                    "At current step", value=True,
                    help="Place the ignition at the current step; untick to "
                         "schedule it for a future step.")
                ig_step = sim.state.step if ig_live else st.number_input(
                    "Step", 0, 5000, 0, key="ig_step")
                ig_rad = st.number_input(
                    "Radius (cells)", 0, 20, 1, key="ig_rad",
                    help=f"0 = a single cell; r ignites a disk of r cells "
                         f"around the click. One cell = {cfg.cell_size_m:g} m "
                         "on the ground.")
                th = st.number_input(
                    "$\\Theta_{ign}$ — ignition threshold", 0.005, 1.0,
                    float(cfg.spread.theta_ign), 0.005, format="%.3f",
                    help="Activation threshold on the ignition influence "
                         "buildup $A_k$ (System Description Sec. 4). Unit: "
                         "accumulated influence = \u215b cell-widths of "
                         "front travel. Default $0.125=1/8$, which makes a "
                         "wind-aligned front advance at exactly "
                         "$R_{spread}$ cells per step. Lower = easier "
                         "ignition of new cells.")
                if abs(th - float(cfg.spread.theta_ign)) > 1e-12:
                    cfg.spread.theta_ign = th

        # ---------- column 2: legend + layers ----------
        with sc2:
            with st.expander("Legend", expanded=True):
                st.markdown(legend_html(), unsafe_allow_html=True)
            with st.expander("Layers", expanded=True):
                # layer keys (ly_*) are shared with the Map editor, so a
                # selection made on either page holds on both
                flags = dict(
                    show_hillshade=st.checkbox("Relief", True, key="ly_relief"),
                    show_fire=st.checkbox("Fire", True, key="ly_fire"),
                    show_value=st.checkbox(
                        "Protection value", True, key="ly_val",
                        help="Tints asset cells by the protection priority "
                             "$V_{prio}$ (System Description Sec. 2.5): pale "
                             "pink = lower, deep purple = higher priority. "
                             "Nothing shows until the map has buildings, "
                             "critical facilities or population."),
                    show_roads=st.checkbox("Roads", True, key="ly_roads"),
                    show_grid=st.checkbox("Grid", False, key="ly_grid"))
            if flags["show_value"] and float(world.value.vbld.max()
                    + world.value.vcrit.max() + world.value.vpop.max()) <= 0:
                st.caption("\u26a0 Protection value: this map has no "
                           "buildings, critical facilities or population "
                           "yet, so the overlay shows nothing. Add them "
                           "with the Asset tool in the Map editor.")

    with view_col:
        _views = ["2D map", "3D terrain"]
        _vcur = st.session_state.get("sim_view_sel", "2D map")
        vmode = st.radio("View", _views,
                         index=_views.index(_vcur) if _vcur in _views else 0,
                         horizontal=True, label_visibility="collapsed")
        st.session_state["sim_view_sel"] = vmode
        scale = _fit_scale(cfg.nx)
        playing = st.session_state.get("anim_on", False)
        if vmode == "3D terrain":
            ign3d = st.checkbox("Click terrain to place ignition "
                                "(off = free rotate / zoom)", value=False,
                                key="ign3d")
            st.caption("Drag to rotate, scroll to zoom. The view (rotation and "
                       "zoom) is kept between steps.")
            fig = viz.fire_surface_figure(world, sim=sim,
                                          pick=(ign3d and not playing),
                                          pick_label="click to place ignition")
            key3d = f"plot3d_{st.session_state.map_version}"

            def _ignite_at(gx, gy):
                world.add_ignition(gx, gy, step=ig_step, radius=int(ig_rad))
                st.toast(f"Ignition at ({gx}, {gy})")
                st.session_state.canvas_key += 1
                st.rerun()

            if ign3d and not playing:
                gxy = _click3d(fig,
                               key=f"{key3d}_{st.session_state.canvas_key}")
                if gxy:
                    _ignite_at(*gxy)
                if not HAS_PLOTLY_EVENTS:
                    _manual_cell_apply("Ignite at (x, y)", _ignite_at, "sim3d")
            else:
                # pure viewer: camera and zoom survive the steps
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
                                drawing_mode="point", display_toolbar=False,
                                point_display_radius=max(3, scale // 2),
                                key=f"simc_{st.session_state.canvas_key}_{scale}")
                if world.ignitions and st.button(
                        "Remove last ignition",
                        help="Deletes the most recently placed ignition "
                             "marker."):
                    world.ignitions.pop()
                    st.session_state.canvas_key += 1
                    st.rerun()
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


_J_TERMS = [
    ("J\u1d47\u1d58\u02b3\u207f", "forest_value_loss", "#2e8b57", "land + forest"),
    ("J\u1d5b\u1d43\u02e1", "building_loss", "#d9822b", "structures"),
    ("J\u2071\u207f\u1da0", "critical_infrastructure_loss", "#c0392b", "critical infra"),
    ("J\u1d56\u1d52\u1d56", "human_cost", "#8e44ad", "population"),
    ("J\u02e2\u1d58\u1d56", "suppression_cost", "#2c3e50", "suppression"),
]


def _cost_panel():
    import matplotlib.pyplot as plt
    rep = compute_costs(sim)
    d = rep.to_dict()
    st.divider()
    st.subheader("Cost function $J_k$")
    st.latex(r"J_k=w_1 J_k^{burn}+w_2 J_k^{val}+w_3 J_k^{inf}"
             r"+w_4 J_k^{pop}+w_5 J_k^{sup}\;(+\,w_6 J_k^{del})")
    st.caption("Realized cost-plus-loss of the run so far (System Description "
               "Sec. 14, all weights 1 in a common currency). The delay term "
               "$J^{del}$ scores candidate allocations inside the decision "
               "layer and is not part of the realized cost.")

    # physical impact
    m = st.columns(4)
    m[0].metric("Burned area (ha)", f"{rep.burned_area_ha:,.1f}")
    m[1].metric("Burned forest (ha)", f"{rep.burned_forest_ha:,.1f}")
    m[2].metric("Population exposed", f"{rep.population_exposed:,.0f}")
    m[3].metric("Expected casualties", f"{rep.expected_casualties:,.2f}")

    # J terms and the total
    cc = st.columns(len(_J_TERMS) + 1)
    for i, (lab, key, _c, sub) in enumerate(_J_TERMS):
        cc[i].metric(f"{lab} \u00b7 {sub}", f"{d[key]:,.0f}")
    cc[-1].metric("J \u00b7 TOTAL", f"{rep.total_economic_cost:,.0f}")

    series = st.session_state.cost_series
    if len(series) > 1:
        steps = [r["step"] for r in series]
        st.markdown("##### $J$ terms over time \u2014 one chart per term")
        # NOTE: chart texts use unicode superscripts instead of mathtext on
        # purpose: the installed matplotlib crashes on "$...$" labels
        titles = {"forest_value_loss": "J\u1d47\u1d58\u02b3\u207f",
                  "building_loss": "J\u1d5b\u1d43\u02e1",
                  "critical_infrastructure_loss": "J\u2071\u207f\u1da0",
                  "human_cost": "J\u1d56\u1d52\u1d56",
                  "suppression_cost": "J\u02e2\u1d58\u1d56"}
        cells = st.columns(3)
        panels = [(lab, key, col, titles[key]) for lab, key, col, _ in _J_TERMS]
        panels.append(("J", "total_economic_cost", "#111111", "J (total)"))
        for i, (lab, key, col, ttl) in enumerate(panels):
            with cells[i % 3]:
                f, a = plt.subplots(figsize=(3.4, 2.0))
                a.plot(steps, [r[key] for r in series], color=col, lw=1.8)
                a.fill_between(steps, [r[key] for r in series],
                               color=col, alpha=0.15)
                a.set_title(ttl, fontsize=11)
                a.set_xlabel("step k", fontsize=8)
                a.tick_params(labelsize=7)
                a.grid(alpha=0.25)
                f.tight_layout()
                st.pyplot(f, use_container_width=True)
                plt.close(f)


# =============================================================== MAP EDITOR ==
def page_editor():
    with st.expander("New map / scenario \u2014 generate, load, import",
                     expanded=False):
        st.caption("Procedural landscapes for experiments. For a real region "
                   "(e.g. an actual Turkish province) import its elevation and "
                   "fuel rasters on the GIS import page; everything else "
                   "(assets, roads, ignition) is then edited here.")
        src = st.radio("Source", ["Landscape type", "Built in scenario",
                                  "Blank grid"])
        nx = st.number_input("Resolution X (nx)", 20, 600,
                             max(int(cfg.nx), 160), 10)
        ny = st.number_input("Resolution Y (ny)", 20, 600,
                             max(int(cfg.ny), 100), 10)
        cell = st.number_input("Cell size (m)", 1.0, 1000.0,
                               float(cfg.cell_size_m), 5.0)
        if src == "Landscape type":
            ltype = st.selectbox("Type", list(terrain.PRESETS.keys()))
            seed = st.number_input("Seed", 0, 99999, 42)
            gen_assets = st.checkbox("Add town, assets and roads", value=True)
            if st.button("Generate map", use_container_width=True,
                         type="primary"):
                _new_simulator(terrain.generate_landscape(
                    SimConfig(nx=int(nx), ny=int(ny), cell_size_m=float(cell)),
                    seed=int(seed), preset=ltype, with_assets=gen_assets,
                    with_roads=gen_assets))
                st.rerun()
        elif src == "Built in scenario":
            scen = st.selectbox("Scenario", list(scenarios.SCENARIOS.keys()))
            if st.button("Load scenario", use_container_width=True,
                         type="primary"):
                _new_simulator(scenarios.SCENARIOS[scen]()); st.rerun()
        else:
            dfuel = st.selectbox("Default fuel", FUEL_TYPES)
            if st.button("Create blank grid", use_container_width=True,
                         type="primary"):
                _new_simulator(World.blank(
                    SimConfig(nx=int(nx), ny=int(ny), cell_size_m=float(cell)),
                    default_fuel=dfuel)); st.rerun()
        st.divider()
        st.markdown("**Resize current map** \u2014 keeps everything on it, "
                    "resamples the grid")
        rz1, rz2, rz3 = st.columns([1, 1, 1])
        rnx = rz1.number_input("nx", 20, 600, int(cfg.nx), 10, key="res_nx")
        rny = rz2.number_input("ny", 20, 600, int(cfg.ny), 10, key="res_ny")
        rz3.markdown("<div style='height:1.75em'></div>",
                     unsafe_allow_html=True)
        if rz3.button("Resize", use_container_width=True,
                      disabled=(int(rnx) == cfg.nx and int(rny) == cfg.ny)):
            _new_simulator(_resize_world(world, int(rnx), int(rny)))
            st.rerun()
        st.divider()
        up = st.file_uploader("Load scenario file", type=["json", "yaml", "yml"])
        if up is not None:
            tmp = os.path.join(os.path.dirname(__file__), "_upload_" + up.name)
            with open(tmp, "wb") as fh:
                fh.write(up.getbuffer())
            _new_simulator(io_utils.load_scenario(tmp)); st.rerun()

    tools_col, view_col, legend_col = st.columns([1.2, 3.0, 0.9])
    with tools_col:
        st.markdown("**Tool palette**")
        _tools = ["Fuel", "Firebreak", "Access", "Asset", "Elevation"]
        _cur = st.session_state.get("tool", "Fuel")
        if _cur not in _tools:
            _cur = "Fuel"
        # a radio needs no rerun, so switching tools never interrupts the run
        # and never resets the other widgets (view mode, brushes, layers)
        tool = st.radio("Tool palette", _tools, index=_tools.index(_cur),
                        label_visibility="collapsed")
        st.session_state.tool = tool
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
                st.caption(f"$r_{{base}}$={mdl.r_base}  $m_{{ext}}$={mdl.m_ext}  "
                           f"$b_{{base}}$={mdl.b_base}")
                kw["load"] = st.slider("$F_{load}$ \u2014 fuel load", 0.0, 1.0,
                                       1.0, 0.05)
                kw["moisture"] = st.slider("$F_{moist}$ \u2014 moisture", 0.0,
                                           0.6, 0.08, 0.01)
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
        with st.expander("Legend", expanded=True):
            st.markdown(legend_html(), unsafe_allow_html=True)
        with st.expander("Layers", expanded=True):
            # same keys as the Simulation page: one shared selection
            eflags = dict(
                show_hillshade=st.checkbox("Relief", True, key="ly_relief"),
                show_fire=st.checkbox("Fire", True, key="ly_fire"),
                show_value=st.checkbox(
                    "Protection value", True, key="ly_val",
                    help="Tints asset cells by the protection priority "
                         "$V_{prio}$ (System Description Sec. 2.5): pale pink "
                         "= lower, deep purple = higher priority. Nothing "
                         "shows until the map has buildings, critical "
                         "facilities or population (Asset tool)."),
                show_roads=st.checkbox("Roads", True, key="ly_roads"),
                show_grid=st.checkbox("Grid", False, key="ly_grid"))
            if eflags["show_value"] and float(world.value.vbld.max()
                    + world.value.vcrit.max() + world.value.vpop.max()) <= 0:
                st.caption("\u26a0 No assets on the map yet \u2014 the "
                           "overlay shows nothing. Use the Asset tool.")

    with view_col:
        _views = ["2D canvas", "3D terrain"]
        _vcur = st.session_state.get("editor_view_sel", "2D canvas")
        vmode = st.radio("Editor view", _views,
                         index=_views.index(_vcur) if _vcur in _views else 0,
                         horizontal=True, label_visibility="collapsed")
        st.session_state["editor_view_sel"] = vmode
        if vmode == "3D terrain":
            edit3d = st.checkbox(
                "Click terrain to apply the active tool "
                "(off = free rotate / zoom)",
                value=st.session_state.get("edit3d_sel", True))
            st.session_state["edit3d_sel"] = edit3d
            _e = world.topo.elev
            u1, u2 = st.columns([1, 2.2])
            if u1.button("Undo edit", use_container_width=True, key="undo3d",
                         disabled=not st.session_state.get("edit_undo")):
                _restore_snapshot(); st.session_state.canvas_key += 1; st.rerun()
            u2.caption(f"Active tool: **{tool}**  \u00b7  elevation "
                       f"{_e.min():.0f}-{_e.max():.0f} m  \u00b7  hover shows "
                       "the elevation of every cell")
            st.caption("Click one of the **light dots** on the terrain: the "
                       "active tool is applied at that cell as a point/brush "
                       "edit (rectangles only in 2D). Elevation edits reshape "
                       "the surface immediately. The view resets after an "
                       "edit; turn clicking off to rotate freely.")
            fig3 = viz.fire_surface_figure(
                world, sim=sim, pick=edit3d,
                pick_label=f"click to apply: {tool}")
            key3e = (f"edit3d_{st.session_state.map_version}_"
                     f"{st.session_state.canvas_key}")
            def _tool_at(gx, gy):
                _push_snapshot()
                _do_point(gx, gy, kw)
                if kw["tool"] == "Elevation":
                    world.recompute_slope_aspect()
                    st.toast(f"Elevation at ({gx}, {gy}) -> "
                             f"{world.topo.elev[gy, gx]:.0f} m")
                else:
                    st.toast(f"{kw['tool']} applied at ({gx}, {gy})")
                st.session_state.canvas_key += 1
                st.rerun()

            if edit3d:
                gxy = _click3d(fig3, key=key3e)
                if gxy:
                    _tool_at(*gxy)
                if not HAS_PLOTLY_EVENTS:
                    _manual_cell_apply(f"Apply {tool} at (x, y)", _tool_at,
                                       "ed3d")
            else:
                st.plotly_chart(fig3, use_container_width=True,
                                config={"scrollZoom": True}, key=key3e)
        else:
            from io import BytesIO
            scale = _fit_scale(cfg.nx)
            bg = viz.render_pil(world, sim=sim, scale=scale, show_labels=True,
                                **eflags)
            # our own controls (the built in canvas toolbar is hidden below):
            # a real PNG download, an edit undo and a stroke clear
            t1, t2, t3 = st.columns(3)
            _buf = BytesIO(); bg.save(_buf, format="PNG")
            t1.download_button("Download PNG", _buf.getvalue(),
                               file_name="map.png", mime="image/png",
                               use_container_width=True)
            if t2.button("Undo edit", use_container_width=True,
                         disabled=not st.session_state.get("edit_undo"),
                         help="Reverts the last APPLIED map edit (fuel, "
                              "firebreak, road, asset, elevation). Up to 12 "
                              "steps back."):
                _restore_snapshot(); st.session_state.canvas_key += 1; st.rerun()
            # Clear strokes only matters when strokes are NOT applied live
            if not live and t3.button(
                    "Clear strokes", use_container_width=True,
                    help="Discards the drawn strokes WITHOUT applying them; "
                         "the map itself never changes."):
                st.session_state.canvas_key += 1; st.rerun()
            if HAS_CANVAS:
                stroke = {"Fuel": "#1f7a1f", "Firebreak": "#3070b0",
                          "Access": "#b08020", "Asset": "#ffd000",
                          "Elevation": "#7a5230"}[tool]
                sw = kw.get("brush", 2) * scale if drawing_mode == "freedraw" else 2
                flagsig = abs(hash(tuple(sorted(eflags.items())))) % 100000
                ckey = f"canvas_{st.session_state.canvas_key}_{scale}_{flagsig}"
                result = st_canvas(fill_color="rgba(255,160,0,0.20)",
                                   stroke_width=int(sw), stroke_color=stroke,
                                   background_image=bg, update_streamlit=True,
                                   height=cfg.ny * scale, width=cfg.nx * scale,
                                   drawing_mode=drawing_mode,
                                   display_toolbar=False,
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
    if e1.button("Clear all assets", disabled=not world.assets,
                 help="Removes every asset marker and its value layers "
                      "(buildings, critical facilities, population, "
                      "evacuation routes)."):
        world.assets.clear()
        from disasteraware.layers import ValueLayer
        world.value = ValueLayer.empty(cfg.ny, cfg.nx); st.rerun()
    if e2.button("Clear ignitions", disabled=not world.ignitions,
                 help="Removes all placed/scheduled ignition markers; "
                      "cells already burning keep burning."):
        world.ignitions.clear(); st.rerun()



# ============================================================== DATA LAYERS ==
# Field registry mirroring the System Description: external inputs (Sec. 2),
# the state vector (Sec. 3) and derived fields.
_FIELD_DEFS = [
    ("W_temp \u2014 air temperature", "U_Meteo (Sec. 2.1)", "temp",
     r"U_{Meteo,k}=[\,W_{temp},W_{rh},W_{ws},W_{wd},W_{gust},W_{prec}\,]^T",
     "\u00b0C"),
    ("W_rh \u2014 relative humidity", "U_Meteo (Sec. 2.1)", "rh",
     r"F_{moist}=EMC(W_{temp},W_{rh})\ \text{(optional mode)}", "%"),
    ("W_ws \u2014 wind speed", "U_Meteo (Sec. 2.1)", "wws",
     r"g_{wind}=1+a_w\tanh(W_{ws}/w_0)", "m/s"),
    ("W_wd \u2014 wind direction", "U_Meteo (Sec. 2.1)", "wwd",
     r"g_{dir}=\max\{0,\cos(W_{wd}-\theta)\}", "rad"),
    ("W_gust \u2014 wind gust", "U_Meteo (Sec. 2.1)", "gust",
     r"\text{exogenous stochastic forcing channel}", "m/s"),
    ("W_prec \u2014 precipitation", "U_Meteo (Sec. 2.1)", "prec",
     r"\text{moistening channel (moisture dynamics mode)}", "mm/h"),
    ("G_elev \u2014 elevation", "U_Geo (Sec. 2.2)", "elev",
     r"U_{Geo}=[\,G_{elev},G_{slope},G_{aspect},G_{access}\,]^T", "m"),
    ("G_slope \u2014 slope", "U_Geo (Sec. 2.2)", "slope",
     r"g_{slope}=1+a_s\tan(G_{slope})", "rad"),
    ("G_aspect \u2014 aspect", "U_Geo (Sec. 2.2)", "aspect",
     r"g_{aspect}=1+a_{asp}\cos(G_{aspect}-W_{wd})", "rad"),
    ("G_access \u2014 accessibility", "U_Geo (Sec. 2.2)", "access",
     r"\eta_{reach}=e^{-\beta_t R_{time}}\,G_{access}", "[0,1]"),
    ("F_type \u2014 fuel class", "U_Fuel (Sec. 2.3)", "ftype",
     r"U_{Fuel,k}=[\,F_{type},F_{load,0},F_{moist,k}\,]^T", "class id"),
    ("F_load,0 \u2014 initial fuel load", "U_Fuel (Sec. 2.3)", "fload0",
     r"F_{load,0}\ \text{initializes the state}\ F_{load,k}", "[0,1] norm."),
    ("F_moist \u2014 fuel moisture", "U_Fuel (Sec. 2.3)", "fmoist",
     r"g_{moist}=\max\{0,\,1-F_{moist}/m_{ext}\}", "mass fraction"),
    ("V_bld \u2014 building footprint", "U_Val (Sec. 2.5)", "vbld",
     r"J^{val}=c_{bld}\lambda_{loss}\textstyle\sum A_k V_{bld}", "[0,1]"),
    ("V_crit \u2014 critical facilities", "U_Val (Sec. 2.5)", "vcrit",
     r"J^{inf}=c_{crit}\lambda_{loss}\textstyle\sum A_k V_{crit}", "[0,1]"),
    ("V_pop \u2014 population density", "U_Val (Sec. 2.5)", "vpop",
     r"P^{exp}=a_{km^2}\textstyle\sum A_k V_{pop}", "person/km\u00b2"),
    ("V_evac \u2014 evacuation distance", "U_Val (Sec. 2.5)", "vevac",
     r"V_{evac}^{norm}=1-\text{minmax}(V_{evac})", "m"),
    ("V_prio \u2014 protection priority", "U_Val (Sec. 2.5)", "vprio",
     r"V_{prio}=w_{bld}V_{bld}+w_{crit}V_{crit}"
     r"+w_{pop}V_{pop}^{norm}+w_{evac}V_{evac}^{norm}", "[0,1]"),
    ("R_cap \u2014 suppression capacity", "U_DSS (Sec. 2.6)", "rcap",
     r"\eta_{cap}=R_{cap}/R_{cap,max}", "capacity/step"),
    ("R_avail \u2014 availability", "U_DSS (Sec. 2.6)", "ravail",
     r"\eta_{avail}=R_{avail}\in\{0,1\}", "{0,1}"),
    ("R_eff \u2014 efficiency", "U_DSS (Sec. 2.6)", "reff",
     r"\eta_{eff}=R_{eff}/(1+\gamma_I I_k)", "[0,1]"),
    ("R_time \u2014 travel time", "U_DSS (Sec. 2.6)", "rtime",
     r"\eta_{reach}=e^{-\beta_t R_{time}}\,G_{access}", "min"),
    ("B_k \u2014 burning status", "State s_k (Sec. 3, 4)", "burning",
     r"B_{k+1}=\max\{B^{pers},B^{prop},I_{Ign}\cdot H\}", "{0,1}"),
    ("F_load,k \u2014 fuel load", "State s_k (Sec. 3, 6)", "fload",
     r"F_{load,k+1}=\max\{0,F_{load,k}-B_kF_{burn,k}F_{load,k}-F_{red,k}\}",
     "[0,1] norm."),
    ("I_k \u2014 fire intensity", "State s_k (Sec. 3, 7)", "intensity",
     r"I_{k+1}=B_{k+1}\tanh\big(\beta(\tilde F+\gamma_W\tilde W"
     r"+\gamma_S\tilde S)\big)", "[0,1]"),
    ("\u03c4_k \u2014 time since ignition", "State s_k (Sec. 3, 8)", "tau",
     r"\tau_{k+1}=\tau_k+\Delta t\ \text{(while burning)}", "steps"),
    ("A_k \u2014 ignition buildup", "Derived (Sec. 4)", "buildup",
     r"A_{k+1}=(1-B_{k+1})\big[(1-\lambda)A_k+\Psi_k\big]", "influence"),
    ("R_spread,k \u2014 rate of spread", "Derived (Sec. 5)", "ros",
     r"R_{spread}=r_{base}\,g_{moist}\,g_{wind}\,g_{slope}\,g_{aspect}",
     "cells/step (= m/min at 30 m / 30 min)"),
    ("t_ign \u2014 time of first ignition", "Derived (Sec. 8)", "tign",
     r"t_{ign}(x,y)=\min\{k:\,B_k(x,y)=1\}", "step index"),
]


def _field_array(key):
    from disasteraware import behavior
    if key == "vprio":
        return world.priority_field()
    if key == "ros":
        return behavior.rate_of_spread_field(world)
    if key == "burning":
        return sim.state.burning
    if key == "intensity":
        return sim.state.intensity
    if key == "tau":
        return sim.state.tau
    if key == "fload":
        return sim.state.fload
    if key == "buildup":
        return sim.ign_buildup
    src = {"temp": world.meteo.temp, "rh": world.meteo.rh,
           "wws": world.meteo.wws, "wwd": world.meteo.wwd,
           "gust": world.meteo.gust, "prec": world.meteo.prec,
           "elev": world.topo.elev, "slope": world.topo.slope,
           "aspect": world.topo.aspect, "access": world.topo.access,
           "ftype": world.fuel.ftype.astype(float),
           "fload0": world.fuel.fload0,
           "fmoist": world.fuel.fmoist,
           "vbld": world.value.vbld, "vcrit": world.value.vcrit,
           "vpop": world.value.vpop, "vevac": world.value.vevac,
           "rcap": world.resource.rcap, "ravail": world.resource.ravail,
           "reff": world.resource.reff, "rtime": world.resource.rtime}
    return src[key]


def page_layers():
    import matplotlib.pyplot as plt
    st.subheader("Data layers")
    st.caption("The exact fields of the System Description: the external "
               "inputs of Sec. 2, the state vector of Sec. 3 and the fields "
               "derived from them. Names, symbols and units match the System "
               "Description page.")

    st.markdown("#### Terrain \u2014 $U_{Geo}$ (Sec. 2.2)")
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
    st.markdown("#### Field viewer")
    names = [f"{grp}  \u00b7  {name}" for name, grp, *_ in _FIELD_DEFS]
    pick = st.selectbox("Field", names, label_visibility="collapsed")
    name, grp, key, latex, unit = _FIELD_DEFS[names.index(pick)]
    st.latex(latex)
    st.caption(f"Defined in System Description {grp.split('(')[-1].rstrip(')')} "
               f"\u00b7 unit: {unit}")
    if key == "tign":
        st.caption("Red = ignites first, blue = ignites later; grey = never "
                   "ignited in this run.")
        st.image(viz.ignition_time_pil(sim, scale=max(4, 700 // max(cfg.nx, 1))))
    else:
        field = _field_array(key)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(field, origin="upper", cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{name}  [{unit}]")
        ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.divider()
    with st.expander("Operational diagnostics (viewer only \u2014 not part "
                     "of the model)"):
        st.caption("Standard fire-behaviour products computed FROM the state "
                   "for interpretation and reporting; they never influence "
                   "the simulation. Fireline intensity is Byram's "
                   "$I_B = H\,w\,R$ (kW/m) and flame length "
                   "$L = 0.0775\,I_B^{0.46}$ (m), the measures used by "
                   "operational tools such as FARSITE/FlamMap; crown fire "
                   "marks burning forest cells above the intensity "
                   "threshold of the Parameters page.")
        bkind = st.selectbox("Diagnostic",
                             ["Fireline intensity", "Flame length",
                              "Rate of spread (m/min)", "Crown fire"])
        bmap = {"Fireline intensity": "fireline_intensity",
                "Flame length": "flame_length",
                "Rate of spread (m/min)": "rate_of_spread",
                "Crown fire": "crown_fire"}
        st.image(viz.behavior_pil(sim, kind=bmap[bkind],
                                  scale=max(4, 700 // max(cfg.nx, 1))))


# ============================================================== PARAMETERS ===
def page_params():
    st.subheader("Model parameters")
    st.caption("These shape the fire behaviour model. Day to day conditions live "
               "in the Simulation tab. Hover the ? on any control. Every symbol "
               "is defined in the System Description page; defaults follow its "
               "tables (Sec. 4\u20137 and 14.4).")
    rc1, rc2 = st.columns([1.2, 3])
    if rc1.button("Reset to defaults", use_container_width=True):
        cfg.spread = SpreadParams()
        cfg.suppression = SuppressionParams()
        cfg.intensity = IntensityParams()
        cfg.value_weights = ValueWeights()
        cfg.cost = CostParams()
        for _i, _m0 in _THESIS_FUELS.items():
            FUEL_MODELS[_i] = dataclasses.replace(_m0)
        st.rerun()
    rc2.caption("Restores every parameter below, including the fuel class table.")
    with st.expander("Optional realism modes (System Description Sec. 9)"):
        st.caption("Literature calibrated speeds, influence buildup and "
                   "the flank/backing floor are always active. The two "
                   "mechanisms below are **on by default**; turning them "
                   "off gives the plain cosine kernel without embers. "
                   "Equations: System Description Sec. 9.")
        a, b = st.columns(2)
        st.markdown("**Elliptical spread kernel** (Cell2Fire / FARSITE)")
        a, b = st.columns(2)
        cfg.spread.elliptical = a.checkbox(
            "Enable elliptical kernel", value=cfg.spread.elliptical,
            help="Replaces the cosine directional weight by a wind-elongated "
                 "ellipse: g_dir = (1-e)/(1-e cos\u0394), with eccentricity "
                 "from the length-to-breadth ratio LB = LB0 + LBw\u00b7W_ws.")
        cfg.spread.lb_ratio_base = b.number_input(
            "$LB_0$ — length/breadth ratio at no wind (–)", 1.0, 3.0,
            float(cfg.spread.lb_ratio_base), 0.05)
        cfg.spread.lb_ratio_wind = b.number_input(
            "$LB_w$ — extra length/breadth per m/s of wind", 0.0, 0.5,
            float(cfg.spread.lb_ratio_wind), 0.01)
        st.markdown("**Ember spotting** (stochastic ignition ahead of the front)")
        a, b = st.columns(2)
        cfg.spread.spotting = a.checkbox(
            "Enable spotting", value=cfg.spread.spotting,
            help="Each intense burning cell (I_k above the threshold) throws "
                 "an ember downwind with the given probability per substep; "
                 "the ember ignites its landing cell if fuel is present.")
        cfg.spread.spot_prob = b.number_input(
            "$p_{spot}$ — spot probability (per cell and substep)", 0.0, 1.0,
            float(cfg.spread.spot_prob), 0.01)
        cfg.spread.spot_distance = int(a.number_input(
            "$d_{spot}$ — spot distance (cells downwind)", 1, 40,
            int(cfg.spread.spot_distance)))
        cfg.spread.spot_intensity_min = b.number_input(
            "$I_{min}$ — minimum $I_k$ to spot (–)", 0.0, 1.0,
            float(cfg.spread.spot_intensity_min), 0.05)
        st.markdown("**Crown fire flag** (diagnostic only)")
        cfg.intensity.crown_fire_threshold = st.number_input(
            "$I_{crown}$ — crown-fire intensity threshold (–)", 0.0, 1.0,
            float(cfg.intensity.crown_fire_threshold), 0.05,
            help="Burning forest cells with I_k above this value are "
                 "reported as crown fire in the Data layers diagnostics; "
                 "it does not change the dynamics.")

    with st.expander("Fire spread and propagation", expanded=True):
        a, b = st.columns(2)
        cfg.spread.w0 = a.number_input("$w_0$ — wind saturation speed (m/s)", 0.5, 30.0,
                                       float(cfg.spread.w0), 0.5,
                                       help="Wind where amplification saturates.")
        cfg.spread.eps_fuel = b.number_input("$\\epsilon_{fuel}$ — extinction fuel threshold (norm. fuel)", 1e-6, 0.1,
                                             float(cfg.spread.eps_fuel), format="%.5f",
                                             help="Fuel below which a cell stops burning.")
        cfg.spread.theta_ign = a.number_input(
            "$\\Theta_{ign}$ — ignition threshold (\u215b cell-widths)",
            0.005, 1.0, float(cfg.spread.theta_ign), 0.005, format="%.3f",
            help="Activation threshold on the ignition influence buildup "
                 "$A_k$ (Eq. 45, System Description Sec. 4). Default "
                 "$0.125=1/8$: the front then advances at exactly "
                 "$R_{spread}$ cells per step.")
        cfg.spread.aniso_wind_full = b.number_input(
            "$w_{aniso}$ — wind for fully directional spread (m/s)", 0.0, 30.0,
            float(cfg.spread.aniso_wind_full), 0.5,
            help="Below this wind speed the directional weight blends toward "
                 "isotropic spread (System Description Sec. 9).")
        cfg.spread.back_frac = a.number_input(
            "$f_{back}$ — flank/backing fraction (–)", 0.0, 1.0,
            float(cfg.spread.back_frac), 0.01,
            help="Directional-weight floor: backing fires run at this "
                 "fraction of the head fire rate (literature ~0.05-0.15).")
        cfg.spread.buildup_leak = b.number_input(
            "$\\lambda$ — influence buildup leak (fraction/step)", 0.0, 1.0,
            float(cfg.spread.buildup_leak), 0.01,
            help="Per reference step decay of the ignition buildup A_k: "
                 "heating dissipates when the fire source disappears.")
        cfg.spread.slope_wind_equiv = a.number_input(
            "$k_{slope}$ \u2014 slope-equivalent wind (m/s per tan)", 0.0,
            30.0, float(getattr(cfg.spread, "slope_wind_equiv", 10.0)), 1.0,
            help="Upslope acts like an extra wind of "
                 "$k_{slope}\\tan(G_{slope})$ m/s blowing uphill "
                 "(FARSITE-style vector combination): fire climbs "
                 "mountainsides even against a light gradient wind.")
        cfg.spread.slope_gain_max = b.number_input(
            "$g_{slope}^{max}$ \u2014 slope factor cap (\u2013)", 1.0,
            10.0, float(getattr(cfg.spread, "slope_gain_max", 3.0)), 0.5,
            help="Saturation of the slope speed factor; prevents tan() "
                 "blow-up on real DEM cliffs.")
        cfg.spread.slope_clip_rad = a.number_input("$G_{slope}$ clip (rad)", 0.1, 1.5,
                                                   float(cfg.spread.slope_clip_rad), 0.05)
        cfg.spread.diagonal_distance_weighting = b.checkbox(
            "Diagonal distance weighting", value=cfg.spread.diagonal_distance_weighting)
    with st.expander("Suppression effectiveness"):
        a, b = st.columns(2)
        cfg.suppression.alpha_s = a.number_input(
            "$\\alpha_s$ — global suppression gain (fraction/step)",
            0.0, 1.0, float(cfg.suppression.alpha_s), 0.05,
            help="Maximum fuel fraction one step of suppression can remove "
                 "when every factor equals 1 (System Description Sec. 6).")
        cfg.suppression.beta_t = b.number_input(
            "$\\beta_t$ — travel-time decay (1/min)", 0.0, 1.0,
            float(cfg.suppression.beta_t), 0.01, format="%.2f",
            help="Exponential decay of suppression effectiveness with "
                 "arrival time: $e^{-\\beta_t R_{time}}$. Default 0.03/min "
                 "halves the effect after about 23 minutes.")
        cfg.suppression.gamma_I = a.number_input(
            "$\\gamma_I$ — intensity resistance (–)", 0.0, 5.0,
            float(cfg.suppression.gamma_I), 0.1,
            help="High-intensity fires resist suppression: effectiveness is "
                 "divided by $1+\\gamma_I I_k$ (Sec. 6).")
        cfg.suppression.rcap_max = b.number_input(
            "$R_{cap,max}$ — reference capacity (same unit as $R_{cap}$)",
            0.1, 100.0, float(cfg.suppression.rcap_max), 0.1,
            help="Normalizes the assigned capacity: "
                 "$\\eta_{cap}=R_{cap}/R_{cap,max}$, clipped to $[0,1]$.")
    with st.expander("Fire intensity"):
        a, b = st.columns(2)
        cfg.intensity.beta = a.number_input(
            "$\\beta$ — global intensity gain (–)", 0.1, 3.0,
            float(cfg.intensity.beta), 0.1,
            help="Scales the $\\tanh$ argument of the intensity proxy "
                 "(Sec. 7); typical range 1–3.")
        cfg.intensity.gamma_w = b.number_input(
            "$\\gamma_W$ — wind weight (–)", 0.0, 1.0,
            float(cfg.intensity.gamma_w), 0.05,
            help="Contribution of normalized wind to the intensity proxy "
                 "(Sec. 7); typical 0–0.7.")
        cfg.intensity.gamma_s = a.number_input(
            "$\\gamma_S$ — slope weight (–)", 0.0, 1.0,
            float(cfg.intensity.gamma_s), 0.05,
            help="Contribution of normalized slope to the intensity proxy "
                 "(Sec. 7); typical 0–0.5.")
        cfg.intensity.wws_max = b.number_input(
            "$W_{ref}$ — wind normalization (m/s)", 1.0, 60.0,
            float(cfg.intensity.wws_max), 1.0,
            help="Wind speed treated as 'extreme' in the intensity "
                 "normalization $\\tilde W=\\min\\{1,W_{ws}/W_{ref}\\}$.")
        cfg.intensity.slope_max_rad = a.number_input(
            "$S_{max}$ — slope normalization (rad)", 0.1, 1.4,
            float(cfg.intensity.slope_max_rad), 0.05,
            help="Slope normalization S_max in Eq. 136. Default 0.7854 (45\u00b0).")
        cfg.intensity.fload_max = b.number_input(
            "$F_{max}$ — fuel normalization (norm. units)", 0.1, 5.0,
            float(cfg.intensity.fload_max), 0.1,
            help="Fuel normalization F_max in Eq. 136. Default 1.0.")
    with st.expander("Protection priority weights (sum to 1)"):
        a, b = st.columns(2)
        st.caption("Aggregation weights of the protection priority "
                   "$V_{prio}$ (Sec. 2.5); dimensionless, renormalized to "
                   "sum to 1.")
        cfg.value_weights.w_crit = a.number_input(
            "$w_{crit}$ — critical facility weight", 0.0, 1.0,
            float(cfg.value_weights.w_crit), 0.05)
        cfg.value_weights.w_pop = b.number_input(
            "$w_{pop}$ — population weight", 0.0, 1.0,
            float(cfg.value_weights.w_pop), 0.05)
        cfg.value_weights.w_bld = a.number_input(
            "$w_{bld}$ — building weight", 0.0, 1.0,
            float(cfg.value_weights.w_bld), 0.05)
        cfg.value_weights.w_evac = b.number_input(
            "$w_{evac}$ — evacuation weight", 0.0, 1.0,
            float(cfg.value_weights.w_evac), 0.05)

    with st.expander("Fuel classes (System Description Sec. 5, Table A.1 + B.1)"):
        st.caption("Per class spread and combustion parameters: r_base "
                   "(cells/step), m_ext (mass fraction), a_w / a_s / a_asp "
                   "(dimensionless), b_base (fraction per step, Table B.1), "
                   "e (economic value per cell unit, used by the cost model). "
                   "Note: b_base is cached when a simulator is created; press "
                   "Reset fire (or make a new map) after editing it.")
        import pandas as _pd
        _rows = [{"id": _i, "fuel": _m.name, "r_base": _m.r_base,
                  "m_ext": _m.m_ext, "a_w": _m.a_w, "a_s": _m.a_s,
                  "a_asp": _m.a_asp, "b_base": _m.b_base,
                  "e": _m.economic_value}
                 for _i, _m in FUEL_MODELS.items()]
        _edit = st.data_editor(_pd.DataFrame(_rows), hide_index=True,
                               disabled=["id", "fuel"], key="fuel_editor")
        for _, _r in _edit.iterrows():
            _m = FUEL_MODELS[int(_r["id"])]
            _m.r_base = float(_r["r_base"]); _m.m_ext = float(_r["m_ext"])
            _m.a_w = float(_r["a_w"]); _m.a_s = float(_r["a_s"])
            _m.a_asp = float(_r["a_asp"]); _m.b_base = float(_r["b_base"])
            _m.economic_value = float(_r["e"])
        if st.button("Reset fuel classes to defaults"):
            for _i, _m0 in _THESIS_FUELS.items():
                FUEL_MODELS[_i] = dataclasses.replace(_m0)
            st.rerun()

    with st.expander("Cost model (cost-plus-loss, System Description Sec. 14)"):
        a, b = st.columns(2)
        cfg.cost.cost_per_burned_ha = a.number_input(
            "$c_{ha}$ — rehabilitation cost per burned ha (currency/ha)", 0.0, 1e7,
            float(cfg.cost.cost_per_burned_ha), 100.0)
        cfg.cost.forest_value_multiplier = b.number_input(
            "$\\lambda_{for}$ — forest value multiplier (–)", 0.0, 100.0,
            float(cfg.cost.forest_value_multiplier), 0.1)
        cfg.cost.building_unit_value = a.number_input(
            "$c_{bld}$ — building unit value (currency/cell)", 0.0, 1e9,
            float(cfg.cost.building_unit_value), 10000.0)
        cfg.cost.critical_unit_value = b.number_input(
            "$c_{crit}$ — critical facility unit value (currency/cell)", 0.0, 1e10,
            float(cfg.cost.critical_unit_value), 100000.0)
        cfg.cost.value_loss_on_burn = a.number_input(
            "$\\lambda_{loss}$ — value loss fraction on burn (–)", 0.0, 1.0,
            float(cfg.cost.value_loss_on_burn), 0.05)
        cfg.cost.statistical_life_value = b.number_input(
            "$v_L$ — value of statistical life (currency/person)", 0.0, 1e9,
            float(cfg.cost.statistical_life_value), 100000.0)
        cfg.cost.population_at_risk_fraction = a.number_input(
            "$\\rho_{risk}$ — population at risk fraction (–)", 0.0, 1.0,
            float(cfg.cost.population_at_risk_fraction), 0.005, format="%.3f")
        cfg.cost.suppression_unit_cost = b.number_input(
            "$c_{sup}$ — suppression unit cost (currency/fuel unit)", 0.0, 1e7,
            float(cfg.cost.suppression_unit_cost), 500.0)


# ====================================================== SYSTEM DESCRIPTION ===
def page_system_description():
    """Full mathematical description of the model (System Description page).

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
# =============================================================== VALIDATION ==
def _load_auto_validate():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "examples", "auto_validate.py")
    spec = importlib.util.spec_from_file_location("auto_validate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def page_validation():
    import types as _types
    st.subheader("Validation \u2014 hindcast against a real fire")
    st.markdown(
        "The simulator receives ONLY the real inputs of a documented fire "
        "and runs **blind**: real terrain (Copernicus GLO-30 DEM), real fuel "
        "map (ESA WorldCover 10 m \u2192 fuel classes), real hourly weather "
        "(ERA5: wind drives the spread, humidity drives the fuel moisture "
        "via EMC), and the real ignition (first NASA FIRMS satellite "
        "detection). The simulated burned area is then scored against the "
        "observed fire footprint with the standard metrics (Sorensen-Dice, "
        "Jaccard/IoU, hit rate, false alarm ratio, area bias, front "
        "position error). Details and the referee protocol: "
        "`03_Codes/VALIDATION.md`.")
    st.caption("First run downloads ~100\u2013200 MB of open data; "
               "everything is cached in `validation_cache/` afterwards.")

    av = None
    try:
        av = _load_auto_validate()
    except Exception as exc:
        st.error(f"Could not load the validation module: {exc}")
        return

    c1, c2, c3 = st.columns([1.3, 1.6, 0.8])
    case_id = c1.selectbox("Case", sorted(av.CASES),
                           format_func=lambda k: av.CASES[k]["label"],
                           help="Documented historical fire to hindcast.")
    key = c2.text_input("NASA FIRMS MAP_KEY", type="password",
                        value=st.session_state.get("firms_key",
                                                   os.environ.get(
                                                       "FIRMS_MAP_KEY", "")),
                        help="Free key (1 minute): "
                             "https://firms.modaps.eosdis.nasa.gov/api/"
                             "map_key/  \u2014 used to fetch the satellite "
                             "fire detections (ignition + burned footprint).")
    st.session_state["firms_key"] = key
    seeds = int(c3.number_input("Seeds", 1, 10, 3,
                                help="Ember spotting is stochastic; the "
                                     "score is reported as mean over seeds."))
    c4, c5, c6 = st.columns(3)
    cell = float(c4.number_input("Cell size (m)", 30.0, 200.0, 90.0, 10.0,
                                 help="90 m matches the satellite truth "
                                      "resolution and keeps the run fast."))
    hours = float(c5.number_input("Hours to simulate", 6.0, 120.0,
                                  float(av.CASES[case_id]["hours"]), 6.0,
                                  help="Documented duration of the main "
                                       "fire run."))
    stepm = float(c6.number_input("Step length (min)", 10.0, 60.0, 30.0,
                                  10.0))
    wens = st.checkbox(
        "Wind-direction uncertainty ensemble (8 extra runs)", value=False,
        help="Gridded reanalysis winds miss local channeling and the "
             "fire's own convection \u2014 the dominant input uncertainty. "
             "This reruns the hindcast with the wind rotated over the full "
             "circle and reports the sensitivity and the best-matching "
             "member.")
    up = st.file_uploader(
        "Optional: official EFFIS/EMS perimeter raster (stronger ground "
        "truth than the FIRMS footprint)", type=["tif", "tiff", "npy"])

    if st.button("Run validation", type="primary",
                 disabled=(not key and up is None)):
        try:
            import rasterio  # noqa: F401
            import requests  # noqa: F401
        except ImportError as exc:
            st.error(f"Missing package: {exc}. Run "
                     "`pip install rasterio requests` in the app "
                     "environment and restart.")
            return
        case = dict(av.CASES[case_id]); case["hours"] = hours
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args = _types.SimpleNamespace(cell=cell, step_minutes=stepm,
                                      seeds=seeds,
                                      cache=os.path.join(root,
                                                         "validation_cache",
                                                         case_id),
                                      out=os.path.join(root,
                                                       "validation_report"))
        os.makedirs(args.cache, exist_ok=True)
        import datetime as _dt2
        import json as _json
        _stamp = _dt2.datetime.now().strftime("%Y%m%d-%H%M%S")
        _rdir = os.path.join(root, "validation_runs", f"{case_id}_{_stamp}")
        os.makedirs(os.path.join(_rdir, "frames"), exist_ok=True)
        # migrate a pre-existing flat cache into the per-case folder once
        _flat = os.path.join(root, "validation_cache")
        for _fn in ("dem_win.npz", "wc_win.npz", "weather.json", "firms.csv"):
            _o, _n = os.path.join(_flat, _fn), os.path.join(args.cache, _fn)
            if os.path.exists(_o) and not os.path.exists(_n):
                os.replace(_o, _n)
        try:
            with st.status("Downloading the real data \u2026",
                           expanded=True) as stat:
                st.write("1/4 terrain \u2014 Copernicus GLO-30 DEM")
                dem, dlons, dlats = av._download_dem(case, args.cache)
                st.write("2/4 fuel \u2014 ESA WorldCover 10 m")
                wc, wlons, wlats = av._download_worldcover(case, args.cache)
                nx, ny, lons, lats = av._grid(case, cell)
                first = None
                fmask = None
                _ign_cells = []
                if key:
                    st.write("3/4 fire truth \u2014 NASA FIRMS detections")
                    pts = av._download_firms(case, key, args.cache)
                    fmask, first, _ign_cells = av._firms_mask_and_ignition(
                        case, pts, nx, ny, lons, lats, cell)
                    if first is not None:
                        # align the weather series with the ignition hour
                        case["t0_hour"] = first[4].hour
                st.write("4/4 weather \u2014 hourly ERA5 at the fire "
                         "(open-meteo)")
                if first is not None:
                    _iglat = case["north"] - (first[1] + 0.5) * cell / 110540.0
                    import math as _math
                    _iglon = case["west"] + (first[0] + 0.5) * cell / (
                        111320.0 * _math.cos(_math.radians(_iglat)))
                    weather = av._download_weather(case, args.cache,
                                                   lat=_iglat, lon=_iglon)
                else:
                    weather = av._download_weather(case, args.cache)
                if up is not None:
                    tmpp = os.path.join(args.cache, "_upl_" + up.name)
                    with open(tmpp, "wb") as fh:
                        fh.write(up.getbuffer())
                    from disasteraware.gis import _read_resampled
                    obs = (np.load(tmpp) if tmpp.endswith(".npy") else
                           _read_resampled(tmpp, ny, nx, nearest=True)) > 0.5
                else:
                    obs = fmask
                if obs is None or first is None and not up:
                    st.error("No ground truth: give a FIRMS key or upload "
                             "a perimeter raster.")
                    return
                ign = _ign_cells if _ign_cells else (nx // 2, ny // 2)
                if first:
                    st.write(f"initial fire front: {len(_ign_cells)} cells "
                             f"from the first overpass {first[2]} "
                             f"{first[3]} UTC (the fire had already grown "
                             "before the satellite first saw it)")
                stat.update(label="Data ready \u2014 running the blind "
                                  "hindcast", state="running")
            bar = st.progress(0.0, text="simulating \u2026")
            # live event view: the fire grows on the real map while it runs
            demg0 = av._sample(dem, dlons, dlats, lons, lats)
            wcg0 = av._sample(wc, wlons, wlats, lons, lats).astype(int)
            ftype0 = np.zeros_like(wcg0)
            for code, (fid, _ld) in av.WORLDCOVER_TO_FUEL.items():
                ftype0[wcg0 == code] = (case.get("tree_fuel", 3)
                                        if code == 10 else fid)
            base0 = av._basemap(ftype0, demg0)
            live = st.empty()
            log_lines = []
            cell_ha = (cell * cell) / 10000.0

            def _cb(seed, k, n):
                bar.progress((seed + k / n) / seeds,
                             text=f"seed {seed + 1}/{seeds} \u00b7 step "
                                  f"{k}/{n}")

            def _frame(k, n, burned, ws_now):
                t_h = k * stepm / 60.0
                img = base0.copy()
                img[obs] = (58, 110, 220)
                img[burned & obs] = (46, 160, 67)
                img[burned & ~obs] = (200, 55, 44)
                _i0 = ign[0] if isinstance(ign, list) else ign
                gx, gy = int(_i0[0]), int(_i0[1])
                rr = max(2, nx // 150)
                img[max(0, gy - rr):gy + rr + 1,
                    max(0, gx - rr):gx + rr + 1] = (255, 235, 60)
                ha = float(burned.sum()) * cell_ha
                live.image(img, use_container_width=True,
                           caption=f"t = {t_h:.1f} h \u00b7 simulated burn "
                                   f"{ha:,.0f} ha \u00b7 wind {ws_now:.1f} "
                                   "m/s \u00b7 red/green = simulation, "
                                   "blue = observed (final), yellow = "
                                   "ignition")
                log_lines.append(f"t={t_h:5.1f} h  burned={ha:9,.0f} ha  "
                                 f"wind={ws_now:4.1f} m/s")
                try:
                    from PIL import Image as _Img
                    _Img.fromarray(img).save(os.path.join(
                        _rdir, "frames", f"t{t_h:06.1f}h.png"))
                except Exception:
                    pass

            n_total = int(round(case["hours"] * 60.0 / stepm))
            runs, shape = av.run_case(case, args, dem, (dlons, dlats),
                                      wc, (wlons, wlats), weather, obs,
                                      ign, progress_cb=_cb,
                                      frame_cb=_frame,
                                      frame_every=max(1, n_total // 24))
            ens = None
            if wens:
                bar.progress(0.0, text="wind ensemble \u2026")

                def _ecb(i, nmem, k, n):
                    bar.progress((i + k / n) / nmem,
                                 text=f"wind ensemble member {i + 1}/{nmem}")

                members, _ = av.run_wind_ensemble(
                    case, args, dem, (dlons, dlats), wc, (wlons, wlats),
                    weather, obs, ign,
                    offsets=[-135, -90, -45, 0, 45, 90, 135, 180],
                    progress_cb=_ecb)
                ens = [{"offset": m["offset_deg"], **m["rep"]}
                       for m in members]
            bar.empty()
            live.empty()
            st.session_state["val_log"] = log_lines
            st.session_state["val_ens"] = ens
            # ---- persistent run archive (for later analysis) ----
            try:
                import numpy as _np2
                from PIL import Image as _Img
                import disasteraware as _da2
                _keys = ["jaccard", "dice", "hit_rate", "false_alarm",
                         "area_bias", "mean_m", "p90_m"]
                _summary = {k: {"mean": float(_np2.mean(
                                    [r[0][k] for r in runs])),
                                "sd": float(_np2.std(
                                    [r[0][k] for r in runs]))}
                            for k in _keys}
                _json.dump({
                    "case_id": case_id, "case": {k: v for k, v in
                                                 case.items()},
                    "settings": {"cell_m": cell, "step_minutes": stepm,
                                 "hours": hours, "seeds": seeds,
                                 "wind_ensemble": bool(wens),
                                 "truth": ("perimeter" if up is not None
                                           else "firms")},
                    "engine_version": getattr(_da2, "__version__", "?"),
                    "initial_front_cells": (len(ign)
                                            if isinstance(ign, list) else 1),
                    "summary": _summary,
                    "runs": [r[0] for r in runs],
                    "wind_ensemble": ens,
                }, open(os.path.join(_rdir, "report.json"), "w"), indent=2)
                with open(os.path.join(_rdir, "run_log.txt"), "w") as _fh:
                    _fh.write("\n".join(log_lines))
                _best = max(runs, key=lambda r: r[0]["dice"])[1]
                _img = base0.copy()
                _img[obs] = (58, 110, 220)
                _img[_best & obs] = (46, 160, 67)
                _img[_best & ~obs] = (200, 55, 44)
                _Img.fromarray(_img).save(os.path.join(_rdir,
                                                       "agreement.png"))
                st.success(f"Run archived: validation_runs/"
                           f"{case_id}_{_stamp}/ (report.json, run_log.txt, "
                           "agreement.png, frames/)")
            except Exception as _exc:
                st.warning(f"Could not archive the run: {_exc}")
            demg = av._sample(dem, dlons, dlats, lons, lats)
            wcg = av._sample(wc, wlons, wlats, lons, lats).astype(int)
            ftype = np.zeros_like(wcg)
            for code, (fid, _ld) in av.WORLDCOVER_TO_FUEL.items():
                ftype[wcg == code] = (case.get("tree_fuel", 3)
                                      if code == 10 else fid)
            st.session_state["val_result"] = dict(
                runs=[r[0] for r in runs],
                best=max(runs, key=lambda r: r[0]["dice"])[1],
                obs=obs, shape=shape, case=case["label"],
                base=av._basemap(ftype, demg), ign=ign,
                truth="perimeter" if up is not None else "firms")
        except Exception as exc:
            st.error(f"Validation failed: {exc}")
            return

    res = st.session_state.get("val_result")
    if res:
        st.markdown(f"#### Result \u2014 {res['case']}")
        import numpy as _np
        keys = [("dice", "Dice"), ("jaccard", "Jaccard"),
                ("hit_rate", "Hit rate"), ("false_alarm", "False alarm"),
                ("area_bias", "Area bias"), ("mean_m", "Front err (m)")]
        cols = st.columns(len(keys))
        for c, (k, lab) in zip(cols, keys):
            vals = [r[k] for r in res["runs"]]
            c.metric(lab, f"{_np.mean(vals):.3f}",
                     delta=f"\u00b1{_np.std(vals):.3f}",
                     delta_color="off")
        # plain-language reading of the numbers
        _m = {k: float(_np.mean([r[k] for r in res["runs"]]))
              for k in ("dice", "jaccard", "hit_rate", "false_alarm",
                        "area_bias", "mean_m")}
        _hr, _ab, _fe = _m["hit_rate"], _m["area_bias"], _m["mean_m"]
        _v = []
        _v.append(f"**Coverage** \u2014 the simulation reproduced "
                  f"**{_hr:.0%}** of the area that was observed to burn"
                  + (" \u2014 strong." if _hr >= 0.7 else
                     " \u2014 moderate." if _hr >= 0.4 else
                     " \u2014 weak: the spread direction or speed misses "
                     "a large part of the real fire."))
        if _ab >= 1:
            _v.append(f"**Size** \u2014 the simulated burn is "
                      f"**{_ab:.1f}\u00d7** the observed area. Values well "
                      "above 1 are expected against satellite truth: the "
                      "real fire was actively suppressed while the "
                      "simulation runs free, and detections undersample "
                      "the burned interior.")
        else:
            _v.append(f"**Size** \u2014 the simulated burn is only "
                      f"**{_ab:.1f}\u00d7** the observed area: the model "
                      "underspreads on this case.")
        _v.append(f"**Front position** \u2014 on average the simulated "
                  f"fire edge sits **{_fe/1000:.1f} km** from the observed "
                  "edge"
                  + (" \u2014 good for a landscape-scale hindcast."
                     if _fe < 2000 else
                     " \u2014 fair; direction is close but the run "
                     "length differs." if _fe < 5000 else
                     " \u2014 large; check wind input and window."))
        _v.append(f"**Overlap (Dice {_m['dice']:.2f})** \u2014 the single "
                  "headline score; against an official perimeter 0.5+ is "
                  "publishable for an uncalibrated model.")
        st.markdown("**What do these numbers say?**")
        for _line in _v:
            st.markdown("- " + _line)
        if res.get("truth") == "firms":
            st.caption("Truth = satellite detections: they undersample the "
                       "burned interior and the real fire was actively "
                       "SUPPRESSED while the simulation runs free, so the "
                       "primary scores here are **hit rate** and the front "
                       "error; false alarm and area bias only become "
                       "meaningful against an official EFFIS/EMS perimeter "
                       "(upload one above). Reference: Sorensen-Dice "
                       "0.5\u20130.7 is typical for uncalibrated "
                       "semi-empirical models (VALIDATION.md).")
        else:
            st.caption("Reference: Sorensen-Dice 0.5\u20130.7 is typical "
                       "for uncalibrated semi-empirical models, 0.7+ after "
                       "calibration on a separate fire (VALIDATION.md).")
        nx, ny = res["shape"]
        img = (res["base"].copy() if res.get("base") is not None
               else _np.zeros((ny, nx, 3), dtype=_np.uint8) + 24)
        best, obs = res["best"], res["obs"]
        img[best & obs] = (46, 160, 67)
        img[best & ~obs] = (200, 55, 44)
        img[~best & obs] = (58, 110, 220)
        if res.get("ign") is not None:
            _i0 = (res["ign"][0] if isinstance(res["ign"], list)
                   else res["ign"])
            gx, gy = int(_i0[0]), int(_i0[1])
            rr = max(2, nx // 150)
            img[max(0, gy - rr):gy + rr + 1,
                max(0, gx - rr):gx + rr + 1] = (255, 235, 60)
        st.image(img, use_container_width=True,
                 caption="Agreement map on the real terrain \u2014 green: "
                         "correctly predicted burn, red: simulated only, "
                         "blue: observed only, yellow: ignition")
        if st.session_state.get("val_log"):
            with st.expander("Run log \u2014 what happened, step by step"):
                st.code("\n".join(st.session_state["val_log"]),
                        language=None)
        _ens = st.session_state.get("val_ens")
        if _ens:
            st.markdown("**Wind-direction uncertainty ensemble** \u2014 "
                        "same fire, wind rotated:")
            rows = ["| rotation | Dice | Hit rate | Front err (km) |",
                    "|---|---|---|---|"]
            for m in sorted(_ens, key=lambda m: m["offset"]):
                mark = " **\u2190 best**" if m is max(
                    _ens, key=lambda x: x["dice"]) else ""
                rows.append(f"| {m['offset']:+.0f}\u00b0 | "
                            f"{m['dice']:.3f} | {m['hit_rate']:.3f} | "
                            f"{m['mean_m']/1000:.1f}{mark} |")
            st.markdown("\n".join(rows))
            bestm = max(_ens, key=lambda x: x["dice"])
            st.caption(f"Best member at {bestm['offset']:+.0f}\u00b0 "
                       f"(Dice {bestm['dice']:.2f} vs "
                       f"{[m for m in _ens if m['offset']==0][0]['dice']:.2f} "
                       "with the raw reanalysis wind). A large gain from "
                       "rotation means the case is limited by the INPUT "
                       "wind (terrain channeling, fire convection), not by "
                       "the spread model \u2014 report it as input "
                       "sensitivity.")


PAGES = {"Simulation": page_simulation, "Map editor": page_editor,
         "Data layers": page_layers, "Parameters": page_params,
         "GIS import": page_gis, "Validation": page_validation,
         "System Description": page_system_description}
PAGES[page]()

# animate: advance one step per rerun while playing (fast: only this page runs)
if st.session_state.get("anim_on", False):
    _pending = any(ev.step >= sim.state.step for ev in world.ignitions)
    _finished = sim.is_quiescent() and (
        sim.ever_burned.any() or (not _pending and sim.state.step > 1))
    if not _finished and sim.state.step < cfg.max_steps:
        sim.step(); _record_costs(); time.sleep(0.08); st.rerun()
    else:
        # cannot write a widget key after the toggle rendered; flag it for
        # the next run (handled just before the toggle in the sidebar)
        st.session_state["anim_stop"] = True
        st.rerun()
