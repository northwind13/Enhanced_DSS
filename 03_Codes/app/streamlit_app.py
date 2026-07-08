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

from disaster_phyengine import (Simulator, World, SimConfig, Asset, compute_costs,
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
    from disaster_phyengine.layers import (MeteoLayer, TopoLayer, FuelLayer,
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


# ---- engine freshness gate: a zombie server or stale module otherwise
# surfaces as confusing TypeErrors deep inside the pages ----
import disaster_phyengine as _dpe
import dss as _dss_pkg
_EXPECTED_ENGINE_BUILD = 4
_EXPECTED_DSS_BUILD = 2
if (getattr(_dpe, "ENGINE_BUILD", 0) != _EXPECTED_ENGINE_BUILD
        or getattr(_dss_pkg, "DSS_BUILD", 0) != _EXPECTED_DSS_BUILD):
    st.error(
        "**Old engine code is still running in this process.**\n\n"
        "The files on disk are up to date, but an older DisasterAware "
        "server is still alive and your browser is talking to it (check "
        "the port in the address bar vs. the one the launcher printed).\n\n"
        "Fix:\n"
        "1. Close **every** DisasterAware terminal window \u2014 or run "
        "`taskkill /F /IM python.exe` in a command prompt,\n"
        "2. start `run_dashboard.bat` again,\n"
        "3. open the **new** URL it prints (usually "
        "http://localhost:8501).")
    st.stop()

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
# the engine's headless safety cap (500) is far too low for interactive use:
# lift it once; the exact value stays adjustable in Parameters
if not st.session_state.get("maxsteps_migrated"):
    if cfg.max_steps <= 500:
        cfg.max_steps = 100_000
    st.session_state["maxsteps_migrated"] = True


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



def _clock_info():
    """(label, brightness) of the simulation clock for the map display."""
    import math as _m
    import datetime as _dt
    sm = float(getattr(cfg, "step_minutes", 1.0))
    t_min = sim.state.step * sm
    mode = st.session_state.get("wx_mode", "Manual")
    if mode == "Real case weather" and st.session_state.get("real_wx"):
        wx = st.session_state["real_wx"]
        ts = (_dt.datetime.fromisoformat(wx["start"])
              + _dt.timedelta(hours=float(wx["h0"]), minutes=t_min))
        hh = ts.hour + ts.minute / 60.0
        label = ts.strftime("%Y-%m-%d %H:%M UTC")
    elif mode == "Diurnal cycle":
        start = float(st.session_state.get("dr_start", 12.0))
        tot = start * 60.0 + t_min
        hh = (tot / 60.0) % 24.0
        label = f"Day {int(tot // 1440) + 1} \u00b7 " \
                f"{int(hh):02d}:{int((hh % 1) * 60):02d}"
    else:
        return f"t = {_fmt_sim_time(t_min)}", 1.0
    bright = 0.60 + 0.40 * max(0.0, _m.cos((hh - 13.0) / 24.0
                                           * 2.0 * _m.pi)) ** 0.8
    return label, bright


def _step_sim(n: int = 1):
    """Advance the simulation, driving the exogenous weather first.

    With the diurnal cycle on, air temperature and humidity follow a daily
    wave (peak mid-afternoon) and dead fuel moisture tracks them through the
    EMC model. Nights are cool and humid, so the moisture can exceed the
    extinction threshold m_ext and the fire stalls or dies out on its own -
    the same mechanism that stops real fires overnight."""
    import math as _m
    diag = None
    mode = st.session_state.get("wx_mode", "Manual")
    for _ in range(int(n)):
        t_min = sim.state.step * float(getattr(cfg, "step_minutes", 1.0))
        if mode == "Real case weather" and st.session_state.get("real_wx"):
            wx = st.session_state["real_wx"]
            h = min(int(wx["h0"] + t_min // 60.0), len(wx["ws"]) - 1)
            world.meteo.wws[:] = float(wx["ws"][h])
            world.meteo.wwd[:] = _m.radians((270.0 - float(wx["wd"][h]))
                                            % 360.0)
            world.meteo.temp[:] = float(wx["t"][h])
            world.meteo.rh[:] = float(wx["rh"][h])
            if st.session_state.get("emc_on", True):
                from disaster_phyengine.fuel_moisture import (
                    update_dead_fuel_moisture)
                update_dead_fuel_moisture(world)
        elif mode == "Diurnal cycle":
            start = float(st.session_state.get("dr_start", 12.0))
            h = (start + t_min / 60.0) % 24.0
            phase = _m.cos((h - 15.0) / 24.0 * 2.0 * _m.pi)  # peak at 15:00
            td = float(st.session_state.get("dr_tday", 34.0))
            tn = float(st.session_state.get("dr_tnight", 22.0))
            rd = float(st.session_state.get("dr_rhday", 20.0))
            rn = float(st.session_state.get("dr_rhnight", 70.0))
            world.meteo.temp[:] = (td + tn) / 2.0 + (td - tn) / 2.0 * phase
            world.meteo.rh[:] = (rd + rn) / 2.0 - (rn - rd) / 2.0 * phase
            if st.session_state.get("emc_on", True):
                from disaster_phyengine.fuel_moisture import (
                    update_dead_fuel_moisture)
                update_dead_fuel_moisture(world)
        diag = sim.step()
        _net = st.session_state.get("dss_network")
        if _net is not None and st.session_state.get("dss_use_obs", True):
            _net.update(sim, float(getattr(cfg, "step_minutes", 1.0)))
    return diag


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
    import disaster_phyengine as _da
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
            _step_sim(); _record_costs(); st.rerun()
        if c2.button(f"Step {xsteps}", use_container_width=True,
                     help="Advance X steps at once."):
            _step_sim(xsteps); _record_costs(); st.rerun()
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
                _d = _step_sim(); _record_costs()
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
            sim.reset(); st.session_state.cost_series = []
            st.session_state.pop("dss_net_sig", None)
            st.rerun()
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

st.title("DisasterAware")
st.caption("Enhanced Decision Support System for Wildfire Disaster "
           "Response and Management")


# ============================================================== SIMULATION ===
def page_simulation():
    view_col, side_col = st.columns([2.9, 1.45], gap="medium")

    def _sv(key, default):
        return st.session_state.get(key, default)

    with side_col:
        # one panel at a time: no scrolling, DSS first
        _panels = ["DSS", "Conditions", "Ignition", "Display"]
        panel = st.radio("Panel", _panels, horizontal=True,
                         label_visibility="collapsed", key="sim_panel")

        if panel == "Conditions":
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
            st.session_state["emc_on"] = emc
            _modes = ["Manual", "Diurnal cycle", "Real case weather"]
            _mcur = st.session_state.get("wx_mode", "Manual")
            wxm = st.radio(
                "Weather source", _modes,
                index=_modes.index(_mcur) if _mcur in _modes else 0,
                help="Manual: the sliders above stay fixed. Diurnal "
                     "cycle: a synthetic day/night wave (afternoon "
                     "peak); humid nights push fuel moisture above "
                     "$m_{ext}$ and the fire can die out overnight. "
                     "Real case weather: the hourly ERA5 series already "
                     "downloaded for a validation case drives wind, "
                     "temperature and humidity with real timestamps.")
            st.session_state["wx_mode"] = wxm
            st.session_state["diurnal_on"] = (wxm == "Diurnal cycle")
            if wxm == "Real case weather":
                import glob as _gl
                import json as _js
                _root = os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))
                _found = {}
                for _wf in _gl.glob(os.path.join(
                        _root, "validation", "cache", "*",
                        "weather_*.json")):
                    _cid = os.path.basename(os.path.dirname(_wf))
                    _found[_cid] = _wf
                if not _found:
                    st.warning("No cached case weather yet \u2014 run "
                               "a case once on the Validation page.")
                else:
                    _cids = sorted(_found)
                    _csel = st.selectbox("Case weather", _cids,
                                         key="wx_case")
                    _h0 = st.number_input(
                        "Start hour (UTC) in the series", 0, 23, 11, 1,
                        key="wx_h0",
                        help="Simulation step 0 maps to this hour of "
                             "the case's first day (fires typically "
                             "start late morning).")
                    try:
                        _av2 = _load_auto_validate()
                        _start = _av2.CASES.get(_csel, {}).get(
                            "start", "2021-01-01")
                    except Exception:
                        _start = "2021-01-01"
                    _js0 = _js.load(open(_found[_csel]))
                    st.session_state["real_wx"] = {
                        "start": _start, "h0": float(_h0),
                        "ws": _js0["wind_speed_10m"],
                        "wd": _js0["wind_direction_10m"],
                        "t": _js0["temperature_2m"],
                        "rh": _js0["relative_humidity_2m"]}
                    st.caption(f"\u2713 {_csel}: {_start} + "
                               f"{int(_h0):02d}:00 UTC \u2014 wind, "
                               "T and RH follow the real series; the "
                               "map clock shows the real date-time.")
            if wxm == "Diurnal cycle":
                st.session_state["dr_start"] = float(st.number_input(
                    "Start hour of day", 0.0, 23.0,
                    float(st.session_state.get("dr_start", 12.0)), 1.0,
                    help="What time of day step 0 corresponds to."))
                dc1, dc2 = st.columns(2)
                st.session_state["dr_tday"] = float(dc1.number_input(
                    "Day $T$ (\u00b0C)", -10.0, 50.0,
                    float(st.session_state.get("dr_tday", 34.0)), 1.0))
                st.session_state["dr_tnight"] = float(dc2.number_input(
                    "Night $T$ (\u00b0C)", -20.0, 40.0,
                    float(st.session_state.get("dr_tnight", 22.0)), 1.0))
                st.session_state["dr_rhday"] = float(dc1.number_input(
                    "Day RH (%)", 5.0, 100.0,
                    float(st.session_state.get("dr_rhday", 20.0)), 5.0))
                st.session_state["dr_rhnight"] = float(dc2.number_input(
                    "Night RH (%)", 5.0, 100.0,
                    float(st.session_state.get("dr_rhnight", 70.0)), 5.0))
                _tmin = sim.state.step * float(getattr(cfg,
                                                       "step_minutes",
                                                       1.0))
                st.caption(f"Sim clock: {(_tmin/60.0) % 24.0:04.1f} h "
                           "of day (fire behaviour peaks ~15:00, "
                           "minimum before dawn).")
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
            from disaster_phyengine.spread import rate_of_spread as _rosf
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
                from disaster_phyengine.fuel_moisture import update_dead_fuel_moisture
                update_dead_fuel_moisture(world)
            else:
                _last_mo = st.session_state.get("last_mo")
                if _last_mo is not None and abs(mo - _last_mo) > 1e-9:
                    world.fuel.fmoist[:] = mo
                st.session_state["last_mo"] = mo

        elif panel == "Ignition":
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
            st.session_state["ig_live_v"] = bool(ig_live)
            st.session_state["ig_step_v"] = int(ig_step)
            st.session_state["ig_rad_v"] = int(ig_rad)

        elif panel == "DSS":
            import dss as _dss
            st.caption("Local DSS = regional decision agent (a block of "
                       "cells); resources are what agents allocate. One "
                       "Global DSS coordinates the regional agents.")
            # ---- sensor network (partial observation) ----
            st.markdown("**Sensor network**")
            use_obs = st.checkbox(
                "Partial observation via sensors",
                value=bool(_sv("dss_use_obs", True)),
                help="ON: agents only know what the sensors deliver "
                     "\u2014 uncovered or stale cells keep old values and "
                     "the observation confidence $conf$ (min of "
                     "observability $\\theta$, coverage "
                     "$\\rho$, freshness $e^{-\\lambda_{conf}\\Delta "
                     "t}$, reliability $\\gamma$) decays. OFF: perfect "
                     "observation (debug mode).")
            st.session_state["dss_use_obs"] = use_obs
            _slist = list(_sv("dss_sensors", []))
            sa1, sa2, sa3, sa4 = st.columns([1.5, 0.9, 0.9, 0.9])
            _kinds = list(_dss.SENSOR_CATALOG)
            _kadd = sa1.selectbox(
                "Type", _kinds,
                format_func=lambda k: _dss.SENSOR_CATALOG[k]["label"],
                key="dss_sens_kind")
            _xadd = int(sa2.number_input("x", 0, cfg.nx - 1, cfg.nx // 2,
                                         key="dss_sens_x"))
            _yadd = int(sa3.number_input("y", 0, cfg.ny - 1, cfg.ny // 2,
                                         key="dss_sens_y"))
            sa4.markdown("<div style='height:1.75em'></div>",
                         unsafe_allow_html=True)
            if sa4.button("Add", use_container_width=True):
                _slist.append(dict(kind=_kadd, x=_xadd, y=_yadd))
                st.session_state["dss_sensors"] = _slist
                st.rerun()
            sb1, sb2 = st.columns(2)
            if sb1.button("Suggest network", use_container_width=True,
                          help="Placement rationale: satellite = national "
                               "capability, always tasked (whole-map B, I "
                               "baseline). Aerial recon over the highest "
                               "spread-potential fuels (where a fire would "
                               "run fastest). In-situ sensors at the "
                               "highest protection-priority asset (fuel "
                               "state where losses matter most). Field "
                               "report post at the best road access "
                               "(crews report from reachable ground)."):
                from disaster_phyengine.behavior import rate_of_spread_field
                _ros = rate_of_spread_field(world)
                _ry, _rx = np.unravel_index(int(np.argmax(_ros)),
                                            _ros.shape)
                _vp = world.priority_field()
                _vy, _vx = np.unravel_index(int(np.argmax(_vp)), _vp.shape)
                _roads = getattr(world, "roads", None)
                if _roads is not None and np.asarray(_roads).any():
                    _rr = np.asarray(_roads, dtype=float)
                    _ky, _kx = np.unravel_index(int(np.argmax(_rr)),
                                                _rr.shape)
                else:
                    _ky, _kx = cfg.ny // 2, cfg.nx // 2
                st.session_state["dss_sensors"] = [
                    dict(kind="satellite", x=0, y=0),
                    dict(kind="aerial", x=int(_rx), y=int(_ry)),
                    dict(kind="in_situ", x=int(_vx), y=int(_vy)),
                    dict(kind="field_report", x=int(_kx), y=int(_ky))]
                st.toast("Suggested: satellite (always) \u00b7 aerial @ "
                         f"max spread ({_rx},{_ry}) \u00b7 in-situ @ max "
                         f"asset priority ({_vx},{_vy}) \u00b7 field "
                         f"report @ road access ({_kx},{_ky})")
                st.rerun()
            if _slist and sb2.button("Clear sensors",
                                     use_container_width=True):
                st.session_state["dss_sensors"] = []
                st.rerun()
            if _slist:
                for _i, _sd in enumerate(_slist):
                    st.caption(f"{_i + 1}. {_sd['kind']} @ ({_sd['x']}, "
                               f"{_sd['y']})")
            elif use_obs:
                st.warning("No sensors placed \u2014 the agents are "
                           "BLIND ($conf \\approx 0$): they keep "
                           "assuming no fire. Add sensors or use 'Suggest "
                           "network'.")
            # (re)build the network when the map or the fleet changes
            _sig = (st.session_state.get("map_version"),
                    tuple((d["kind"], d["x"], d["y"]) for d in _slist))
            if st.session_state.get("dss_net_sig") != _sig:
                _net = _dss.SensorNetwork(
                    [_dss.Sensor(d["kind"], d["x"], d["y"])
                     for d in _slist], cfg.ny, cfg.nx, cfg.cell_size_m)
                _net.update(sim, 0.0)
                st.session_state["dss_network"] = _net
                st.session_state["dss_net_sig"] = _sig
            _net = st.session_state.get("dss_network")
            st.session_state["dss_sensors_draw"] = [
                (d["x"], d["y"],
                 (None if _dss.SENSOR_CATALOG[d["kind"]]["radius_m"] is None
                  else max(1, int(round(_dss.SENSOR_CATALOG[d["kind"]]
                                        ["radius_m"] / cfg.cell_size_m)))),
                 d["kind"], f"S{_i + 1} {d['kind']}")
                for _i, d in enumerate(_slist)] or None
            _obsnet = _net if (use_obs and _net is not None) else None
            n_agents = int(st.number_input(
                "Number of local DSS agents", 1, 12,
                int(_sv("dss_n", 1)), 1,
                help="The map is split into exactly this many regions "
                     "covering every cell (near-square blocks, Agent_1 "
                     "at the north-west)."))
            st.session_state["dss_n"] = n_agents
            _regs = _dss.partition_n(cfg.nx, cfg.ny, n_agents)
            show_all = st.checkbox(
                "Show all regions on the map",
                value=bool(_sv("dss_show_all", True)))
            st.session_state["dss_show_all"] = show_all
            st.session_state["dss_regions_all"] = (
                [(*r.box, r.name) for r in _regs] if show_all else None)
            _names = [r.name for r in _regs] + ["Global DSS"]
            _icur = min(int(_sv("dss_sel_i", 0)), len(_names) - 1)
            _selA = st.selectbox("Agent", _names, index=_icur)
            st.session_state["dss_sel_i"] = _names.index(_selA)
            if _selA == "Global DSS":
                st.session_state["dss_region"] = None
                _thr = float(st.slider(
                    "Attention threshold (urgency)", 0.0, 1.0,
                    float(_sv("dss_attn_thr", 0.35)), 0.05,
                    help="The coordinator ATTENDS a region when its "
                         "temporal urgency reaches this value (fire "
                         "inside or approaching); the rest are only "
                         "monitored. Attended regions burn hot on the "
                         "map, ignored ones stay dim."))
                st.session_state["dss_attn_thr"] = _thr
                _feats = {_r.name: _dss.ten_features(sim, _r,
                                                     network=_obsnet)
                          for _r in _regs}
                _att_flags = {n: f["temporal_urgency"] >= _thr
                              for n, f in _feats.items()}
                # full table: all 10 features as rows, one column per agent
                # (the attention mark travels in the column header)
                _short = {"fire_intensity": "fire intensity",
                          "spread_potential": "spread potential",
                          "weather_severity": "weather severity",
                          "ignition_proximity": "ignition proximity",
                          "fuel_load": "fuel load",
                          "asset_exposure": "asset exposure",
                          "resource_accessibility": "accessibility",
                          "access_road_status": "roads/egress",
                          "suppression_availability": "supp. availability",
                          "temporal_urgency": "temporal urgency"}
                _head = "| feature |"
                _sep = "|---|"
                for _r in _regs:
                    _mark = "\u25cf" if _att_flags[_r.name] else "\u2013"
                    _head += f" {_r.name} {_mark} |"
                    _sep += "---|"
                _rowsmd = [_head, _sep]
                if _obsnet is not None:
                    _kcells = " | ".join(
                        f"{_obsnet.region_conf(_r):.2f}" for _r in _regs)
                    _rowsmd.append(f"| **conf** | {_kcells} |")
                for _k in _dss.FEATURE_ORDER:
                    _cells = " | ".join(f"{_feats[_r.name][_k]:.2f}"
                                        for _r in _regs)
                    _lab = _short[_k]
                    if _k == "temporal_urgency":
                        _lab = f"**{_lab}**"
                    _rowsmd.append(f"| {_lab} | {_cells} |")
                st.markdown("\n".join(_rowsmd))
                _natt = sum(1 for v in _att_flags.values() if v)
                st.caption(f"\u25cf attended: {_natt} \u00b7 monitored "
                           f"only: {len(_regs) - _natt} (urgency \u2265 "
                           f"{_thr:.2f} \u21d2 attended). Allocation "
                           "logic arrives in the next phase.")
                if show_all:
                    st.session_state["dss_regions_all"] = [
                        (*_r.box, _r.name, bool(_att_flags[_r.name]))
                        for _r in _regs]
            else:
                _reg = _regs[_names.index(_selA)]
                st.session_state["dss_region"] = (*_reg.box, _reg.name)
                st.caption(f"{_reg.name}: x {_reg.x0}\u2013{_reg.x1 - 1}, "
                           f"y {_reg.y0}\u2013{_reg.y1 - 1} "
                           f"({_reg.n_cells} cells) \u2014 highlighted "
                           "on the map.")
                if _obsnet is not None:
                    _cf = _obsnet.region_conf(_reg)
                    st.progress(min(1.0, _cf),
                                text=f"conf \u2014 observation confidence: "
                                     f"{_cf:.2f}")
                    _cc = _obsnet.region_conf_components(_reg)
                    st.caption(" \u00b7 ".join(
                        f"{_ch}: {_v:.2f}" for _ch, _v in _cc.items())
                        + " \u00b7 " + _obsnet.coverage_note(_reg))
                    st.caption("Sensors are shared infrastructure; this "
                               "agent reads the fused observation "
                               "restricted to its own region "
                               "$\\Omega_i$.")
                _f = _dss.ten_features(sim, _reg, network=_obsnet)
                for _k in _dss.FEATURE_ORDER:
                    st.progress(min(1.0, float(_f[_k])),
                                text=f"{_k.replace('_', ' ')}: "
                                     f"{_f[_k]:.2f}")

        else:  # Display: layers + legend
            st.markdown("**Layers** \u2014 shared with the Map editor")
            _lyd = [("ly_relief_v", "Relief", True),
                    ("ly_fire_v", "Fire", True),
                    ("ly_val_v", "Protection value", True),
                    ("ly_roads_v", "Roads", True),
                    ("ly_grid_v", "Grid", False),
                    ("ly_per_v", "Fire perimeter", True)]
            for _k, _lab, _d in _lyd:
                st.session_state[_k] = st.checkbox(
                    _lab, value=bool(_sv(_k, _d)))
            st.markdown("**Legend**")
            st.markdown(legend_html(), unsafe_allow_html=True)

    # values needed by the map regardless of which panel is open
    flags = dict(show_hillshade=bool(_sv("ly_relief_v", True)),
                 show_fire=bool(_sv("ly_fire_v", True)),
                 show_value=bool(_sv("ly_val_v", True)),
                 show_roads=bool(_sv("ly_roads_v", True)),
                 show_grid=bool(_sv("ly_grid_v", False)),
                 show_perimeter=bool(_sv("ly_per_v", True)))
    ig_live = bool(_sv("ig_live_v", True))
    ig_step = sim.state.step if ig_live else int(_sv("ig_step_v", 0))
    ig_rad = int(_sv("ig_rad_v", 1))
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
            # 3D is a pure viewer by design: browsers do not deliver click
            # events on 3D charts. Place ignitions on the 2D map view.
            _clk3, _ = _clock_info()
            st.caption(f"\u23f1 {_clk3} \u00b7 drag to rotate, scroll to "
                       "zoom; the camera survives the steps. To place "
                       "ignitions, switch to the 2D map and tick 'Click map "
                       "to place ignition'.")
            fig = viz.fire_surface_figure(world, sim=sim)
            key3d = f"plot3d_{st.session_state.map_version}"
            st.plotly_chart(fig, use_container_width=True,
                            config={"scrollZoom": True}, key=key3d)
        else:
            place = st.checkbox("Click map to place ignition "
                                "(off = scroll to zoom / pan)", value=False,
                                key="sim_place")
            _clk, _nf = _clock_info()
            _dreg = st.session_state.get("dss_region")
            _rb, _rl = (_dreg[:4], _dreg[4]) if _dreg else (None, None)
            _rall = st.session_state.get("dss_regions_all")
            _sens = st.session_state.get("dss_sensors_draw")
            if place and HAS_CANVAS and not playing:
                bg = viz.render_pil(world, sim=sim, scale=scale,
                                    show_labels=True, clock_text=_clk,
                                    night_factor=_nf, region_box=_rb,
                                    region_label=_rl, regions=_rall,
                                    sensors=_sens, **flags)
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
                                        show_labels=True, clock_text=_clk,
                                        night_factor=_nf, region_box=_rb,
                                        region_label=_rl, regions=_rall,
                                        sensors=_sens, **flags))
            else:
                # paused: zoomable, pannable plotly with click to ignite; zoom is
                # preserved across steps via the figure uirevision
                ev2 = st.plotly_chart(
                    viz.map_figure_2d(world, sim=sim, scale=scale,
                                      clock_text=_clk, night_factor=_nf,
                                      region_box=_rb, region_label=_rl,
                                      regions=_rall, sensors=_sens,
                                      **flags),
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


# normalized [0,1] cost terms (Table 2.4); superscript labels because the
# installed matplotlib crashes on "$...$" mathtext titles
_J_TERMS = [
    ("J\u1d47\u1d58\u02b3\u207f", "j_burn", "#2e8b57", "burned area"),
    ("J\u1d43\u02e2\u02e2\u1d49\u1d57", "j_asset", "#d9822b", "asset loss"),
    ("J\u1d56\u1d52\u1d56", "j_pop", "#8e44ad", "population"),
    ("J\u02b3\u1d49\u02e2\u1d56", "j_resp", "#2c3e50", "response cost"),
    ("J\u1d48\u1d49\u02e1", "j_delay", "#7f8c8d", "response delay"),
]


def _cost_panel():
    import matplotlib.pyplot as plt
    rep = compute_costs(sim)
    d = rep.to_dict()
    thr = float(sim.cfg.cost.acceptance_fraction)
    st.divider()
    st.subheader("Cost function $J_k$")
    st.latex(r"J_k=w_1 J_k^{burn}+w_2 J_k^{asset}+w_3 J_k^{pop}"
             r"+w_4 J_k^{resp}+w_5 J_k^{del}")
    st.caption("Normalized cost-plus-loss of the run so far (System "
               "Description Sec. 14): each term is divided by its scenario "
               "reference scale, so every term and the weighted total lie in "
               "$[0,1]$. The weights encode operational priority. The dashed "
               f"line marks the acceptance threshold ({thr:g} of the "
               "do-nothing cost).")

    # physical impact
    m = st.columns(4)
    m[0].metric("Burned area (ha)", f"{rep.burned_area_ha:,.1f}")
    m[1].metric("Burned forest (ha)", f"{rep.burned_forest_ha:,.1f}")
    m[2].metric("Population exposed", f"{rep.population_exposed:,.0f}")
    m[3].metric("Asset value lost",
                f"{rep.asset_value_lost:,.1f} / {rep.asset_value_total:,.1f}")

    # normalized J terms and the total, all in [0,1]
    cc = st.columns(len(_J_TERMS) + 1)
    for i, (lab, key, _c, sub) in enumerate(_J_TERMS):
        cc[i].metric(f"{lab} \u00b7 {sub}", f"{d[key]:.3f}")
    cc[-1].metric("J \u00b7 TOTAL", f"{rep.j_total:.3f}")

    series = st.session_state.cost_series
    if len(series) > 1:
        steps = [r["step"] for r in series]
        st.markdown("##### $J$ terms over time \u2014 one chart per term")
        titles = {k: lab for lab, k, _c, _s in _J_TERMS}
        cells = st.columns(3)
        panels = [(lab, key, col, titles[key]) for lab, key, col, _ in _J_TERMS]
        panels.append(("J", "j_total", "#111111", "J (total)"))
        for i, (lab, key, col, ttl) in enumerate(panels):
            with cells[i % 3]:
                f, a = plt.subplots(figsize=(3.4, 2.0))
                a.plot(steps, [r[key] for r in series], color=col, lw=1.8)
                a.fill_between(steps, [r[key] for r in series],
                               color=col, alpha=0.15)
                if key == "j_total":
                    a.axhline(thr, color="#c0392b", lw=1.0, ls="--")
                a.set_title(ttl, fontsize=11)
                a.set_xlabel("step k", fontsize=8)
                a.set_ylim(-0.02, 1.02)
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
                from disaster_phyengine import fuels_standard
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
            # shadow-state shared with the Simulation page's Display panel:
            # a selection made on either page holds on both
            def _lyv(key, default):
                return bool(st.session_state.get(key, default))

            st.session_state["ly_relief_v"] = st.checkbox(
                "Relief", value=_lyv("ly_relief_v", True))
            st.session_state["ly_fire_v"] = st.checkbox(
                "Fire", value=_lyv("ly_fire_v", True))
            st.session_state["ly_val_v"] = st.checkbox(
                "Protection value", value=_lyv("ly_val_v", True),
                help="Tints asset cells by the protection priority "
                     "$V_{prio}$ (System Description Sec. 2.5): pale pink "
                     "= lower, deep purple = higher priority. Nothing "
                     "shows until the map has buildings, critical "
                     "facilities or population (Asset tool).")
            st.session_state["ly_roads_v"] = st.checkbox(
                "Roads", value=_lyv("ly_roads_v", True))
            st.session_state["ly_grid_v"] = st.checkbox(
                "Grid", value=_lyv("ly_grid_v", False))
            eflags = dict(show_hillshade=_lyv("ly_relief_v", True),
                          show_fire=_lyv("ly_fire_v", True),
                          show_value=_lyv("ly_val_v", True),
                          show_roads=_lyv("ly_roads_v", True),
                          show_grid=_lyv("ly_grid_v", False))
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
            # 3D is a pure preview: all editing happens on the 2D canvas
            _e = world.topo.elev
            st.caption(f"3D preview \u00b7 elevation {_e.min():.0f}\u2013"
                       f"{_e.max():.0f} m \u00b7 drag to rotate, scroll to "
                       "zoom. All tools work on the 2D canvas view; switch "
                       "back there to edit.")
            fig3 = viz.fire_surface_figure(world, sim=sim)
            key3e = f"edit3d_{st.session_state.map_version}"
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
        from disaster_phyengine.layers import ValueLayer
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
    from disaster_phyengine import behavior
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
                   r"$I_B = H\,w\,R$ (kW/m) and flame length "
                   r"$L = 0.0775\,I_B^{0.46}$ (m), the measures used by "
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

    with st.expander("Run limit"):
        cfg.max_steps = int(st.number_input(
            "max_steps \u2014 safety stop (steps)", 100, 1_000_000,
            int(cfg.max_steps), 100,
            help="Hard upper bound for 'Run to end' and the animation. "
                 "Purely a safety net against endless runs; the fire "
                 "normally stops by itself when it burns out."))
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
                 "$A_k$ (System Description Sec. 4). Default "
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
            help="Slope normalization S_max (System Description Sec. 7). "
                 "Default 0.7854 (45\u00b0).")
        cfg.intensity.fload_max = b.number_input(
            "$F_{max}$ — fuel normalization (norm. units)", 0.1, 5.0,
            float(cfg.intensity.fload_max), 0.1,
            help="Fuel normalization F_max (System Description Sec. 7). "
                 "Default 1.0.")
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

    with st.expander("Cost model (normalized cost-plus-loss, System Description Sec. 14)"):
        st.caption("Five priority weights over the normalized $[0,1]$ terms, "
                   "plus the reference scales and safeguards.")
        a, b = st.columns(2)
        cfg.cost.w_burn = a.number_input(
            "$w_1$ — burned area weight", 0.0, 10.0,
            float(cfg.cost.w_burn), 0.1, format="%.2f")
        cfg.cost.w_asset = b.number_input(
            "$w_2$ — asset loss weight", 0.0, 10.0,
            float(cfg.cost.w_asset), 0.1, format="%.2f")
        cfg.cost.w_pop = a.number_input(
            "$w_3$ — population exposure weight", 0.0, 10.0,
            float(cfg.cost.w_pop), 0.1, format="%.2f")
        cfg.cost.w_resp = b.number_input(
            "$w_4$ — response cost weight", 0.0, 10.0,
            float(cfg.cost.w_resp), 0.1, format="%.2f")
        cfg.cost.w_delay = a.number_input(
            "$w_5$ — response delay weight", 0.0, 10.0,
            float(cfg.cost.w_delay), 0.1, format="%.2f")
        cfg.cost.acceptance_fraction = b.number_input(
            "acceptance threshold (fraction of do-nothing)", 0.0, 1.0,
            float(cfg.cost.acceptance_fraction), 0.05, format="%.2f")
        cfg.cost.population_at_risk_fraction = a.number_input(
            "$\\rho_{risk}$ — population at risk fraction", 0.0, 1.0,
            float(cfg.cost.population_at_risk_fraction), 0.005, format="%.3f")
        cfg.cost.horizon_steps = b.number_input(
            "$H$ — scenario horizon (steps)", 1.0, 5000.0,
            float(cfg.cost.horizon_steps), 10.0)
        cfg.cost.capacity_reference = a.number_input(
            "total available capacity (response-cost reference)", 0.0, 1e6,
            float(cfg.cost.capacity_reference), 10.0)
        cfg.cost.delay_reference = b.number_input(
            "reference delay (response-delay reference)", 0.0, 1e5,
            float(cfg.cost.delay_reference), 5.0)


# ====================================================== SYSTEM DESCRIPTION ===
def page_system_description():
    """Full mathematical description of the model (System Description page).

    The content lives in app/system_description.py to keep this file small."""
    from system_description import render
    render()


# ============================================================== GIS IMPORT ===
def page_gis():
    st.subheader("Real-world maps")
    st.markdown("#### Automatic \u2014 download a real area "
                "(same open data as the Validation page)")
    st.caption("Terrain: Copernicus GLO-30 DEM \u00b7 fuel: ESA WorldCover "
               "10 m \u2192 fuel classes. Needs rasterio + internet on the "
               "first download; areas are cached under validation/cache/ "
               "and load offline afterwards.")
    try:
        _av = _load_auto_validate()
    except Exception:
        _av = None
    if _av is not None:
        rw1, rw2 = st.columns([1.4, 1])
        _opts = ["Custom bounding box"] + sorted(_av.CASES)
        _sel = rw1.selectbox(
            "Area", _opts,
            format_func=lambda k: (_av.CASES[k]["label"] + " (case area)"
                                   if k in _av.CASES else k))
        rw_cell = float(rw2.number_input("Cell size (m)", 30.0, 300.0, 90.0,
                                         10.0, key="rw_cell"))
        if _sel in _av.CASES:
            bbox = {k: _av.CASES[_sel][k]
                    for k in ("west", "south", "east", "north")}
            cdir = _sel
        else:
            b1, b2, b3, b4 = st.columns(4)
            bbox = dict(
                west=float(b1.number_input("West (lon)", -180.0, 180.0,
                                           31.30, 0.05, key="bb_w")),
                south=float(b2.number_input("South (lat)", -85.0, 85.0,
                                            36.72, 0.05, key="bb_s")),
                east=float(b3.number_input("East (lon)", -180.0, 180.0,
                                           31.80, 0.05, key="bb_e")),
                north=float(b4.number_input("North (lat)", -85.0, 85.0,
                                            37.08, 0.05, key="bb_n")))
            cdir = (f"custom_{bbox['west']:.2f}_{bbox['south']:.2f}_"
                    f"{bbox['east']:.2f}_{bbox['north']:.2f}")
        if st.button("Download and load into the simulator", type="primary"):
            if bbox["east"] <= bbox["west"] or bbox["north"] <= bbox["south"]:
                st.error("Bounding box is inverted: need west < east and "
                         "south < north.")
            else:
                try:
                    import rasterio  # noqa: F401
                    root = os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__)))
                    cache = os.path.join(root, "validation", "cache", cdir)
                    with st.status("Building the real-world map \u2026",
                                   expanded=True):
                        st.write("terrain + fuel (cache-aware)")
                        wnew = _av.build_real_world(bbox, rw_cell, cache)
                        st.write(f"grid {wnew.config.nx} \u00d7 "
                                 f"{wnew.config.ny} at {rw_cell:g} m")
                    _new_simulator(wnew)
                    st.success("Loaded. Paint assets and ignitions in the "
                               "Map editor, then simulate.")
                    st.rerun()
                except ImportError:
                    st.error("Needs rasterio: pip install rasterio, then "
                             "restart.")
                except Exception as exc:
                    st.error(f"Download/build failed: {exc}")
    st.divider()
    st.markdown("#### Manual \u2014 your own GeoTIFF rasters")
    st.caption("Load a real elevation and optional fuel raster. Needs rasterio.")
    g1, g2, g3 = st.columns(3)
    gnx = g1.number_input("grid nx", 10, 400, 120, key="gnx")
    gny = g2.number_input("grid ny", 10, 400, 80, key="gny")
    gcell = g3.number_input("cell size (m)", 1.0, 1000.0, 30.0, key="gcell")
    dem_file = st.file_uploader("Elevation raster (GeoTIFF)", type=["tif", "tiff"], key="dem")
    fuel_file = st.file_uploader("Fuel raster (GeoTIFF)", type=["tif", "tiff"], key="fuelr")
    if st.button("Import rasters", type="primary"):
        try:
            from disaster_phyengine import gis
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
                        "..", "validation", "auto_validate.py")
    spec = importlib.util.spec_from_file_location("auto_validate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def page_validation():
    import types as _types
    st.subheader("Validation \u2014 hindcast against a real fire")
    st.markdown(
        "A **hindcast**: the simulator receives ONLY the real inputs of a "
        "documented fire \u2014 real terrain (Copernicus GLO-30 DEM), real "
        "fuel map (ESA WorldCover 10 m \u2192 fuel classes), real hourly "
        "weather at the fire (ERA5: wind drives the spread, humidity drives "
        "fuel moisture via EMC) and the real initial fire front (first NASA "
        "FIRMS overpass) \u2014 and runs **blind**, without seeing the "
        "answer. The simulated burn is then scored against what actually "
        "burned.")
    with st.expander("Method \u2014 what is the ground truth?"):
        st.markdown(
            "**Primary ground truth** is the observed fire footprint. Two "
            "levels:\n\n"
            "1. **NASA FIRMS satellite detections** (automatic here): every "
            "375 m VIIRS pixel seen actively burning inside the simulated "
            "time window, consolidated by a morphological closing. Real, "
            "timestamped, citable \u2014 but it undersamples the burned "
            "interior and the real fire was actively **suppressed** while "
            "the simulation runs free, so hit rate and front error are the "
            "primary scores against it.\n"
            "2. **Official EFFIS / Copernicus EMS perimeter** (upload "
            "below): the mapped polygon of the final burned area \u2014 "
            "the referee-grade truth that makes Dice, false alarm and area "
            "bias meaningful. Sources: forest-fire.emergency.copernicus.eu "
            "(burnt areas), emergency.copernicus.eu (activations), "
            "mtbs.gov (US).\n\n"
            "Inputs come from open archives: Copernicus GLO-30 DEM (AWS), "
            "ESA WorldCover 2021 (AWS), ERA5 hourly via open-meteo, FIRMS "
            "VIIRS archive. Everything is fetched automatically and cached "
            "per case.")
    with st.expander("Metrics \u2014 definitions and published targets"):
        st.markdown(
            "$A$ = simulated burned area, $B$ = observed burned area.\n\n"
            "| Metric | Definition | Published reference |\n"
            "|---|---|---|\n"
            r"| Sorensen-Dice | $2\lvert A\cap B\rvert\,/\,(\lvert A\rvert+\lvert B\rvert)$ | $0.7$\u2013$0.9$ calibrated (Cell2Fire); $0.5$\u2013$0.7$ uncalibrated |"
            "\n"
            r"| Jaccard / IoU | $\lvert A\cap B\rvert\,/\,\lvert A\cup B\rvert$ | $>0.5$ good agreement |"
            "\n"
            r"| Hit rate (POD) | $\lvert A\cap B\rvert\,/\,\lvert B\rvert$ | $>0.7$ |"
            "\n"
            r"| False alarm ratio | $\lvert A\setminus B\rvert\,/\,\lvert A\rvert$ | $<0.3$ (perimeter truth) |"
            "\n"
            r"| Area bias | $\lvert A\rvert\,/\,\lvert B\rvert$ | $0.8$\u2013$1.2$ (perimeter truth) |"
            "\n"
            r"| Front error | $\overline{d}(\partial A,\partial B)$ \u2014 mean edge-to-edge distance | $1$\u2013$3$ cells |")
    with st.expander("Referee protocol \u2014 how to report this in the thesis"):
        st.markdown(
            "1. **Calibration/validation split**: tune free parameters on "
            "ONE fire only, freeze them, hindcast the other fires blind and "
            "report those.\n"
            "2. **Multi-seed**: spotting is stochastic \u2014 report mean "
            "\u00b1 sd over \u2265 5 seeds.\n"
            "3. **Sensitivity**: \u00b125% on wind, moisture, "
            "$\\Theta_{ign}$; plus the wind-direction ensemble below.\n"
            "4. **Baselines**: compare against a no-wind isotropic run and "
            "published FARSITE/Cell2Fire scores.\n"
            "5. **Temporal check**: simulated arrival times ($t_{ign}$ "
            "layer) vs FIRMS detection times.\n"
            "6. Archive every run (`validation_runs/`) for "
            "reproducibility.")
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
    _root0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _ccache = os.path.join(_root0, "validation", "cache", case_id)
    _f_new = os.path.join(_ccache,
                          f"firms_{av.CASES[case_id]['start']}.csv")
    _f_old = os.path.join(_ccache, "firms.csv")
    cached_truth = (os.path.exists(_f_new)
                    or (os.path.exists(_f_old)
                        and sum(1 for _ in open(_f_old)) > 1))
    key = c2.text_input("NASA FIRMS MAP_KEY"
                        + (" (not needed \u2014 case is cached)"
                           if cached_truth else ""),
                        type="password",
                        value=st.session_state.get("firms_key",
                                                   os.environ.get(
                                                       "FIRMS_MAP_KEY", "")),
                        help="Free key (1 minute): "
                             "https://firms.modaps.eosdis.nasa.gov/api/"
                             "map_key/ \u2014 only needed the FIRST time a "
                             "case is run; afterwards the satellite data is "
                             "cached and the run starts directly.")
    st.session_state["firms_key"] = key
    if cached_truth:
        st.caption(f"\u2713 {av.CASES[case_id]['label']}: all data cached "
                   "\u2014 press Run, no key required.")
    if cached_truth and not key:
        key = "cached"   # never used: the FIRMS file is read from cache
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
                                                         "validation",
                                                         "cache", case_id),
                                      out=os.path.join(root, "validation",
                                                       "runs", "report"))
        os.makedirs(args.cache, exist_ok=True)
        import datetime as _dt2
        import json as _json
        _stamp = _dt2.datetime.now().strftime("%Y%m%d-%H%M%S")
        _rdir = os.path.join(root, "validation", "runs",
                             f"{case_id}_{_stamp}")
        os.makedirs(os.path.join(_rdir, "frames"), exist_ok=True)
        # migrate any pre-restructure cache locations once
        for _flat in (os.path.join(root, "validation_cache", case_id),
                      os.path.join(root, "validation_cache")):
            if not os.path.isdir(_flat):
                continue
            for _fn in os.listdir(_flat):
                _o, _n = os.path.join(_flat, _fn), os.path.join(args.cache,
                                                                _fn)
                if os.path.isfile(_o) and not os.path.exists(_n):
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
                    from disaster_phyengine.gis import _read_resampled
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
                import disaster_phyengine as _da2
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
                st.success(f"Run archived: validation/runs/"
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
                       "semi-empirical models (see Metrics above).")
        else:
            st.caption("Reference: Sorensen-Dice 0.5\u20130.7 is typical "
                       "for uncalibrated semi-empirical models, 0.7+ after "
                       "calibration on a separate fire (see Metrics above).")
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
        _step_sim(); _record_costs(); time.sleep(0.08); st.rerun()
    else:
        # cannot write a widget key after the toggle rendered; flag it for
        # the next run (handled just before the toggle in the sidebar)
        if sim.state.step >= cfg.max_steps:
            st.toast(f"Animation stopped at the max_steps cap "
                     f"({cfg.max_steps}); raise it in Parameters "
                     "> Run limit.")
        else:
            st.toast("Animation stopped: the fire is out.")
        st.session_state["anim_stop"] = True
        st.rerun()
