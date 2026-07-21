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
_REFERENCE_FUELS = {fid: dataclasses.replace(m) for fid, m in FUEL_MODELS.items()}


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
    w2.resource = ResourceLayer(
        rcap=R(w.resource.rcap), ravail=R(w.resource.ravail),
        reff=R(w.resource.reff), rtime=R(w.resource.rtime),
        rair=(None if getattr(w.resource, "rair", None) is None
              else R(w.resource.rair)))
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
    # a NEW MAP invalidates the whole DSS setup: sensors and depots
    # carry coordinates of the old terrain, so everything is
    # cleared and 'Apply decisions' drops to OFF (a fire RESET, in
    # contrast, keeps the infrastructure and clears only the
    # decisions)
    for _k in ("dss_sensors", "dss_sens_edit", "dss_network",
               "dss_net_sig", "dss_sensors_draw", "dss_res_items",
               "dss_res_base", "dss_res_base_v", "dss_res_sig",
               "dss_res_why", "res_edit", "dss_depots_draw",
               "dss_suggest_why"):
        st.session_state.pop(_k, None)
    st.session_state["dss_apply"] = False
    try:
        _reset_dss_state(drop_engine=True)
    except Exception:
        pass


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
    _reset_dss_state()


def _reset_dss_state(drop_engine: bool = False) -> None:
    """A fire reset clears the DECISION state (gating priors, feature
    histories, per-run transients) but the engine SURVIVES: learned
    rules, membership moves and the controller value table are knowledge, not
    decisions, and persist across fires. drop_engine=True (map
    regeneration) discards the engine too."""
    for _k in list(st.session_state.keys()):
        if _k.startswith(("l3_gate_", "dss_featprev_")):
            del st.session_state[_k]
    _eng_fr = st.session_state.get("dss_engine")
    if drop_engine or _eng_fr is None:
        for _k in ("dss_engine", "dss_engine_sig"):
            st.session_state.pop(_k, None)
    else:
        try:
            _eng_fr.new_fire()
        except Exception:
            for _k in ("dss_engine", "dss_engine_sig"):
                st.session_state.pop(_k, None)


def _record_costs() -> None:
    st.session_state.cost_series.append(compute_costs(st.session_state.sim).to_dict())


def _fit_scale(nx) -> int:
    return int(max(4, min(16, 900 // max(nx, 1))))


_IV_COLOR = {"suppression_effort": "#2878ff",
             "resource_deployment": "#9aa0a6",
             "containment_line": "#96501e",
             "asset_protection": "#28dc5a",
             "evacuation": "#ff8c00",
             "public_warning": "#e6c400"}


def _iv_bar(label: str, value: float, color: str) -> str:
    """One colored intensity bar (the intervention palette matches
    the map icons and the legend)."""
    v = max(0.0, min(1.0, float(value)))
    return (
        "<div style='margin:3px 0'>"
        f"<span style='display:inline-block;width:11px;height:11px;"
        f"background:{color};border-radius:2px;margin-right:6px'>"
        "</span>"
        f"<span style='font-size:0.86em'>{label}: "
        f"<b>{v:.2f}</b></span>"
        "<div style='background:#8882;border-radius:3px;height:7px;"
        "margin-top:2px'>"
        f"<div style='width:{v * 100:.0f}%;background:{color};"
        "height:7px;border-radius:3px'></div></div></div>")


def _legend_swatch(hexc: str, glyph: str, px: int = 13) -> str:
    """One legend swatch that mimics the MAP icon of the item."""
    base = (f"width:{px}px;height:{px}px;display:inline-block;"
            "flex:none;box-sizing:border-box;")
    if glyph == "sq":
        return (f"<span style='{base}background:{hexc};"
                "border:1px solid #555'></span>")
    if glyph == "dot":
        return (f"<span style='{base}background:{hexc};"
                "border:1px solid #444;border-radius:50%'></span>")
    if glyph == "ring":
        return (f"<span style='{base}border:2.5px solid {hexc};"
                "border-radius:50%;background:transparent;"
                "box-shadow:0 0 0 1px #7773'></span>")
    if glyph == "box":
        return (f"<span style='{base}border:2.5px solid {hexc};"
                "background:transparent'></span>")
    if glyph == "tri":
        h = px
        return ("<span style='display:inline-block;flex:none;width:0;"
                f"height:0;border-left:{h // 2}px solid transparent;"
                f"border-right:{h // 2}px solid transparent;"
                f"border-bottom:{h}px solid {hexc};"
                "filter:drop-shadow(0 0 1px #555)'></span>")
    # literal text badge (e.g. the S/D/C/P/E/W order chip)
    return ("<span style='display:inline-block;flex:none;"
            f"background:#000c;color:{hexc};font-size:0.75em;"
            "padding:0 3px;border-radius:2px;font-family:monospace'>"
            f"{glyph}</span>")


def legend_html(horizontal: bool = False) -> str:
    groups = {}
    for grp, lab, hexc, glyph in viz.legend_entries():
        groups.setdefault(grp, []).append((lab, hexc, glyph))
    if horizontal:
        # one line per CATEGORY: the group name anchors the row and
        # its items stay together (wrapping within the row only)
        html = "<div style='font-size:0.8em;margin-top:2px'>"
        for grp, items in groups.items():
            html += ("<div style='display:flex;flex-wrap:wrap;"
                     "gap:3px 12px;align-items:center;margin:2px 0;"
                     "padding:1px 0;border-bottom:1px solid #8882'>"
                     f"<span style='font-weight:600;min-width:110px'>"
                     f"{grp}</span>")
            for lab, hexc, glyph in items:
                html += ("<span style='display:inline-flex;align-items:"
                         "center;gap:4px'>"
                         + _legend_swatch(hexc, glyph, px=11)
                         + f"<span>{lab}</span></span>")
            html += "</div>"
        return html + "</div>"
    html = "<div style='font-size:0.9em'>"
    for grp, items in groups.items():
        html += f"<div style='font-weight:600;margin:6px 0 2px'>{grp}</div>"
        for lab, hexc, glyph in items:
            html += ("<div style='display:flex;align-items:center;"
                     "gap:6px;margin:1px 0'>"
                     + _legend_swatch(hexc, glyph)
                     + f"<span>{lab}</span></div>")
    return html + "</div>"


# ---- engine freshness gate: a zombie server or stale module otherwise
# surfaces as confusing TypeErrors deep inside the pages ----
import disaster_phyengine as _dpe
import dss as _dss_pkg
_EXPECTED_ENGINE_BUILD = 35
_EXPECTED_DSS_BUILD = 58
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

    def _g(px, py):
        return px / scale, py / scale

    n = 0
    for obj in objects:
        otype = obj.get("type")
        if otype == "rect":
            left, top = obj.get("left", 0), obj.get("top", 0)
            w = obj.get("width", 0) * obj.get("scaleX", 1)
            h = obj.get("height", 0) * obj.get("scaleY", 1)
            x0, y0 = _clip(*_g(left, top))
            x1, y1 = _clip(*_g(left + w, top + h))
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
                cur = _g(px, py)
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
            gx, gy = _clip(*_g(left + rad, top + rad))
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
    m = int(round(minutes))
    d, rem = divmod(m, 1440)
    h, mm = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d} d")
    if h:
        parts.append(f"{h} h")
    if mm or not parts:
        parts.append(f"{mm} min")
    return " ".join(parts)


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


def _drive_weather(wobj, t_min: float) -> None:
    """Exogenous weather at simulated minute t_min, applied to wobj.

    Deterministic in t_min and the UI settings, so the counterfactual
    replay drives its CLONE through exactly the weather history the
    factual run saw (diurnal wave, EMC fuel-moisture drying, wind
    veer, shower windows). Without this the replay froze the weather
    at the rewind point and burned a different fire."""
    import math as _m
    mode = st.session_state.get("wx_mode", "Manual")
    if mode == "Real case weather" and st.session_state.get("real_wx"):
        wx = st.session_state["real_wx"]
        h = min(int(wx["h0"] + t_min // 60.0), len(wx["ws"]) - 1)
        wobj.meteo.wws[:] = float(wx["ws"][h])
        wobj.meteo.wwd[:] = _m.radians((270.0 - float(wx["wd"][h]))
                                       % 360.0)
        wobj.meteo.temp[:] = float(wx["t"][h])
        wobj.meteo.rh[:] = float(wx["rh"][h])
        if st.session_state.get("emc_on", True):
            from disaster_phyengine.fuel_moisture import (
                update_dead_fuel_moisture)
            update_dead_fuel_moisture(wobj)
    elif mode == "Diurnal cycle":
        start = float(st.session_state.get("dr_start", 12.0))
        h = (start + t_min / 60.0) % 24.0
        phase = _m.cos((h - 15.0) / 24.0 * 2.0 * _m.pi)  # peak 15:00
        td = float(st.session_state.get("dr_tday", 34.0))
        tn = float(st.session_state.get("dr_tnight", 22.0))
        rd = float(st.session_state.get("dr_rhday", 20.0))
        rn = float(st.session_state.get("dr_rhnight", 70.0))
        wobj.meteo.temp[:] = (td + tn) / 2.0 + (td - tn) / 2.0 * phase
        wobj.meteo.rh[:] = (rd + rn) / 2.0 - (rn - rd) / 2.0 * phase
        if st.session_state.get("emc_on", True):
            from disaster_phyengine.fuel_moisture import (
                update_dead_fuel_moisture)
            update_dead_fuel_moisture(wobj)
        # diurnal wind: calmer at night, stronger mid-afternoon,
        # direction veers; spatial pattern preserved by scaling a
        # captured base field rather than overwriting it
        _mv = st.session_state.get("map_version", 0)
        _bkey = f"_wind_base_{_mv}"
        if _bkey not in st.session_state:
            st.session_state[_bkey] = (wobj.meteo.wws.copy(),
                                       wobj.meteo.wwd.copy())
        _bws, _bwd = st.session_state[_bkey]
        _dayf = 0.55 + 0.55 * max(0.0, phase)
        _veer = 0.6 * _m.sin((h - 9.0) / 24.0 * 2.0 * _m.pi)
        wobj.meteo.wws[:] = np.clip(_bws * _dayf, 0.0, None)
        wobj.meteo.wwd[:] = _bwd + _veer
        wobj.meteo.gust[:] = wobj.meteo.wws * 1.4
        # optional shower window: only sets W_prec; the ENGINE wets
        # the fuel from precipitation every step and stops ember
        # spotting above 1 mm/h
        _mm = float(st.session_state.get("dr_rain_mm", 0.0))
        _rs = float(st.session_state.get("dr_rain_start", 18.0))
        _rd_ = float(st.session_state.get("dr_rain_dur", 3.0))
        _raining = _mm > 0 and ((h - _rs) % 24.0) < _rd_
        wobj.meteo.prec[:] = _mm if _raining else 0.0


def _step_sim(n: int = 1):
    """Advance the simulation, driving the exogenous weather first.

    With the diurnal cycle on, air temperature and humidity follow a daily
    wave (peak mid-afternoon) and dead fuel moisture tracks them through the
    EMC model. Nights are cool and humid, so the moisture can exceed the
    extinction threshold m_ext and the fire stalls or dies out on its own -
    the same mechanism that stops real fires overnight."""
    diag = None
    for _ in range(int(n)):
        t_min = sim.state.step * float(getattr(cfg, "step_minutes", 1.0))
        _drive_weather(world, t_min)
        _ov = None
        if (st.session_state.get("dss_apply")
                and st.session_state.get("dss_res_base") is not None
                and st.session_state.get("dss_res_base_v")
                == st.session_state.get("map_version")):
            import dss as _dss_step
            _net0 = st.session_state.get("dss_network")
            _obs0 = (_net0 if (st.session_state.get("dss_use_obs", True)
                               and _net0 is not None) else None)
            _sv0 = st.session_state.get
            _esig = (st.session_state.get("map_version"),
                     int(_sv0("dss_n", 1)),
                     float(_sv0("dss_cycle_min", 1.0)),
                     float(_sv0("dss_horizon_min", 30.0)),
                     float(_sv0("dss_jth", 0.35)),
                     float(_sv0("dss_eta", 0.60)),
                     bool(_sv0("dss_adapt_on", True)),
                     bool(_sv0("dss_genai_on", True)),
                     bool(_sv0("dss_evfis_on", True)),
                     float(_sv0("dss_evfis_step", 0.05)),
                     float(_sv0("dss_ctrl_eps", 0.10)),
                     float(_sv0("dss_ctrl_lr", 0.05)),
                     float(_sv0("dss_attn_thr", 0.35)),
                     float(_sv0("dss_min_gain", 0.05)),
                     str(_sv0("dss_seed_profile", "full")),)
            _eng = st.session_state.get("dss_engine")
            if _eng is None or st.session_state.get(
                    "dss_engine_sig") != _esig:
                import os as _os_rl
                _lg = _dss_step.RunLogger(
                    _os_rl.path.join(_os_rl.path.dirname(
                        _os_rl.path.dirname(_os_rl.path.abspath(
                            __file__))), "logs"),
                    tag=f"m{st.session_state.map_version}")
                _eng = _dss_step.DecisionEngine(
                    _dss_step.partition_n(cfg.nx, cfg.ny, _esig[1]),
                    base_pool=st.session_state["dss_res_base"],
                    network=_obs0, j_threshold=_esig[4], eta=_esig[5],
                    cycle_min=_esig[2], horizon_min=_esig[3],
                    evfis_step=_esig[9], adapt_on=_esig[6],
                    genai_on=_esig[7], evfis_on=_esig[8],
                    ctrl_eps=_esig[10], ctrl_lr=_esig[11],
                    attention_thr=_esig[12], min_gain=_esig[13],
                    seed_profile=_esig[14],
                    learned_store=_os_rl.path.join(
                        _os_rl.path.dirname(_os_rl.path.dirname(
                            _os_rl.path.abspath(__file__))),
                        "logs", "learned_rules.json"),
                    run_logger=_lg)
                try:
                    _lg.write_meta(dict(
                        map=dict(nx=cfg.nx, ny=cfg.ny,
                                 cell_m=cfg.cell_size_m,
                                 step_min=float(cfg.step_minutes)),
                        engine=dict(
                            regions=_esig[1], cycle_min=_esig[2],
                            horizon_min=_esig[3], j_th=_esig[4],
                            eta=_esig[5], adapt=_esig[6],
                            genai=_esig[7], evfis=_esig[8],
                            evfis_step=_esig[9],
                            ctrl_eps=_esig[10], ctrl_lr=_esig[11],
                            attn=_esig[12], min_gain=_esig[13],
                            seed_profile=_esig[14]),
                        weather=dict(
                            wx_mode=_sv0("wx_mode", "Manual"),
                            emc_on=bool(_sv0("emc_on", True)),
                            dr_start=float(_sv0("dr_start", 12.0)),
                            dr_tday=float(_sv0("dr_tday", 34.0)),
                            dr_tnight=float(_sv0("dr_tnight", 22.0)),
                            dr_rhday=float(_sv0("dr_rhday", 20.0)),
                            dr_rhnight=float(_sv0("dr_rhnight", 70.0)),
                            dr_rain_mm=float(_sv0("dr_rain_mm", 0.0)),
                            dr_rain_start=float(_sv0("dr_rain_start",
                                                     18.0)),
                            dr_rain_dur=float(_sv0("dr_rain_dur", 3.0))),
                        sensors=list(_sv0("dss_sensors", []) or []),
                        depots=list(_sv0("dss_res_items", []) or [])))
                    # the snapshot must be the t=0 BASELINE even if
                    # the engine is (re)built mid-run: swap in the
                    # pristine fuel state around the dump
                    _fl_bk = world.fuel.fload
                    _fm_bk = world.fuel.fmoist
                    world.fuel.fload = world.fuel.fload0.copy()
                    world.fuel.fmoist = getattr(
                        sim, "_fmoist0", world.fuel.fmoist).copy()
                    try:
                        _lg.save_world(world)
                    finally:
                        world.fuel.fload = _fl_bk
                        world.fuel.fmoist = _fm_bk
                except Exception:
                    pass
                _eng_prev = st.session_state.get("dss_engine")
                if (_eng_prev is not None
                        and getattr(_eng_prev, "seed_profile", None)
                        == _eng.seed_profile
                        and st.session_state.get("dss_engine_map")
                        == st.session_state.get("map_version")):
                    # a SETTINGS change rebuilds the engine, but the
                    # learned knowledge survives: rules and the controller
                    # value table transplant (membership moves live in
                    # the global registry, which the new engine
                    # resets; the tuned consequents ride along in
                    # the rules)
                    _eng.rules = _eng_prev.rules
                    _eng.controller.q = _eng_prev.controller.q
                st.session_state["dss_engine"] = _eng
                st.session_state["dss_engine_sig"] = _esig
                st.session_state["dss_engine_map"] = \
                    st.session_state.get("map_version")
            _eng.network = _obs0
            _eng.base_pool = st.session_state["dss_res_base"]
            _dtm = float(getattr(cfg, "step_minutes", 1.0))
            _cycm = (_eng.cycle_min if _eng.cycle_min is not None
                     else _eng.cycle_steps * _dtm)
            _due = (_eng.last_override is None
                    or (sim.state.step - _eng.last_cycle_step)
                    >= max(1, int(round(_cycm / _dtm))))
            if _due:
                with st.spinner("DSS decision cycle: shadow "
                                "forecasts + adaptation..."):
                    _ov = _eng.maybe_decide(sim)
            else:
                _ov = _eng.maybe_decide(sim)
        diag = sim.step(resource_override=_ov)
        _engL = st.session_state.get("dss_engine")
        if _engL is not None and _engL.run_logger is not None:
            try:
                _engL.run_logger.log_step(sim, compute_costs(sim),
                                          override=_ov)
            except Exception:
                pass
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
    if not _engine_ok:
        st.caption("\u26a0 engine outdated in memory")
    if not _engine_ok:
        st.error("The simulation engine changed on disk but the running "
                 "process still uses the old version (Python caches "
                 "imported modules). Close the app completely and start "
                 "it again (run_dashboard.bat). Until then rewind and "
                 "time-step scaling stay inactive.")

    # --- simulation control panel ---
    if st.session_state.pop("anim_stop", False):
        st.session_state.anim_on = False
        st.session_state.runend_on = False
    if st.session_state.pop("runend_stop", False):
        st.session_state.runend_on = False
    with st.container(border=True):
        st.markdown("**Simulation**")
        c1, c2, c0 = st.columns([1.0, 1.2, 0.9])
        _c0v = c0.container()
        xsteps = int(_c0v.number_input(
            "X", 1, 1000, 10, key="step_x",
            label_visibility="collapsed",
            help="How many steps the 'Step X' button advances at "
                 "once."))
        if c1.button("Step", use_container_width=True,
                     help="Advance the fire by one time step."):
            _step_sim(); _record_costs(); st.rerun()
        if c2.button(f"Step {xsteps}", use_container_width=True,
                     help="Advance X steps at once (set X in the box "
                          "beside)."):
            _step_sim(xsteps); _record_costs(); st.rerun()
        st.toggle("Animate step by step", key="anim_on",
                  help="Advance automatically, one step per refresh, until "
                       "the fire is over.")
        c3, c4 = st.columns(2)
        c3.toggle("Run to end", key="runend_on",
                  help="Latches ON and keeps running until the fire "
                       "is out or the step cap (max_steps) is "
                       "reached; press again to stop. The map "
                       "refreshes after every chunk.")
        if c4.button("Reset fire", use_container_width=True,
                     help="Clear the fire and the cost series; the map and "
                          "all edits stay."):
            sim.reset(); st.session_state.cost_series = []
            _reset_dss_state()
            st.session_state.pop("dss_net_sig", None)
            st.rerun()
        _telap = float(getattr(
            sim, "t_elapsed_min",
            sim.state.step * float(getattr(cfg, "step_minutes", 30.0))))
        st.caption(f"Step {sim.state.step} \u00b7 "
                   f"t = {_fmt_sim_time(_telap)}")
        st.caption("active fire "
                   f"{int((sim.state.burning > 0.5).sum())} cells")
        if st.session_state.get("runend_on"):
            _t0 = time.time()
            _limit = int(cfg.max_steps)
            _done = None
            while sim.state.step < _limit and time.time() - _t0 < 2.5:
                _d = _step_sim(); _record_costs()
                if (_d.n_burning == 0 and sim.state.step > 1
                        and sim.ever_burned.any()
                        and not any(ev.step >= sim.state.step
                                    for ev in world.ignitions)):
                    _done = "fire is out"
                    break
            if sim.state.step >= _limit:
                _done = f"step cap {_limit} reached"
            if _done:
                st.session_state["runend_stop"] = True
                st.toast(f"Run to end stopped at step "
                         f"{sim.state.step}: {_done}")
            st.rerun()
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
                    # the DSS rolls back WITH the physics: standing
                    # orders, overlays, logs and gating priors after
                    # the rewind point disappear (learned rules stay,
                    # they are knowledge, not decisions)
                    _eng_rw = st.session_state.get("dss_engine")
                    if _eng_rw is not None:
                        try:
                            _eng_rw.rewind_to(int(_k))
                        except Exception:
                            pass
                    for _kk in list(st.session_state.keys()):
                        if _kk.startswith(("l3_gate_",
                                           "dss_featprev_")):
                            del st.session_state[_kk]
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
        _panels = ["Layer 1 \u00b7 Input", "Layer 2 \u00b7 Perception",
                   "Layer 3 \u00b7 Concepts", "Layer 4 \u00b7 Decision",
                   "Rules", "Log", "Time", "Ignition", "Display"]
        # shadow state: a Step press ends its run with st.rerun() BEFORE
        # this radio is drawn again, so plain widget state would be
        # dropped and the panel would snap back to Layer 1. The shadow
        # variable keeps the selection across steps.
        _pcur = st.session_state.get("sim_panel_v", _panels[0])
        panel = st.radio("Panel", _panels, horizontal=True,
                         index=(_panels.index(_pcur)
                                if _pcur in _panels else 0),
                         label_visibility="collapsed", key="sim_panel3")
        st.session_state["sim_panel_v"] = panel
        import dss as _dss
        # the sensor network lives OUTSIDE the panels: the map overlay
        # and Layer 2 need it no matter which panel is open
        _slist = list(_sv("dss_sensors", []))
        use_obs = bool(_sv("dss_use_obs", True))
        # the resource pool is rasterized from its editable item rows
        # whenever the rows change (signature check), panel-independent
        _rit = st.session_state.get("dss_res_items")
        if _rit and (st.session_state.get("dss_res_base_v")
                     != st.session_state.get("map_version")):
            # SELF-HEAL: rows exist but the bookkeeping says another
            # map. If every row still fits this map, adopt the rows
            # (the user DID stage a pool; a stale version stamp must
            # not un-stage it); only rows that fall off the map void
            # the pool for real.
            if all((0 <= int(it.get("x", 0)) < cfg.nx
                    and 0 <= int(it.get("y", 0)) < cfg.ny)
                   for it in _rit if "x" in it):
                st.session_state["dss_res_base_v"] = \
                    st.session_state.get("map_version")
            else:
                st.session_state["dss_res_items"] = None
                st.session_state["dss_res_base"] = None
                _rit = None
        if _rit and (st.session_state.get("dss_res_base_v")
                     == st.session_state.get("map_version")):
            _rsig = tuple(sorted((it["kind"], it.get("x", -1),
                                  it.get("y", -1), it.get("cap", 0),
                                  it.get("radius", 0),
                                  it.get("avail", 1.0),
                                  it.get("t_disp", 10.0))
                                 for it in _rit))
            if st.session_state.get("dss_res_sig") != _rsig:
                _rl_new = _dss.build_resource_layer(world, _rit)
                st.session_state["dss_res_base"] = _rl_new
                st.session_state["dss_res_sig"] = _rsig
                # the response-cost term is charged as the FRACTION
                # of the staged pool committed (with the 1.2 surge),
                # so J_resp reads "how much of what exists is in the
                # field", not an absolute vs an arbitrary constant
                cfg.cost.capacity_reference = max(
                    100.0, 1.2 * float((_rl_new.rcap
                                        * _rl_new.ravail).sum()))
        elif st.session_state.get("dss_res_base") is not None \
                and not _rit:
            st.session_state["dss_res_base"] = None
        st.session_state["dss_depots_draw"] = ([
            (int(it["x"]), int(it["y"]), int(it.get("radius", 4)),
             float(it.get("cap", 0.8)),
             f"D{_k + 1}")
            for _k, it in enumerate(_rit or [])
            if it.get("kind") == "depot"] or None) if _rit else None
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

        if panel == "Layer 1 \u00b7 Input":
            st.caption("Layer 1 \u2014 input space: heterogeneous, "
                       "spatio-temporal sources. Sensing assets are "
                       "managed here; terrain, fuel, values, roads and "
                       "resources are known priors; weather drives "
                       "$U_{Meteo}$.")
            # ---- sensor network (partial observation) ----
            st.markdown("**Sensor network**")
            use_obs = st.checkbox(
                "Partial observation via sensors",
                value=bool(_sv("dss_use_obs", True)),
                help="ON: epistemic uncertainty enters the system "
                     "through partial observation \u2014 agents only know "
                     "what the sensors deliver, uncovered or stale cells "
                     "keep old values, and the per-cell confidence "
                     "$conf=\\min\\{\\gamma_{obs},\\gamma_{cov},"
                     "\\gamma_{fre},\\gamma_{rel}\\}$ decays "
                     "(observability, coverage, freshness "
                     "$e^{-\\lambda_{conf}\\Delta t}$, source "
                     "reliability). OFF: perfect observation (debug).")
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
            _covt = st.slider(
                "Coverage target (%)", 0, 100,
                int(_sv("dss_cov_target", 60)), 5,
                help="Suggest network keeps adding the best next "
                     "asset (families in rotation, greedy maximum "
                     "weighted coverage) until this fraction of "
                     "the risk-weighted map is covered. 0 = only "
                     "the satellite; 100 = blanket the risk.")
            st.session_state["dss_cov_target"] = int(_covt)
            if (st.session_state.get("dss_suggest_why")
                    and st.session_state.get("dss_cov_applied")
                    is not None
                    and int(_covt) != int(
                        st.session_state["dss_cov_applied"])):
                # the network came from the optimizer: a moved target
                # re-runs it live (adds or removes assets), exactly
                # like pressing 'Suggest network' again
                _pl, _why = _dss.suggest_network(
                    world, coverage_target=float(_covt))
                st.session_state["dss_sensors"] = _pl
                st.session_state["dss_suggest_why"] = _why
                st.session_state["dss_cov_applied"] = int(_covt)
                st.rerun()
            sb1, sb2 = st.columns(2)
            if sb1.button(
                    "Suggest network", use_container_width=True,
                    help="Optimized field deployment: greedy maximum "
                         "weighted coverage, i.e. place assets one at a "
                         "time to maximize $\\sum_{x,y} r(x,y)\\,"
                         "c(x,y)$ where the risk field "
                         "$r=0.45\\,\\hat R_{spread}+0.35\\,"
                         "\\hat V_{prio}+0.20\\,F_{load}$ and each "
                         "asset lifts the coverage $c$ of its footprint "
                         "by its sensing quality. Constraints: cameras "
                         "prefer high ground (line of sight), field "
                         "posts sit on the road network, public-report "
                         "sources are pinned to the settlements, one "
                         "satellite is always tasked, same-type assets "
                         "keep a footprint apart."):
                _pl, _why = _dss.suggest_network(
                    world,
                    coverage_target=float(_covt))
                st.session_state["dss_sensors"] = _pl
                st.session_state["dss_suggest_why"] = _why
                st.session_state["dss_cov_applied"] = int(_covt)
                st.rerun()
            if st.session_state.get("dss_suggest_why"):
                with st.expander("Why these positions (optimization "
                                 "trace)"):
                    for _ln in st.session_state["dss_suggest_why"]:
                        st.caption(_ln)
            if _slist and sb2.button("Clear sensors",
                                     use_container_width=True):
                st.session_state["dss_sensors"] = []
                st.rerun()
            if _slist:
                _net0 = st.session_state.get("dss_network")
                _stat = None
                if (_net0 is not None
                        and len(getattr(_net0, "sensors", [])) == len(_slist)
                        and hasattr(_net0, "status")):
                    _stat = _net0.status()
                for _i, _sd in enumerate(list(_slist)):
                    _c1, _c2, _c3 = st.columns([5, 0.7, 0.7])
                    _scol = {"satellite": "#aa78ff",
                             "aerial": "#00dcff",
                             "ground_camera": "#ffffff",
                             "in_situ": "#ffdc00",
                             "field_report": "#ff9640",
                             "public_report": "#ff69b4"}.get(
                                 _sd["kind"], "#78c8ff")
                    _full = _dss.SENSOR_CATALOG[_sd["kind"]]["label"]
                    _tx = (f"S{_i + 1} \u2014 {_full} @ "
                           f"({_sd['x']}, {_sd['y']})")
                    if _stat is not None:
                        _si = _stat[_i]
                        _lr = _si["last_report_min"]
                        _tx += (f" \u00b7 next pass "
                                f"{_si['next_pass_min']:.0f} min")
                        _tx += (f" \u00b7 data {_lr:.0f} min old"
                                if _lr is not None else
                                " \u00b7 no data yet")
                        if _si["in_transit"]:
                            _tx += (f" \u00b7 {_si['in_transit']} report "
                                    f"en route ({_si['latency_min']:.0f} "
                                    f"min latency)")
                    _c1.markdown(
                        f"<span style='color:{_scol}'>\u25cf</span> "
                        f"<small>{_tx}</small>",
                        unsafe_allow_html=True)
                    if _c2.button("\u270e", key=f"dss_sed_{_i}",
                                  help="Edit this sensor"):
                        st.session_state["dss_sens_edit"] = (
                            None if st.session_state.get("dss_sens_edit")
                            == _i else _i)
                        st.rerun()
                    if _c3.button("\u2716",
                                  key=(f"dss_srm_{_i}_{_sd['kind']}_"
                                       f"{_sd['x']}_{_sd['y']}"),
                                  help="Remove this sensor"):
                        _slist.pop(_i)
                        st.session_state["dss_sensors"] = _slist
                        st.session_state["dss_sens_edit"] = None
                        st.rerun()
                    if st.session_state.get("dss_sens_edit") == _i:
                        e1, e2, e3, e4 = st.columns([1.6, 0.9, 0.9, 0.8])
                        _nk = e1.selectbox(
                            "Type", _kinds,
                            index=_kinds.index(_sd["kind"]),
                            format_func=lambda k:
                                _dss.SENSOR_CATALOG[k]["label"],
                            key=f"dss_se_k_{_i}")
                        _nx_ = int(e2.number_input(
                            "x", 0, cfg.nx - 1, int(_sd["x"]),
                            key=f"dss_se_x_{_i}"))
                        _ny_ = int(e3.number_input(
                            "y", 0, cfg.ny - 1, int(_sd["y"]),
                            key=f"dss_se_y_{_i}"))
                        e4.markdown("<div style='height:1.75em'></div>",
                                    unsafe_allow_html=True)
                        if e4.button("Save", key=f"dss_se_s_{_i}",
                                     use_container_width=True):
                            _slist[_i] = dict(kind=_nk, x=_nx_, y=_ny_)
                            st.session_state["dss_sensors"] = _slist
                            st.session_state["dss_sens_edit"] = None
                            st.rerun()
            elif use_obs:
                st.warning("No sensors placed \u2014 the agents are "
                           "BLIND ($conf \\approx 0$): they keep "
                           "assuming no fire. Add sensors or use 'Suggest "
                           "network'.")
            st.divider()
            st.markdown("**Resource pool \u2014 $U_{Res}$ "
                        "($R_{cap}, R_{avail}, R_{eff}, R_{time}$)**")
            _ritems = st.session_state.get("dss_res_items")
            if (st.session_state.get("dss_res_base_v")
                    != st.session_state.map_version):
                _ritems = None      # map changed: the old pool is void
            st.session_state["dss_eff_target"] = int(st.slider(
                "Target effectiveness (%)", 10, 90,
                int(_sv("dss_eff_target", 50)), 5,
                help="The planner stages the baseline pool (depots + "
                     "road corridor + one helibase), then keeps ADDING "
                     "aerial units on the worst risk-weighted reach "
                     "gaps until the expected intervention "
                     "effectiveness meets this target (up to 10 "
                     "additions). Every added unit stays an editable "
                     "row below."))
            if (st.session_state.get("dss_res_why") and _ritems
                    and st.session_state.get("dss_eff_applied")
                    is not None
                    and int(st.session_state["dss_eff_target"])
                    != int(st.session_state["dss_eff_applied"])):
                _its, _rwhy = _dss.suggest_resource_items(
                    world, efficiency_target=float(
                        st.session_state["dss_eff_target"]) / 100.0)
                st.session_state["dss_res_items"] = _its
                st.session_state["dss_res_why"] = _rwhy
                st.session_state["dss_res_base_v"] = \
                    st.session_state.map_version
                st.session_state["dss_eff_applied"] = int(
                    st.session_state["dss_eff_target"])
                st.rerun()
            rp1, rp2 = st.columns(2)
            if rp1.button(
                    "Suggest resources", use_container_width=True,
                    help="Builds the baseline pool as EDITABLE rows: a "
                         "depot at every settlement / critical facility "
                         "(capacity $0.8\\,R_{cap}^{max}$, station "
                         "radius) plus a thin road-corridor capacity. "
                         "$R_{eff}$ comes from the terrain access "
                         "field, $R_{time}$ from the road-network "
                         "distance (10 min dispatch + 2 min per "
                         "off-road cell). The Layer 4 decisions "
                         "allocate THIS pool and cannot exceed 1.5x "
                         "the staged capacity anywhere."):
                _its, _rwhy = _dss.suggest_resource_items(
                    world, efficiency_target=float(
                        st.session_state["dss_eff_target"]) / 100.0)
                st.session_state["dss_res_items"] = _its
                st.session_state["dss_res_why"] = _rwhy
                st.session_state["dss_res_base_v"] = \
                    st.session_state.map_version
                st.session_state["dss_eff_applied"] = int(
                    st.session_state["dss_eff_target"])
                st.rerun()
            if _ritems and rp2.button("Clear pool",
                                      use_container_width=True):
                st.session_state["dss_res_items"] = None
                st.session_state["dss_res_base"] = None
                st.rerun()
            if st.session_state.get("dss_res_why") and _ritems:
                for _ln in st.session_state["dss_res_why"]:
                    st.caption(_ln)
            if _ritems:
                _bpe = st.session_state.get("dss_res_base")
                if _bpe is None:   # first render after Suggest
                    _bpe = _dss.build_resource_layer(world, _ritems)
                _pe, _ped = _dss.pool_efficiency(world, _bpe)
                st.progress(min(1.0, _pe),
                            text=f"Expected intervention "
                                 f"effectiveness: {_pe:.0%}")
                st.caption(f"= reach {_ped['reach']:.0%} \u00d7 "
                           f"capacity {_ped['capacity']:.0%} \u00b7 "
                           f"aerial covers {_ped.get('air', 0.0):.0%} "
                           "of the risk \u2014 "
                           "risk-weighted: can the crews reach "
                           "the ground that matters, and is the "
                           "staged pool big enough for it? Add "
                           "depots near the risk or raise their "
                           "capacity to push this up.")
            if _ritems:
                for _j, _it in enumerate(list(_ritems)):
                    _c1, _c2, _c3 = st.columns([5, 0.7, 0.7])
                    if _it["kind"] == "road_corridor":
                        _c1.caption(
                            f"{_j + 1}. road corridor \u00b7 "
                            f"$R_{{cap}}$ {_it['cap']:.2f} \u00b7 "
                            f"$R_{{avail}}$ "
                            f"{_it.get('avail', 1.0):.2f}")
                    else:
                        _c1.caption(
                            f"{_j + 1}. D{_j + 1} depot @ ({_it['x']}, "
                            f"{_it['y']}) \u00b7 $R_{{cap}}$ "
                            f"{_it['cap']:.2f} \u00b7 $R_{{avail}}$ "
                            f"{_it.get('avail', 1.0):.2f} \u00b7 "
                            f"dispatch {_it.get('t_disp', 10.0):.0f} min "
                            f"\u00b7 r {_it['radius']} \u00b7 "
                            f"{_it.get('label', '')}")
                    if _c2.button("\u270e", key=f"res_ed_{_j}",
                                  help="Edit this row"):
                        st.session_state["res_edit"] = (
                            None if st.session_state.get("res_edit")
                            == _j else _j)
                        st.rerun()
                    if _c3.button("\u2716", key=f"res_rm_{_j}",
                                  help="Remove this row"):
                        _ritems.pop(_j)
                        st.session_state["dss_res_items"] = _ritems
                        st.session_state["res_edit"] = None
                        st.rerun()
                    if st.session_state.get("res_edit") == _j:
                        if _it["kind"] == "road_corridor":
                            f1, f2, f3 = st.columns([1.0, 1.0, 0.8])
                            _nc = float(f1.number_input(
                                "$R_{cap}$", 0.0, 1.0, float(_it["cap"]),
                                0.05, key=f"res_e_c_{_j}"))
                            _na = float(f2.number_input(
                                "$R_{avail}$", 0.0, 1.0,
                                float(_it.get("avail", 1.0)), 0.05,
                                key=f"res_e_a_{_j}"))
                            f3.markdown("<div style='height:1.75em'>"
                                        "</div>", unsafe_allow_html=True)
                            if f3.button("Save", key=f"res_e_s_{_j}",
                                         use_container_width=True):
                                _it.update(cap=_nc, avail=_na)
                                st.session_state["res_edit"] = None
                                st.rerun()
                        else:
                            f1, f2, f3, f4 = st.columns(4)
                            _nx_ = int(f1.number_input(
                                "x", 0, cfg.nx - 1, int(_it["x"]),
                                key=f"res_e_x_{_j}"))
                            _ny_ = int(f2.number_input(
                                "y", 0, cfg.ny - 1, int(_it["y"]),
                                key=f"res_e_y_{_j}"))
                            _nc = float(f3.number_input(
                                "$R_{cap}$", 0.0, 1.0, float(_it["cap"]),
                                0.05, key=f"res_e_c_{_j}"))
                            _nr = int(f4.number_input(
                                "r (cells)", 1, 20, int(_it["radius"]),
                                key=f"res_e_r_{_j}"))
                            f5, f6, f7 = st.columns([1.0, 1.0, 0.8])
                            _na = float(f5.number_input(
                                "$R_{avail}$", 0.0, 1.0,
                                float(_it.get("avail", 1.0)), 0.05,
                                key=f"res_e_a_{_j}"))
                            _nt = float(f6.number_input(
                                "dispatch (min)", 0.0, 120.0,
                                float(_it.get("t_disp", 10.0)), 1.0,
                                key=f"res_e_t_{_j}"))
                            f7.markdown("<div style='height:1.75em'>"
                                        "</div>", unsafe_allow_html=True)
                            if f7.button("Save", key=f"res_e_s_{_j}",
                                         use_container_width=True):
                                _it.update(x=_nx_, y=_ny_, cap=_nc,
                                           radius=_nr, avail=_na,
                                           t_disp=_nt)
                                st.session_state["res_edit"] = None
                                st.rerun()
                a1, a2, a3, a4, a5 = st.columns(
                    [0.9, 0.9, 0.9, 0.9, 0.8])
                _ax = int(a1.number_input("x", 0, cfg.nx - 1,
                                          cfg.nx // 2, key="res_a_x"))
                _ay = int(a2.number_input("y", 0, cfg.ny - 1,
                                          cfg.ny // 2, key="res_a_y"))
                _ac = float(a3.number_input("cap", 0.0, 1.0, 0.8, 0.05,
                                            key="res_a_c"))
                _ar = int(a4.number_input("r", 1, 20, 4, key="res_a_r"))
                a5.markdown("<div style='height:1.75em'></div>",
                            unsafe_allow_html=True)
                if a5.button("Add", key="res_a_b",
                             use_container_width=True):
                    _ritems.append(dict(kind="depot", x=_ax, y=_ay,
                                        cap=_ac, radius=_ar, avail=1.0,
                                        t_disp=10.0,
                                        label="manual depot"))
                    st.session_state["dss_res_items"] = _ritems
                    st.rerun()
            else:
                st.caption("No pool staged \u2014 without a pool the "
                           "Layer 4 decisions have nothing to allocate.")
            st.divider()
            st.markdown("**Weather / environment drivers \u2014 $U_{Meteo}$**")
            ws = st.slider(
                "$W_{ws}$ — wind speed (m/s)", 0.0, 30.0,
                float(world.meteo.wws.mean()), 0.5,
                help="Synoptic (large-scale) wind. The engine modulates it "
                     "per cell by the terrain: exposed ridges speed it up, "
                     "valleys shelter it "
                     "($\\mathrm{clip}(1+g_{tw}(2\\hat e-1),0.4,1.8)$) "
                     "and on steep ground the direction veers toward the "
                     "local valley axis. Spread saturates toward $w_0$ "
                     "(Parameters).")
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
                       "(0\u00b0 = east, 90\u00b0 = north). Terrain bends "
                       "this synoptic direction toward valley axes on "
                       "steep ground.")
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
                rr1, rr2, rr3 = st.columns(3)
                st.session_state["dr_rain_mm"] = float(rr1.number_input(
                    "Rain (mm/h)", 0.0, 30.0,
                    float(st.session_state.get("dr_rain_mm", 0.0)), 1.0,
                    help="0 = no rain. Rain wets the fuel toward "
                         "extinction moisture and stops ember spotting "
                         "\u2014 fires die out under sustained rain."))
                st.session_state["dr_rain_start"] = float(rr2.number_input(
                    "Rain start (h)", 0.0, 23.0,
                    float(st.session_state.get("dr_rain_start", 18.0)),
                    1.0))
                st.session_state["dr_rain_dur"] = float(rr3.number_input(
                    "Duration (h)", 0.5, 24.0,
                    float(st.session_state.get("dr_rain_dur", 3.0)), 0.5))
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
            mo = st.slider(
                "$F_{moist}$ — fuel moisture, whole map (mass fraction)",
                0.0, 0.6,
                float(world.fuel.fmoist.mean()), 0.01, disabled=emc,
                help="Surface fuel moisture as a mass fraction, applied "
                     "uniformly to every cell. It is a daily weather "
                     "condition, which is why it lives here; spatially "
                     "varying moisture can be painted in the Map editor "
                     "(moving this slider overwrites painted values). "
                     "At the extinction moisture $m_{ext}$ of a fuel class "
                     "the spread stops entirely.")
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

        elif panel == "Layer 3 \u00b7 Concepts":
            st.caption("Layer 3 \u2014 concept space: the features are "
                       "fuzzified on the five-term partition and "
                       "aggregated up the four-level hierarchy with "
                       "weights $\\omega$; every activation is gated "
                       "by the observation confidence and blended with "
                       "a persistence prior before any rule reads it.")
            _regsC = _dss.partition_n(cfg.nx, cfg.ny, int(_sv("dss_n", 1)))
            _namesC = [r.name for r in _regsC] + ["All agents (table)"]
            _iC = min(int(_sv("dss_sel_i", 0)), len(_namesC) - 1)
            _selC = st.selectbox("Agent (region)", _namesC, index=_iC,
                                 key="l3_agent")
            if _selC == "All agents (table)":
                _pool3 = st.session_state.get("dss_res_base")
                _head = "| concept |" + "".join(
                    f" {_r.name} |" for _r in _regsC)
                _sep = "|---|" + "---|" * len(_regsC)
                _cols = {}
                _eng_h = getattr(st.session_state.get("dss_engine"),
                                 "hierarchy", None)
                for _r in _regsC:
                    _f3 = _dss.ten_features(sim, _r, network=_obsnet,
                                            pool=_pool3)
                    _g3 = _dss.concept_gates(
                        _dss.feature_confidence(_obsnet, _r),
                        hierarchy=_eng_h)
                    _gt = st.session_state.get(f"l3_gate_{_r.name}")
                    if _gt is None:
                        _gt = _dss.GatedConcepts()
                        st.session_state[f"l3_gate_{_r.name}"] = _gt
                    _cols[_r.name] = _dss.crisp(_gt.gate(
                        _dss.infer_concepts(_f3, hierarchy=_eng_h),
                        _g3, step=int(sim.state.step)))
                _rows3 = [_head, _sep]
                for _cn, (_l, _ins3) in (_eng_h
                                         or _dss.HIERARCHY).items():
                    _lab = _dss.CONCEPT_LABEL.get(
                        _cn, _cn.replace("_", " ") + " \U0001F7E9")
                    if _cn in _dss.DECISION_CONCEPTS:
                        _lab = f"**{_lab}** \u2605"
                    _cells = " | ".join(f"{_cols[_r.name][_cn]:.2f}"
                                        for _r in _regsC)
                    _rows3.append(f"| L{_l} {_lab} | {_cells} |")
                st.markdown("\n".join(_rows3))
                st.caption("gated (effective) activations, crisp "
                           "readout \u00b7 \u2605 = the five decision "
                           "concepts")
            else:
                _regC = _regsC[_namesC.index(_selC)]
                st.session_state["dss_region"] = (*_regC.box, _regC.name)
                _fC = _dss.ten_features(
                    sim, _regC, network=_obsnet,
                    pool=st.session_state.get("dss_res_base"))
                _fcC = _dss.feature_confidence(_obsnet, _regC)
                _eng_hC = getattr(st.session_state.get("dss_engine"),
                                  "hierarchy", None)
                _gamC = _dss.concept_gates(_fcC, hierarchy=_eng_hC)
                _gkey = f"l3_gate_{_regC.name}"
                _gater = st.session_state.get(_gkey)
                if _gater is None:
                    _gater = _dss.GatedConcepts()
                    st.session_state[_gkey] = _gater
                _actC = _dss.infer_concepts(_fC, hierarchy=_eng_hC)
                _effC = _gater.gate(_actC, _gamC, step=int(sim.state.step))
                _crO = _dss.crisp(_actC)
                _crE = _dss.crisp(_effC)
                st.caption("per-concept gate $\\gamma$ = min of the "
                           "feature confidences feeding the concept \u00b7 "
                           f"persistence $\\rho={_dss.RHO_PERSIST}$ \u00b7 "
                           "observed \u2192 gated (effective)")
                _lvlname = {1: "Level 1 \u2014 base", 2: "Level 2",
                            3: "Level 3", 4: "Level 4 \u2014 coordination"}
                for _lvl in (1, 2, 3, 4):
                    st.markdown(f"**{_lvlname[_lvl]}**")
                    for _cn, (_l, _ins) in (_eng_hC
                                            or _dss.HIERARCHY).items():
                        if _l != _lvl:
                            continue
                        _dec = _cn in _dss.DECISION_CONCEPTS
                        _lab = _dss.CONCEPT_LABEL.get(
                            _cn, _cn.replace("_", " ") + " \U0001F7E9")
                        if _dec:
                            _lab = f"{_lab} \u2605"
                        st.progress(min(1.0, _crE[_cn]),
                                    text=f"{_lab}: {_crO[_cn]:.2f} "
                                         f"\u2192 {_crE[_cn]:.2f} "
                                         f"(\u03b3 {_gamC[_cn]:.2f})")
                st.caption("\u2605 = the five decision concepts the "
                           "intervention rules read.")
            with st.expander("Feature \u2192 concept mapping "
                             "($\\omega$, Eq. 40) \u2014 explicit"):
                _fn = dict(_dss.FEATURE_NAME)
                _fn["access_road_status_inv"] = \
                    "access / road status (inverted)"
                for _cn, (_l, _ins) in _dss.HIERARCHY.items():
                    _parts = " + ".join(
                        f"{_fn.get(_src, _dss.CONCEPT_LABEL.get(_src, _src))}"
                        f" ({_w:.2f})" for _src, _w in _ins)
                    st.caption(f"L{_l} **{_dss.CONCEPT_LABEL[_cn]}** "
                               f"\u2190 {_parts}")
                st.caption("Single source of truth: dss/concepts.py "
                           "HIERARCHY \u00b7 weights are nonnegative "
                           "and sum to one per concept \u00b7 the same "
                           "table goes into the thesis as Table 4.X "
                           "(update prompt, item 2).")
        elif panel == "Layer 4 \u00b7 Decision":
            st.caption("Layer 4 \u2014 decision space (evolution). "
                       "These knobs configure the staged adaptation; "
                       "the stages themselves arrive with the decision "
                       "layer.")
            st.session_state["dss_apply"] = bool(st.toggle(
                "Apply decisions to the simulation",
                value=bool(_sv("dss_apply", False)),
                help="ON: every simulation step, each Local DSS turns "
                     "its intervention intensities into its region's "
                     "slice of $U_{DSS}=[R_{cap},R_{avail},R_{eff},"
                     "R_{time}]$ and the composed layer enters "
                     "sim.step(resource_override=...). The suppression "
                     "mapping (Eq. 130) converts it into fuel "
                     "reduction; nothing else touches the physics. "
                     "Requires a staged resource pool (Layer 1)."))
            if (st.session_state["dss_apply"]
                    and st.session_state.get("dss_res_base") is None):
                _rit_fix = st.session_state.get("dss_res_items")
                if _rit_fix:
                    # rows are staged but the raster is missing (a
                    # condition the bookkeeping missed): build it NOW
                    st.session_state["dss_res_base"] = \
                        _dss.build_resource_layer(world, _rit_fix)
                    st.session_state["dss_res_base_v"] = \
                        st.session_state.get("map_version")
                    st.caption("Resource pool rebuilt from the "
                               "staged rows.")
                else:
                    st.warning("No resource pool staged \u2014 go to "
                               "Layer 1 and press 'Suggest "
                               "resources'.")
            dc1, dc2, dc3 = st.columns(3)
            st.session_state["dss_cycle_min"] = float(dc1.number_input(
                "Decision cycle (min)", 1.0, 240.0,
                float(_sv("dss_cycle_min", 1.0)), 1.0,
                help="A decision is recomputed every this many SIM "
                     "minutes regardless of the step length; between "
                     "cycles the last allocation holds. NOTE: every "
                     "cycle boundary runs shadow forecasts (about "
                     "1-3 s wall time); with a very short cycle the "
                     "animation will stall on every frame. The 1 min "
                     "default reacts fastest; raise it if "
                     "Animate feels slow."))
            st.session_state["dss_horizon_min"] = float(dc2.number_input(
                "Forecast horizon (min)", 10.0, 480.0,
                float(_sv("dss_horizon_min", 30.0)), 5.0,
                help="Lookahead used to judge every candidate. With "
                     "the 1-min decision cycle the decision is "
                     "re-checked every minute but each check looks 30 "
                     "min ahead (adaptation trials and the no-harm "
                     "guard always re-check at >= 45 min). The "
                     "shadow run steps at a coarse 5-min tick (the "
                     "reference-step scaling keeps the physics "
                     "identical), so a 1-minute live tick still gets "
                     "a real horizon."))
            st.session_state["dss_min_gain"] = float(dc3.number_input(
                "Required forecast gain", 0.0, 0.5,
                float(_sv("dss_min_gain", 0.05)), 0.01,
                help="ADAPTIVE satisficing: a candidate must clear "
                     "$J_{TH}$ OR beat the no-action forecast by this "
                     "relative margin; otherwise the adaptation stages "
                     "engage. 0 restores the absolute-threshold-only "
                     "behaviour."))
            st.markdown("**Decision mode**")
            _md1, _md2 = st.columns(2)
            st.session_state["dss_evfis_on"] = bool(_md1.toggle(
                "evFIS adaptation (stages \u2460\u2461)",
                value=bool(_sv("dss_evfis_on", True)),
                help="Stage \u2460 tunes memberships/consequents, "
                     "stage \u2461 instantiates a finer rule cell. "
                     "OFF removes both from the controller's menu."))
            st.session_state["dss_genai_on2"] = bool(_md2.toggle(
                "GenAI proposals (stage \u2462)",
                value=bool(_sv("dss_genai_on2",
                               _sv("dss_genai_on", True))),
                help="Stage \u2462 asks the generative proposer for "
                     "a brand-new rule (4 admission gates). OFF "
                     "removes it from the controller's menu."))
            st.session_state["dss_genai_on"] =                 st.session_state["dss_genai_on2"]
            st.caption("Mode now: " + (
                "Fuzzy only \u2014 the seed rule base decides, no "
                "adaptation" if not (
                    st.session_state["dss_evfis_on"]
                    or st.session_state["dss_genai_on"])
                else "evFIS + GenAI \u2014 the stage controller picks among stages "
                     "\u2460\u2461\u2462" if (
                    st.session_state["dss_evfis_on"]
                    and st.session_state["dss_genai_on"])
                else "evFIS only \u2014 stages \u2460\u2461"
                if st.session_state["dss_evfis_on"]
                else "GenAI only \u2014 stage \u2462"))
            st.session_state["dss_adapt_on"] = bool(
                st.session_state["dss_evfis_on"]
                or st.session_state["dss_genai_on"])
            with st.expander(f"Rule base \u2014 all "
                             f"{len(_dss.SEED_RULES)} seed rules "
                             "(dss/rules.py)"):
                st.caption("The base is SPARSE: these seeds anchor the "
                           "response surface; the adaptation loop "
                           "instantiates further rules only for the "
                           "situations that demand them. The antecedent "
                           "space has $5^5=3125$ combinations; none of "
                           "the enumeration is needed.")
                for _r in _dss.SEED_RULES:
                    st.caption(_r.text())
            st.markdown("**evFIS \u2014 evolving fuzzy rule base**")
            st.session_state["dss_jth"] = float(st.slider(
                "$J_{TH}$ \u2014 acceptable cost threshold", 0.0, 1.0,
                float(_sv("dss_jth", 0.35)), 0.01,
                help="A candidate decision is applied as-is when its "
                     "cost satisfies $J \\le J_{TH}$; otherwise the "
                     "adaptation stages engage."))
            st.session_state["dss_eta"] = float(st.slider(
                "$\\eta$ \u2014 decision quality gate", 0.0, 1.0,
                float(_sv("dss_eta", 0.60)), 0.01,
                help="Adapted rules apply only while the decision "
                     "quality keeps $Q \\ge \\eta$; below it the "
                     "graduated fail-safe attenuates toward the safe "
                     "baseline."))
            st.session_state["dss_evfis_step"] = float(st.slider(
                "Membership adaptation step", 0.0, 0.2,
                float(_sv("dss_evfis_step", 0.05)), 0.01,
                help="How far stage \u2460 may move the membership "
                     "parameters (a,b,c,d) and the consequents per "
                     "adaptation."))
            st.markdown("**GenAI \u2014 rule proposer (stage \u2462)**")
            st.session_state["dss_genai_on"] = bool(st.checkbox(
                "Enable generative rule proposals",
                value=bool(_sv("dss_genai_on", True)),
                help="Proposed rules pass 4 gates before use: G1 format "
                     "\u00b7 G2 constraints \u00b7 G3 simulated "
                     "$\\Delta J<0$ \u00b7 G4 A/B."))
            st.session_state["dss_genai_temp"] = float(st.slider(
                "Proposal temperature", 0.0, 1.0,
                float(_sv("dss_genai_temp", 0.3)), 0.05,
                disabled=not st.session_state["dss_genai_on"]))
            # --- GenAI status: is stage 3 genuinely wired to Claude? ---
            _gcfg = _dss.genai_config()
            if _gcfg["key_present"]:
                st.success(
                    f"GenAI link: **Claude via API** \u2014 model "
                    f"`{_gcfg['model']}`, key `{_gcfg['key_masked']}`, "
                    f"transport {_gcfg['transport']}.")
            else:
                st.warning(
                    "GenAI link: **inactive** \u2014 no "
                    "`ANTHROPIC_API_KEY` in the environment, so stage "
                    "\u2462 is skipped (stages \u2460\u2461 still run). "
                    "Set the key and restart to enable Claude.")
            if st.button("Test Claude connection",
                         help="Sends ONE real, live request to the "
                              "configured model and shows Claude's own "
                              "reply \u2014 proof the generative stage "
                              "is wired to the API, not mocked."):
                with st.spinner("Calling Claude over the API\u2026"):
                    _pr = _dss.genai_probe()
                if _pr["ok"]:
                    _u = _pr.get("usage") or {}
                    st.success(
                        f"Live reply from Claude in {_pr['latency_ms']} ms "
                        f"(model `{_pr['reported_model']}`, "
                        f"tokens in/out "
                        f"{_u.get('input_tokens', '?')}/"
                        f"{_u.get('output_tokens', '?')}):")
                    st.code(_pr["reply"] or "(empty)", language=None)
                    st.caption(
                        "This text was generated by Claude just now over "
                        f"{_pr['endpoint']} \u2014 the same path stage "
                        "\u2462 uses to propose rules.")
                else:
                    st.error(f"No live link: {_pr['error']}")
            st.markdown("**Stage controller \u2014 associative search**")
            st.session_state["dss_ctrl_eps"] = float(st.slider(
                "$\\epsilon$ \u2014 exploration", 0.0, 1.0,
                float(_sv("dss_ctrl_eps", 0.10)), 0.01,
                help="Probability of trying a non-greedy adaptation "
                     "stage; reward = realized cost reduction."))
            st.session_state["dss_ctrl_lr"] = float(st.slider(
                "Learning rate", 0.001, 0.5,
                float(_sv("dss_ctrl_lr", 0.05)), 0.001, format="%.3f"))
            with st.expander("How the learning works \u2014 controller \u00b7 "
                             "evFIS \u00b7 GenAI, and what persists"):
                st.markdown(
                    "**Stage controller (associative search)** is an "
                    "$\\epsilon$-greedy contextual bandit. State = "
                    "the cost-deficit bucket (low / mid / high), "
                    "actions = the enabled adaptation stages, reward "
                    "= the realized forecast improvement of the "
                    "chosen stage. It is NOT pre-trained: it learns "
                    "online during the run, every untried "
                    "(bucket, stage) pair is tried once before "
                    "greedy takes over, and the value table is written "
                    "to the run log (cycles.jsonl) every cycle.\n\n"
                    "**evFIS (stages \u2460\u2461)** modifies the "
                    "ENGINE's own copy of the rule base: consequent "
                    "nudges and membership-shoulder moves that "
                    "survive for the lifetime of this engine. The "
                    "thesis seed catalog in dss/rules.py is never "
                    "touched.\n\n"
                    "**GenAI (stage \u2462)** proposes a brand-new "
                    "rule from the live concept situation \u2014 "
                    "with ANTHROPIC_API_KEY set the proposal comes "
                    "from Claude via the API (see the status line "
                    "above); a reachable model is required, so "
                    "without one the stage is skipped and logged, "
                    "never faked. Admitted rules (gates G1-G4) join "
                    "the runtime base with a G prefix.\n\n"
                    "**Lifecycle**: everything learned lives in the "
                    "running engine and its logs. A rebuilt engine "
                    "(changed settings, new map) starts from the "
                    "thesis seed state again. The button below "
                    "resets by hand.")
            if st.button("Reset learned adaptations",
                         help="Drops the adaptation-born rules "
                              "(A*/G*), restores every consequent "
                              "and membership to the thesis tables "
                              "(D.3 / E.1) and clears the controller "
                              "value table."):
                from dss.adapt import reset_partitions as _rsp
                _rsp()
                _eng_rs = st.session_state.get("dss_engine")
                if _eng_rs is not None:
                    _eng_rs.rules = _dss.make_runtime_rules()
                    _eng_rs.controller.q.clear()
                st.toast("Seed state restored: Table D.3 "
                         "memberships, Table E.1 rules, empty "
                         "value table.")
                st.rerun()
            st.divider()
            st.markdown("**Candidate intervention \u2014 rule base "
                        "(Appendix D seeds)**")
            _regs4 = _dss.partition_n(cfg.nx, cfg.ny,
                                      int(_sv("dss_n", 1)))
            _names4 = [r.name for r in _regs4]
            _eng4r = st.session_state.get("dss_engine")

            _eng_hX = getattr(_eng4r, "hierarchy", None)

            def _cand_for(_regX):
                _fX = _dss.ten_features(sim, _regX, network=_obsnet)
                _gamX = _dss.concept_gates(
                    _dss.feature_confidence(_obsnet, _regX),
                    hierarchy=_eng_hX)
                _gX = st.session_state.get(f"l3_gate_{_regX.name}")
                if _gX is None:
                    _gX = _dss.GatedConcepts()
                    st.session_state[f"l3_gate_{_regX.name}"] = _gX
                _effX = _gX.gate(
                    _dss.infer_concepts(_fX, hierarchy=_eng_hX),
                    _gamX, step=int(sim.state.step))
                return _dss.evaluate_rules(
                    _effX, _fX,
                    _eng4r.rules if _eng4r is not None else None,
                    macros=getattr(_eng4r, "macros", None) or None)

            _glbT = getattr(st.session_state.get("dss_engine"),
                            "last_global", None)
            st.markdown("**Global DSS**")
            if _glbT:
                st.markdown(_glbT["statement"])
                st.caption("Ranking by operational priority: "
                           + " > ".join(f"{n} ({p:.2f})"
                                        for n, p in
                                        _glbT["ranking"])
                           + " \u00b7 shares steer both the "
                           "offensive tempo and the budget "
                           "concentration. Every cycle is logged to "
                           "global.csv and cycles.jsonl.")
                if _glbT.get("thresholds"):
                    st.caption("Acceptance gate \u03b7 per region "
                               "(monitored regions carry a tighter "
                               "gate): " + ", ".join(
                                   f"{n}: {v:.2f}" for n, v in
                                   _glbT["thresholds"].items()))
            else:
                st.caption("No global decision yet \u2014 enable "
                           "'Apply decisions' and step; the Global "
                           "DSS decides every cycle (with a single "
                           "agent the ranking is trivial but still "
                           "logged).")
            _all4 = (st.checkbox("All agents (table)",
                                 value=bool(_sv("l4_all", True)),
                                 key="l4_all")
                     if len(_regs4) > 1 else False)
            if _all4:
                _res4 = {r.name: _cand_for(r) for r in _regs4}
                _head4 = "| intervention |" + "".join(
                    f" {n} |" for n in _names4)
                _sep4 = "|---|" + "---|" * len(_names4)
                _rows4 = [_head4, _sep4]
                for _iv in _dss.INTERVENTIONS:
                    _cells4 = " | ".join(
                        f"{_res4[n][0][_iv]:.2f}" for n in _names4)
                    _chip4 = ("<span style='color:"
                              f"{_IV_COLOR.get(_iv, '#999')}'>"
                              "\u25a0</span> ")
                    _rows4.append(
                        f"| {_chip4}{_dss.INTERVENTION_LABEL[_iv]} | "
                        f"{_cells4} |")
                _rows4.append("| fired rules | " + " | ".join(
                    str(sum(1 for _r, _w in _res4[n][1] if _w > 0.01))
                    for n in _names4) + " |")
                st.markdown("\n".join(_rows4),
                            unsafe_allow_html=True)
                st.caption("Every Local DSS agent side by side: the "
                           "candidate order intensities its rule "
                           "base produces from its own region right "
                           "now, and how many rules fired. Untick "
                           "for the single-agent detail with the "
                           "rule trace.")

            else:
                _i4 = min(int(_sv("dss_sel_i", 0)), len(_names4) - 1)
                _sel4 = st.selectbox("Agent (region)", _names4,
                                     index=_i4, key="l4_agent")
                _reg4 = _regs4[_names4.index(_sel4)]
                _u4, _tr4 = _cand_for(_reg4)
                st.markdown("".join(
                    _iv_bar(_dss.INTERVENTION_LABEL[_iv], _u4[_iv],
                            _IV_COLOR.get(_iv, "#999"))
                    for _iv in _dss.INTERVENTIONS),
                    unsafe_allow_html=True)
                _fired = [(r, w) for r, w in _tr4 if w > 0.01]
                with st.expander(f"Fired rules ({len(_fired)}) \u2014 "
                                 "traceability"):
                    if not _fired:
                        st.caption("No rule fires in this region "
                                   "right now.")
                    for _r, _w in sorted(_fired, key=lambda t: -t[1]):
                        st.caption(f"[{_w:.2f}] {_r.text()}")
        elif panel == "Rules":
            st.markdown("**Rule catalog \u2014 thesis seeds + "
                        "everything this run has learned**")
            _profs = {
                "full \u2014 whole Table E.1 (40 seeds)": "full",
                "core \u2014 doctrine R1-R22 only": "core",
                "minimal \u2014 5 strongest seeds (one per "
                "intervention family), LEARN the rest by trial":
                "minimal"}
            _pcur_r = str(_sv("dss_seed_profile", "full"))
            _pidx = [i for i, v in enumerate(_profs.values())
                     if v == _pcur_r]
            _psel = st.selectbox(
                "Seed profile \u2014 how much doctrine the run "
                "starts with", list(_profs),
                index=(_pidx[0] if _pidx else 0),
                help="'minimal' starts nearly naked: the single "
                     "strongest seed per intervention family, 5 "
                     "rules in total (one seed answers two "
                     "families). The adaptation "
                     "stages (resolution + GenAI) must then DISCOVER "
                     "the missing rules by trial, which is exactly "
                     "the controlled-experiment setting for showing "
                     "that the staged adaptation works. Changing "
                     "the profile rebuilds the engine (learned "
                     "rules are dropped).")
            if _profs[_psel] != _pcur_r:
                st.session_state["dss_seed_profile"] = _profs[_psel]
                st.rerun()
            import os as _os_st
            _store_p = _os_st.path.join(_os_st.path.dirname(
                _os_st.path.dirname(_os_st.path.abspath(__file__))),
                "logs", "learned_rules.json")
            _rb1, _rb2 = st.columns(2)
            if _rb1.button("Reset (keep the strongest)",
                           key="rules_reset",
                           help="Restores Table D.3 memberships and "
                                "the seed rules, clears the controller "
                                "value table, and PRUNES the persistent "
                                "learned store to its 10 strongest "
                                "rules (strength = accumulated fired "
                                "weight of applied decisions). The "
                                "survivors reload immediately: this "
                                "is the natural-selection reset."):
                from dss.adapt import reset_partitions as _rspR
                _rspR()
                _nkeep = _dss.prune_learned(
                    _store_p, keep=10,
                    profile=str(_sv("dss_seed_profile", "full")))
                _engRs = st.session_state.get("dss_engine")
                if _engRs is not None:
                    _engRs.rules = _dss.make_runtime_rules(
                        str(_sv("dss_seed_profile", "full")))
                    _dss.merge_learned(
                        _engRs.rules, _store_p,
                        profile=str(_sv("dss_seed_profile",
                                        "full")))
                    _engRs.controller.q.clear()
                st.toast(f"Reset: seeds restored, the {_nkeep} "
                         "strongest learned rules survive.")
                st.rerun()
            if _rb2.button("Wipe learned store",
                           key="rules_wipe",
                           help="Deletes logs/learned_rules.json and "
                                "returns to the pure seed profile "
                                "\u2014 the clean-room start for a "
                                "new convergence experiment."):
                from dss.adapt import reset_partitions as _rspW
                _rspW()
                _dss.wipe_learned(
                    _store_p,
                    profile=str(_sv("dss_seed_profile", "full")))
                _engRw = st.session_state.get("dss_engine")
                if _engRw is not None:
                    _engRw.rules = _dss.make_runtime_rules(
                        str(_sv("dss_seed_profile", "full")))
                    _engRw.controller.q.clear()
                st.toast("Learned store wiped; pure seed profile.")
                st.rerun()
            _engR = st.session_state.get("dss_engine")
            import os as _os_rs
            _store_v = _os_rs.path.join(_os_rs.path.dirname(
                _os_rs.path.dirname(_os_rs.path.abspath(__file__))),
                "logs", "learned_rules.json")
            if _engR is not None:
                _rlist = _engR.rules
            else:
                # no engine (fresh map / apply off): the DISPLAY must
                # still show the persistent store merged into the
                # seed profile, otherwise the learned rules LOOK
                # lost even though the file has them
                _rlist = _dss.make_runtime_rules(
                    str(_sv("dss_seed_profile", "full")))
                _dss.merge_learned(
                    _rlist, _store_v,
                    profile=str(_sv("dss_seed_profile", "full")))
                st.caption("No engine running yet \u2014 showing "
                           "the seed profile MERGED with the "
                           "persistent learned store. Enable "
                           "'Apply decisions' and step: the engine "
                           "starts from exactly this base.")
            _bornN, _tunedN = _dss.load_learned(
                _store_v,
                profile=str(_sv("dss_seed_profile", "full")))
            if _os_rs.path.exists(_store_v):
                import time as _t_rs
                _mt = _t_rs.strftime(
                    "%H:%M:%S", _t_rs.localtime(
                        _os_rs.path.getmtime(_store_v)))
                st.caption(f"Persistent store: `logs/learned_rules"
                           f".json` \u00b7 lineage of THIS profile: "
                           f"{len(_bornN)} born rules, "
                           f"{len(_tunedN)} tuned seeds \u00b7 last "
                           f"saved {_mt} \u00b7 saved EVERY decision "
                           "cycle; survives fires, engines and maps. "
                           "Each seed profile keeps its OWN lineage: "
                           "the selected profile's rules are the "
                           "base, evFIS/GenAI grow on top of it, and "
                           "nothing leaks between profiles.")
            else:
                st.caption("Persistent store: not created yet "
                           "(first accepted adaptation or decision "
                           "cycle writes logs/learned_rules.json).")

            def _origin_of(_r):
                if _r.name.startswith("G"):
                    return 3, "\U0001F7E9 GenAI (stage \u2462)"
                if _r.name.startswith("A"):
                    return 2, ("\U0001F7E7 resolution "
                               "(stage \u2461)")
                if "evFIS" in (_r.note or ""):
                    return 1, ("\U0001F7E8 seed, evFIS-tuned "
                               "(stage \u2460)")
                return 0, "\U0001F7E6 seed (Table E.1)"

            _tblR = [dict(
                origin=_origin_of(_r)[1], name=_r.name,
                IF=" AND ".join(f"{v} is {t}"
                                for v, t in _r.antecedents),
                THEN=", ".join(f"{iv} {x:.2f}"
                               for iv, x in _r.consequents),
                active="yes" if _r.active else "no",
                strength=round(float(getattr(_r, "strength", 0.0)), 1),
                note=(_r.note or "")) for _r in _rlist]
            st.dataframe(_tblR, use_container_width=True, height=420)
            _nnew = sum(1 for _r in _rlist if _r.name[0] in "AG")
            _ntun = sum(1 for _r in _rlist if "evFIS" in (_r.note or ""))
            _fullR = _dss.make_runtime_rules("full")
            _cellsN = [set(_r.antecedents) for _r in _rlist
                       if _r.active]
            _hitR = sum(1 for _fr in _fullR if _fr.active and any(
                set(_fr.antecedents) & _cn for _cn in _cellsN))
            _totR = sum(1 for _fr in _fullR if _fr.active)
            st.caption(f"{len(_rlist)} rules \u00b7 adaptation-born: "
                       f"{_nnew} \u00b7 evFIS-tuned consequents: "
                       f"{_ntun} \u00b7 convergence toward the "
                       f"Table E.1 doctrine: {_hitR}/{_totR} doctrine "
                       f"cells touched ({_hitR / max(_totR, 1):.0%})."
                       " Learned rules persist in logs/"
                       "learned_rules.json across fires, engines and "
                       "MAPS; strength = accumulated fired weight.")
            from dss.fuzzy import REGISTRY as _REGR
            _dcL = (list(getattr(_engR, "decision_concepts", []))
                    or list(_dss.DECISION_CONCEPTS))
            _catN = 1
            for _dcC in _dcL:
                _catN *= len(_REGR.get(_dcC))
            _newC = [c for c in getattr(_engR, "hierarchy", {}) or {}
                     if c not in _dss.HIERARCHY] if _engR else []
            _newM = list(getattr(_engR, "macros", {}) or {}) \
                if _engR else []
            st.caption("LEARNED VOCABULARY \u00b7 concepts: "
                       + (", ".join(_newC) or "none yet")
                       + " \u00b7 macro interventions: "
                       + (", ".join(_newM) or "none yet")
                       + " \u2014 vocabulary packages (new object + "
                       "a rule using it, gates G2/G2b/G3/G4/G5) come "
                       "from the LIVE Claude proposer only. Set "
                       "ANTHROPIC_API_KEY to enable.")
            st.caption(f"Linguistic catalog: {_catN:,} antecedent "
                       "cells over the five decision concepts "
                       "(5\u2075 = 3,125 at the seed partition; "
                       "stage \u2461 term insertions GROW it).")
            st.markdown("**Membership modifications (evFIS stage "
                        "\u2460 + inserted terms, stage \u2461) "
                        "\u2014 registry vs Table D.3**")
            from dss.fuzzy import default_partition as _defpR
            _dpR = _defpR()
            _modsR = []
            for _var in _REGR.variables():
                for _term, _abcd in _REGR.get(_var).items():
                    _d0 = _dpR.get(_term)
                    if _d0 is None:
                        _modsR.append(dict(
                            variable=_var, term=_term,
                            default="(INSERTED \u2014 catalog "
                                    "grew)",
                            current=str(tuple(round(float(v), 3)
                                              for v in _abcd))))
                        continue
                    if tuple(np.round(np.asarray(_abcd, float), 4)) \
                            != tuple(np.round(np.asarray(_d0, float),
                                              4)):
                        _modsR.append(dict(
                            variable=_var, term=_term,
                            default=str(tuple(round(float(v), 3)
                                              for v in _d0)),
                            current=str(tuple(round(float(v), 3)
                                              for v in _abcd))))
            if _modsR:
                st.dataframe(_modsR, use_container_width=True)
            else:
                st.caption("Every membership still sits on its "
                           "Table D.3 default \u2014 stage \u2460 "
                           "has not moved anything (yet).")
        elif panel == "Log":
            _eng4 = st.session_state.get("dss_engine")
            with st.expander("Saved runs \u2014 load & replay"):
                import os as _os_rp
                import json as _js_rp
                import gzip as _gz_rp
                _logroot = _os_rp.path.join(_os_rp.path.dirname(
                    _os_rp.path.dirname(_os_rp.path.abspath(__file__))),
                    "logs")
                _runs = sorted((d for d in (_os_rp.listdir(_logroot)
                                if _os_rp.path.isdir(_logroot) else [])
                                if _os_rp.path.exists(_os_rp.path.join(
                                    _logroot, d, "world.json.gz"))),
                               reverse=True)
                if not _runs:
                    st.caption("No replayable runs yet \u2014 every "
                               "run with 'Apply decisions' ON saves "
                               "itself under 03_Codes/logs "
                               "(world.json.gz + meta.json + "
                               "cycles.jsonl).")
                else:
                    _rsel_rp = st.selectbox("Run", _runs, key="rp_run")
                    if st.button("Load this run for replay",
                                 help="Rebuilds the exact map, "
                                      "sensors, resource pool, "
                                      "engine and weather settings "
                                      "of the saved run. Then press "
                                      "'Run to end': the engine and "
                                      "the rng are deterministic, so "
                                      "the run reproduces itself "
                                      "(when GenAI is off; live "
                                      "Claude proposals are not "
                                      "deterministic)."):
                        _rd = _os_rp.path.join(_logroot, _rsel_rp)
                        with _gz_rp.open(_os_rp.path.join(
                                _rd, "world.json.gz"), "rt") as _f:
                            _wnew = World.from_dict(_js_rp.load(_f))
                        try:
                            _meta = _js_rp.load(open(_os_rp.path.join(
                                _rd, "meta.json")))
                        except Exception:
                            _meta = {}
                        _new_simulator(_wnew)
                        _me = _meta.get("engine", {}) or {}
                        for _mk, _sk in (
                                ("regions", "dss_n"),
                                ("cycle_min", "dss_cycle_min"),
                                ("horizon_min", "dss_horizon_min"),
                                ("j_th", "dss_jth"),
                                ("eta", "dss_eta"),
                                ("genai", "dss_genai_on"),
                                ("evfis", "dss_evfis_on"),
                                ("evfis_step", "dss_evfis_step"),
                                ("ctrl_eps", "dss_ctrl_eps"),
                                ("ctrl_lr", "dss_ctrl_lr"),
                                ("attn", "dss_attn_thr"),
                                ("min_gain", "dss_min_gain")):
                            if _mk in _me:
                                st.session_state[_sk] = _me[_mk]
                        for _wk, _wv in (_meta.get("weather", {})
                                         or {}).items():
                            st.session_state[_wk] = _wv
                        if _meta.get("sensors"):
                            st.session_state["dss_sensors"] =                                 list(_meta["sensors"])
                        if _meta.get("depots"):
                            st.session_state["dss_res_items"] =                                 list(_meta["depots"])
                            st.session_state["dss_res_base_v"] =                                 st.session_state.map_version
                        st.session_state["dss_apply"] = True
                        st.toast(f"{_rsel_rp} loaded \u2014 press "
                                 "'Run to end' to replay.")
                        st.rerun()
            st.markdown("**Decision log \u2014 backward trace & "
                        "counterfactual**")
            if _eng4 is None or not _eng4.log.records:
                st.caption("No decisions logged yet \u2014 stage a "
                           "pool, toggle 'Apply decisions' and step "
                           "the simulation.")
            else:
                _cyc = _eng4.log.cycles()
                st.caption(f"{len(_cyc)} decision cycles \u00b7 "
                           f"runtime rules: {len(_eng4.rules)} "
                           f"(adaptation-born: "
                           f"{sum(1 for r in _eng4.rules if r.name[0] in 'AG')})"
                           f" \u00b7 controller values: " + ", ".join(
                               f"{k[0]}/s{k[1]}={v:.2f}"
                               for k, v in sorted(_eng4.controller.q.items())))
                _ksel = st.selectbox("Cycle (step)", list(reversed(_cyc)),
                                     key="dlog_k")
                _recs = _eng4.log.at(int(_ksel))
                _ropt = (["All agents (table)"]
                         if len(_recs) > 1 else []) \
                    + [r.region for r in _recs]
                _rsel = st.selectbox("Region", _ropt, key="dlog_r")
                if _rsel == "All agents (table)":
                    _hd = "| |" + "".join(f" {r.region} |"
                                          for r in _recs)
                    _rws = [_hd, "|---|" + "---|" * len(_recs)]
                    for _ivL in _dss.INTERVENTIONS:
                        _chipL = ("<span style='color:"
                                  f"{_IV_COLOR.get(_ivL, '#999')}'>"
                                  "\u25a0</span> ")
                        _rws.append(
                            f"| {_chipL}"
                            f"{_dss.INTERVENTION_LABEL[_ivL]} | "
                            + " | ".join(
                                f"{r.intensities.get(_ivL, 0.0):.2f}"
                                for r in _recs) + " |")
                    _rws.append("| quality Q | " + " | ".join(
                        f"{r.quality:.2f}" for r in _recs) + " |")
                    _rws.append("| global share | " + " | ".join(
                        f"{getattr(r, 'coord_share', 1.0):.2f}"
                        for r in _recs) + " |")
                    _rws.append("| attended | " + " | ".join(
                        ("\u25cf" if getattr(r, "attended", True)
                         else "\u2013") for r in _recs) + " |")
                    st.markdown("\n".join(_rws),
                                unsafe_allow_html=True)
                    _glbL = getattr(_eng4, "last_global", None)
                    if _glbL:
                        st.markdown("**" + _glbL["statement"] + "**")
                    _hotG = max(_recs, key=lambda r: getattr(
                        r, "coord_share", 0.0))
                    st.caption(
                        "GLOBAL coordination \u00b7 the Global DSS "
                        "reads every region's operational priority, "
                        "assigns the shares above (budget "
                        "concentration follows them) and keeps the "
                        f"hotspot in focus: {_hotG.region} carries "
                        "the largest share this cycle"
                        + (" \u00b7 NO-HARM withheld the offensive "
                           "orders" if getattr(_eng4, "last_withheld",
                                               False) else "") + ".")
                    _rec = _recs[0]
                else:
                    _rec = next(r for r in _recs
                                if r.region == _rsel)
                    st.markdown(f"**{_eng4.log.stage_story(_rec)}**")
                    for _ln in _eng4.log.why(_rec)[2:]:
                        st.caption(_ln)
                st.markdown("**Cycle chronicle \u2014 one decision "
                            "cycle, the full story**")
                _cycsel = st.selectbox(
                    "Cycle k", [c["step"] for c in
                                reversed(_eng4.cycles)],
                    key="chron_k") if _eng4.cycles else None
                if _cycsel is not None:
                    _cy = next(c for c in _eng4.cycles
                               if c["step"] == _cycsel)
                    _sm0 = _cy["sim"]
                    st.caption(
                        f"SITUATION \u00b7 k={_cy['step']} "
                        f"(t={_cy['t_min']:.0f} min) \u00b7 "
                        f"{_sm0['burning']} cells burning, "
                        f"{_sm0['burned']} burned so far \u00b7 wind "
                        f"{_sm0['wws_mean']:.1f} m/s \u00b7 rain "
                        f"{_sm0['prec_mean']:.1f} mm/h \u00b7 fuel "
                        f"moisture {_sm0['fmoist_mean']:.3f}")
                    _fc0 = _cy["forecast"]
                    st.caption(
                        f"FORECAST \u00b7 with the decision "
                        f"J={_fc0['j_candidate']:.3f}, doing nothing "
                        f"J={_fc0['j_noaction']:.3f} \u00b7 "
                        f"satisficing bound "
                        f"{_fc0['satisficing_bound']:.3f} "
                        f"(J_TH {_fc0['j_threshold']:.2f}) \u00b7 "
                        + " ".join(f"{k}={v:.3f}" for k, v in
                                   _cy["costs"].items()))
                    _gain0 = (_fc0["j_noaction"]
                              - _fc0["j_candidate"])
                    st.caption(
                        f"PERFORMANCE \u00b7 expected gain vs no "
                        f"action \u0394J={_gain0:+.3f} "
                        + ("(the decision helps)" if _gain0 > 1e-4
                           else "(no measurable gain at this "
                                "horizon)"))
                    _rl0 = _cy["stage_controller"]
                    st.caption(
                        "Stage controller \u00b7 " + (
                            f"selected adaptation stage "
                            f"{_rl0['selected_stage']} (deficit "
                            f"bucket {_rl0['bucket']}, "
                            f"\u03b5={_rl0['eps']})"
                            if _rl0["selected_stage"] else
                            "no stage selected (the seed rule base "
                            "satisficed)") + " \u00b7 Q: "
                        + (", ".join(f"{k}={v}" for k, v in
                                     _rl0["value_table"].items())
                           or "empty"))
                    _gd0 = _cy.get("global_dss")
                    if _gd0:
                        st.caption("GLOBAL \u00b7 "
                                   + _gd0.get("statement", ""))
                    _ad0 = _cy["adaptation"]
                    if not _ad0["tried"]:
                        st.caption("ADAPTATION \u00b7 not attempted "
                                   "\u2014 the seed rule-base "
                                   "decision was good enough (or the "
                                   "trial cooldown is running); its "
                                   "orders are the ones applied "
                                   "below")
                    if _ad0["tried"]:
                        st.markdown(
                            ("\u2705 " if _ad0["accepted"] else
                             "\u274c ") + f"stage {_ad0['tried']}: "
                            f"{_ad0['detail']} (dJ {_ad0['dJ']:+.4f})")
                        if _ad0.get("info"):
                            with st.expander("Adaptation trials \u2014 "
                                             "every attempt, every "
                                             "reject reason"):
                                st.json(_ad0["info"])
                    _rgsel = st.selectbox("Region detail",
                                          list(_cy["regions"]),
                                          key="chron_r")
                    _rg = _cy["regions"][_rgsel]
                    _zrow = " \u00b7 ".join(
                        f"{_dss.FEATURE_SYM[k].replace('_', '')}="
                        f"{v:.2f}" for k, v in _rg["features"].items())
                    st.caption("z: " + _zrow)
                    st.caption("concepts (gated): " + " \u00b7 ".join(
                        f"{k.replace('_', ' ')}={v:.2f}"
                        for k, v in
                        _rg["concepts_effective"].items()))
                    st.caption("fired: " + (" ".join(
                        f"{n}[{w:.2f}]" for n, w in _rg["fired"])
                        or "none"))
                    st.caption(
                        "DECISION APPLIED (orders rules\u2192final): "
                        + " \u00b7 ".join(
                            f"{k.split('_')[0]} "
                            f"{_rg['orders_from_rules'][k]:.2f}"
                            f"\u2192{v:.2f}"
                            for k, v in _rg["orders_final"].items())
                        + f" \u00b7 Q={_rg['quality']:.2f}"
                        + (" \u00b7 FAILSAFE" if _rg["failsafe"]
                           else "")
                        + f" \u00b7 share {_rg['coord_share']:.2f}")
                    import json as _js_c
                    st.download_button(
                        "Download this cycle (JSON)",
                        _js_c.dumps(_cy, indent=1).encode(),
                        file_name=f"cycle_k{_cy['step']}.json",
                        mime="application/json", key="chron_dl")
                st.divider()
                st.markdown("**Trace table (all cycles, Excel-ready)**")
                _sm_l = float(getattr(cfg, "step_minutes", 1.0))
                _tbl = [dict(
                    step=r.step, t_min=int(r.step * _sm_l),
                    region=r.region,
                    provenance=_eng4.log.stage_story(r),
                    fired=" ".join(f"{n}[{w:.2f}]"
                                   for n, w in r.fired[:4]),
                    S=round(r.intensities.get("suppression_effort", 0), 2),
                    D=round(r.intensities.get("resource_deployment", 0), 2),
                    C=round(r.intensities.get("containment_line", 0), 2),
                    P=round(r.intensities.get("asset_protection", 0), 2),
                    E=round(r.intensities.get("evacuation", 0), 2),
                    W=round(r.intensities.get("public_warning", 0), 2),
                    Q=round(r.quality, 2),
                    J_fc=round(r.j_forecast, 3),
                    J_no=round(r.j_noaction, 3))
                    for r in _eng4.log.records]
                st.dataframe(_tbl, use_container_width=True, height=280)
                import io as _io_l
                import csv as _csv_l
                _buf_l = _io_l.StringIO()
                _wr = _csv_l.DictWriter(_buf_l,
                                        fieldnames=list(_tbl[0].keys()),
                                        delimiter=";")
                _wr.writeheader()
                _wr.writerows(_tbl)
                st.download_button(
                    "Download trace (CSV, opens in Excel)",
                    _buf_l.getvalue().encode("utf-8-sig"),
                    file_name="dss_decision_trace.csv",
                    mime="text/csv")
                with st.expander("Global DSS history (all "
                                 "cycles)"):
                    _gcyc = [(c.get("step"), c.get("t_min"),
                              c.get("global_dss"))
                             for c in _eng4.cycles
                             if c.get("global_dss")]
                    if not _gcyc:
                        st.caption("No global decisions logged yet.")
                    for _gs, _gt, _gg in _gcyc[-40:]:
                        st.caption(f"k={_gs} (t={_gt:.0f} min): "
                                   + _gg.get("statement", ""))
                    st.caption("Full history: `global.csv` in the "
                               "run folder (one row per cycle: "
                               "hotspot, attended set, shares, "
                               "statement).")
                with st.expander("Adaptation history (all cycles)"):
                    _seen_h = set()
                    for _r_h in _eng4.log.records:
                        if _r_h.step in _seen_h:
                            continue
                        _seen_h.add(_r_h.step)
                        st.caption(f"k={_r_h.step}: "
                                   + _eng4.log.stage_story(_r_h))
                if _eng4.run_logger is not None:
                    st.caption("Persistent run log: "
                               f"`{_eng4.run_logger.dir}` "
                               "(steps.csv + decisions.jsonl \u2014 "
                               "every simulation step and every "
                               "decision cycle).")
                _cfsc = st.radio(
                    "Counterfactual scope", [
                        "no orders AT ALL (replay from the very "
                        "beginning)",
                        f"only from this decision on (k={_rec.step})"],
                    index=0, key="cf_scope",
                    help="First option answers 'what if the DSS had "
                         "never intervened': the clone rewinds to "
                         "step 0 and replays the whole history with "
                         "no resource orders. Second option keeps "
                         "everything up to the selected decision and "
                         "withdraws only the orders from that cycle "
                         "on.")
                if st.button("What if these orders were NOT taken? "
                             "(counterfactual replay)",
                             help="Clones the live simulation, rewinds "
                                  "the CLONE and replays history "
                                  "without the selected orders; the "
                                  "live run is untouched. Fuel "
                                  "moisture and the rng state travel "
                                  "with the snapshot, so the "
                                  "difference is attributable to the "
                                  "withdrawn orders alone."):
                    _cf_from = (0 if _cfsc.startswith("no orders")
                                else int(_rec.step))
                    _dtm_cf = float(getattr(cfg, "step_minutes", 1.0))
                    with st.spinner("Replaying without orders..."):
                        _cf, _rcf = _dss.counterfactual(
                            sim, _cf_from,
                            step_hook=lambda w2, k2:
                                _drive_weather(w2, k2 * _dtm_cf))
                    if _cf is None:
                        st.warning("No snapshot left at that step "
                                   "(memory-capped history).")
                    else:
                        from disaster_phyengine.costs import (
                            compute_costs as _cc4)
                        _ract = _cc4(sim)
                        _pa = float(getattr(_ract, "j_physical", 0.0))
                        _pc = float(getattr(_rcf, "j_physical", 0.0))
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Jφ actual (with DSS)",
                                  f"{_pa:.3f}",
                                  help="Fair outcome metric: "
                                       "physical loss only, both "
                                       "sides (a no-order run pays "
                                       "no response cost by "
                                       "definition, so the decision "
                                       "cost would be biased).")
                        m2.metric("Jφ without orders",
                                  f"{_pc:.3f}",
                                  delta=f"{_pc - _pa:+.3f}")
                        m3.metric("Burned cells: actual vs without",
                                  f"{int(sim.ever_burned.sum())} / "
                                  f"{int(_cf.ever_burned.sum())}")
                        i1, i2, i3 = st.columns(3)
                        _sc4 = max(3, 380 // max(cfg.nx, 1))
                        i1.image(viz.render_pil(world, sim=sim,
                                                scale=_sc4),
                                 caption="actual (orders applied)")
                        i2.image(viz.render_pil(_cf.world, sim=_cf,
                                                scale=_sc4),
                                 caption="counterfactual (no orders)")
                        from PIL import Image as _PILImg
                        _ba = sim.ever_burned
                        _bc = _cf.ever_burned
                        _dif = np.zeros((cfg.ny, cfg.nx, 3),
                                        dtype=np.uint8)
                        _dif[...] = 34
                        _dif[_ba & _bc] = (90, 90, 90)
                        _dif[_bc & ~_ba] = (40, 200, 90)
                        _dif[_ba & ~_bc] = (230, 60, 60)
                        _im = _PILImg.fromarray(_dif).resize(
                            (cfg.nx * _sc4, cfg.ny * _sc4),
                            _PILImg.NEAREST)
                        i3.image(_im, caption="difference \u2014 "
                                 "green: cells the orders SAVED \u00b7 "
                                 "red: burned only WITH orders \u00b7 "
                                 "grey: burned in both")
                        st.caption("Reading the maps: black = burned "
                                   "OUT, orange = still burning now; a "
                                   "slowed fire can look 'more orange' "
                                   "while having burned far less. The "
                                   "difference map is the honest "
                                   "comparison.")
        elif panel == "Time":
            st.caption("Simulation time: what one step represents and "
                       "where the clock stands.")
            st.session_state["dr_start"] = float(st.number_input(
                "Start hour of day", 0.0, 23.0,
                float(st.session_state.get("dr_start", 12.0)), 1.0,
                help="What time of day step 0 corresponds to (drives "
                     "the map clock, night dimming and the diurnal "
                     "weather wave)."))
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
            st.caption("Elapsed fire time so far: "
                       f"{_fmt_sim_time(getattr(sim, 't_elapsed_min', sim.state.step * cfg.step_minutes))}"
                       f" in {sim.state.step} steps. Changing the "
                       "step length applies from the NEXT step on and "
                       "the elapsed total is PRESERVED (the clock "
                       "integrates each executed step). The physical "
                       "speed in m/min stays the same (System "
                       "Description Sec. 9, note 8).")
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
            _tmin = sim.state.step * float(getattr(cfg,
                                                   "step_minutes", 1.0))
            st.caption(f"Sim clock: {(_tmin/60.0) % 24.0:04.1f} h of "
                       "day (fire behaviour peaks ~15:00, minimum "
                       "before dawn).")
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

        elif panel == "Layer 2 \u00b7 Perception":
            st.caption("Layer 2 \u2014 perception: multi-source data "
                       "fusion + feature extraction. Each Local DSS "
                       "reads the fused observation restricted to its "
                       "region and extracts the ten features "
                       "$z_1..z_{10}$.")
            if _obsnet is None:
                st.warning("OBSERVATION BYPASS \u2014 'Partial "
                           "observation via sensors' is OFF: every "
                           "feature reads the TRUE simulation state "
                           "with confidence 1. This is the idealized "
                           "physics-test mode, not the thesis "
                           "pipeline. Turn it on in Layer 1 for the "
                           "sensor-fused feed.")
            elif not _sv("dss_sensors", []):
                st.warning("No sensors staged: the fire-driven "
                           "features ($z_1$ intensity, $z_4$ "
                           "proximity, $z_{10}$ urgency) read 0 with "
                           "ZERO confidence \u2014 the DSS is blind "
                           "to the fire. The values you still see "
                           "come from sensor-independent sources "
                           "(see below). Stage a network in Layer 1.")
            st.caption("Where each value comes from \u00b7 "
                       "$z_1, z_4, z_{10}$: SENSOR fusion (burning/"
                       "intensity channels) \u00b7 $z_5$: pre-fire "
                       "fuel map prior, aged \u00b7 $z_2$: spread "
                       "model over maps + weather \u00b7 $z_3$: "
                       "weather service \u00b7 $z_6, z_7, z_8$: GIS "
                       "(assets, access, roads) \u00b7 $z_9$: own "
                       "resource ledger (staged pool).")
            n_agents = int(st.number_input(
                "Number of local DSS agents", 1, 12,
                int(_sv("dss_n", 1)), 1,
                help="The map is split into exactly this many regions "
                     "covering every cell (near-square blocks, Agent_1 "
                     "at the north-west)."))
            st.session_state["dss_n"] = n_agents
            _nrec = max(1, min(12, round((cfg.nx * cfg.ny) / 2500.0)))
            st.caption(f"Suggested: {_nrec} agent(s) \u2014 one "
                       "Local DSS per ~50\u00d750-cell "
                       "responsibility block. More agents pay off "
                       "when regional situations diverge (big maps, "
                       "several fronts); a single agent suffices on "
                       "small maps.")
            if _nrec != n_agents and st.button(
                    f"Use suggested ({_nrec})",
                    key="dss_n_sugg"):
                st.session_state["dss_n"] = int(_nrec)
                st.rerun()
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
                _short = {k: f"${_dss.FEATURE_SYM[k].replace('_10', '_{10}')}$ "
                             + _dss.FEATURE_NAME[k]
                          for k in _dss.FEATURE_ORDER}
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
                    _rowsmd.append(f"| $conf=\\min\\gamma$ | {_kcells} |")
                for _k in _dss.FEATURE_ORDER:
                    _cells = " | ".join(f"{_feats[_r.name][_k]:.2f}"
                                        for _r in _regs)
                    _lab = _short[_k]
                    if _k == "temporal_urgency":
                        _lab = f"**{_lab}**"
                    _rowsmd.append(f"| {_lab} | {_cells} |")
                st.markdown("\n".join(_rowsmd))
                with st.expander("What each $z_i$ measures"):
                    for _k, _sym, _nm, _ms in _dss.FEATURE_META:
                        _symtex = _sym.replace("_10", "_{10}")
                        st.markdown(f"${_symtex}$ **{_nm}** \u2014 {_ms}")
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
                                text=f"conf \u2014 observation confidence "
                                     f"(Sec. 11): {_cf:.2f}")
                    _cc = _obsnet.region_conf_components(_reg)
                    st.caption("per channel " + " \u00b7 ".join(
                        f"{_dss.CHANNEL_SYMBOL[_ch]}: {_v:.2f}"
                        for _ch, _v in _cc.items())
                        + " \u00b7 " + _obsnet.coverage_note(_reg))
                    _ra = _obsnet.region_age(_reg)
                    st.caption("data age (median) " + " \u00b7 ".join(
                        f"{_dss.CHANNEL_SYMBOL[_ch]}: "
                        + ("\u221e" if _v > 9e5 else f"{_v:.0f} min")
                        for _ch, _v in _ra.items()))
                    _syx = _reg.slices()
                    _tb = int((sim.state.burning[_syx] > 0.5).sum())
                    _ob = int((_obsnet.obs["burning"][_syx] > 0.5).sum())
                    st.caption(f"burning cells \u2014 observed: {_ob} "
                               f"vs actual: {_tb}. The gap IS the "
                               "epistemic uncertainty the DSS lives "
                               "with; it closes when fresh reports "
                               "arrive.")
                    st.caption("Sensors are shared infrastructure; this "
                               "agent reads the fused observation "
                               "restricted to its own region "
                               "$\\Omega_i$.")
                _f = _dss.ten_features(sim, _reg, network=_obsnet)
                _zsub = {"z_1": "z\u2081", "z_2": "z\u2082",
                         "z_3": "z\u2083", "z_4": "z\u2084",
                         "z_5": "z\u2085", "z_6": "z\u2086",
                         "z_7": "z\u2087", "z_8": "z\u2088",
                         "z_9": "z\u2089", "z_10": "z\u2081\u2080"}
                # change arrows: compare with the features at the PREVIOUS
                # simulation step (not the previous rerun)
                _pk = f"dss_featprev_{_reg.name}"
                _prev_rec = st.session_state.get(_pk)
                _step_now = int(sim.state.step)
                if _prev_rec is None or _prev_rec[0] == _step_now:
                    _fprev = (_prev_rec[1] if _prev_rec else dict(_f))
                else:
                    _fprev = _prev_rec[1]
                _fcL = _dss.feature_confidence(_obsnet, _reg)
                for _k in _dss.FEATURE_ORDER:
                    _d = float(_f[_k]) - float(_fprev.get(_k, _f[_k]))
                    _arr = ("\u2191" if _d > 0.005
                            else "\u2193" if _d < -0.005 else "")
                    _dtx = f"  {_arr}{abs(_d):.2f}" if _arr else ""
                    _ctx = ("" if _fcL[_k] >= 0.999
                            else f" \u00b7 conf {_fcL[_k]:.2f}")
                    st.progress(min(1.0, float(_f[_k])),
                                text=f"{_zsub[_dss.FEATURE_SYM[_k]]} "
                                     f"{_dss.FEATURE_NAME[_k]}: "
                                     f"{_f[_k]:.2f}{_dtx}{_ctx}")
                if _prev_rec is None or _prev_rec[0] != _step_now:
                    st.session_state[_pk] = (_step_now, dict(_f))
                with st.expander("What each $z_i$ measures"):
                    for _k, _sym, _nm, _ms in _dss.FEATURE_META:
                        _symtex = _sym.replace("_10", "_{10}")
                        st.markdown(f"${_symtex}$ **{_nm}** \u2014 {_ms}")

        else:  # Display: layer visibility
            st.markdown("**Layers** \u2014 shared with the Map editor")
            _lyd = [("ly_relief_v", "Relief", True),
                    ("ly_fire_v", "Fire", True),
                    ("ly_val_v", "Protection value", True),
                    ("ly_roads_v", "Roads", True),
                    ("ly_grid_v", "Grid", True),
                    ("ly_per_v", "Fire perimeter", True),
                    ("ly_orders_v", "DSS orders (icons)", True),
                    ("ly_alloc_v", "DSS allocation glow", True),
                    ("ly_sens_v", "Sensors + coverage", True),
                    ("ly_deps_v", "Resource depots + service areas",
                     True),
                    ("ly_agents_v", "Agent regions (Local DSS "
                     "boundaries)", True)]
            for _k, _lab, _d in _lyd:
                st.session_state[_k] = st.checkbox(
                    _lab, value=bool(_sv(_k, _d)))

    # values needed by the map regardless of which panel is open
    flags = dict(show_hillshade=bool(_sv("ly_relief_v", True)),
                 show_fire=bool(_sv("ly_fire_v", True)),
                 show_value=bool(_sv("ly_val_v", True)),
                 show_roads=bool(_sv("ly_roads_v", True)),
                 show_grid=bool(_sv("ly_grid_v", True)),
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
            place = st.checkbox(
                "Click map to place ignition", value=False, key="sim_place",
                help="On: the map switches to a click canvas (same "
                     "mechanism as the Map editor) and every click drops "
                     "an ignition marker at that cell. Off: the map is the "
                     "pan / zoom view like 3D \u2014 drag = pan, mouse "
                     "wheel = zoom, toolbar top-right.")
            _clk, _nf = _clock_info()
            _dreg = st.session_state.get("dss_region")
            _rb, _rl = (_dreg[:4], _dreg[4]) if _dreg else (None, None)
            _rall = (st.session_state.get("dss_regions_all")
                     if bool(_sv("ly_agents_v", True)) else None)
            _sens = (st.session_state.get("dss_sensors_draw")
                     if bool(_sv("ly_sens_v", True)) else None)
            _deps = (st.session_state.get("dss_depots_draw")
                     if bool(_sv("ly_deps_v", True)) else None)
            if not bool(_sv("ly_agents_v", True)):
                _rb, _rl = None, None
            _eng_m = st.session_state.get("dss_engine")
            _alloc = None
            _acts = None
            if (st.session_state.get("dss_apply")
                    and _eng_m is not None
                    and _eng_m.last_override is not None):
                if bool(_sv("ly_alloc_v", True)):
                    _alloc = _eng_m.last_override.rcap
                if bool(_sv("ly_orders_v", True)):
                    _acts = _eng_m.last_actions
            if playing:
                # fast image frames while animating (keeps the loop
                # responsive); pause to pan / zoom
                st.image(viz.render_pil(world, sim=sim, scale=scale,
                                        show_labels=True, clock_text=_clk,
                                        night_factor=_nf, region_box=_rb,
                                        region_label=_rl, regions=_rall,
                                        sensors=_sens, depots=_deps,
                                        alloc=_alloc, actions=_acts,
                                        **flags))
            elif place and HAS_CANVAS:
                # ignition placement works exactly like the Map editor:
                # a click canvas over the rendered map; each click drops a
                # marker that is applied as an ignition
                bg = viz.render_pil(world, sim=sim, scale=scale,
                                    show_labels=True, clock_text=_clk,
                                    night_factor=_nf, region_box=_rb,
                                    region_label=_rl, regions=_rall,
                                    sensors=_sens, depots=_deps,
                                    alloc=_alloc, actions=_acts,
                                    **flags)
                res = st_canvas(stroke_width=2, stroke_color="#a200de",
                                background_image=bg, update_streamlit=True,
                                height=cfg.ny * scale, width=cfg.nx * scale,
                                drawing_mode="point", display_toolbar=False,
                                point_display_radius=max(3, scale // 2),
                                key=(f"simc_{st.session_state.canvas_key}_"
                                     f"{scale}"))
                if world.ignitions and st.button(
                        "Remove last ignition",
                        help="Deletes the most recently placed ignition "
                             "marker."):
                    world.ignitions.pop()
                    st.session_state.canvas_key += 1
                    st.session_state.sim_applied = 0
                    st.rerun()
                objs = (res.json_data or {}).get("objects", []) if res else []
                new = objs[st.session_state.get("sim_applied", 0):]
                if new:
                    for o in new:
                        if o.get("type") == "circle":
                            rad = o.get("radius", 0)
                            gx, gy = _clip((o["left"] + rad) / scale,
                                           (o["top"] + rad) / scale)
                            world.add_ignition(gx, gy, step=ig_step,
                                               radius=int(ig_rad))
                    # the canvas is NOT remounted between clicks (a
                    # remount swallowed the next click, so placing a
                    # second ignition needed re-ticking the box);
                    # instead only the objects beyond the applied
                    # counter are consumed, exactly like the editor
                    st.session_state.sim_applied = len(objs)
                    st.rerun()
            else:
                # plotly map exactly like the 3D view: top-right modebar
                # (zoom / pan / reset), drag to pan, wheel to zoom; the view
                # survives the steps via the figure uirevision
                st.plotly_chart(
                    viz.map_figure_2d(world, sim=sim, scale=scale,
                                      clock_text=_clk, night_factor=_nf,
                                      region_box=_rb, region_label=_rl,
                                      regions=_rall, sensors=_sens,
                                      depots=_deps, alloc=_alloc,
                                      actions=_acts,
                                      uirevision=f"m{st.session_state.map_version}",
                                      **flags),
                    use_container_width=True,
                    key=f"plot2d_{st.session_state.map_version}",
                    config={"scrollZoom": True,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
                if world.ignitions and st.button(
                        "Remove last ignition",
                        help="Deletes the most recently placed ignition "
                             "marker."):
                    world.ignitions.pop()
                    st.rerun()
        if _alloc is not None and float(np.asarray(_alloc).max()) > 1e-9:
            st.caption("DSS orders on the map: blue dots = suppression "
                       "(water on the engaged cells) | dark-brown "
                       "squares = containment line being cut | green "
                       "rings = asset protection | orange arrow+EVAC "
                       "at settlements = evacuation | yellow ! = "
                       "public warning | region badge S/D/C/P/E/W = "
                       "ordered intensities | cyan glow = allocated "
                       "$R_{cap}$ | D1.. = staged depots.")
        st.markdown(legend_html(horizontal=True), unsafe_allow_html=True)

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
    ("Jᵇᵘʳⁿ", "j_burn", "#2e8b57", "burned area"),
    ("Jᵃˢˢᵉᵗ", "j_asset", "#d9822b", "asset loss"),
    ("Jᵖᵒᵖ", "j_pop", "#8e44ad", "population"),
    ("Jʳᵉˢᵖ", "j_resp", "#2c3e50", "response cost"),
    ("Jᵈᵉˡ", "j_delay", "#7f8c8d", "response delay"),
]


def _cost_panel():
    import matplotlib.pyplot as plt
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

    # ---- operational priority: doctrine presets for the weights ----
    _wpre = {
        "Balanced (default)":            (0.294, 0.294, 0.294,
                                          0.059, 0.059),
        "Life first \u2014 \u00f6nce insan":  (0.20, 0.20, 0.45,
                                          0.075, 0.075),
        "Assets & infrastructure first": (0.20, 0.45, 0.25,
                                          0.05, 0.05),
        "Environment first \u2014 forest":    (0.45, 0.20, 0.25,
                                          0.05, 0.05),
        "Manual":                        None,
    }
    _wmode = st.selectbox(
        "Operational priority \u2014 how the five losses are "
        "weighted", list(_wpre), key="cost_wmode",
        help="A doctrine choice assigns consistent weights (they "
             "sum to 1). 'Life first' makes population exposure "
             "dominant, 'Assets first' the built environment, "
             "'Environment first' the burned area. Manual frees "
             "the five weights; the live caption shows the "
             "normalized shares the engine actually uses. The "
             "choice steers the DSS: forecasts, satisficing, "
             "adaptation gates and the counterfactual all read "
             "this J.")
    _cwm = sim.cfg.cost
    if _wpre[_wmode] is not None:
        (_cwm.w_burn, _cwm.w_asset, _cwm.w_pop,
         _cwm.w_resp, _cwm.w_delay) = _wpre[_wmode]
    else:
        _mw = st.columns(5)
        _defs = [("burn", _cwm.w_burn), ("asset", _cwm.w_asset),
                 ("pop", _cwm.w_pop), ("resp", _cwm.w_resp),
                 ("delay", _cwm.w_delay)]
        _vals = []
        for _i_w, (_nm_w, _dv_w) in enumerate(_defs):
            _vals.append(float(_mw[_i_w].number_input(
                f"w_{_nm_w}", 0.0, 1.0,
                float(_sv(f"cost_w_{_nm_w}",
                          round(float(_dv_w), 3))),
                0.05, key=f"cost_w_{_nm_w}")))
        (_cwm.w_burn, _cwm.w_asset, _cwm.w_pop,
         _cwm.w_resp, _cwm.w_delay) = _vals
    _wsm = (_cwm.w_burn + _cwm.w_asset + _cwm.w_pop
            + _cwm.w_resp + _cwm.w_delay)
    if _wsm <= 1e-9:
        st.error("All weights are zero \u2014 J would be "
                 "undefined; falling back to Balanced.")
        (_cwm.w_burn, _cwm.w_asset, _cwm.w_pop,
         _cwm.w_resp, _cwm.w_delay) = _wpre["Balanced (default)"]
        _wsm = 1.0
    st.caption(
        "Normalized shares (always renormalized to sum 1): burn "
        f"{(_cwm.w_burn / _wsm):.0%} \u00b7 asset "
        f"{(_cwm.w_asset / _wsm):.0%} \u00b7 population "
        f"{(_cwm.w_pop / _wsm):.0%} \u00b7 response "
        f"{(_cwm.w_resp / _wsm):.0%} \u00b7 delay "
        f"{(_cwm.w_delay / _wsm):.0%}")
    if (_cwm.w_pop / _wsm) < 0.15:
        st.warning("Life-safety share below 15% is unusual for an "
                   "operational doctrine \u2014 make sure this is "
                   "intended.")
    if ((_cwm.w_resp + _cwm.w_delay) / _wsm) > 0.4:
        st.warning("Response cost + delay above 40% makes the "
                   "optimizer reluctant to field resources at all "
                   "\u2014 the fire terms should dominate.")
    rep = compute_costs(sim)
    d = rep.to_dict()

    # physical impact
    m = st.columns(5)
    m[0].metric("Burned area (ha)", f"{rep.burned_area_ha:,.1f}")
    m[1].metric("Burned forest (ha)", f"{rep.burned_forest_ha:,.1f}")
    m[2].metric("Population exposed", f"{rep.population_exposed:,.0f}")
    m[3].metric("Evacuated (safe)",
                f"{getattr(rep, 'population_evacuated', 0.0):,.0f}",
                help="People moved out by evacuation orders; they "
                     "leave the exposure and J_pop accounting. Cells "
                     "in or beside active flame empty at ~30%/min "
                     "under an order, elsewhere ~5%/min.")
    m[4].metric("Asset value lost",
                f"{rep.asset_value_lost:,.1f} / {rep.asset_value_total:,.1f}")

    with st.expander("How each term is computed \u2014 actual "
                     "value, unit, reference, normalization"):
        _cw = sim.cfg.cost
        _bref_c = float(getattr(_cw, "burn_reference_fraction", 0.02))
        _aref_ha = _bref_c * rep.burnable_cells * sim.cfg.cell_area_ha
        _H = float(_cw.horizon_steps)
        _wsum = (_cw.w_burn + _cw.w_asset + _cw.w_pop
                 + _cw.w_resp + _cw.w_delay)
        _rows = [
            ("J_burn", f"{rep.burned_area_ha:,.1f} ha burned "
             f"({rep.burned_forest_ha:,.1f} ha forest)",
             f"A_ref = {_aref_ha:,.0f} ha (5% of burnable; 'major "
             "fire')",
             "x/(1+x), x = A/A_ref \u2014 rational saturation: "
             "0.5 at the reference fire, gradient survives at ANY "
             "size (the old 1-exp form went numerically flat past "
             "3\u00d7 the reference and the optimizer lost its "
             "incentive on big fires)",
             rep.j_burn, _cw.w_burn),
            ("J_asset", f"{rep.asset_value_lost:,.2f} of "
             f"{rep.asset_value_total:,.2f} value units lost",
             "total building + critical-infrastructure value",
             "lost / total", rep.j_asset, _cw.w_asset),
            ("J_pop", f"{rep.population_person_steps:,.0f} "
             "person-steps in burning cells "
             f"(now exposed: {rep.population_exposed:,.0f})",
             f"total population \u00d7 H={_H:g} steps",
             "\u03a3 exposed / (pop \u00d7 H)", rep.j_pop,
             _cw.w_pop),
            ("J_resp", f"{rep.committed_capacity:,.0f} capacity now "
             "fielded (effort integrates over time)",
             f"staged pool \u00d7 1.2 = {rep.available_capacity:,.0f}"
             f", \u00d7 H={_H:g}",
             "\u03a3 committed / (ref \u00d7 H) \u2014 an army "
             "held for hours costs more than a strike", rep.j_resp,
             _cw.w_resp),
            ("J_del", f"{rep.mean_response_delay:,.1f} min mean "
             "dispatch (capacity-weighted)",
             f"delay_reference = {_cw.delay_reference:g} min",
             "mean delay / reference", rep.j_delay, _cw.w_delay),
        ]
        for _nm, _act, _ref, _frm, _jv, _wv in _rows:
            st.markdown(
                f"**{_nm}** = {_jv:.3f} \u00b7 weight {_wv:g} "
                f"\u2192 contribution {(_wv * _jv / _wsum):.3f}  \n"
                f"actual: {_act}  \n"
                f"reference: {_ref}  \n"
                f"normalization: {_frm}")
        st.caption(
            "J_resp and J_del price the response, never forbid it: "
            "their weights are small (0.2 vs 1.0) and J_burn is "
            "deliberately expensive, so 'never intervene' loses "
            "whenever the fire is real. The no-harm guard compares "
            "candidates on the PHYSICAL terms only (burn + asset + "
            "pop).")

    # normalized J terms and the total, all in [0,1]
    cc = st.columns(len(_J_TERMS) + 2)
    for i, (lab, key, _c, sub) in enumerate(_J_TERMS):
        cc[i].metric(f"{lab} · {sub}", f"{d[key]:.3f}")
    cc[-2].metric("J · DECISION", f"{rep.j_total:.3f}",
                  help="The 5-term cost the DSS optimizes (response "
                       "cost and delay INCLUDED: a commander "
                       "economizes the fleet).")
    cc[-1].metric("Jφ · OUTCOME",
                  f"{getattr(rep, 'j_physical', 0.0):.3f}",
                  help="Physical loss only (burn + asset + pop). "
                       "THE fair metric for comparing runs: a no-DSS "
                       "run pays no response cost by definition, so "
                       "J decision would reward doing nothing. Every "
                       "with/without comparison reads THIS.")

    series = st.session_state.cost_series
    if len(series) > 1:
        steps = [r["step"] for r in series]
        st.markdown("##### $J$ terms over time — one chart per term")
        titles = {k: lab for lab, k, _c, _s in _J_TERMS}
        cells = st.columns(3)
        panels = [(lab, key, col, titles[key]) for lab, key, col, _ in _J_TERMS]
        panels.append(("J", "j_total", "#111111", "J (decision)"))
        panels.append(("Jφ", "j_physical", "#b03a2e",
                       "Jφ (outcome, fair)"))
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
        nx = st.number_input("Resolution X (nx)", 20, 600, 200, 10)
        ny = st.number_input("Resolution Y (ny)", 20, 600, 200, 10)
        cell = st.number_input(
            "Cell size (m)", 1.0, 1000.0, 30.0, 1.0,
            help="30 m is the wildfire reference calibration: the "
                 "spread physics (m/min), travel times, station and "
                 "flight radii and the substep logic all reference "
                 "30 m and AUTO-SCALE to other sizes. Pick the cell "
                 "size for the DETAIL you want; pick nx/ny for the "
                 "AREA you want.")
        _ext_x = nx * cell / 1000.0
        _ext_y = ny * cell / 1000.0
        st.caption(f"Physical extent: **{_ext_x:.1f} \u00d7 "
                   f"{_ext_y:.1f} km** ({_ext_x * _ext_y * 100:.0f} "
                   "ha). Everything cell-size-dependent scales "
                   "automatically (ROS, dispatch/flight minutes, "
                   "service radii, burn reference = % of burnable "
                   "area). A smaller cell does NOT change the "
                   "physics \u2014 it only refines the grid of a "
                   "smaller or same area.")
        if cell < 15:
            st.info(f"{cell:.0f} m cells over {nx}\u00d7{ny} gives "
                    f"only {_ext_x:.1f}\u00d7{_ext_y:.1f} km \u2014 "
                    "a neighbourhood, not a landscape. For wildfire "
                    "scenarios 30 m cells are recommended; increase "
                    "nx/ny instead if you need a larger area.")
        if src == "Landscape type":
            seed = st.number_input("Seed", 0, 99999, 42)
            _D2R = 0.017453292519943295

            st.markdown("**Terrain — $U_{Geo}$**")
            g1, g2 = st.columns(2)
            relief = g1.slider(
                "Relief: plain ↔ mountain (m)", 20, 1200, 450, 10,
                help="Peak-to-valley elevation. Low = flat plain, high = "
                     "steep mountains where fire runs faster uphill.")
            access = g2.slider(
                "Accessibility", 0.0, 1.0, 1.0, 0.05,
                help="Scales the per-cell access field $A(x,y)\\in[0,1]$: "
                     "how quickly ground crews can reach EACH cell, "
                     "including forest interior (terrain sets it from "
                     "slope, roads force $A=1$ along them). $A=1$ means "
                     "every cell is reachable at nominal speed; lower "
                     "values slow suppression and raise the response-cost "
                     "and delay terms. It is not about villages: it is the "
                     "crew-reachability of the ground itself.")

            st.markdown("**Weather — $U_{Meteo}$**")
            m1, m2, m3 = st.columns(3)
            wspd = m1.slider("Wind speed (m/s)", 0.0, 30.0, 8.0, 0.5)
            wdir = m2.slider(
                "Wind direction (°)", 0, 359, 0, 5,
                help="Direction the wind blows toward (0 = +x / east).")
            moist = m3.slider(
                "Fuel moisture", 0.02, 0.40, 0.08, 0.01,
                help="Higher moisture slows spread and lowers intensity.")
            prec = st.slider(
                "Precipitation (mm/h)", 0.0, 30.0, 0.0, 1.0,
                help="Rain falls in scattered showers. While it rains the "
                     "engine drives fuel moisture up toward $0.35 > m_{ext}$ "
                     "(so the fire stalls and dies under sustained rain) and "
                     "ember spotting stops above $1$ mm/h.")

            st.markdown("**Vegetation / fuel — $U_{Fuel}$**")
            f1, f2 = st.columns(2)
            forest = f1.slider(
                "Forest density", 0.0, 1.0, 0.45, 0.05,
                help="Fraction of land covered by forest fuel clusters.")
            water = f2.slider(
                "Water level", 0.0, 0.40, 0.06, 0.02,
                help="Fraction of the lowest cells turned into water "
                     "(non-flammable).")
            fw1, fw2 = st.columns(2)
            river = fw1.checkbox("River", value=False)
            coast = fw2.checkbox("Coast (sea on east edge)", value=False)

            st.markdown("**Settlements & values at risk — $U_{Val}$**")
            v1, v2 = st.columns(2)
            nvill = v1.slider(
                "Number of settlements", 0, 50, 50, 1,
                help="Villages plus a central town (the first one). "
                     "0 = wildland only, no people or structures. "
                     "Settlements are scattered with blue-noise spacing "
                     "on suitable land (flat, low, near water).")
            popv = v2.slider(
                "Total population", 0, 500000, 60000, 5000,
                help="TOTAL population of the whole map, split across the "
                     "settlements with a skewed share: the town takes the "
                     "largest part, villages get smaller shares; the parts "
                     "sum exactly to this value.")
            bscale = st.slider(
                "Building / critical facility density", 0.2, 2.0, 1.0, 0.1,
                help="Scales the footprint of buildings and critical "
                     "facilities (hospital, power substation).")

            if st.button("Generate map", use_container_width=True,
                         type="primary"):
                _new_simulator(terrain.generate_landscape(
                    SimConfig(nx=int(nx), ny=int(ny), cell_size_m=float(cell)),
                    seed=int(seed), relief_m=float(relief),
                    forest_density=float(forest), base_moisture=float(moist),
                    water_level=float(water), wind_speed=float(wspd),
                    wind_dir_rad=float(wdir) * _D2R,
                    precipitation=float(prec),
                    with_assets=int(nvill) > 0, with_roads=True,
                    n_settlements=int(nvill),
                    population_per_settlement=int(popv),
                    building_scale=float(bscale), accessibility=float(access),
                    coast=bool(coast), river=bool(river)))
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
                "Grid", value=_lyv("ly_grid_v", True))
            eflags = dict(show_hillshade=_lyv("ly_relief_v", True),
                          show_fire=_lyv("ly_fire_v", True),
                          show_value=_lyv("ly_val_v", True),
                          show_roads=_lyv("ly_roads_v", True),
                          show_grid=_lyv("ly_grid_v", True))
            if eflags["show_value"] and float(world.value.vbld.max()
                    + world.value.vcrit.max() + world.value.vpop.max()) <= 0:
                st.caption("\u26a0 No assets on the map yet \u2014 the "
                           "overlay shows nothing. Use the Asset tool.")

    with view_col:
        _views = ["2D canvas", "2D pan / zoom", "3D terrain"]
        _vcur = st.session_state.get("editor_view_sel", "2D canvas")
        vmode = st.radio("Editor view", _views,
                         index=_views.index(_vcur) if _vcur in _views else 0,
                         horizontal=True, label_visibility="collapsed")
        st.session_state["editor_view_sel"] = vmode
        if vmode == "2D pan / zoom":
            # view-only inspection with the same top-right toolbar as the
            # 3D view: drag = pan, wheel = zoom. Editing happens on the
            # 2D canvas view.
            st.caption("Inspect view \u00b7 drag to pan, wheel to zoom, "
                       "toolbar top-right. Switch back to the 2D canvas "
                       "to edit.")
            st.plotly_chart(
                viz.map_figure_2d(world, sim=sim,
                                  scale=_fit_scale(cfg.nx), **eflags),
                use_container_width=True,
                key=f"edpz_{st.session_state.map_version}",
                config={"scrollZoom": True,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
        elif vmode == "3D terrain":
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
        _u = unit
        if key in ("wwd", "slope", "aspect"):
            field = np.degrees(field)
            _u = "°"
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(field, origin="upper", cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{name}  [{_u}]")
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
        for _i, _m0 in _REFERENCE_FUELS.items():
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
            for _i, _m0 in _REFERENCE_FUELS.items():
                FUEL_MODELS[_i] = dataclasses.replace(_m0)
            st.rerun()

    with st.expander("Cost model (normalized cost-plus-loss, System Description Sec. 14)"):
        st.caption("Five priority weights over the normalized $[0,1]$ terms, "
                   "plus the reference scales and safeguards.")
        a, b = st.columns(2)
        cfg.cost.w_burn = a.number_input(
            "$w_1$ - burned area weight", 0.0, 10.0,
            float(cfg.cost.w_burn), 0.1, format="%.2f")
        cfg.cost.w_asset = b.number_input(
            "$w_2$ - asset loss weight", 0.0, 10.0,
            float(cfg.cost.w_asset), 0.1, format="%.2f")
        cfg.cost.w_pop = a.number_input(
            "$w_3$ - population exposure weight", 0.0, 10.0,
            float(cfg.cost.w_pop), 0.1, format="%.2f")
        cfg.cost.w_resp = b.number_input(
            "$w_4$ - response cost weight", 0.0, 10.0,
            float(cfg.cost.w_resp), 0.1, format="%.2f")
        cfg.cost.w_delay = a.number_input(
            "$w_5$ - response delay weight", 0.0, 10.0,
            float(cfg.cost.w_delay), 0.1, format="%.2f")
        cfg.cost.acceptance_fraction = b.number_input(
            "acceptance threshold (fraction of do-nothing)", 0.0, 1.0,
            float(cfg.cost.acceptance_fraction), 0.05, format="%.2f")
        cfg.cost.population_at_risk_fraction = a.number_input(
            "$\\rho_{risk}$ - population at risk fraction", 0.0, 1.0,
            float(cfg.cost.population_at_risk_fraction), 0.005, format="%.3f")
        cfg.cost.horizon_steps = b.number_input(
            "$H$ - scenario horizon (steps)", 1.0, 5000.0,
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
    with st.expander("Validation criteria \u2014 definitions"):
        st.markdown(
            "$A$ = simulated burn, $B$ = observed burn. A free-spread "
            "model (no suppression) is validated on FRONT PROPAGATION, not "
            "final area.\n\n"
            "| Criterion | Definition | Checks |\n"
            "|---|---|---|\n"
            r"| Coverage (POD) | $\lvert A\cap B\rvert\,/\,\lvert B\rvert$ | fraction of the observed burn reproduced (target $>0.7$) |"
            "\n"
            r"| Front position error | mean / p90 of $\overline{d}(\partial A,\partial B)$ (m) | how far the simulated front is from the observed front |"
            "\n"
            "| Arrival-time agreement | mean abs. difference of simulated "
            "vs FIRMS first-detection time (h) and Spearman $\\rho$ of the "
            "arrival order | the rate-of-spread (propagation) test |\n\n"
            "Area-overlap scores (Dice, Jaccard, false alarm, area bias) "
            "are omitted: a free run overpredicts a suppressed real fire. "
            "Upload an official EFFIS/EMS perimeter for referee-grade area "
            "metrics.")
    with st.expander("Referee protocol \u2014 how to report these results"):
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
    seeds = int(c3.number_input(
        "Seeds", 1, 12, 5,
        help="Ember spotting is stochastic, so each criterion is reported "
             "as mean +/- sd over this many seeds. 5 is a good default "
             "(3 minimum for a spread, 8-10 for final thesis numbers)."))
    c4, c5, c6 = st.columns(3)
    cell = float(c4.number_input("Cell size (m)", 30.0, 200.0, 90.0, 10.0,
                                 help="90 m matches the satellite truth "
                                      "resolution and keeps the run fast."))
    hours = float(c5.number_input("Time window (h)", 1.0, 120.0,
                                  float(av.CASES[case_id]["hours"]), 1.0,
                                  help="The time window that bounds BOTH the "
                                       "observed footprint (only what was seen "
                                       "burning by this hour) AND the "
                                       "simulation. Lower it to compare an "
                                       "earlier phase; the observed (blue) "
                                       "area shrinks with it."))
    stepm = float(c6.number_input(
        "Step length (min)", 10.0, 60.0, 30.0, 5.0,
        help="Temporal resolution of the spread. 30 min matches the hourly "
             "ERA5 weather and keeps runs fast; 15 min is finer. Do not "
             "exceed 60 min (the weather is hourly)."))
    stop_mode = st.selectbox(
        "Stopping rule (when does the simulation end?)",
        ["Match observed area (recommended)",
         "Fixed hours"],
        help="Both use the time window above for the observed footprint. "
             "'Match observed area' stops as soon as the simulated burn "
             "reaches the observed area OR the time window elapses, "
             "whichever comes first, so the shapes are compared at a "
             "comparable extent (the suppression-driven size difference "
             "cancels). 'Fixed hours' always runs the whole window and lets "
             "the free burn overpredict.")
    # a wind-direction offset can be adopted from the ensemble table below
    # (the "use this wind" button) and applied to the next run; apply the
    # pending pick BEFORE the widgets are created so it takes effect
    st.session_state.setdefault("val_wind_offset", 0.0)
    st.session_state.setdefault("val_wens", False)
    if st.session_state.get("_val_adopt_offset") is not None:
        st.session_state["val_wind_offset"] = float(
            st.session_state.pop("_val_adopt_offset"))
        st.session_state["val_wens"] = False    # clean, ensemble-off rerun
    c7, c8 = st.columns([1, 2])
    wind_offset = float(c7.number_input(
        "Wind-direction offset (deg)", -180.0, 180.0, step=15.0,
        key="val_wind_offset",
        help="Rotate the reanalysis wind direction by this angle for THIS "
             "run. Run the ensemble first, then adopt the best-matching "
             "member below and run again (ensemble off) to lock that wind."))
    if abs(wind_offset) > 1e-6:
        c8.caption(f"This run rotates the reanalysis wind by "
                   f"{wind_offset:+.0f}\u00b0.")
    wens = st.checkbox(
        "Wind-direction uncertainty ensemble (8 extra runs)", key="val_wens",
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
                    # download on the documented window (fetches all pts)
                    _case_ign = dict(case)
                    _case_ign["hours"] = float(av.CASES[case_id]["hours"])
                    pts = av._download_firms(_case_ign, key, args.cache)
                    # OBSERVED footprint bounded by the TIME WINDOW the user
                    # picked: the detections seen burning within [t0, t0 + T].
                    # _firms_mask_and_ignition windows the detections by
                    # case["hours"] and morphologically closes them, so the
                    # blue mask is a FILLED footprint that grows with T_max.
                    case["hours"] = float(hours)
                    fmask, first, _ign_cells = \
                        av._firms_mask_and_ignition(
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

            _pts = locals().get("pts")
            _obs_arr = None
            if _pts and first is not None and fmask is not None:
                _obs_arr = av.firms_arrival_hours(case, _pts, nx, ny, cell,
                                                  fmask, first[4])
            # observed arrival window = time span of the observed detections
            # (last minus first). The arrival-time error is judged against
            # THIS real fire duration, not the sim's (possibly short) run.
            _obs_window = None
            if _obs_arr is not None and np.isfinite(_obs_arr).any():
                _obs_window = float(_obs_arr[np.isfinite(_obs_arr)].max())
            # STOPPING RULE: keep the suppression-free run from overgrowing
            # the suppressed real fire (see the selectbox)
            # Match observed area: stop when the sim reaches the (windowed)
            # observed area OR the time window elapses, whichever first.
            # Fixed hours: run the whole window (stop_area = None).
            _stop_area = None
            if stop_mode.startswith("Match observed area"):
                _stop_area = float(np.asarray(obs, dtype=bool).sum())
            n_total = int(round(case["hours"] * 60.0 / stepm))
            weather_run = weather
            if abs(wind_offset) > 1e-6:
                weather_run = dict(weather)
                weather_run["wind_direction_10m"] = [
                    (d + wind_offset) % 360.0
                    for d in weather["wind_direction_10m"]]
                st.write(f"wind-direction offset {wind_offset:+.0f}° "
                         "applied to this run")
            runs, shape = av.run_case(case, args, dem, (dlons, dlats),
                                      wc, (wlons, wlats), weather_run, obs,
                                      ign, progress_cb=_cb,
                                      frame_cb=_frame,
                                      frame_every=max(1, n_total // 24),
                                      obs_arrival=_obs_arr,
                                      stop_area=_stop_area)
            # representative wind for the on-map arrow: mean DOWNWIND
            # direction over the simulated window (where the wind pushes the
            # fire). ERA5 gives the "from" direction; downwind = fire heading.
            _wtoward = None
            try:
                _wd = weather_run["wind_direction_10m"]
                _h0w = int(case.get("t0_hour", 0))
                _hi = min(len(_wd), _h0w + max(1, int(round(case["hours"]))))
                _ang = [np.radians((270.0 - float(_wd[h])) % 360.0)
                        for h in range(_h0w, max(_h0w + 1, _hi))]
                _u = float(np.mean([np.cos(a) for a in _ang]))
                _v = float(np.mean([np.sin(a) for a in _ang]))
                _wtoward = float(np.degrees(np.arctan2(_v, _u)) % 360.0)
            except Exception:
                _wtoward = None
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
                    progress_cb=_ecb, stop_area=_stop_area)
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
                _keys = ["hit_rate", "mean_m", "p90_m",
                         "arrival_mae_h", "arrival_rho"]
                _summary = {k: {"mean": float(_np2.nanmean(
                                    [r[0].get(k, _np2.nan) for r in runs])),
                                "sd": float(_np2.nanstd(
                                    [r[0].get(k, _np2.nan) for r in runs]))}
                            for k in _keys}
                _json.dump({
                    "case_id": case_id, "case": {k: v for k, v in
                                                 case.items()},
                    "settings": {"cell_m": cell, "step_minutes": stepm,
                                 "hours": hours, "seeds": seeds,
                                 "wind_ensemble": bool(wens),
                                 "wind_offset_deg": wind_offset,
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
                _best = max(runs, key=lambda r: r[0]["hit_rate"])[1]
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
                best=max(runs, key=lambda r: r[0]["hit_rate"])[1],
                obs=obs, shape=shape, case=case["label"], cell=cell,
                base=av._basemap(ftype, demg), ign=ign,
                wind_offset=wind_offset, obs_window_h=_obs_window,
                wind_toward_deg=_wtoward, stop_rule=stop_mode,
                seeds=seeds, cap_hours=hours,
                truth="perimeter" if up is not None else "firms")
        except Exception as exc:
            st.error(f"Validation failed: {exc}")
            return

    res = st.session_state.get("val_result")
    if res:
        st.markdown(f"#### Result \u2014 {res['case']}")
        import numpy as _np
        # exact settings that produced THIS result, so two runs can be
        # compared at a glance (only these change the outcome; the engine is
        # deterministic and a longer run is a superset of a shorter one)
        _woff = float(res.get("wind_offset", 0.0) or 0.0)
        _shr = res.get("runs", [{}])
        _sh0 = float(_np.mean([r.get("stop_hours", float("nan"))
                               for r in _shr])) if _shr else float("nan")
        _Bcells0 = int(_np.asarray(res.get("obs"), dtype=bool).sum()) \
            if res.get("obs") is not None else 0
        st.info(
            f"**Run settings** \u2014 cell **{float(res.get('cell', 0)):.0f} m** \u00b7 "
            f"wind offset **{_woff:+.0f}\u00b0** \u00b7 stopping "
            f"**{res.get('stop_rule', '?')}** \u00b7 stopped at "
            f"**{_sh0:.1f} h** (cap {float(res.get('cap_hours', 0)):.0f} h) \u00b7 "
            f"seeds **{int(res.get('seeds', 0))}** \u00b7 |B| **{_Bcells0}** cells. "
            "If two runs differ, one of THESE changed (most often the wind "
            "offset, which can linger from an earlier ensemble-adopt). The "
            "max-hours cap alone cannot move or reshape the burn; the engine "
            "is deterministic and a longer run is a superset of a shorter "
            "one.")

        def _cm(k):
            vs = [r.get(k, float("nan")) for r in res["runs"]]
            vs = [v for v in vs if v == v]
            return (float(_np.mean(vs)) if vs else float("nan"),
                    float(_np.std(vs)) if vs else float("nan"))

        def _fmtv(k, p):
            mv, sv = _cm(k)
            return "n/a" if mv != mv else f"{mv:.{p}f} \u00b1 {sv:.{p}f}"
        _sh = _cm("stop_hours")[0]
        if _sh == _sh:
            st.caption(f"Simulation stopped at ~{_sh:.1f} h per the stopping "
                       "rule, so the fronts are scored at a comparable "
                       "extent and the run always terminates.")
        # normalization context: fire equivalent radius and effective time
        _obsm = _np.asarray(res["obs"], dtype=bool)
        _cellm = float(res.get("cell", 90.0))
        _Bc = float(_obsm.sum())
        _R = ((_Bc * _cellm * _cellm / 3.141592653589793) ** 0.5
              if _Bc > 0 else 1.0)
        # arrival error is judged against the OBSERVED fire duration (span of
        # the satellite detections), not the sim's own run length; this is a
        # per-case, physically meaningful reference for every case
        _T = res.get("obs_window_h")
        if not (_T and _T == _T and _T > 0):
            _T = _sh if (_sh == _sh and _sh > 0) else 24.0

        def _clip01(x):
            return max(0.0, min(1.0, float(x)))

        def _assess(key):
            mv, _sv = _cm(key)
            if mv != mv:
                return ("n/a", None, "n/a", "")
            if key == "hit_rate":
                pct = 100 * _clip01(mv); tgt = "\u2265 0.70"
                vd = ("\u2713 pass" if mv >= 0.7 else
                      "~ partial" if mv >= 0.4 else "\u2717 fail")
            elif key == "mean_m":
                pct = 100 * _clip01(1 - mv / _R); tgt = f"\u2264 {0.25*_R:.0f} m"
                vd = ("\u2713 pass" if mv <= 0.25*_R else
                      "~ partial" if mv <= 0.5*_R else "\u2717 fail")
            elif key == "p90_m":
                pct = 100 * _clip01(1 - mv / _R); tgt = f"\u2264 {0.5*_R:.0f} m"
                vd = ("\u2713 pass" if mv <= 0.5*_R else
                      "~ partial" if mv <= _R else "\u2717 fail")
            elif key == "arrival_mae_h":
                pct = 100 * _clip01(1 - mv / _T); tgt = f"\u2264 {0.25*_T:.1f} h"
                vd = ("\u2713 pass" if mv <= 0.25*_T else
                      "~ partial" if mv <= 0.5*_T else "\u2717 fail")
            elif key == "arrival_rho":
                # the observed detection times are satellite-overpass gated;
                # with few distinct time levels the rank correlation is
                # dominated by ties and is not meaningful, so mark it n/a and
                # keep it OUT of the overall score
                _lv = _cm("arrival_obs_levels")[0]
                if _lv != _lv or _lv < 5:
                    return ("n/a", None, "n/a (overpass-limited)",
                            "\u2265 0.60")
                pct = 100 * _clip01(mv); tgt = "\u2265 0.60"
                vd = ("\u2713 pass" if mv >= 0.6 else
                      "~ partial" if mv >= 0.3 else "\u2717 fail")
            elif key == "arrival_n":
                pct = 100 * _clip01(mv / 50.0); tgt = "\u2265 30 cells"
                vd = ("\u2713 pass" if mv >= 30 else
                      "~ partial" if mv >= 10 else "\u2717 fail")
            else:
                pct = 0.0; tgt = ""; vd = "n/a"
            return (f"{pct:.0f}%", pct, vd, tgt)
        _CR = [
            ("hit_rate", "Coverage (POD)", 3, "|A\u2229B| / |B|",
             "fraction of the observed burn reproduced"),
            ("mean_m", "Front error, mean (m)", 0, "mean d(\u2202A,\u2202B)",
             "average front-to-front distance"),
            ("p90_m", "Front error, p90 (m)", 0, "P90 d(\u2202A,\u2202B)",
             "90th-percentile front distance"),
            ("arrival_mae_h", "Arrival MAE (h)", 2, "mean |t_sim \u2212 t_obs|",
             "arrival-time error over shared cells"),
        ]
        _left, _right, _pcts = [], [], []
        for _key, _lab, _pr, _eqn, _meas in _CR:
            _score_s, _pct, _vd, _tgt = _assess(_key)
            if _pct is not None:
                _pcts.append(_pct)
            _left.append({"Criterion": _lab,
                          "Value (mean \u00b1 sd)": _fmtv(_key, _pr),
                          "Score": _score_s, "Verdict": _vd})
            _right.append({"Criterion": _lab, "Equation": _eqn,
                           "What it measures": _meas, "Target": _tgt})
        _overall = (sum(_pcts) / len(_pcts)) if _pcts else float("nan")
        # accumulate this case's Table 5.5 row so an export can hold all four
        # scenarios together (keyed by case, so a re-run updates its own row)
        _acell = float(res.get("cell", 0.0)) ** 2
        st.session_state.setdefault("val_rows", {})
        st.session_state["val_rows"][str(res.get("case", ""))] = {
            "POD": _cm("hit_rate")[0], "mean_m": _cm("mean_m")[0],
            "p90_m": _cm("p90_m")[0], "MAE": _cm("arrival_mae_h")[0],
            "R": _R, "T": _T, "A_cell": _acell,
            "cell": float(res.get("cell", 0.0))}
        st.markdown("**Success metrics** \u2014 normalized score = agreement with "
                    "the target (100% is ideal), pass / partial / fail per "
                    "criterion")
        _lc, _rc = st.columns([1.15, 1])
        _lc.table(_left)
        _rc.table(_right)
        if _overall == _overall:
            _ov = ("SUCCESS" if _overall >= 70 else
                   "PARTIAL" if _overall >= 45 else "WEAK")
            st.markdown(f"**Overall validation score: {_overall:.0f}% "
                        f"({_ov})** \u2014 mean of the four normalized criteria.")
        st.caption("Normalization: coverage and \u03c1 are already 0..1; the front "
                   f"errors are scored against the fire equivalent radius "
                   f"R = {_R:.0f} m (\u221a(|B|\u00b7cell\u00b2/\u03c0)); the arrival MAE against "
                   f"the OBSERVED fire duration T = {_T:.1f} h (span of the "
                   "satellite detections), so the timing error is judged "
                   "relative to how long the real fire actually spread. "
                   "Suppression-free model: validated on front propagation, "
                   "not final area (Dice/Jaccard/FAR/area-bias omitted; "
                   "upload an EFFIS/EMS perimeter for area metrics).")
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
        # keep this case's agreement map so the export can embed the PNG of
        # EVERY scenario run so far, not just the current one
        st.session_state.setdefault("val_imgs", {})
        st.session_state["val_imgs"][str(res.get("case", ""))] = img.copy()
        # suggested wind-direction offset to steer the sim toward the
        # OBSERVED spread: rotate the wind by (observed heading - simulated
        # heading), both measured from the ignition to the burn centroid
        _sugg = None
        try:
            _obsm2 = _np.asarray(res["obs"], dtype=bool)
            _simm2 = _np.asarray(res["best"], dtype=bool)
            _i0s = (res["ign"][0] if isinstance(res.get("ign"), list)
                    else res.get("ign"))
            if _i0s is not None and _obsm2.any() and _simm2.any():
                _gx0, _gy0 = float(_i0s[0]), float(_i0s[1])
                _oy, _ox = _np.where(_obsm2)
                _sy, _sx = _np.where(_simm2)

                def _brg2(cx, cy):
                    _e = cx - _gx0
                    _n = -(cy - _gy0)          # image row grows south
                    return _np.degrees(_np.arctan2(_e, _n)) % 360.0
                _ob = _brg2(_ox.mean(), _oy.mean())
                _sb = _brg2(_sx.mean(), _sy.mean())
                # residual heading gap of THIS run, plus the offset already
                # applied, gives the ABSOLUTE offset to set (so it converges
                # instead of oscillating around the leftover gap)
                _applied = float(res.get("wind_offset", 0.0) or 0.0)
                _gap = ((_ob - _sb + 180.0) % 360.0) - 180.0
                _sugg = ((_applied + _gap + 180.0) % 360.0) - 180.0
        except Exception:
            _sugg = None
            _gap = 0.0
        # wind direction (mean over the run) shown as a compact line ABOVE
        # the map, so the map itself stays clean
        _wt = res.get("wind_toward_deg")
        if _wt is not None and _wt == _wt:
            _brg = (90.0 - float(_wt)) % 360.0
            _pts8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            _gly8 = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]
            _bi = int((_brg + 22.5) // 45) % 8
            _msg = (f"**Mean wind over the run:** {_gly8[_bi]} toward "
                    f"**{_pts8[_bi]}** (bearing {_brg:.0f}°, downwind). The "
                    "spread also feels upslope, so the burn direction can "
                    "differ from the raw wind.")
            if _sugg is not None and abs(_gap) >= 8:
                _msg += (f"  \n**Set the wind-direction offset to about "
                         f"{_sugg:+.0f}°** to steer the sim toward the "
                         f"observed spread (current residual gap "
                         f"{_gap:+.0f}°). It may take one more pass since "
                         "slope also steers the fire; the wind ensemble is "
                         "the reliable optimizer.")
            elif _sugg is not None:
                _msg += ("  \nHeading matches the observed spread "
                         "(residual gap under 8°).")
            st.markdown(_msg)
        # interactive agreement map: its OWN zoom/pan (scroll + toolbar),
        # independent of the browser zoom, on a fixed-height canvas
        try:
            import plotly.graph_objects as _go
            _figm = _go.Figure(_go.Image(z=img))
            _figm.update_xaxes(visible=False,
                               scaleanchor="y", constrain="domain")
            _figm.update_yaxes(visible=False)
            _figm.update_layout(height=640, dragmode="pan",
                                margin=dict(l=0, r=0, t=6, b=0))
            st.plotly_chart(_figm, use_container_width=True,
                            config={"scrollZoom": True,
                                    "displayModeBar": True,
                                    "displaylogo": False})
        except Exception:
            st.image(img, use_container_width=True)
        st.caption("Agreement map on the real terrain \u2014 green: correctly "
                   "predicted burn, red: simulated only, blue: observed only, "
                   "yellow: ignition. Scroll or use the toolbar to zoom and "
                   "pan; double-click resets.")
        # --- Export: a Word report (Table 5.5 + computation) + map PNG ---
        st.markdown("**Export for the thesis** \u2014 a Word report (Table 5.5 for "
                    "all cases run so far, plus the R / A_cell / % / target "
                    "computation for this case) and the agreement map as PNG")
        _accum = st.session_state.get("val_rows", {})
        _ec1, _ec2 = st.columns([2, 1])
        _ec1.caption("Table 5.5 currently holds: "
                     + (", ".join(_accum.keys()) if _accum else "(none yet)")
                     + f"  ({len(_accum)}/4 scenarios). Run each case once, "
                     "then export; the table and the per-case maps accumulate.")
        if _ec2.button("Clear accumulated table"):
            # wipe EVERYTHING: the accumulated Table 5.5 rows + per-case maps,
            # and the current run so nothing lingers in the view or the export
            for _k in ("val_rows", "val_imgs", "val_result", "val_log",
                       "val_ens", "val_wens", "val_adopt_pick"):
                st.session_state.pop(_k, None)
            st.rerun()
        if st.button("Export to Word + PNG", type="primary"):
            try:
                _dbytes, _pbytes, _odir = _make_validation_report(
                    res, _left, _right, _overall, _R, _T, img,
                    st.session_state.get("val_rows", {}),
                    st.session_state.get("val_imgs", {}))
                st.success(f"Saved to {os.path.relpath(_odir)}")
                st.download_button("\u2b07 Download Word report (.docx)", _dbytes,
                                   file_name="validation_report.docx")
                if _pbytes:
                    st.download_button("\u2b07 Download agreement map (.png)",
                                       _pbytes, file_name="agreement_map.png")
            except Exception as _exc:
                st.error(f"Export failed: {_exc}")
        if st.session_state.get("val_log"):
            with st.expander("Run log \u2014 what happened, step by step"):
                st.code("\n".join(st.session_state["val_log"]),
                        language=None)
        _ens = st.session_state.get("val_ens")
        if _ens:
            st.markdown("**Wind-direction ensemble** - same fire, wind "
                        "rotated; ranked by coverage (POD):")
            _best_e = max(_ens, key=lambda x: x["hit_rate"])
            rows = ["| rotation | Coverage (POD) | Front err (km) |",
                    "|---|---|---|"]
            for m in sorted(_ens, key=lambda m: m["offset"]):
                mark = " **\u2190 best**" if m is _best_e else ""
                rows.append(f"| {m['offset']:+.0f}\u00b0 | "
                            f"{m['hit_rate']:.3f} | "
                            f"{m['mean_m']/1000:.1f}{mark} |")
            st.markdown("\n".join(rows))
            _zero = [m for m in _ens if m["offset"] == 0]
            _z = _zero[0]["hit_rate"] if _zero else float("nan")
            st.caption(f"Best member at {_best_e['offset']:+.0f}\u00b0 "
                       f"(coverage {_best_e['hit_rate']:.2f} vs {_z:.2f} "
                       "with the raw reanalysis wind). A large gain from "
                       "rotation means the case is limited by the INPUT "
                       "wind (terrain channeling, fire convection), not by "
                       "the spread model; report it as input sensitivity.")
            # --- adopt a wind offset and rerun without the ensemble ---
            _offs = sorted({m["offset"] for m in _ens})
            _dmap = {m["offset"]: m["hit_rate"] for m in _ens}
            _ca, _cb = st.columns([1.5, 1])
            _pick = _ca.selectbox(
                "Adopt a wind offset for a clean (ensemble-off) rerun",
                _offs, index=_offs.index(_best_e["offset"]),
                format_func=lambda o: f"{o:+.0f}\u00b0  (POD {_dmap[o]:.3f})",
                key="val_adopt_pick")
            _cb.write("")
            if _cb.button("Use this wind \u2192 rerun", type="primary"):
                st.session_state["_val_adopt_offset"] = float(_pick)
                st.rerun()
            st.caption("Adopting sets the wind-direction offset field above "
                       "and turns the ensemble off. Then press "
                       "**Run validation** to score that single wind against "
                       "the truth.")


def _make_validation_report(res, left, right, overall, R, T, img,
                            all_rows=None, all_imgs=None):
    """Build a Word report (Table 5.5 row + detailed R/%/target computation +
    the agreement map of EVERY case run so far) and a PNG of the map, always
    in the same format. Returns (docx_bytes, png_bytes, out_dir)."""
    import datetime as _dt
    import numpy as _np2
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from PIL import Image as _PILImg

    def _agg(k):
        vs = [r.get(k, float("nan")) for r in res["runs"]]
        vs = [v for v in vs if v == v]
        return (float(_np2.mean(vs)) if vs else float("nan"),
                float(_np2.std(vs)) if vs else float("nan"))

    def _f(v, p):
        return "n/a" if (v != v) else f"{v:.{p}f}"

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _slug = "".join(c for c in str(res.get("case", "case"))
                    if c.isalnum() or c in "-_")[:40]
    out_dir = os.path.join(root, "validation", "runs",
                           f"export_{_slug}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, "agreement_map.png")
    try:
        _PILImg.fromarray(img).save(png_path)
    except Exception:
        png_path = None

    d = Document()
    d.styles["Normal"].font.name = "Times New Roman"
    d.styles["Normal"].font.size = Pt(11)
    d.add_heading(f"Simulation Core Validation: {res.get('case', '')}", 1)

    _woff = float(res.get("wind_offset", 0.0) or 0.0)
    _sh = _agg("stop_hours")[0]
    _Bc = int(_np2.asarray(res.get("obs"), dtype=bool).sum()) \
        if res.get("obs") is not None else 0
    d.add_paragraph(
        f"Run settings: cell {float(res.get('cell', 0)):.0f} m, time window "
        f"{float(res.get('cap_hours', 0)):.0f} h, wind offset {_woff:+.0f} deg, "
        f"stopping {res.get('stop_rule', '?')}, stopped at {_sh:.1f} h, seeds "
        f"{int(res.get('seeds', 0))}, |B| {_Bc} cells, R {R:.0f} m, T {T:.1f} h "
        f"(exported {stamp}).")

    # ---- Table 5.5 (all cases run so far, paste into the thesis table) ----
    d.add_heading("Table 5.5 (paste into the thesis)", 2)
    # accumulate the current case into the row set so a single run still works
    _acell_now = float(res.get("cell", 0.0)) ** 2
    rows = dict(all_rows or {})
    rows.setdefault(str(res.get("case", "")), {
        "POD": _agg("hit_rate")[0], "mean_m": _agg("mean_m")[0],
        "p90_m": _agg("p90_m")[0], "MAE": _agg("arrival_mae_h")[0],
        "R": R, "T": T, "A_cell": _acell_now,
        "cell": float(res.get("cell", 0.0))})
    _order = list(rows.keys())
    t5 = d.add_table(rows=1 + len(_order) + 1, cols=6); t5.style = "Table Grid"
    for i, x in enumerate(["Real Sce. (Table 5.4)", "cell (m)", "POD (-)",
                           "d_Front mean (m)", "d_90 (m)", "MAE (h)"]):
        c = t5.rows[0].cells[i]; c.text = ""
        c.paragraphs[0].add_run(x).bold = True
    for ri, ck in enumerate(_order, start=1):
        rd = rows[ck]
        vals = [ck, _f(rd.get("cell", float("nan")), 0),
                _f(rd.get("POD", float("nan")), 3),
                _f(rd.get("mean_m", float("nan")), 0),
                _f(rd.get("p90_m", float("nan")), 0),
                _f(rd.get("MAE", float("nan")), 2)]
        for i, x in enumerate(vals):
            t5.rows[ri].cells[i].text = str(x)
    sc = t5.rows[-1].cells
    sc[0].text = ""; sc[0].paragraphs[0].add_run("Success Criteria").bold = True
    for i, x in enumerate(["-", ">= 0.70", "<= 0.25 R", "<= 0.50 R",
                           "<= 0.25 T"], start=1):
        sc[i].text = x
    d.add_paragraph("The front and arrival targets are relative to the "
                    "per-case R and T listed below, since these differ by "
                    "case.").italic = True

    # ---- per-case normalization constants ----
    d.add_heading("Per-case normalization (R, T, A_cell)", 2)
    nt = d.add_table(rows=1 + len(_order), cols=6); nt.style = "Table Grid"
    for i, x in enumerate(["Real Sce.", "cell (m)", "A_cell (m^2)", "R (m)",
                           "0.25 R / 0.50 R (m)", "T (h)  /  0.25 T (h)"]):
        c = nt.rows[0].cells[i]; c.text = ""
        c.paragraphs[0].add_run(x).bold = True
    for ri, ck in enumerate(_order, start=1):
        rd = rows[ck]; _Rc = rd.get("R", 0.0); _Tc = rd.get("T", 0.0)
        _ac = rd.get("A_cell", 0.0); _cl = rd.get("cell", 0.0)
        vals = [ck, f"{_cl:.0f}", f"{_ac:.0f}", f"{_Rc:.0f}",
                f"{0.25*_Rc:.0f} / {0.50*_Rc:.0f}",
                f"{_Tc:.1f} / {0.25*_Tc:.1f}"]
        for i, x in enumerate(vals):
            nt.rows[ri].cells[i].text = str(x)

    # ---- detailed computation for EVERY case run so far ----
    def _clip01(v):
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    def _detail_rows(rd):
        """(Criterion, Value, Normalization, Score, Target, Verdict) list for
        one case, computed from its stored POD / front / MAE and R / T."""
        _Rc = float(rd.get("R", 0.0)) or float("nan")
        _Tc = float(rd.get("T", 0.0)) or float("nan")
        pod = rd.get("POD", float("nan"))
        fm = rd.get("mean_m", float("nan"))
        f9 = rd.get("p90_m", float("nan"))
        mae = rd.get("MAE", float("nan"))
        out = []
        # Coverage (POD)
        _s = 100 * _clip01(pod) if pod == pod else float("nan")
        _v = ("pass" if pod >= 0.70 else "partial" if pod >= 0.50 else "fail") \
            if pod == pod else "n/a"
        out.append(("Coverage (POD)", _f(pod, 3), "value (already 0..1)",
                    _f(_s, 0) + "%" if _s == _s else "n/a", ">= 0.70", _v))
        # Front mean
        _s = 100 * _clip01(1 - fm / _Rc) if (fm == fm and _Rc == _Rc) \
            else float("nan")
        _v = ("pass" if fm <= 0.25 * _Rc else "partial" if fm <= 0.50 * _Rc
              else "fail") if (fm == fm and _Rc == _Rc) else "n/a"
        out.append(("Front error, mean (m)", _f(fm, 0), "1 - value / R",
                    _f(_s, 0) + "%" if _s == _s else "n/a",
                    f"<= 0.25 R = {0.25*_Rc:.0f} m" if _Rc == _Rc else "n/a",
                    _v))
        # Front p90
        _s = 100 * _clip01(1 - f9 / _Rc) if (f9 == f9 and _Rc == _Rc) \
            else float("nan")
        _v = ("pass" if f9 <= 0.50 * _Rc else "partial" if f9 <= _Rc
              else "fail") if (f9 == f9 and _Rc == _Rc) else "n/a"
        out.append(("Front error, p90 (m)", _f(f9, 0), "1 - value / R",
                    _f(_s, 0) + "%" if _s == _s else "n/a",
                    f"<= 0.50 R = {0.50*_Rc:.0f} m" if _Rc == _Rc else "n/a",
                    _v))
        # Arrival MAE
        _s = 100 * _clip01(1 - mae / _Tc) if (mae == mae and _Tc == _Tc) \
            else float("nan")
        _v = ("pass" if mae <= 0.25 * _Tc else "partial" if mae <= 0.50 * _Tc
              else "fail") if (mae == mae and _Tc == _Tc) else "n/a"
        out.append(("Arrival MAE (h)", _f(mae, 2), "1 - value / T",
                    _f(_s, 0) + "%" if _s == _s else "n/a",
                    f"<= 0.25 T = {0.25*_Tc:.1f} h" if _Tc == _Tc else "n/a",
                    _v))
        return out

    for ck in _order:
        rd = rows[ck]
        _Rc = float(rd.get("R", 0.0)); _Tc = float(rd.get("T", 0.0))
        _ac = float(rd.get("A_cell", 0.0)); _cl = float(rd.get("cell", 0.0))
        d.add_heading(f"Detailed computation: {ck}", 2)
        p = d.add_paragraph()
        p.add_run(
            "A_cell = cell area = (cell size)^2 = "
            f"{_ac:.0f} m^2 (cell = {_cl:.0f} m). "
            f"R = sqrt(|B| * A_cell / pi) = {_Rc:.0f} m is the equivalent "
            "radius of the observed burned area. "
            f"T = {_Tc:.1f} h is the observed fire duration (span of the "
            "satellite detections). Coverage is already in [0, 1]; front "
            "errors are scored as 1 - value/R; the arrival error as "
            "1 - value/T.").italic = True
        drows = _detail_rows(rd)
        dt = d.add_table(rows=1 + len(drows), cols=6); dt.style = "Table Grid"
        for i, x in enumerate(["Criterion", "Value", "Normalization",
                               "Score", "Target", "Verdict"]):
            c = dt.rows[0].cells[i]; c.text = ""
            c.paragraphs[0].add_run(x).bold = True
        _scores = []
        for ri, row6 in enumerate(drows, start=1):
            for i, x in enumerate(row6):
                dt.rows[ri].cells[i].text = str(x)
            try:
                _scores.append(float(str(row6[3]).rstrip("%")))
            except ValueError:
                pass
        if _scores:
            d.add_paragraph(
                f"Overall validation score: {sum(_scores)/len(_scores):.0f}% "
                "(mean of the four normalized criteria).")

    # ---- basis of the acceptance targets ----
    d.add_heading("Basis of the acceptance targets", 2)
    bt = d.add_table(rows=1, cols=3); bt.style = "Table Grid"
    for i, x in enumerate(["Criterion", "Target", "Basis"]):
        c = bt.rows[0].cells[i]; c.text = ""
        c.paragraphs[0].add_run(x).bold = True
    _basis = [
        ("Coverage (POD)", ">= 0.70",
         "Literature level for good agreement between simulated and observed "
         "fire footprints: FARSITE reached a Sorensen coefficient of 0.70 with "
         "improved fuel inputs (Price et al., 2022); calibrated cell-based "
         "models reach 0.7-0.9 and uncalibrated semi-empirical models 0.5-0.7 "
         "(Pais et al., 2021). The POD/Sorensen index set follows Filippi et "
         "al. (2014)."),
        ("Front error, mean (m)", f"<= 0.25 R = {0.25*R:.0f} m",
         "Acceptance threshold adopted in this study: the mean front lies "
         "within a quarter of the fire equivalent radius R, so the criterion "
         "is scale aware. Front position error in metres itself is a standard "
         "FARSITE-style output (Finney, 1998)."),
        ("Front error, p90 (m)", f"<= 0.50 R = {0.50*R:.0f} m",
         "Acceptance threshold adopted in this study: the worst-matching "
         "tenth of the front lies within half of R."),
        ("Arrival MAE (h)", f"<= 0.25 T = {0.25*T:.1f} h",
         "Acceptance threshold adopted in this study: the arrival-time error "
         "is within a quarter of the observed fire duration T (scale aware)."),
    ]
    for cr, tg, bs in _basis:
        cells = bt.add_row().cells
        cells[0].text = cr; cells[1].text = tg; cells[2].text = bs

    d.add_heading("References", 3)
    for _r in [
        "Filippi, J.-B., Mallet, V., & Nader, B. (2014). Representation and "
        "evaluation of wildfire propagation simulations. International Journal "
        "of Wildland Fire, 23(1), 46-57.",
        "Finney, M. A. (1998). FARSITE: Fire Area Simulator, model development "
        "and evaluation. USDA Forest Service, RMRS-RP-4.",
        "Pais, C., Carrasco, J., Martell, D. L., Weintraub, A., & Woodruff, "
        "D. L. (2021). Cell2Fire: a cell-based forest fire growth model to "
        "support strategic landscape management planning. Frontiers in Forests "
        "and Global Change, 4, 692706.",
        "Price, S., et al. (2022). Modeling of fire spread in sagebrush "
        "steppe using FARSITE: an approach to improving input data and "
        "simulation accuracy. Fire Ecology, 18, 22.",
    ]:
        _p = d.add_paragraph(_r)
        _p.paragraph_format.left_indent = Pt(18)
        _p.paragraph_format.first_line_indent = Pt(-18)

    # ---- figures: the agreement map of EVERY case run so far ----
    # build the image set: the accumulated per-case maps, and always the
    # current one (so a single run still exports its map).
    _imgs = dict(all_imgs or {})
    _imgs.setdefault(str(res.get("case", "")), img)
    d.add_heading("Agreement maps", 2)
    d.add_paragraph(
        "Green: correctly predicted burn, Red: simulated only, Blue: "
        "observed only, Yellow: ignition.").italic = True
    _fno = 0
    for _ck, _im in _imgs.items():
        if _im is None:
            continue
        _fno += 1
        _cslug = "".join(c for c in str(_ck) if c.isalnum() or c in "-_")[:40]
        _fp = os.path.join(out_dir, f"agreement_map_{_cslug or _fno}.png")
        try:
            _PILImg.fromarray(_np2.asarray(_im)).save(_fp)
        except Exception:
            continue
        d.add_heading(str(_ck), 3)
        d.add_picture(_fp, width=Inches(5.6))
        cap = d.add_paragraph()
        cap.add_run(f"Figure 5.{_fno} Agreement map for {_ck}.").italic = True

    docx_path = os.path.join(out_dir, "validation_report.docx")
    d.save(docx_path)
    with open(docx_path, "rb") as fh:
        docx_bytes = fh.read()
    png_bytes = open(png_path, "rb").read() if png_path else b""
    return docx_bytes, png_bytes, out_dir


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
