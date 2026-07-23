"""System Description page of the DisasterAware dashboard.

A self contained, hand computable description of the whole framework in three
parts: the simulator, the decision support
system, and the cost function that scores decisions. Every function is written
out explicitly; no closed form is left opaque.
"""

from __future__ import annotations

import streamlit as st

from disaster_phyengine import FUEL_MODELS, SimConfig


# ------------------------------------------------------- architecture figure
_ARCH_SVG = """<svg viewBox="0 0 880 780" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>
    </marker>
  </defs>
  <style>
    .box{stroke-width:1.4;} .lbl{font-size:12px;fill:#111;} .sm{font-size:11px;fill:#111;}
    .eq{font-size:12.5px;font-style:italic;fill:#111;font-family:'Cambria Math','STIX Two Math',Georgia,serif;}
    .fn{font-size:11.5px;font-style:italic;fill:#111;font-family:'Cambria Math','STIX Two Math',Georgia,serif;}
    .hdr{font-size:12.5px;font-weight:bold;fill:#111;} .edge{stroke:#333;stroke-width:1.3;fill:none;marker-end:url(#arr);}
  </style>

  <!-- ===================== Web UI Dashboard ===================== -->
  <rect x="120" y="10" width="280" height="186" fill="#ffffff" stroke="#6c8ebf" class="box"/>
  <rect x="120" y="10" width="280" height="26" fill="#dae8fc" stroke="#6c8ebf" class="box"/>
  <text x="260" y="27" text-anchor="middle" class="hdr">Web UI Dashboard</text>
  <rect x="140" y="46"  width="240" height="30" rx="14" fill="#f5f5f5" stroke="#999"/>
  <text x="260" y="65"  text-anchor="middle" class="lbl">Map View</text>
  <rect x="140" y="82"  width="240" height="30" rx="14" fill="#f5f5f5" stroke="#999"/>
  <text x="260" y="101" text-anchor="middle" class="lbl">Map &amp; Data Editor</text>
  <rect x="140" y="118" width="240" height="30" rx="14" fill="#f5f5f5" stroke="#999"/>
  <text x="260" y="137" text-anchor="middle" class="lbl">Scenario Manager</text>
  <rect x="140" y="154" width="240" height="30" rx="14" fill="#f5f5f5" stroke="#999"/>
  <text x="260" y="173" text-anchor="middle" class="lbl">Simulation Settings</text>

  <!-- ===================== clock + time equations ===================== -->
  <circle cx="638" cy="32" r="15" fill="none" stroke="#1e88e5" stroke-width="2.5"/>
  <path d="M 638 24 L 638 33 L 645 36" stroke="#1e88e5" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <text x="600" y="68" class="eq">t<tspan dy="3.2" font-size="9.5">sim</tspan><tspan dy="-3.2">​</tspan> = kΔt<tspan dy="3.2" font-size="9.5">sim</tspan><tspan dy="-3.2">​</tspan> ,  k ∈ ℕ</text>
  <text x="600" y="90" class="eq">t<tspan dy="3.2" font-size="9.5">k+1</tspan><tspan dy="-3.2">​</tspan> = t<tspan dy="3.2" font-size="9.5">k</tspan><tspan dy="-3.2">​</tspan> + Δt<tspan dy="3.2" font-size="9.5">sim</tspan><tspan dy="-3.2">​</tspan></text>
  <text x="600" y="112" class="eq">S<tspan dy="3.2" font-size="9.5">k</tspan><tspan dy="-3.2">​</tspan>(x,y) = {{ s<tspan dy="3.2" font-size="9.5">c,k</tspan><tspan dy="-3.2">​</tspan> }}<tspan dy="3.2" font-size="9.5">c∈&#119970;</tspan><tspan dy="-3.2">​</tspan></text>
  <text x="600" y="134" class="eq">c ≡ (x,y)</text>

  <!-- ===================== External Sources ===================== -->
  <rect x="560" y="215" width="240" height="26" fill="#ffb570" stroke="#d79b00" class="box"/>
  <text x="680" y="232" text-anchor="middle" class="hdr">External Sources</text>
  <rect x="560" y="241" width="240" height="30" fill="#ffe6cc" stroke="#d79b00" class="box"/>
  <text x="680" y="254" text-anchor="middle" class="lbl" font-weight="bold">Fire Spread Model</text>
  <text x="680" y="267" text-anchor="middle" class="lbl" font-weight="bold">External Sources</text>
  <rect x="560" y="271" width="240" height="30" fill="#fff2cc" stroke="#d6b656" class="box"/>
  <text x="572" y="290" class="eq">Meteorology  U<tspan dy="3.2" font-size="9.5">Meteo,k</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>
  <rect x="560" y="301" width="240" height="30" fill="#fff2cc" stroke="#d6b656" class="box"/>
  <text x="572" y="320" class="eq">Topography  U<tspan dy="3.2" font-size="9.5">Geo</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>
  <rect x="560" y="331" width="240" height="30" fill="#fff2cc" stroke="#d6b656" class="box"/>
  <text x="572" y="350" class="eq">Fuel  U<tspan dy="3.2" font-size="9.5">Fuel,k</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>
  <rect x="560" y="361" width="240" height="26" fill="#ffe6cc" stroke="#d79b00" class="box"/>
  <text x="680" y="378" text-anchor="middle" class="lbl" font-weight="bold">Decisional External Sources</text>
  <rect x="560" y="387" width="240" height="30" fill="#fff2cc" stroke="#d6b656" class="box"/>
  <text x="572" y="406" class="eq">Values  U<tspan dy="3.2" font-size="9.5">Val,k</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>
  <rect x="560" y="417" width="240" height="30" fill="#fff2cc" stroke="#d6b656" class="box"/>
  <text x="572" y="436" class="eq">Resource  U<tspan dy="3.2" font-size="9.5">Res,k</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>

  <!-- ===================== Simulation Core ===================== -->
  <rect x="30" y="230" width="440" height="235" fill="none" stroke="#b85450" stroke-width="1.6"/>
  <rect x="30" y="230" width="22" height="235" fill="#f08705" opacity="0.75"/>
  <text x="45" y="348" text-anchor="middle" class="hdr" transform="rotate(-90 45 348)">Simulation Core</text>
  <rect x="100" y="252" width="290" height="52" rx="8" fill="#f8cecc" stroke="#b85450" class="box"/>
  <text x="245" y="273" text-anchor="middle" class="lbl" font-weight="bold">Hybrid Fire Spread Model</text>
  <text x="245" y="291" text-anchor="middle" class="sm">(Deterministic + Stochastic)</text>
  <rect x="100" y="390" width="290" height="52" rx="8" fill="#f8cecc" stroke="#b85450" class="box"/>
  <text x="245" y="410" text-anchor="middle" class="lbl" font-weight="bold">Grid State Manager</text>
  <text x="245" y="429" class="eq" text-anchor="middle">S<tspan dy="3.2" font-size="9.5">k+1</tspan><tspan dy="-3.2">​</tspan>(x,y) = Φ( S<tspan dy="3.2" font-size="9.5">k</tspan><tspan dy="-3.2">​</tspan>(x,y), ℱ<tspan dy="3.2" font-size="9.5">in,k</tspan><tspan dy="-3.2">​</tspan> )</text>
  <path class="edge" d="M 170 390 L 170 304"/>
  <text x="122" y="352" class="eq">S<tspan dy="3.2" font-size="9.5">k</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>
  <path class="edge" d="M 320 304 L 320 390"/>
  <text x="328" y="352" class="eq">S<tspan dy="3.2" font-size="9.5">k+1</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>

  <!-- ===================== DSS Core ===================== -->
  <rect x="30" y="520" width="440" height="230" fill="none" stroke="#82b366" stroke-width="1.6"/>
  <rect x="30" y="520" width="22" height="230" fill="#9ac7bf" opacity="0.9"/>
  <text x="45" y="635" text-anchor="middle" class="hdr" transform="rotate(-90 45 635)">DSS Core</text>
  <rect x="75"  y="545" width="140" height="90" rx="10" fill="#d5e8d4" stroke="#82b366" class="box"/>
  <text x="145" y="583" text-anchor="middle" class="lbl" font-weight="bold">Actions</text>
  <text x="145" y="601" text-anchor="middle" class="sm">(Decisions)</text>
  <rect x="285" y="545" width="160" height="60" rx="10" fill="#d5e8d4" stroke="#82b366" class="box"/>
  <text x="365" y="570" text-anchor="middle" class="lbl" font-weight="bold">Observations</text>
  <text x="365" y="588" text-anchor="middle" class="sm">(Measurements)</text>
  <rect x="75"  y="665" width="370" height="60" rx="10" fill="#d5e8d4" stroke="#82b366" class="box"/>
  <text x="260" y="690" text-anchor="middle" class="lbl" font-weight="bold">Local DSS</text>
  <text x="260" y="708" text-anchor="middle" class="lbl" font-weight="bold">+ Global DSS</text>

  <!-- ===================== edges ===================== -->
  <path class="edge" d="M 400 100 L 640 100 L 640 215"/>
  <text x="420" y="93" class="eq">Θ<tspan dy="3.2" font-size="9.5">UI</tspan><tspan dy="-3.2">​</tspan> → ( ℱ<tspan dy="3.2" font-size="9.5">in,k</tspan><tspan dy="-3.2">​</tspan> , ℱ<tspan dy="3.2" font-size="9.5">DSS,k</tspan><tspan dy="-3.2">​</tspan> )</text>
  <path class="edge" d="M 560 278 L 390 278"/>
  <text x="462" y="271" class="eq">ℱ<tspan dy="3.2" font-size="9.5">in,k</tspan><tspan dy="-3.2">​</tspan></text>
  <text x="505" y="266" class="fn">*(1)</text>
  <path class="edge" d="M 640 447 L 640 695 L 445 695"/>
  <text x="648" y="480" class="eq">ℱ<tspan dy="3.2" font-size="9.5">DSS,k</tspan><tspan dy="-3.2">​</tspan></text>
  <text x="705" y="475" class="fn">*(2)</text>
  <path class="edge" d="M 100 416 L 12 416 L 12 60 L 120 60"/>
  <text x="18" y="50" class="eq">S<tspan dy="3.2" font-size="9.5">k</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>
  <path class="edge" d="M 355 442 L 355 545"/>
  <text x="363" y="500" class="eq">S<tspan dy="3.2" font-size="9.5">k</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>
  <path class="edge" d="M 365 605 L 365 665"/>
  <text x="373" y="640" class="eq">&#119978;<tspan dy="3.2" font-size="9.5">k</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>
  <text x="437" y="635" class="fn">*(3)</text>
  <path class="edge" d="M 145 665 L 145 635"/>
  <path class="edge" d="M 145 545 L 145 442"/>
  <text x="152" y="500" class="eq">U<tspan dy="3.2" font-size="9.5">DSS,k</tspan><tspan dy="-3.2">​</tspan>(x,y)</text>
  <text x="238" y="495" class="fn">*(4)</text>

  <!-- ===================== footnotes ===================== -->
  <text x="480" y="560" class="fn">*(1)  ℱ<tspan dy="3.2" font-size="8.5">in,k</tspan><tspan dy="-3.2">​</tspan> = {{ U<tspan dy="3.2" font-size="8.5">Meteo,k</tspan><tspan dy="-3.2">​</tspan> , U<tspan dy="3.2" font-size="8.5">Geo</tspan><tspan dy="-3.2">​</tspan> , U<tspan dy="3.2" font-size="8.5">Fuel,k</tspan><tspan dy="-3.2">​</tspan> , U<tspan dy="3.2" font-size="8.5">Ign,k</tspan><tspan dy="-3.2">​</tspan> , U<tspan dy="3.2" font-size="8.5">DSS,k</tspan><tspan dy="-3.2">​</tspan> }}</text>
  <text x="480" y="586" class="fn">*(2)  ℱ<tspan dy="3.2" font-size="8.5">DSS,k</tspan><tspan dy="-3.2">​</tspan> = {{ U<tspan dy="3.2" font-size="8.5">Val,k</tspan><tspan dy="-3.2">​</tspan> , U<tspan dy="3.2" font-size="8.5">Res,k</tspan><tspan dy="-3.2">​</tspan> }}</text>
  <text x="480" y="612" class="fn">*(3)  &#119978;<tspan dy="3.2" font-size="8.5">k</tspan><tspan dy="-3.2">​</tspan>(x,y) = h( S<tspan dy="3.2" font-size="8.5">k</tspan><tspan dy="-3.2">​</tspan>(x,y), ε<tspan dy="3.2" font-size="8.5">k</tspan><tspan dy="-3.2">​</tspan> )</text>
  <text x="480" y="638" class="fn">*(4)  U<tspan dy="3.2" font-size="8.5">DSS,k</tspan><tspan dy="-3.2">​</tspan>(x,y) = π<tspan dy="3.2" font-size="8.5">DSS</tspan><tspan dy="-3.2">​</tspan>( &#119978;<tspan dy="3.2" font-size="8.5">k</tspan><tspan dy="-3.2">​</tspan>(x,y), ℱ<tspan dy="3.2" font-size="8.5">DSS,k</tspan><tspan dy="-3.2">​</tspan> )</text>
</svg>"""


# ------------------------------------------------------------------ helpers
def _eq(latex: str, defs=None, note: str = None):
    """Render an equation followed by the definition of every symbol in it."""
    st.latex(latex)
    if defs:
        st.markdown("\n".join(f"- {d}" for d in defs))
    if note:
        st.caption(note)


def _table(md: str):
    st.markdown(md)


def _cfg() -> SimConfig:
    if "sim" in st.session_state:
        return st.session_state.sim.cfg
    return SimConfig()


# ---------------------------------------------------------------- sections
def _sec_0(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "Three cores exchange data in a closed loop. The **Web UI "
        "Dashboard** edits scenarios; the **Simulation Core** advances the "
        "wildfire state; the **DSS Core** observes the state and decides "
        "the intervention. No block ever writes another block's memory: "
        "all coupling happens through the labelled interfaces.")
    st.image(_ARCH_SVG, use_container_width=True)
    st.markdown("The four interfaces, written out:")
    _eq(r"\Theta_{UI}\;\rightarrow\;\big(\mathcal{F}_{in,k},\,"
        r"\mathcal{F}_{DSS,k}\big)",
        [r"$\Theta_{UI}$ — the user interaction operator: every dashboard "
         r"action (painting fuel, dropping an ignition, moving a resource) "
         r"is an admissible, bounded modification of the *input sets*, "
         r"never of the state $S_k$ itself (state immutability)"])
    _eq(r"\mathcal{F}_{in,k}=\{\,U_{Meteo,k},\;U_{Geo},\;U_{Fuel,k},\;"
        r"U_{Ign,k},\;U_{DSS,k}\,\}",
        [r"$\mathcal{F}_{in,k}$ — the physical input set consumed by the "
         r"transition operator $\Phi$ (Part II,)"])
    _eq(r"\mathcal{F}_{DSS,k}=\{\,U_{Val,k},\;U_{Res,k}\,\}",
        [r"$\mathcal{F}_{DSS,k}$ — the decisional context: values at risk "
         r"and the available resource pool. It informs the DSS but never "
         r"enters the fire physics"])
    _eq(r"\mathcal{O}_k(x,y)=h\big(S_k(x,y),\,\epsilon_k\big)",
        [r"$\mathcal{O}_k$ — the observation the DSS reasons on (Part III, "
         r"); $h$ projects the state, $\epsilon_k$ is bounded "
         r"observation noise"])
    _eq(r"U_{DSS,k}(x,y)=\pi_{DSS}\big(\mathcal{O}_k(x,y),\,"
        r"\mathcal{F}_{DSS,k}\big)",
        [r"$\pi_{DSS}$ — the decision policy (Part III,): it maps "
         r"observations and decisional context to the resource allocation "
         r"$U_{DSS,k}$, which re-enters the simulator as an input"])
    st.caption(
        "Time advances as $t_{sim}=k\\,\\Delta t_{sim}$; the global state "
        "is the collection $S_k(x,y)=\\{s_{c,k}\\}_{c\\in G}$ with "
        "$c\\equiv(x,y)$.")


def _sec_1(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "**Grid space.** The world is a rectangular grid of square cells. "
        "A cell is addressed by its integer indices $(x,y)$:")
    _eq(r"G=\{(x,y)\in\mathbb{Z}^2 \mid 0\le x<n_x,\;0\le y<n_y\}",
        [r"$G$ — the set of all grid cells (the simulation domain)",
         r"$n_x,\,n_y$ — number of cells along the horizontal and vertical axes",
         r"$\Delta x$ — edge length of one square cell, in meters; the area "
         r"of a cell is $\Delta x^2$"])
    st.markdown(
        "**Discrete time.** The simulation does not evolve continuously; it "
        "advances in steps indexed by an integer $k$. Physical time at step "
        "$k$ is")
    _eq(r"t_k = k\,\Delta t,\qquad k=0,1,2,\dots",
        [r"$k$ — the time step index (a counter, dimensionless)",
         r"$\Delta t$ — length of one simulation step, in model time units "
         r"(e.g. minutes). All *per step* parameters below are calibrated "
         r"relative to this $\Delta t$."])
    st.markdown(
        "Every spatio-temporal quantity in this document carries the cell "
        "index $(x,y)$ and the step index $k$; e.g. $B_k(x,y)$ is the value "
        "of $B$ at cell $(x,y)$ at step $k$. Static quantities omit $k$.")
    _table(
        "| Quantity | Symbol | Current value |\n"
        "|---|---|---|\n"
        f"| Grid width (cells) | $n_x$ | {cfg.nx} |\n"
        f"| Grid height (cells) | $n_y$ | {cfg.ny} |\n"
        f"| Cell size | $\\Delta x$ | {cfg.cell_size_m:g} m |\n"
        f"| Time step | $\\Delta t$ | {cfg.dt:g} model unit "
            f"(one step \u2248 {getattr(cfg, 'step_minutes', 30):g} min real "
            "time, set in the Simulation page) |")
    st.caption("The step length actively rescales the per-step dynamics; "
               "see, note 8 for the exact scaling law.")


def _sec_2(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "The transition operator consumes five external input groups, "
        "collected in the input set")
    _eq(r"\mathcal{F}_{in,k}=\{\,U_{Meteo,k},\;U_{Geo},\;U_{Fuel,k},\;"
        r"U_{Ign,k},\;U_{DSS,k}\,\}",
        [r"$U_{Meteo,k}$ — meteorology fields (spatio-temporal)",
         r"$U_{Geo}$ — terrain fields (static)",
         r"$U_{Fuel,k}$ — fuel description (type static, load and moisture dynamic)",
         r"$U_{Ign,k}$ — external ignition injection (spatio-temporal)",
         r"$U_{DSS,k}$ — suppression resource fields set by the decision layer"])
    st.markdown(
        "The **decisional context** $\\mathcal{F}_{DSS,k}=\\{U_{Val,k},"
        "U_{Res,k}\\}$ (values at risk and resource pool) does not enter "
        "the fire physics; it feeds the DSS (Part III). Each group is "
        "detailed below.")

    # ---- 2.1 meteorology
    st.markdown("#### 2.1 Meteorology — $U_{Meteo,k}$")
    _eq(r"U_{Meteo,k}(x,y)=\begin{bmatrix}"
        r"W_{temp,k}(x,y)\\ W_{rh,k}(x,y)\\ W_{ws,k}(x,y)\\"
        r"W_{wd,k}(x,y)\\ W_{gust,k}(x,y)\\ W_{prec,k}(x,y)\end{bmatrix}")
    _table(
        "| Symbol | Field | Type | Unit | Role in the model |\n"
        "|---|---|---|---|---|\n"
        "| $W_{temp,k}$ | Air temperature | spatio-temporal | °C | Drives fuel drying; enters the optional equilibrium moisture model |\n"
        "| $W_{rh,k}$ | Relative humidity | spatio-temporal | % (0–100) | Drives fuel drying; enters the optional equilibrium moisture model |\n"
        "| $W_{ws,k}$ | Wind speed | spatio-temporal | m/s | Amplifies rate of spread and fire intensity |\n"
        "| $W_{wd,k}$ | Wind direction | spatio-temporal | rad | Direction the wind blows **toward** (math convention, $0$ = +x, counter-clockwise positive); sets the axis of anisotropic spread |\n"
        "| $W_{gust,k}$ | Wind gust speed | spatio-temporal | m/s | Exogenous stochastic forcing channel; not used by the deterministic core equations |\n"
        "| $W_{prec,k}$ | Precipitation | spatio-temporal | mm/h | Stops ember spotting above 1 mm/h and drives fuel moisture toward extinction (, note 13) |")
    st.caption(
        "Only wind speed and wind direction enter the transition equations "
        "directly. Temperature, humidity, gust and precipitation act "
        "indirectly through the fuel moisture field.")

    # ---- 2.2 terrain
    st.markdown("#### 2.2 Terrain — $U_{Geo}$")
    _eq(r"U_{Geo}(x,y)=\begin{bmatrix}"
        r"G_{elev}(x,y)\\ G_{slope}(x,y)\\ G_{aspect}(x,y)\\ G_{access}(x,y)"
        r"\end{bmatrix}")
    _table(
        "| Symbol | Field | Type | Unit | Role in the model |\n"
        "|---|---|---|---|---|\n"
        "| $G_{elev}$ | Elevation | static | m | Source for slope/aspect derivation; context for visualization and GIS import |\n"
        "| $G_{slope}$ | Terrain slope | static | rad | Accelerates uphill spread and raises intensity |\n"
        "| $G_{aspect}$ | Slope orientation (aspect) | static | rad | Interacts with wind direction in the spread rate |\n"
        "| $G_{access}$ | Accessibility index | static | –, $[0,1]$ | Gates suppression reachability; $1$ = fully accessible, $0$ = unreachable |")

    # ---- 2.3 fuel
    st.markdown("#### 2.3 Fuel — $U_{Fuel,k}$")
    _eq(r"U_{Fuel,k}(x,y)=\begin{bmatrix}"
        r"F_{type}(x,y)\\ F_{load,0}(x,y)\\ F_{moist,k}(x,y)\end{bmatrix}")
    _table(
        "| Symbol | Field | Type | Unit | Role in the model |\n"
        "|---|---|---|---|---|\n"
        "| $F_{type}$ | Fuel class | static | – (categorical id) | Selects the per class parameters $r_{base}, m_{ext}, a_w, a_s, a_{asp}, b_{base}$ (Tables in–6) |\n"
        "| $F_{load,0}$ | Initial fuel load | initial condition | – , normalized $[0,1]$ | Initializes the dynamic state $F_{load,k}$; $1$ corresponds to a reference dry biomass of about 2 kg/m² |\n"
        "| $F_{moist,k}$ | Surface fuel moisture | spatio-temporal | –, mass fraction $[0,1]$ | Damps spread rate and combustion fraction |")
    st.caption(
        "The fuel load itself is part of the *state*; only its "
        "initial value is an external input. Moisture is a static exogenous "
        "field by default; an optional mode recomputes it each step from "
        "temperature and humidity (Simard equilibrium moisture).")

    # ---- 2.4 ignition
    st.markdown("#### 2.4 Ignition — $U_{Ign,k}$")
    _eq(r"U_{Ign,k}(x,y)=\big[\,I_{Ign,k}(x,y)\,\big],\qquad "
        r"I_{Ign,k}(x,y)\in\{0,1\}",
        [r"$I_{Ign,k}$ — binary ignition injection: $1$ forces cell $(x,y)$ "
         r"to start burning at step $k$ (lightning, arson, scenario event, "
         r"or an interactive click), $0$ means no external ignition"])
    st.caption(
        "Ignition injection is an external *trigger*, not a state: it starts "
        "a fire but does not describe its strength. If the cell has no fuel, "
        "the injection has no lasting effect.")

    # ---- 2.5 values
    st.markdown("#### 2.5 Values at risk — $U_{Val}$ (decisional context)")
    _table(
        "| Symbol | Field | Type | Unit | Role |\n"
        "|---|---|---|---|---|\n"
        "| $V_{bld}$ | Building footprint | static | – , $[0,1]$ | Presence/density of structures; identifies wildland–urban zones |\n"
        "| $V_{crit}$ | Critical facility index | static | – , $[0,1]$ | Hospitals, power plants, depots, water infrastructure |\n"
        "| $V_{pop}$ | Population density | static | person/km² | Exposed population per cell |\n"
        "| $V_{evac}$ | Distance to evacuation route | static | m | Proximity to evacuation infrastructure (smaller = better served) |\n"
        "| $V_{prio}$ | Protection priority score | derived | – , $[0,1]$ | Single scalar ranking of each cell, computed below |")
    st.markdown(
        "The four indicators are aggregated into one priority score. The two "
        "unbounded fields are first normalized over the whole grid:")
    _eq(r"V_{pop}^{norm}(x,y)=\frac{V_{pop}(x,y)-\min V_{pop}}"
        r"{\max V_{pop}-\min V_{pop}},\qquad "
        r"V_{evac}^{norm}(x,y)=1-\frac{V_{evac}(x,y)-\min V_{evac}}"
        r"{\max V_{evac}-\min V_{evac}}",
        [r"$V_{pop}^{norm}$ — min–max normalized population density, in $[0,1]$",
         r"$V_{evac}^{norm}$ — **inverted** normalized evacuation distance: a "
         r"cell close to an evacuation route gets a value near $1$ (higher "
         r"priority), a remote cell near $0$"])
    _eq(r"V_{prio}(x,y)=w_{bld}V_{bld}(x,y)+w_{crit}V_{crit}(x,y)"
        r"+w_{pop}V_{pop}^{norm}(x,y)+w_{evac}V_{evac}^{norm}(x,y)",
        [r"$w_{bld},w_{crit},w_{pop},w_{evac}$ — aggregation weights with "
         r"$w_{bld}+w_{crit}+w_{pop}+w_{evac}=1$",
         rf"current values: $w_{{bld}}={vw.w_bld:g}$, $w_{{crit}}={vw.w_crit:g}$, "
         rf"$w_{{pop}}={vw.w_pop:g}$, $w_{{evac}}={vw.w_evac:g}$"])

    # ---- 2.6 resources
    st.markdown("#### 2.6 Suppression resources — $U_{DSS,k}$ (the decision channel)")
    _eq(r"U_{DSS,k}(x,y)=\begin{bmatrix}"
        r"R_{cap,k}(x,y)\\ R_{avail,k}(x,y)\\ R_{eff,k}(x,y)\\ R_{time,k}(x,y)"
        r"\end{bmatrix}")
    _table(
        "| Symbol | Field | Type | Unit | Role in the model |\n"
        "|---|---|---|---|---|\n"
        "| $R_{cap,k}$ | Suppression capacity | spatio-temporal | m³ water-equivalent per step (normalized by $R_{cap,max}$) | How much suppression capability is assigned to the cell |\n"
        "| $R_{avail,k}$ | Availability | spatio-temporal | – , $\\{0,1\\}$ | Whether the assigned resource is actually deployable ($0$ ⇒ no effect at all) |\n"
        "| $R_{eff,k}$ | Suppression efficiency | spatio-temporal | – , $[0,1]$ | How effectively the resource reduces fuel (agent type, crew skill) |\n"
        "| $R_{time,k}$ | Travel time | spatio-temporal | min (model time) | Estimated arrival time to the cell; late arrival decays the effect exponentially |")
    st.markdown(
        "**These four fields are the only lever of the Decision Support "
        "System.** The DSS decides *where* to allocate capacity, *whether* "
        "resources are available, *how effective* they are and *how long* "
        "they need to arrive. Everything else in the model is physics; the "
        "DSS influences the fire exclusively through "
        "$U_{DSS,k} \\rightarrow F_{red,k}$ and never overwrites "
        "the fire state directly.")
    st.caption(
        "Distinguish $U_{Res,k}$ from $U_{DSS,k}$: $U_{Res,k}$ is the "
        "*external resource pool* (what exists — part of "
        "$\\mathcal{F}_{DSS,k}$), while $U_{DSS,k}$ is the *allocation* the "
        "DSS derives from it (what is deployed where). With no DSS active, "
        "a static $U_{Res,k}$ can be applied directly as $U_{DSS,k}$ for "
        "what-if studies.")


def _sec_3(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "Each cell carries a local state vector of four variables. This is "
        "the *complete* memory of the simulation: everything else is either "
        "an external input or is recomputed from these four fields.")
    _eq(r"s_k(x,y)=\begin{bmatrix}B_k(x,y)\\ F_{load,k}(x,y)\\ I_k(x,y)\\ "
        r"\tau_k(x,y)\end{bmatrix}")
    _table(
        "| Symbol | State | Range / Unit | Meaning |\n"
        "|---|---|---|---|\n"
        "| $B_k$ | Burning status | $\\{0,1\\}$ | $1$ ⇔ the cell is actively burning at step $k$; $0$ otherwise. Never a fraction: a cell either burns or it does not |\n"
        "| $F_{load,k}$ | Fuel load | $[0,1]$, normalized | Remaining combustible mass. $1$ = full initial reference load (≈ 2 kg/m² dry biomass), $0$ = fully depleted. Monotonically non-increasing |\n"
        "| $I_k$ | Fire intensity proxy | $[0,1]$, dimensionless | Bounded indicator of local combustion strength. It does **not** drive spread; it modulates suppression effectiveness and feeds the decision layer |\n"
        "| $\\tau_k$ | Time since ignition | $\\ge 0$, model time units | Elapsed burning duration of the current combustion phase; pure bookkeeping for decision features and explainability |")
    st.markdown("The global state is the collection of all local states:")
    _eq(r"S_k=\{\,s_k(x,y)\;\mid\;(x,y)\in G\,\}")
    st.markdown(
        "One simulation step applies the deterministic transition operator "
        "$\\Phi$ to every cell simultaneously (synchronous update):")
    _eq(r"S_{k+1}=\Phi\big(S_k,\;\mathcal{F}_{in,k}\big)",
        [r"$\Phi$ — the transition operator: the four update rules of "
         r"Sections 4, 6, 7 and 8 applied to all cells at once",
         r"$\mathcal{F}_{in,k}$ — the external input set of"],
        note="Φ itself is deterministic. Randomness can only enter through "
             "the exogenous inputs (e.g. gust fluctuations), never through "
             "the update rules.")


def _sec_4(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "A cell burns at the next step if **any** of three mechanisms fires: "
        "it keeps burning (persistence), it is ignited by its neighbours "
        "(propagation), or it is ignited externally (injection). The maximum "
        "realizes the logical OR of the three binary signals:")
    _eq(r"B_{k+1}(x,y)=\max\Big\{\,B_{k+1}^{pers}(x,y),\;B_{k+1}^{prop}(x,y),"
        r"\;I_{Ign,k}(x,y)\cdot H(x,y)\,\Big\}",
        [r"$B_{k+1}^{pers}$ — persistence term, defined below",
         r"$B_{k+1}^{prop}$ — neighbour propagation term, defined below",
         r"$I_{Ign,k}$ — external ignition injection ",
         r"$H(x,y)=\mathbb{1}\big[F_{load,k}(x,y)>\epsilon_{fuel}\big]$ — the "
         r"fuel availability gate: no mechanism can ignite a cell that holds "
         r"no combustible fuel"])
    st.markdown("**(a) Persistence.** A burning cell keeps burning while fuel remains:")
    _eq(r"B_{k+1}^{pers}(x,y)=B_k(x,y)\cdot"
        r"\mathbb{1}\big[F_{load,k}(x,y)>\epsilon_{fuel}\big]",
        [r"$\mathbb{1}[\cdot]$ — indicator function: $1$ if the condition "
         r"holds, $0$ otherwise",
         rf"$\epsilon_{{fuel}}$ — extinction fuel threshold, currently "
         rf"${sp.eps_fuel:g}$ (normalized fuel units). It prevents a cell "
         r"from burning forever on a vanishing residue: once "
         r"$F_{load}\le\epsilon_{fuel}$ the cell extinguishes at the next step"])
    st.markdown(
        "**(b) Propagation.** Neighbouring burning cells push *influence* "
        "into the cell; ignition occurs when the accumulated influence "
        "crosses a threshold:")
    _eq(r"B_{k+1}^{prop}(x,y)=\mathbb{1}\big[A_k(x,y)>\Theta_{ign}\big]"
        r"\cdot\mathbb{1}\big[F_{load,k}(x,y)>\epsilon_{fuel}\big]",
        [r"$A_k(x,y)$ — the **ignition influence buildup**: neighbour "
         r"influence integrated over time, defined below",
         rf"$\Theta_{{ign}}$ — ignition threshold, currently "
         rf"${sp.theta_ign:g}$. Unit: accumulated influence, i.e. "
         r"$\tfrac{1}{8}\times$ cell-widths of front travel. With "
         r"$\Theta_{ign}=1/8$ the $\tfrac{1}{8}$ neighbourhood "
         r"normalization cancels and the buildup crosses the threshold "
         r"exactly when the front has travelled one cell width — so a "
         r"front driven by one aligned burning neighbour advances at "
         r"exactly $R_{spread}$ cells per step"])
    st.markdown(
        "The buildup integrates the instantaneous influence $\\Psi_k$ and "
        "leaks slowly, so heating dissipates if the fire source disappears; "
        "cells that are burning carry no buildup:")
    _eq(r"A_{k+1}(x,y)=\big(1-B_{k+1}(x,y)\big)\Big[(1-\lambda)\,A_k(x,y)"
        r"+\Psi_k(x,y)\Big]",
        [rf"$\lambda$ — buildup leak per reference step, currently "
         rf"${sp.buildup_leak:g}$ (dimensionless fraction). It models the "
         r"dissipation of pre-heating once the neighbouring fire is gone",
         r"$(1-B_{k+1})$ — reset: once a cell burns, its buildup is spent",
         r"physical reading: the time a cell needs to ignite is "
         r"$\approx\Theta_{ign}/\Psi_k$ steps, i.e. the front crossing "
         r"time $\Delta x/R_{spread}$ — slow fuels ignite late rather "
         r"than never"])
    st.markdown(
        "The influence is a wind-aligned weighted average over the eight "
        "surrounding cells (the 8-connected neighbourhood):")
    _eq(r"\Psi_k(x,y)=\frac{1}{8}\sum_{(i,j)\in N^8(x,y)}"
        r"B_k(i,j)\cdot R_{spread,k}(i,j)\cdot g_{dir,k}^{(i,j)\to(x,y)}",
        [r"$N^8(x,y)$ — the 8 cells surrounding $(x,y)$ (N, NE, E, SE, S, "
         r"SW, W, NW)",
         r"$B_k(i,j)$ — only burning neighbours contribute (the factor is "
         r"$0$ for non-burning ones)",
         r"$R_{spread,k}(i,j)$ — rate of spread of the **burning neighbour** "
         r"(cells per step,): a fast-burning neighbour pushes more "
         r"influence",
         r"$g_{dir,k}^{(i,j)\to(x,y)}$ — directional wind weight of the "
         r"specific neighbour-to-cell geometry, defined next",
         r"$\tfrac{1}{8}$ — normalization over the neighbourhood size, so "
         r"that $\Psi_k$ stays on the same scale as $R_{spread}$"])
    st.markdown(
        "**Directional weight.** Spread is strongest along the wind and "
        "suppressed against it:")
    _eq(r"g_{dir,k}^{(i,j)\to(x,y)}=\max\Big\{0,\;\cos\big(W_{wd,k}(i,j)"
        r"-\theta_{(i,j)\to(x,y)}\big)\Big\}",
        [r"$W_{wd,k}$ — wind direction at the burning neighbour (rad)",
         r"$\theta_{(i,j)\to(x,y)}$ — geometric direction of the vector "
         r"pointing from the neighbour $(i,j)$ to the target $(x,y)$ (rad, "
         r"same convention as $W_{wd}$: $0$ = +x/east, $\pi/2$ = north)",
         r"$\max\{0,\cdot\}$ — clipping: a neighbour strictly downwind of "
         r"the target contributes nothing (no upwind spread through this "
         r"term; upwind creep can still occur via the isotropic blend at "
         r"low wind, see)"])
    st.markdown(
        "The eight geometric angles are fixed by the grid (direction *from* "
        "the neighbour *to* the target):")
    _table(
        "| Neighbour is to the … | W | SW | S | SE | E | NE | N | NW |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        "| $\\theta_{(i,j)\\to(x,y)}$ | $0$ | $\\pi/4$ | $\\pi/2$ | $3\\pi/4$ | $\\pi$ | $-3\\pi/4$ | $-\\pi/2$ | $-\\pi/4$ |")
    st.caption(
        "Example: wind blowing due east ($W_{wd}=0$) and a burning neighbour "
        "directly to the west gives $g_{dir}=\\cos(0-0)=1$ (full weight); a "
        "diagonal neighbour gives $\\cos(\\pi/4)\\approx0.71$; a neighbour "
        "to the east gives $\\cos(\\pi)=-1 \\to 0$ (clipped).")
    st.markdown("**Activation defaults:**")
    _table(
        "| Parameter | Symbol | Unit | Default | Current |\n"
        "|---|---|---|---|---|\n"
        f"| Ignition threshold | $\\Theta_{{ign}}$ | accumulated influence (\u215b cell-widths) | 0.125 = 1/8 | {sp.theta_ign:g} |\n"
        f"| Buildup leak | $\\lambda$ | fraction per reference step | 0.05 | {sp.buildup_leak:g} |\n"
        f"| Extinction fuel threshold | $\\epsilon_{{fuel}}$ | normalized fuel | $10^{{-4}}$ | {sp.eps_fuel:g} |")
    st.caption("Both are adjustable in the **Parameters** page "
               "(Fire spread and propagation).")


def _sec_5(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "The rate of spread converts local fuel, moisture, wind and terrain "
        "into a forward propagation potential. It is a product of five "
        "factors, each tied to one physical effect. The output unit is "
        "**cells per step**: it is a grid-consistent abstraction, not a "
        "metric speed.")
    _eq(r"R_{spread,k}(x,y)=r_{base}\big(F_{type}(x,y)\big)\cdot "
        r"g_{moist,k}(x,y)\cdot g_{wind,k}(x,y)\cdot g_{slope}(x,y)\cdot "
        r"g_{aspect,k}(x,y)",
        [r"$r_{base}(F_{type})$ — base spread rate of the fuel class under "
         r"nominal (dry, calm, flat) conditions, in cells/step (table below)",
         r"$g_{moist},g_{wind},g_{slope},g_{aspect}$ — dimensionless "
         r"modifiers, each written out below"])
    st.markdown("**(a) Moisture damping** — wet fuel spreads slower, and not at all "
                "beyond the extinction moisture:")
    _eq(r"g_{moist,k}(x,y)=\max\Big\{0,\;1-\frac{F_{moist,k}(x,y)}"
        r"{m_{ext}(F_{type}(x,y))}\Big\}",
        [r"$F_{moist,k}$ — surface fuel moisture (mass fraction,)",
         r"$m_{ext}(F_{type})$ — extinction moisture threshold of the fuel "
         r"class (mass fraction, table below). "
         r"$F_{moist}\ge m_{ext}\Rightarrow g_{moist}=0$: no spread"])
    st.markdown("**(b) Wind amplification** — spread accelerates with wind but "
                "saturates instead of growing without bound:")
    _eq(r"g_{wind,k}(x,y)=1+a_w\big(F_{type}(x,y)\big)\cdot"
        r"\tanh\!\Big(\frac{W_{ws,k}(x,y)}{w_0}\Big)",
        [r"$W_{ws,k}$ — wind speed (m/s)",
         r"$a_w(F_{type})$ — wind sensitivity of the fuel class "
         r"(dimensionless, table below)",
         rf"$w_0$ — wind saturation scale, currently ${sp.w0:g}$ m/s: at "
         r"$W_{ws}=w_0$ the $\tanh$ term is already $\approx0.76$ of its "
         r"maximum; $\tanh(\cdot)\!\to\!1$ keeps the factor bounded at "
         r"$1+a_w$ in extreme wind"])
    st.markdown("**(c) Slope acceleration** — flames preheat upslope fuel, so "
                "spread accelerates uphill:")
    _eq(r"g_{slope}(x,y)=1+a_s\big(F_{type}(x,y)\big)\cdot"
        r"\tan\!\big(G_{slope}(x,y)\big)",
        [r"$G_{slope}$ — terrain slope (rad,); in the code the "
         r"slope is clipped to $\pm1.4$ rad so $\tan(\cdot)$ stays finite, "
         rf"and the whole factor is bounded to $[0, {sp.slope_gain_max:g}]$ "
         r"(real slope response saturates; unbounded $\tan$ explodes on "
         r"DEM cliffs)",
         r"$a_s(F_{type})$ — slope sensitivity of the fuel class "
         r"(dimensionless, table below)"])
    st.markdown("**(d) Aspect–wind alignment** — terrain orientation interacts "
                "with wind direction:")
    _eq(r"g_{aspect,k}(x,y)=1+a_{asp}\big(F_{type}(x,y)\big)\cdot"
        r"\cos\!\big(G_{aspect}(x,y)-W_{wd,k}(x,y)\big)",
        [r"$G_{aspect}$ — slope orientation (rad,)",
         r"$W_{wd,k}$ — wind direction (rad)",
         r"$a_{asp}(F_{type})$ — aspect sensitivity of the fuel class "
         r"(dimensionless, table below). Aligned wind and aspect "
         r"($\cos=1$) boost spread; opposed ($\cos=-1$) damp it; the factor "
         r"is floored at $0$"])
    st.markdown("#### Table A.1 + B.1 — Fuel class parameters")
    st.caption(
        "$r_{base}$ in cells/step — at the 30 m / 30 min reference grid "
        "1 cell/step = 1 m/min, so the numbers are metric no-wind spread "
        "rates; $m_{ext}$ as mass fraction; $a_w$, $a_s$, $a_{asp}$ "
        "dimensionless sensitivities; $b_{base}$ as fraction of fuel "
        "consumed per step; $e$ as economic value per cell unit "
        ". Values are calibrated to the fire-behaviour literature: "
        "the Anderson (1982) fuel models (grass FM1/FM3, shrub FM5/FM6, "
        "long-needle litter FM9, compact hardwood litter FM8), Scott & "
        "Burgan (2005) and the Rothermel (1972) spread model. With the wind "
        "factor of(b) the benchmark head-fire rates are reproduced: "
        "grass tens of m/min, shrub 5\u201310, pine litter 2\u20134, "
        "hardwood litter below 1.")
    rows = ["| id | Fuel class | $r_{base}$ | $m_{ext}$ | $a_w$ | $a_s$ | "
            "$a_{asp}$ | $b_{base}$ | forest | $e$ |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for fid, m in FUEL_MODELS.items():
        rows.append(f"| {fid} | {m.name} | {m.r_base:g} | {m.m_ext:g} | "
                    f"{m.a_w:g} | {m.a_s:g} | {m.a_asp:g} | {m.b_base:g} | "
                    f"{'yes' if m.is_forest else 'no'} | {m.economic_value:g} |")
    _table("\n".join(rows))
    st.caption(
        "Columns $r_{base}, m_{ext}, a_w, a_s, a_{asp}$ form Table "
        "A.1; $b_{base}$ is Table B.1; $e$ feeds the cost model "
        ". Classes with $r_{base}=0$ (non_fuel, water) can never "
        "propagate fire; they act as natural firebreaks. The whole table "
        "is editable in the **Parameters** page (Fuel classes).")


def _sec_6(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "Fuel decreases through two simultaneous mechanisms: combustion in "
        "burning cells and suppression applied by the decision layer. Fuel "
        "can never become negative and never increases:")
    _eq(r"F_{load,k+1}(x,y)=\max\Big\{0,\;F_{load,k}(x,y)"
        r"-\underbrace{B_k(x,y)\,F_{burn,k}(x,y)\,F_{load,k}(x,y)}_{\text{combustion}}"
        r"-\underbrace{F_{red,k}(x,y)}_{\text{suppression}}\Big\}",
        [r"$B_k$ — burning status: combustion consumes fuel **only** in "
         r"actively burning cells",
         r"$F_{burn,k}\in[0,1]$ — fraction of the available fuel consumed "
         r"in one step (defined below)",
         r"$F_{red,k}\in[0,1]$ — suppression-driven fuel reduction in one "
         r"step (defined below); it acts on burning **and** non-burning "
         r"cells, which is how preventive fuel treatment and firebreak "
         r"cutting are modelled",
         r"$\max\{0,\cdot\}$ — non-negativity guard"])
    st.markdown("**(a) Combustion fraction.** Computed algebraically each step; "
                "it is not a state:")
    _eq(r"F_{burn,k}(x,y)=\min\Big\{1,\;b_{base}\big(F_{type}(x,y)\big)\cdot"
        r"\big(1-F_{moist,k}(x,y)\big)\Big\}",
        [r"$b_{base}(F_{type})$ — baseline combustion coefficient of the "
         r"fuel class, in fraction of fuel per step (Table in): "
         r"grass burns off fast (0.25/step), hardwood slowly (0.05/step)",
         r"$(1-F_{moist,k})$ — linear moisture damping: fully wet fuel "
         r"($F_{moist}=1$) does not burn, dry fuel burns at the baseline rate",
         r"$\min\{1,\cdot\}$ — saturation, keeping the fraction in $[0,1]$"])
    st.markdown(
        "**(b) Suppression mapping.** The decision inputs are converted into "
        "a per step fuel reduction by a product of interpretable factors. "
        "In fully explicit form:")
    _eq(r"F_{red,k}(x,y)=\min\Bigg\{\;\alpha_s\cdot"
        r"\frac{R_{cap,k}(x,y)}{R_{cap,max}}\cdot R_{avail,k}(x,y)\cdot "
        r"e^{-\beta_t R_{time,k}(x,y)}\,G_{access}(x,y)\cdot"
        r"\frac{R_{eff,k}(x,y)}{1+\gamma_I\,I_k(x,y)}\;,\;\;"
        r"F_{load,k}(x,y)\Bigg\}")
    st.markdown("Reading the factors one by one (each is in $[0,1]$ except the "
                "gain $\\alpha_s$ which sets the overall scale):")
    _eq(r"\eta_{cap,k}=\frac{R_{cap,k}(x,y)}{R_{cap,max}}",
        [r"capacity factor — fraction of the reference capacity actually "
         r"assigned to the cell",
         rf"$R_{{cap,max}}$ — reference maximum capacity, currently "
         rf"${su.rcap_max:g}$ (same unit as $R_{{cap}}$); the ratio is "
         r"clipped to $[0,1]$"])
    _eq(r"\eta_{avail,k}=R_{avail,k}(x,y)\in\{0,1\}",
        [r"availability gate — a strict necessary condition: if the "
         r"resource is not deployable the entire product is $0$, whatever "
         r"the other factors say"])
    _eq(r"\eta_{reach,k}(x,y)=e^{-\beta_t R_{time,k}(x,y)}\cdot G_{access}(x,y)",
        [r"reachability factor — late arrival decays the effect "
         r"exponentially and inaccessible terrain scales it down",
         rf"$\beta_t$ — travel-time decay rate, currently ${su.beta_t:g}$ "
         r"per minute (unit: 1/min): effectiveness halves after "
         r"$\ln 2/\beta_t\approx 23$ min, matching initial-attack "
         r"response curves. E.g. $\beta_t=0.03$, $R_{time}=30$ min: "
         r"$e^{-0.9}\approx0.41$",
         r"$G_{access}\in[0,1]$ — static accessibility "])
    _eq(r"\eta_{eff,k}(x,y)=\frac{R_{eff,k}(x,y)}{1+\gamma_I\,I_k(x,y)}",
        [r"effectiveness factor — nominal efficiency, degraded by the "
         r"**current** fire intensity: intense fires resist suppression",
         rf"$\gamma_I$ — intensity resistance coefficient, currently "
         rf"${su.gamma_I:g}$ (dimensionless). With $\gamma_I=2$ and "
         r"$I_k=0.5$ the efficiency is halved",
         r"$I_k$ — the intensity **before** the update (decisions act on "
         r"what is observable at step $k$)"])
    _eq(r"F_{red,k}=\min\big\{\alpha_s\,\eta_{cap,k}\,\eta_{avail,k}\,"
        r"\eta_{reach,k}\,\eta_{eff,k},\;F_{load,k}\big\}",
        [rf"$\alpha_s$ — global suppression gain, currently ${su.alpha_s:g}$ "
         r"(fraction of fuel per step): the maximum achievable reduction in "
         r"one step when every factor equals $1$",
         r"$\min\{\cdot,F_{load,k}\}$ — suppression can never remove more "
         r"fuel than the cell holds"])
    st.markdown("**Table B.2 — Suppression mapping calibration parameters "
                ":**")
    _table(
        "| Parameter | Symbol | Typical range | Current | Effect when increased |\n"
        "|---|---|---|---|---|\n"
        f"| Global suppression gain | $\\alpha_s$ | 0.01 – 0.30 | {su.alpha_s:g} | Stronger overall mitigation per step |\n"
        f"| Travel-time decay | $\\beta_t$ | 0.01 – 0.10 /min | {su.beta_t:g} | Effectiveness drops faster with delay |\n"
        f"| Intensity resistance | $\\gamma_I$ | 0.5 – 5.0 | {su.gamma_I:g} | Intense fires become harder to suppress |")
    st.caption(
        "Operational reading: no resources ⇒ zero effect; unreachable cell "
        "⇒ effect decays smoothly to zero; poor access ⇒ proportional "
        "reduction; intense fire ⇒ diminishing returns. Suppression removes "
        "fuel — it never switches $B_k$ off directly. Extinction then "
        "follows physically through the persistence condition of "
        "All three coefficients are adjustable in the **Parameters** page "
        "(Suppression effectiveness).")


def _sec_7(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "The intensity proxy is a bounded indicator of combustion strength "
        "in $[0,1]$. It does not model temperature or heat flux; it exists "
        "to (i) make intense fires resist suppression, (ii) rank "
        "risk, and (iii) feed the decision layer. It is built from three "
        "normalized drivers:")
    _eq(r"\tilde F_k(x,y)=\min\Big\{1,\frac{F_{load,k}(x,y)}{F_{max}}\Big\},"
        r"\quad \tilde W_k(x,y)=\min\Big\{1,\frac{W_{ws,k}(x,y)}{W_{ref}}\Big\},"
        r"\quad \tilde S(x,y)=\min\Big\{1,\frac{\tan G_{slope}(x,y)}"
        r"{\tan S_{max}}\Big\}",
        [rf"$F_{{max}}$ — reference maximum fuel load, currently "
         rf"${ip.fload_max:g}$ (normalized units)",
         rf"$W_{{ref}}$ — reference maximum wind speed, currently "
         rf"${ip.wws_max:g}$ m/s",
         rf"$S_{{max}}$ — reference maximum slope, currently "
         rf"${ip.slope_max_rad:g}$ rad ($45°$)",
         r"all three drivers are clipped into $[0,1]$, which makes the "
         r"proxy scale- and resolution-independent"])
    _eq(r"I_{k+1}(x,y)=B_{k+1}(x,y)\cdot\tanh\!\Big(\beta\big(\tilde F_k(x,y)"
        r"+\gamma_W\,\tilde W_k(x,y)+\gamma_S\,\tilde S(x,y)\big)\Big)",
        [r"$B_{k+1}$ — the **updated** burning status: non-burning cells "
         r"are forced to zero intensity",
         r"$\tilde F_k$ — evaluated with the current fuel $F_{load,k}$ "
         r"(before depletion); $I_k$ itself is *not* an argument — "
         r"intensity is recomputed, not integrated",
         rf"$\beta$ — global intensity gain, currently ${ip.beta:g}$ "
         r"(dimensionless, typical $1\le\beta\le3$)",
         rf"$\gamma_W$ — wind weight, currently ${ip.gamma_w:g}$ "
         r"(typical $0\le\gamma_W\le0.7$)",
         rf"$\gamma_S$ — slope weight, currently ${ip.gamma_s:g}$ "
         r"(typical $0\le\gamma_S\le0.5$)",
         r"$\tanh(\cdot)$ — keeps $I_{k+1}$ bounded in $[0,1)$; fuel is the "
         r"dominant contributor, wind and slope are amplifying modifiers"])
    st.markdown("**Table C.1 — Fire intensity evolution parameters "
                ":**")
    _table(
        "| Parameter | Symbol | Range | Default | Current |\n"
        "|---|---|---|---|---|\n"
        f"| Global intensity gain | $\\beta$ | $1 - 3$ | 2.0 | {ip.beta:g} |\n"
        f"| Wind weighting | $\\gamma_W$ | $0 - 0.7$ | 0.5 | {ip.gamma_w:g} |\n"
        f"| Slope weighting | $\\gamma_S$ | $0 - 0.5$ | 0.3 | {ip.gamma_s:g} |\n"
        f"| Fuel normalization | $F_{{max}}$ | – | 1.0 | {ip.fload_max:g} |\n"
        f"| Wind normalization | $W_{{ref}}$ | – | 20 m/s | {ip.wws_max:g} m/s |\n"
        f"| Slope normalization | $S_{{max}}$ | – | 0.7854 rad (45°) | {ip.slope_max_rad:g} rad |")
    st.caption("Adjustable in the **Parameters** page (Fire intensity).")


def _sec_8(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "The ignition clock measures how long the current combustion phase "
        "has lasted. It never feeds back into the fire physics; it is a "
        "temporal memory for decision features and explainability:")
    _eq(r"\tau_{k+1}(x,y)=\begin{cases}"
        r"0, & B_{k+1}(x,y)=1\;\wedge\;B_k(x,y)=0 \quad\text{(new ignition)}\\[2pt]"
        r"\tau_k(x,y)+\Delta t, & B_{k+1}(x,y)=1\;\wedge\;B_k(x,y)=1 "
        r"\quad\text{(continued burning)}\\[2pt]"
        r"0, & B_{k+1}(x,y)=0 \quad\text{(not burning / extinguished)}"
        r"\end{cases}",
        [r"new ignition — the clock starts at $0$ the step the cell ignites",
         r"continued burning — the clock advances by one step length "
         r"$\Delta t$ per step",
         r"extinction — the clock resets: extinguished cells carry no "
         r"residual temporal memory"])


def _sec_9(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "The implementation follows the equations above with a small number "
        "of documented refinements:\n\n"
        "1. **Fuel gate on all ignition paths.** Propagation and external "
        "injection are both multiplied by "
        "$\\mathbb{1}[F_{load,k}>\\epsilon_{fuel}]$, so a fuel-depleted "
        "cell can never (re)ignite. This is what makes suppression-created "
        "firebreaks effective.\n"
        "2. **Diagonal distance weighting.** In $\\Psi_k$, contributions "
        "from the four diagonal neighbours are divided by $\\sqrt{2}$, "
        "compensating for their larger centre-to-centre distance "
        f"(currently {'on' if sp.diagonal_distance_weighting else 'off'}).\n"
        "3. **Low-wind isotropic blend with a flank/backing floor.** The "
        "directional weight is applied as "
        "$g_{dir}^{eff}=(1-a)+a\\cdot g_{dir}$ with "
        "$a=\\min\\{W_{ws}/w_{aniso},1\\}\\cdot(1-f_{back})$, "
        f"$w_{{aniso}}={sp.aniso_wind_full:g}$ m/s and "
        f"$f_{{back}}={getattr(sp, 'back_frac', 0.10):g}$. At zero wind the "
        "spread is isotropic (fuel and slope driven); at full wind the "
        "head fire uses the wind-aligned weight of while flanking "
        "and backing spread continue at the floor $f_{back}$ — backing "
        "fires run at roughly 5\u201315% of the head fire rate "
        "(Rothermel 1972; Cheney & Sullivan, Grassfires 2008).\n"
        "4. **Slope clipping.** $G_{slope}$ is clipped to "
        f"$\\pm{sp.slope_clip_rad:g}$ rad before $\\tan(\\cdot)$ to avoid "
        "numerical blow-up near vertical terrain.\n"
        "5. **Optional elliptical kernel** "
        f"(currently {'on' if sp.elliptical else 'off'}). When enabled, the "
        "directional weight of is replaced by the wind-elongated "
        "ellipse used by Cell2Fire/FARSITE: "
        "$g_{dir}^{ell}=\\dfrac{1-e}{1-e\\cos\\Delta}$ with "
        "$\\Delta = W_{wd}-\\theta_{(i,j)\\to(x,y)}$, eccentricity "
        "$e=\\sqrt{1-1/LB^2}$ and length-to-breadth ratio "
        f"$LB = LB_0 + LB_w\\,W_{{ws}}$ (currently $LB_0="
        f"{sp.lb_ratio_base:g}$, $LB_w={sp.lb_ratio_wind:g}$ per m/s). "
        "$g^{ell}_{dir}=1$ at the head, $(1-e)/(1+e)$ at the back.\n"
        "5b. **Optional ember spotting** "
        f"(currently {'on' if sp.spotting else 'off'}). Each burning cell "
        f"with $I_k > {sp.spot_intensity_min:g}$ throws an ember with "
        f"probability $p_{{spot}}={sp.spot_prob:g}$ per reference step "
        "(compounded consistently over substeps as $1-(1-p_{spot})^{s^*}$); "
        f"the ember lands $d_{{spot}}={sp.spot_distance:g}$ cells downwind "
        "along $W_{wd}$ and ignites the landing cell only if it still holds "
        "fuel ($F_{load}>\\epsilon_{fuel}$) **and** is drier than its "
        "extinction moisture ($F_{moist}<m_{ext}$). This is the only "
        "stochastic mechanism in the transition.\n"
        "6. **Optional moisture dynamics.** By default $F_{moist}$ is a "
        "static exogenous field; a toggle recomputes it each step from "
        "temperature and humidity with the Simard (1968) equilibrium "
        "moisture model.\n"
        "7. **Synchronous update.** All cells are updated from the same "
        "$S_k$; no cell sees a neighbour's $k{+}1$ value within a step.\n"
        "8. **Step-length scaling.** One step represents "
        f"$\\Delta t_{{real}}={getattr(cfg, 'step_minutes', 30):g}$ min "
        "(set in the Simulation page). Rates are calibrated at a 30 min "
        "reference; with $s=\\Delta t_{real}/30$ the engine runs "
        "$\\lceil s\\rceil$ internal substeps of scale "
        "$s^*=s/\\lceil s\\rceil$: the substep spread rate is "
        "$s^*\\cdot R_{spread}$ and the per-substep fractions compound "
        "as $F_{burn} \\leftarrow 1-(1-F_{burn})^{s^*}$ (same for "
        "$F_{red}$). A one-day step therefore advances the front by up to "
        "$\\sim s$ cells and consumes fuel accordingly, while at "
        "$\\Delta t_{real}=30$ min the reference equations hold exactly. "
        "The engine also rescales with the cell size, "
        "$R_{cells}=R\\cdot(30\\,\\mathrm{m}/\\Delta x)$, so the "
        "metric speed in m/min is independent of the grid resolution. "
        "For very short steps ($s\\ll 1$) the scaled influence can drop "
        "below $\\Theta_{ign}$: the fire then persists without "
        "spreading, which is consistent with a threshold activation "
        "model.\n"
        "9. **Adaptive substepping.** The substep count also covers the "
        "fastest local spread rate: "
        "$n_{sub}=\\max\\{\\lceil s\\rceil,\\;\\lceil s\\cdot"
        "\\max R_{spread}\\rceil\\}$ (capped at 200), so a grass fire "
        "running at tens of cells per step is resolved cell by cell instead "
        "of being capped at one cell per step.\n"
        "10. **Viewer diagnostics.** The Data layers page derives Byram's "
        "fireline intensity $I_B=H\\,w\\,R$ (kW/m, with heat of "
        f"combustion $H={ip.heat_content:g}$ kJ/kg and $w$ from the "
        f"reference biomass {ip.biomass_ref:g} kg/m\u00b2 at "
        "$F_{load}=1$), the flame length $L=0.0775\\,I_B^{0.46}$ m, and "
        f"a crown-fire flag for burning forest cells with $I_k>"
        f"{ip.crown_fire_threshold:g}$. These are reporting products only "
        "and never feed back into the transition.\n"
        "12. **Terrain-modified wind field.** Exposed ridges accelerate "
        "the wind and valleys shelter it: the wind speed entering the "
        "spread model is scaled per cell by "
        "$\\mathrm{clip}\\big(1+g_{tw}(2\\hat e-1),\\,0.4,\\,"
        "1.8\\big)$ where $\\hat e$ is the normalized elevation and "
        f"$g_{{tw}}={getattr(sp, 'terrain_wind_gain', 0.35):g}$ "
        "(0 disables), a linearized adjustment in the spirit of "
        "diagnostic terrain wind models. On steep ground the DIRECTION "
        "is channeled toward the local slope axis as well: "
        "$W_{wd}^{eff}=\\arg\\big((1-c)e^{iW_{wd}}+c\\,"
        "e^{iG_{aspect}}\\big)$ with "
        "$c=\\tfrac12\\,\\mathrm{clip}(G_{slope}/0.6,0,1)$, so a "
        "uniform user-set wind becomes a valley-following field.\n"
        "13. **Precipitation.** Handled by the engine for every weather "
        "source. Rain above $1$ mm/h disables ember spotting; while it "
        "rains, fuel moisture relaxes toward at least $0.35$ (above "
        "every $m_{ext}$) as "
        "$F_{moist}\\leftarrow F_{moist}+(\\max(F_{moist},0.35)-"
        "F_{moist})\\,\\mathrm{clip}(W_{prec}/2,0,1)\\,"
        "\\min(1,s)$ ($\\sim$30 min time constant), stopping new "
        "spread; and under sustained rain of $W_{prec}\\ge 3$ mm/h a "
        "burning cell is quenched once its moisture reaches $0.34$, so "
        "the front dies out gradually. Light rain only slows the fire.\n"
        "14. **Suppression knockdown.** Crews extinguish flames, not "
        "only fuel: a burning cell is quenched when the suppression "
        "pressure (the $\\eta$ product of the mapping, without the "
        "fuel-removal gain $\\alpha_s$) exceeds "
        f"$k_{{dn}}={getattr(su, 'knockdown_ratio', 0.15):g}$ times the "
        "cell's burn fierceness ($F_{burn}/0.10$ per reference step). "
        "Running heads in cured grass stay unquenchable, as in real "
        "operations; moderate surface fire within reach of committed "
        "capacity goes out, so ordered suppression can visibly contain "
        "and extinguish. Engaged cells are also WETTED: suppression is "
        "water, so fuel moisture relaxes toward $0.35$ at pressure "
        f"$\\times g_{{wet}}={getattr(su, 'wet_gain', 2.0):g}$ per "
        "reference step \u2014 a held line refuses ignition "
        "($g_{moist}\\to 0$) and even a grass fire becomes "
        "quenchable once its fuel is soaked.\n"
        "11. **Slope-directional spread (effective wind).** The directional "
        "weight uses an effective vector that combines the wind with a "
        "slope-equivalent wind blowing uphill (the FARSITE virtual-wind "
        "construction): $\\vec u_{eff}=W_{ws}\\hat e(W_{wd})+"
        "k_{slope}\\tan(G_{slope})\\,\\hat e_{up}$ with "
        f"$k_{{slope}}={getattr(sp, 'slope_wind_equiv', 10):g}$ m/s per "
        "unit $\\tan$. Fire therefore climbs mountainsides even against "
        "a light gradient wind. Only the direction and anisotropy strength "
        "come from $\\vec u_{eff}$; the scalar factors $g_{wind}$ and "
        "$g_{slope}$ stay as defined, so nothing is double counted.")


def _sec_10(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "Setting: a **hardwood litter** cell $C_{src}$ is burning; the wind "
        "blows due east ($W_{wd}=0$) at $W_{ws}=6$ m/s. The target cell "
        "$C_{tgt}$ lies directly east of $C_{src}$ (so $C_{src}$ is its "
        "western neighbour, $\\theta=0$). Both cells: $F_{moist}=0.08$, "
        "$G_{slope}=0.1$ rad, $G_{aspect}=0$, $F_{load}=1.0$. Default "
        "parameters (hardwood row of Table A.1: $r_{base}=0.3$, "
        "$m_{ext}=0.30$, $a_w=4$, $a_s=1.8$, $a_{asp}=0.15$, "
        "$b_{base}=0.08$; $w_0=10$, $\\Theta_{ign}=0.125$, "
        "$\\lambda=0.05$, $\\alpha_s=0.2$, $\\beta_t=0.03$, "
        "$\\gamma_I=2$, $\\beta=2$, $\\gamma_W=0.5$, "
        "$\\gamma_S=0.3$, $W_{ref}=20$, $S_{max}=\\pi/4$). Hardwood "
        "is chosen because its $R_{spread}<1$ keeps the whole arithmetic "
        "in single steps ($n_{sub}=1$); note $W_{ws}=6\\ge w_{aniso}=6$, "
        "so the wind-aligned weight applies (head direction, "
        "$g_{dir}=1$).")
    st.markdown("**Step 1 — rate of spread at the burning source:**")
    st.latex(r"g_{moist}=1-\tfrac{0.08}{0.30}=0.7333,\qquad "
             r"g_{wind}=1+4\tanh(\tfrac{6}{10})=1+4\times0.5370=3.1482")
    st.latex(r"g_{slope}=1+1.8\tan(0.1)=1+1.8\times0.1003=1.1806,\qquad "
             r"g_{aspect}=1+0.15\cos(0-0)=1.150")
    st.latex(r"R_{spread}=0.30\times0.7333\times3.1482\times1.1806"
             r"\times1.150=0.9403\ \text{cells/step}"
             r"\;(=0.94\ \text{m/min at the 30 m / 30 min reference})")
    st.markdown("**Step 2 — influence buildup at the target:** only "
                "$C_{src}$ burns; it sits west, so $\\theta=0$ and "
                "$g_{dir}=\\max\\{0,\\cos(0-0)\\}=1$ (non-diagonal, "
                "no $\\sqrt2$ correction):")
    st.latex(r"\Psi=\tfrac{1}{8}\times1\times0.9403\times1=0.11754")
    st.latex(r"A_1=0.11754<\Theta_{ign}=0.125\;\Rightarrow\;"
             r"\text{no ignition after one step}")
    st.latex(r"A_2=(1-0.05)\times0.11754+0.11754=0.22921"
             r"\;>\;0.125\;\Rightarrow\;B^{prop}=1")
    st.markdown("The target has fuel ($1.0>\\epsilon_{fuel}$), so it "
                "ignites **on the second step** — matching the physical "
                "front crossing time $\\Delta x/R_{spread}=1/0.94"
                "\\approx1.06$ steps. (A cell strictly upwind of the "
                "source receives only the flank/backing floor "
                "$f_{back}=0.1$ of the influence and ignites an order of "
                "magnitude later.)")
    st.markdown("**Step 3 — fuel update at the burning source,** "
                "with a suppression assignment $R_{cap}=0.5$, "
                "$R_{avail}=1$, $R_{eff}=0.8$, $R_{time}=5$ min, "
                "$G_{access}=1$, and current intensity $I_k=0.4$:")
    st.latex(r"F_{burn}=\min\{1,\,0.08(1-0.08)\}=0.0736")
    st.latex(r"\eta_{cap}=\tfrac{0.5}{1.0}=0.500,\quad \eta_{avail}=1,"
             r"\quad \eta_{reach}=e^{-0.03\times5}\times1=0.8607,"
             r"\quad \eta_{eff}=\tfrac{0.8}{1+2\times0.4}=0.4444")
    st.latex(r"F_{red}=0.2\times0.500\times1\times0.8607\times0.4444"
             r"=0.0383")
    st.latex(r"F_{load,k+1}=\max\{0,\;1.0-1\times0.0736\times1.0"
             r"-0.0383\}=0.8881")
    st.markdown("**Step 4 — intensity of the newly ignited target,** "
                "using its current fuel $F_{load,k}=1.0$:")
    st.latex(r"\tilde F=1.0,\quad \tilde W=\tfrac{6}{20}=0.300,\quad "
             r"\tilde S=\tfrac{\tan 0.1}{\tan(\pi/4)}=0.1003")
    st.latex(r"I_{k+1}=1\times\tanh\!\big(2\,(1.0+0.5\times0.300"
             r"+0.3\times0.1003)\big)=\tanh(2.3602)=0.9823")
    st.markdown("**Step 5 — ignition clocks:** the target is newly "
                "ignited ($B_k=0\\to B_{k+1}=1$) so "
                "$\\tau_{k+1}(C_{tgt})=0$; the source keeps burning so "
                "$\\tau_{k+1}(C_{src})=\\tau_k+\\Delta t$.")
    st.caption("Every number above can be reproduced with a pocket "
               "calculator; this is the complete arithmetic of one cell "
               "update, including the buildup mechanism.")


def _sec_11(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "The DSS never reads the authoritative state $S_k$ directly. It "
        "reasons on an observable projection produced by the observation "
        "function, which injects epistemic uncertainty at the observation "
        "stage only:")
    _eq(r"\mathcal{O}_k(x,y)=h\big(S_k(x,y),\,\epsilon_k\big)"
        r"=\begin{bmatrix}B_k(x,y)\\ F_{load,k}(x,y)+\epsilon_k^{F}(x,y)\\ "
        r"I_k(x,y)+\epsilon_k^{I}(x,y)\\ \tau_k(x,y)\end{bmatrix}",
        [r"$h$ — the observation function: a read-only copy of the four "
         r"state fields, restricted to the agent's region if the DSS is "
         r"regional",
         r"$\epsilon_k^{F},\epsilon_k^{I}$ — bounded, zero-mean observation "
         r"disturbances added to the *continuous* fields only (fuel load "
         r"and intensity), then clipped back into $[0,1]$; magnitude "
         r"$\epsilon\ge0$ is a scenario parameter ($\epsilon=0$ ⇒ perfect "
         r"observation)",
         r"the binary field $B_k$ and the clock $\tau_k$ are reported "
         r"exactly in the present implementation"])
    st.caption(
        "This boundary preserves state immutability: observation never "
        "mutates $S_k$, and every decision downstream is based on "
        "$\\mathcal{O}_k$, not on privileged internal data.")
    st.markdown(
        "#### The sensor network \u2014 structural incompleteness model")
    st.markdown(
        "The dominant observation uncertainty in wildfire response is not "
        "statistical noise but **structural incompleteness**: limited "
        "coverage, revisit periods, reporting latency and missing "
        "components. The DSS senses the state components "
        "$j \\in \\{B, F, I, \\tau\\}$ through a network of placed "
        "sensors; static prior maps (terrain, fuel type, values at risk, "
        "own resources), fuel moisture and the weather field come from "
        "maps and the meteorological service. The source families are the "
        "standard wildfire information sources:")
    _table(
        "| Type | Senses $j$ | $r_s$ | $T_s$ | $\\ell_s$ | "
        "$\\bar\\epsilon_j$ |\n"
        "|---|---|---|---|---|---|\n"
        "| Satellite imagery + hot-spot detection | $B, I$ | whole map | 1 min | 20 min | 0.05 |\n"
        "| UAV / aerial thermal recon | $B, I$ | 2500 m | 1 min | 2 min | 0.03 |\n"
        "| Fixed lookout camera (smoke/flame) | $B, I$ | 4000 m | 1 min | 1 min | 0.06 |\n"
        "| Environmental ground sensors | $F_{load}$ | 1500 m | 1 min | 0 | 0.02 |\n"
        "| First-responder field data | $B, \\tau$ | 1000 m | 1 min | 10 min | 0.10 |\n"
        "| Public reports / emergency calls | $B$ | 1200 m | 1 min | 15 min | 0.20 |")
    st.caption(
        "Every source reads once per simulation minute ($T_s=1$ min, "
        "the software's minimum step), so no source lags the "
        "simulation; the sources differ by footprint, latency and "
        "reliability. Cells OUTSIDE every footprint still age freely, "
        "so the freshness factor $\\gamma_{fre}$ keeps degrading "
        "uncovered ground.")
    st.caption(
        "The remaining input-space sources are not sensing assets: "
        "terrain (DEM), the fuel/vegetation map, the road/access "
        "network, values at risk and own resources / suppression "
        "sources are KNOWN PRIORS entering $U_{Geo}, U_{Fuel}, U_{Val}, "
        "U_{Res}$ directly; meteorological stations and forecasts drive "
        "$U_{Meteo}$. The pre-fire fuel map additionally acts as an aged "
        "prior source for the $F$ channel.")
    st.markdown(
        "**Network deployment (Suggest network).** The field layout is "
        "chosen by greedy maximum weighted coverage: assets are placed "
        "one at a time to maximize")
    _eq(r"\Delta\!\sum_{x,y} r(x,y)\,c(x,y),\qquad "
        r"r=0.45\,\hat R_{spread}+0.35\,\hat V_{prio}+0.20\,F_{load}",
        [r"$r(x,y)$ — observation-worth (risk) of the cell: spread "
         r"danger, protection priority and standing fuel",
         r"$c(x,y)$ — composed coverage, each asset lifting its "
         r"footprint by its sensing quality $w_s=0.9\,|J_s|/4$ as "
         r"$c \leftarrow 1-(1-c)(1-w_s)$",
         r"greedy placement on a submodular objective: the standard "
         r"$(1-1/e)$-approximation",
         r"constraints — lookout cameras prefer high ground (line of "
         r"sight), field posts sit on the road network, public-report "
         r"sources are pinned to settlements, one satellite is always "
         r"tasked, same-type assets keep one footprint apart"])
    st.markdown(
        "Reports are not instantaneous: each source samples on its own "
        "revisit clock (first passes staggered within a family), and the "
        "sampled picture is **delivered only after the source latency** "
        "$\\ell_s$ \u2014 what arrives describes the situation at "
        "sampling time, not now. "
        "Every component keeps its **last observed field** and its data "
        "age $\\Delta t_{rep}$. The per-component observation confidence "
        "aggregates four independent degradation factors by the "
        "conservative minimum, the cell-level confidence is the "
        "weakest component, and the bounded disturbance shrinks "
        "with confidence:")
    _eq(r"conf_{j,k}^{i}(x,y)=\min\Big\{\,\theta_{j,k}^{i}(x,y),\;"
        r"\rho_{k}^{i}(x,y),\;e^{-\lambda_{conf}\,\Delta t_{rep,k}^{i}"
        r"(x,y)},\;\gamma_{k}^{i}\,\Big\}",
        [r"$\theta_{j,k}^{i} \in [0,1]$ — observability weight of "
         r"component $j$: $1$ = a sensor observing $j$ covers "
         r"the cell, $0$ = complete coverage gap",
         r"$\rho_{k}^{i} \in [0,1]$ — normalized sensor coverage "
         r"density at the cell",
         r"$\Delta t_{rep}$ — time since the most recent observation; "
         r"$\lambda_{conf} = \ln 2 / 90$ min$^{-1}$ (freshness halves "
         r"every 90 min)",
         r"$\gamma_{k}^{i} \in [0,1]$ — source reliability of the best "
         r"covering sensor ($1-\bar\epsilon_s$)",
         r"figure notation: the four factors are "
         r"$\gamma_{obs}=\theta$, $\gamma_{cov}=\rho$, "
         r"$\gamma_{fre}=e^{-\lambda_{conf}\Delta t}$, "
         r"$\gamma_{rel}=\gamma$, so "
         r"$conf=\min\{\gamma_{obs},\gamma_{cov},\gamma_{fre},"
         r"\gamma_{rel}\}$"])
    _eq(r"conf_{k}^{i}(x,y)=\min_{j\in\{B,F,I,\tau\}} "
        r"conf_{j,k}^{i}(x,y),\qquad "
        r"\big|\epsilon_{j,k}^{i}(x,y)\big|\le "
        r"\big(1-conf_{j,k}^{i}(x,y)\big)\,\bar\epsilon_{j}^{i}",
        [r"cell confidence = weakest component (conservative principle, "
         r") — the model value that bounds the disturbance; the "
         r"region-level scalar DISPLAYED next to each agent is the "
         r"component-mean over its cells $\Omega_i$ (a component with "
         r"no source at all would otherwise pin the display to zero)",
         r"never-observed cells start at $\Delta t_{rep}=\infty$: the "
         r"agent assumes no fire until a source says otherwise — a blind "
         r"or stale region genuinely misleads its agent",
         r"the ten bounded features of the decision layer are computed "
         r"from the fused observation $\hat z$, and each Local DSS reads "
         r"it restricted to its own region $\Omega_i$"])


def _sec_12(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown("The decision policy maps observations and decisional "
                "context to the resource allocation:")
    _eq(r"U_{DSS,k}(x,y)=\pi_{DSS}\big(\mathcal{O}_k(x,y),\,"
        r"\mathcal{F}_{DSS,k}\big),\qquad "
        r"\mathcal{F}_{DSS,k}=\{U_{Val,k},\,U_{Res,k}\}",
        [r"$\pi_{DSS}$ — the decision policy (rule based, optimization "
         r"based, or learned); its output is the four-field allocation of "
         r"",
         r"$U_{Val,k}$ — values at risk: *what to protect*",
         r"$U_{Res,k}$ — the external resource pool: *what exists to "
         r"deploy* (capacities, positions, availabilities)"])
    st.markdown("The full causal chain of an intervention is:")
    _eq(r"U_{DSS,k}=\big[R_{cap,k},R_{avail,k},R_{eff,k},R_{time,k}\big]"
        r"\;\xrightarrow{\text{}}\;F_{red,k}"
        r"\;\xrightarrow{\text{}}\;F_{load,k+1}"
        r"\;\xrightarrow{\text{}}\;B_{k+m}"
        r"\;\xrightarrow{\text{}}\;I_{k+m}")
    st.markdown(
        "- The DSS sets the four resource fields; the suppression mapping "
        "turns them into a fuel reduction $F_{red,k}$.\n"
        "- Reduced fuel weakens persistence (a depleted cell extinguishes) "
        "and blocks propagation (a cell below $\\epsilon_{fuel}$ cannot be "
        "ignited) — this is exactly how firebreaks emerge in the model.\n"
        "- The DSS **never** writes $B$, $I$ or $\\tau$ directly. All "
        "interventions travel through the physics, so their effect is "
        "always physically consistent and explainable.\n"
        "- Conversely the DSS observes $\\mathcal{O}_k$ (and derived "
        "features such as $V_{prio}$, $\\tau_k$, $I_k$) to decide the next "
        "allocation.")


def _sec_13(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "A decision is an *intervention*: the DSS works with a small, fixed "
        "vocabulary of intervention types and sets an intensity for each "
        "type as a field over the grid. Six types cover the levers of "
        "wildfire response:")
    _table(
        "| Intervention type | Operational meaning | Driven by | Materializes in the simulator as |\n"
        "|---|---|---|---|\n"
        "| Suppression effort | Direct / indirect attack on the active front | suppression feasibility, fire threat | $R_{cap}, R_{eff}$ on front cells |\n"
        "| Resource deployment | Assignment of crews, engines, aircraft | suppression feasibility, fire threat | $R_{avail}, R_{time}$ (positioning) |\n"
        "| Containment line | Firebreaks and control lines | spread | preventive $F_{red}$ on unburned cells (fuel removal) |\n"
        "| Asset protection | Point protection of high-value structures | asset exposure | concentrated $R_{cap}$ around high $V_{prio}$ cells |\n"
        "| Evacuation | Move / shelter / stage exposed population | evacuation pressure | decision output; lowers realized population exposure in the cost |\n"
        "| Public warning | Alerts to the affected public | intervention urgency | decision output; affects the delay term of the cost |")
    st.caption(
        "Fixing this vocabulary keeps the action space finite and "
        "interpretable: every rule consequent and every cost term "
        "has a concrete referent. The first four types act physically "
        "through $U_{DSS,k}$; the last two act on exposure and timing and "
        "are scored by the cost function.")


def _sec_14(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "**Mathematical basis.** The decision cost follows the "
        "*cost-plus-loss* principle of wildfire economics: the value of a "
        "response is the effort it spends plus the losses it fails to "
        "prevent. The cost is therefore an additive, weighted sum of five "
        "terms. Because the terms are measured in different units (area, "
        "value, persons, capacity, time), **each is normalized to $[0,1]$ "
        "against a scenario reference scale** before the weights apply, so "
        "the terms are dimensionless and mutually summable. The **weights**, "
        "not the list, encode operational priority.")

    st.markdown("#### 14.1 Bookkeeping quantities (how the terms are measured)")
    st.markdown(
        "The terms are read from the cumulative fields the simulator "
        "maintains and the static value layers:")
    _eq(r"A_k(x,y)=\max_{0\le\kappa\le k} B_\kappa(x,y)\;\in\{0,1\}",
        [r"$A_k$ — the **burned mask**: $1$ if the cell has burned at any "
         r"step up to $k$ (it never resets, unlike $B_k$)"])
    _eq(r"E_k=\sum_{\kappa=0}^{k}\;a_{km^2}\!\!\sum_{(x,y)\in G}"
        r"A_\kappa(x,y)\,V_{pop}(x,y)",
        [r"$E_k$ — cumulative **person-steps** of exposure inside the "
         r"burned footprint (persons summed over steps): the base quantity "
         r"of the population term"])
    _eq(r"a_{ha}=\frac{\Delta x^2}{10^4}\ \text{[ha/cell]},\qquad "
        r"a_{km^2}=\frac{\Delta x^2}{10^6}\ \text{[km}^2\text{/cell]}",
        [rf"cell area conversions; with $\Delta x={cfg.cell_size_m:g}$ m: "
         rf"$a_{{ha}}={cfg.cell_area_ha:g}$ ha, "
         rf"$a_{{km^2}}={cfg.cell_area_ha/100.0:g}$ km²"])

    st.markdown("#### 14.2 The total decision cost")
    _eq(r"J_k=w_1 J_k^{burn}+w_2 J_k^{asset}+w_3 J_k^{pop}"
        r"+w_4 J_k^{resp}+w_5 J_k^{del}",
        [r"$J_k$ — total decision cost at step $k$; with normalized weights "
         r"it lies in $[0,1]$",
         r"$w_1,\dots,w_5\ge0$ — non-negative priority weights; the "
         r"dashboard uses equal weights by default and normalizes them so "
         r"the total is a weighted average of the five terms",
         r"each term $J^{(\cdot)}\in[0,1]$ is defined explicitly below"])

    st.markdown("**Term 1 — burned area $J^{burn}$** (land and ecological loss):")
    _eq(r"J_k^{burn}=\frac{\sum_{(x,y)\in G}A_k(x,y)}"
        r"{\sum_{(x,y)\in G}\mathbb{1}\big[F_{load,0}(x,y)>0\big]}",
        [r"burned cells divided by the **burnable cells** of the scenario "
         r"(cells that carry any initial fuel); pure area fraction, so a "
         r"burned hectare of grass and of forest count the same here"])
    st.markdown("**Term 2 — asset loss $J^{asset}$** "
                "(structures together with critical infrastructure):")
    _eq(r"J_k^{asset}=\frac{\sum_{(x,y)\in G}A_k(x,y)\,"
        r"\big(V_{bld}(x,y)+V_{crit}(x,y)\big)}"
        r"{\sum_{(x,y)\in G}\big(V_{bld}(x,y)+V_{crit}(x,y)\big)}",
        [r"asset value lost divided by the **total exposed asset value**; "
         r"$V_{bld}$ is building footprint and $V_{crit}$ the critical "
         r"facility index, both in $[0,1]$",
         r"structures and infrastructure are priced in this single term; "
         r"burned area (Term 1) prices land only, so no value is counted "
         r"twice"])
    st.markdown("**Term 3 — population exposure $J^{pop}$** (life safety):")
    _eq(r"J_k^{pop}=\frac{E_k}{\big(\sum_{(x,y)}a_{km^2}V_{pop}\big)\,H}",
        [r"at-risk person-steps over the at-risk population times the "
         r"horizon $H$; the risk fraction cancels, so the term is the "
         r"**exposed share of the available population-time**",
         rf"$\rho_{{risk}}$ — casualty share of the exposed population, "
         rf"currently {cp.population_at_risk_fraction:g} (used for the raw "
         r"casualty display)",
         rf"$H$ — scenario horizon, currently {cp.horizon_steps:g} steps",
         r"an effective evacuation lowers $E_k$; beyond its "
         rf"weight, a candidate whose exposure exceeds a hard ceiling "
         rf"({cp.population_ceiling:g}) is rejected at the acceptance gate"])
    st.markdown("**Term 4 — response cost $J^{resp}$** (committed resources):")
    _eq(r"J_k^{resp}=\frac{\sum_{(x,y)}R_{cap,k}(x,y)\,R_{avail,k}(x,y)}"
        r"{C_{avail}}",
        [r"committed capacity divided by the **total available capacity** "
         r"$C_{avail}$; deployed capacity is priced regardless of its "
         r"eventual effectiveness, so protective actions carry a nonzero "
         r"cost for the adaptation loop",
         rf"$C_{{avail}}$ — total available capacity, currently "
         rf"{cp.capacity_reference:g}"])
    st.markdown("**Term 5 — response delay $J^{del}$** (timeliness):")
    _eq(r"J_k^{del}=\frac{1}{t_{ref}}\cdot"
        r"\frac{\sum_{(x,y)}R_{cap,k}(x,y)\,R_{time,k}(x,y)}"
        r"{\sum_{(x,y)}R_{cap,k}(x,y)}\quad\text{(0 if no allocation)}",
        [r"capacity-weighted mean travel time of the allocated resources, "
         r"divided by a reference delay $t_{ref}$; late-arriving "
         r"allocations are penalized because timeliness changes every "
         r"other loss",
         rf"$t_{{ref}}$ — reference delay scale, currently "
         rf"{cp.delay_reference:g}"])
    st.caption(
        "Terms 1–3 are read from the physical run. Terms 4–5 are "
        "decision-layer quantities computed from the resource field "
        "currently applied; they are zero when no allocation is active. "
        "Every denominator is bounded below and every term is clipped to "
        "$[0,1]$.")

    st.markdown("#### 14.3 Satisficing acceptance and rollout evaluation")
    st.markdown(
        "The cost serves as an **acceptance criterion**, not an objective "
        "for global optimization. A candidate intervention is accepted as "
        "soon as its expected cost clears an adaptive threshold:")
    _eq(rf"J_k(U)\le J_k^{{acc}},\qquad "
        rf"J_k^{{acc}}={cp.acceptance_fraction:g}\cdot J_k^{{do-nothing}}",
        [rf"$J_k^{{acc}}$ — acceptance threshold: a fixed fraction "
         rf"({cp.acceptance_fraction:g} by default) of the uncontrolled, "
         r"do-nothing cost over the operational period; it tightens while "
         r"confidence and capacity are high and relaxes as they degrade",
         r"when no candidate clears the threshold, a graduated fail-safe "
         r"attenuates the best available intervention toward a conservative "
         r"baseline rather than forcing a binary act/abstain choice"])
    st.markdown(
        "Because the cost is not differentiable in the decision, candidates "
        "are compared by **forward rollout**: a candidate allocation $U$ is "
        "applied to a copy of the simulator for a short horizon and its "
        "cost is read at the end.")
    _eq(r"J^{(H)}(U)=J_{k+H}\ \text{under}\ \Phi\ \text{with}\ U_{DSS}=U,"
        r"\qquad \Delta J = J^{(H)}(U)-J^{(H)}(U_{base})",
        [r"$U_{base}$ — a conservative baseline allocation (e.g. current "
         r"doctrine or no change)",
         r"$\Delta J<0$ — the candidate is an improvement; every adaptation "
         r"is admitted **only** when it lowers the cost"])

    st.markdown("#### 14.4 Default cost parameters")
    _table(
        "| Parameter | Symbol | Current value | Unit |\n"
        "|---|---|---|---|\n"
        f"| Burned area weight | $w_1$ | {cp.w_burn:g} | – |\n"
        f"| Asset loss weight | $w_2$ | {cp.w_asset:g} | – |\n"
        f"| Population exposure weight | $w_3$ | {cp.w_pop:g} | – |\n"
        f"| Response cost weight | $w_4$ | {cp.w_resp:g} | – |\n"
        f"| Response delay weight | $w_5$ | {cp.w_delay:g} | – |\n"
        f"| Acceptance threshold fraction | – | {cp.acceptance_fraction:g} | – |\n"
        f"| Population at risk fraction | $\\rho_{{risk}}$ | {cp.population_at_risk_fraction:g} | – |\n"
        f"| Scenario horizon | $H$ | {cp.horizon_steps:g} | steps |\n"
        f"| Total available capacity | $C_{{avail}}$ | {cp.capacity_reference:g} | – |\n"
        f"| Reference delay | $t_{{ref}}$ | {cp.delay_reference:g} | – |")
    st.caption(
        "All values are adjustable in the **Parameters** page (Cost model). "
        "The weights default to equal priority; adjust them to encode the "
        "operational hierarchy for a given incident.")

    st.markdown("#### 14.5 Mini example — cost by hand")
    st.markdown(
        "Suppose a scenario with 5000 burnable cells and, by step $k$: "
        "1000 cells burned; the burned footprint holds asset value "
        "$\\sum A_k(V_{bld}+V_{crit})=8$ out of a scenario total of $40$; "
        "cumulative exposure $E_k=600$ person-steps against an available "
        "population-time of $\\big(\\sum a_{km^2}V_{pop}\\big)H=6000$; no "
        "resources deployed. With equal weights:")
    st.latex(r"J^{burn}=\tfrac{1000}{5000}=0.20,\qquad "
             r"J^{asset}=\tfrac{8}{40}=0.20,\qquad "
             r"J^{pop}=\tfrac{600}{6000}=0.10")
    st.latex(r"J^{resp}=0,\qquad J^{del}=0")
    st.latex(r"J=\tfrac{0.20+0.20+0.10+0+0}{5}=0.10")
    st.caption(
        "With equal weights the total is the average of the five terms; "
        f"here $J=0.10$ sits below the acceptance threshold "
        f"({cp.acceptance_fraction:g}), so the do-nothing baseline would "
        "already be acceptable. Raising $w_3$ makes the population term "
        "dominate and pushes the DSS toward earlier protective action.")


def _sec_15(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    st.markdown(
        "Cost measures *outcome*; a complementary quality score measures "
        "how closely a candidate intervention matches the situation "
        "assessment it should serve:")
    _eq(r"Q_k=\sum_j \omega_j\,q_j,\qquad "
        r"q_j=1-\overline{\big|\,u_j-\tilde c_j\,\big|}",
        [r"$q_j\in[0,1]$ — per-component match between intervention "
         r"component $u_j$ (e.g. suppression effort intensity) and the "
         r"decision concept $\tilde c_j$ it serves (e.g. fire threat), "
         r"both normalized to $[0,1]$; the bar denotes the spatial mean "
         r"over the region",
         r"$\omega_j\ge0$ — component weights with $\sum_j\omega_j=1$",
         r"$\eta$ — quality threshold: a decision with $Q_k\ge\eta$ is "
         r"applied as-is"])
    st.markdown(
        "A deficient decision is not simply rejected; it is attenuated "
        "toward a conservative baseline in proportion to its quality "
        "deficit (graduated fail-safe):")
    _eq(r"U_{DSS}=U_{base}+\min\!\big\{1,\;Q_k/\eta\big\}\cdot"
        r"\big(U-U_{base}\big)",
        [r"$U$ — the candidate allocation produced by the policy",
         r"$U_{base}$ — the conservative baseline (doctrine allocation "
         r"protecting the most exposed assets)",
         r"$\min\{1,Q_k/\eta\}$ — attenuation factor: at $Q_k\ge\eta$ the "
         r"candidate acts at full strength; at $Q_k=0$ only the baseline "
         r"acts. The most exposed assets therefore always receive "
         r"protection, while a low-quality decision is prevented from "
         r"acting at full strength"])
    st.caption(
        "Together, and close the loop: the cost drives the "
        "*search* for better decisions (accept only $\\Delta J<0$), the "
        "quality gate governs the *application* of the chosen one.")


def _sec_16(ctx):
    cfg, sp, su, ip, vw, cp = ctx
    _table(
        "| Symbol | Meaning | Unit / Range | Defined in |\n"
        "|---|---|---|---|\n"
        "| $G,\\ (x,y)$ | Grid domain, cell index | – | |\n"
        "| $k,\\ \\Delta t,\\ \\Delta x$ | Step index, step length, cell size | –, time, m | |\n"
        "| $\\Theta_{UI}$ | User interaction operator | – | |\n"
        "| $\\mathcal{F}_{in,k}$ | Physical input set | – |, 2 |\n"
        "| $\\mathcal{F}_{DSS,k}$ | Decisional context set | – |, 12 |\n"
        "| $W_{temp,k}, W_{rh,k}$ | Air temperature, relative humidity | °C, % | |\n"
        "| $W_{ws,k}, W_{wd,k}$ | Wind speed, wind direction | m/s, rad | |\n"
        "| $W_{gust,k}, W_{prec,k}$ | Wind gust, precipitation | m/s, mm/h | |\n"
        "| $G_{elev}, G_{slope}, G_{aspect}$ | Elevation, slope, aspect | m, rad, rad | |\n"
        "| $G_{access}$ | Accessibility index | $[0,1]$ | |\n"
        "| $F_{type}$ | Fuel class id | categorical | |\n"
        "| $F_{load,k}$ | Fuel load (state) | $[0,1]$ norm. |, 3, 6 |\n"
        "| $F_{moist,k}$ | Fuel moisture | mass fraction | |\n"
        "| $I_{Ign,k}$ | Ignition injection | $\\{0,1\\}$ | |\n"
        "| $V_{bld}, V_{crit}, V_{pop}, V_{evac}, V_{prio}$ | Values at risk, priority score | see | |\n"
        "| $w_{bld}, w_{crit}, w_{pop}, w_{evac}$ | Priority weights (sum = 1) | – | |\n"
        "| $R_{cap,k}, R_{avail,k}, R_{eff,k}, R_{time,k}$ | DSS resource fields | see | |\n"
        "| $U_{Res,k}$ | External resource pool | – |, 12 |\n"
        "| $s_k, S_k$ | Local / global state | – | |\n"
        "| $B_k$ | Burning status (state) | $\\{0,1\\}$ |, 4 |\n"
        "| $I_k$ | Fire intensity proxy (state) | $[0,1]$ |, 7 |\n"
        "| $\\tau_k$ | Time since ignition (state) | time |, 8 |\n"
        "| $\\Phi$ | Transition operator | – | |\n"
        "| $\\mathbb{1}[\\cdot]$ | Indicator function | $\\{0,1\\}$ | |\n"
        "| $\\epsilon_{fuel}$ | Extinction fuel threshold | norm. fuel | |\n"
        "| $\\Psi_k$ | Propagation influence | cells/step | |\n"
        "| $\\Theta_{ign}$ | Ignition threshold | cells/step | |\n"
        "| $N^8(x,y)$ | 8-connected neighbourhood | – | |\n"
        "| $g_{dir}, \\theta_{(i,j)\\to(x,y)}$ | Directional weight, geometry angle | $[0,1]$, rad | |\n"
        "| $A_k$ (buildup) | Ignition influence buildup | \u215b cell-widths | |\n"
        "| $\\lambda$ | Buildup leak | fraction/step | |\n"
        "| $f_{back}$ | Flank/backing floor of $g_{dir}$ | – | |\n"
        "| $k_{slope},\\ \\vec u_{eff}$ | Slope-equivalent wind, effective vector | m/s, – | |\n"
        "| $g_{slope}^{max}$ | Slope factor cap | – | |\n"
        "| $s,\\ n_{sub}$ | Step-length scale, substep count | –, – | |\n"
        "| $LB,\\ e$ | Ellipse length/breadth, eccentricity | –, – | |\n"
        "| $p_{spot}, d_{spot}$ | Spotting probability, distance | –, cells | |\n"
        "| $R_{spread,k}$ | Rate of spread | cells/step | |\n"
        "| $r_{base}$ | Base spread rate (per fuel class) | cells/step | |\n"
        "| $g_{moist}, g_{wind}, g_{slope}, g_{aspect}$ | Spread modifiers | – | |\n"
        "| $m_{ext}$ | Extinction moisture (per fuel class) | mass fraction | |\n"
        "| $a_w, a_s, a_{asp}$ | Wind / slope / aspect sensitivity | – | |\n"
        "| $w_0$ | Wind saturation scale | m/s | |\n"
        "| $e(F_{type})$ | Economic value of fuel class | currency/cell unit |, 14 |\n"
        "| $F_{burn,k}$ | Combustion fraction per step | $[0,1]$ | |\n"
        "| $b_{base}$ | Baseline combustion coefficient | fraction/step | |\n"
        "| $F_{red,k}$ | Suppression fuel reduction | $[0,1]$ | |\n"
        "| $\\alpha_s$ | Global suppression gain | fraction/step | |\n"
        "| $\\eta_{cap}, \\eta_{avail}, \\eta_{reach}, \\eta_{eff}$ | Suppression factors | $[0,1]$ | |\n"
        "| $R_{cap,max}$ | Reference max capacity | as $R_{cap}$ | |\n"
        "| $\\beta_t$ | Travel-time decay rate | 1/time | |\n"
        "| $\\gamma_I$ | Intensity resistance | – | |\n"
        "| $\\tilde F, \\tilde W, \\tilde S$ | Normalized fuel / wind / slope | $[0,1]$ | |\n"
        "| $F_{max}, W_{ref}, S_{max}$ | Normalization references | norm., m/s, rad | |\n"
        "| $\\beta, \\gamma_W, \\gamma_S$ | Intensity gain and weights | – | |\n"
        "| $\\mathcal{O}_k,\\ h,\\ \\epsilon_k$ | Observation, obs. function, obs. noise | – | |\n"
        "| $r_s, T_s, \\ell_s, \\bar\\epsilon_j$ | Sensor footprint, revisit, latency, disturbance bound | m, min, min, – | |\n"
        "| $\\theta_{j,k}^{i},\\ \\rho_k^i,\\ \\gamma_k^i$ | Observability weight, coverage density, source reliability | $[0,1]$ | |\n"
        "| $conf_{j,k}^{i},\\ \\lambda_{conf},\\ \\Delta t_{rep}$ | Observation confidence, freshness decay, data age | $[0,1]$, 1/min, min | |\n"
        "| $\\pi_{DSS}$ | Decision policy | – | |\n"
        "| $A_k$ (burned mask) | Cumulative burned mask | $\\{0,1\\}$ | |\n"
        "| $M_{cons,k}, M_{supp,k}$ | Cumulative consumed / suppressed fuel | norm. fuel | |\n"
        "| $a_{ha}, a_{km^2}$ | Cell area conversions | ha, km² | |\n"
        "| $J_k,\\ J^{burn},J^{val},J^{inf},J^{pop},J^{sup},J^{del}$ | Total cost and its six terms | currency | |\n"
        "| $w_1,\\dots,w_6$ | Cost term weights | – | |\n"
        "| $c_{ha}, c_{bld}, c_{crit}, c_{sup}$ | Unit costs | currency | |\n"
        "| $\\lambda_{for}, \\lambda_{loss}$ | Forest multiplier, loss fraction | – | |\n"
        "| $P^{exp}, N^{cas}, \\rho_{risk}, v_L$ | Exposure, casualties, risk fraction, VSL | persons, persons, –, currency | |\n"
        "| $\\bar t^{\\,resp}$ | Capacity-weighted mean response time | time | |\n"
        "| $H,\\ U_{base},\\ \\Delta J$ | Rollout horizon, baseline, cost gain | steps, –, currency | |\n"
        "| $Q_k, q_j, \\omega_j, \\eta$ | Quality score, components, weights, threshold | $[0,1]$ | |")


SECTIONS = [
    ("Part I — Architecture", "0", "The closed simulation–decision loop", True, _sec_0),
    ("Part II — The Simulator (the system)", "1", "Grid space and discrete time", True, _sec_1),
    ("Part II — The Simulator (the system)", "2", "External input sources (the data tables the simulation reads)", False, _sec_2),
    ("Part II — The Simulator (the system)", "3", "System state — what the simulation remembers", False, _sec_3),
    ("Part II — The Simulator (the system)", "4", "State transition I — burning status $B_k \\rightarrow B_{k+1}$", False, _sec_4),
    ("Part II — The Simulator (the system)", "5", "Rate of spread $R_{spread,k}$ — the Rothermel-type kernel", False, _sec_5),
    ("Part II — The Simulator (the system)", "6", "State transition II — fuel mass $F_{load,k} \\rightarrow F_{load,k+1}$", False, _sec_6),
    ("Part II — The Simulator (the system)", "7", "State transition III — fire intensity $I_k \\rightarrow I_{k+1}$", False, _sec_7),
    ("Part II — The Simulator (the system)", "8", "State transition IV — ignition time $\\tau_k \\rightarrow \\tau_{k+1}$", False, _sec_8),
    ("Part II — The Simulator (the system)", "9", "Implementation notes — exact behaviour of the code", False, _sec_9),
    ("Part II — The Simulator (the system)", "10", "Worked example — one update step by hand", False, _sec_10),
    ("Part III — The Decision Support System", "11", "Observation interface — $\\mathcal{O}_k=h(S_k,\\epsilon_k)$", False, _sec_11),
    ("Part III — The Decision Support System", "12", "Decision policy — how $U_{DSS,k}$ is produced and how it acts", False, _sec_12),
    ("Part III — The Decision Support System", "13", "Intervention vocabulary — what a decision can order", False, _sec_13),
    ("Part III — The Decision Support System", "14", "Cost function — how a decision is scored", False, _sec_14),
    ("Part III — The Decision Support System", "15", "Decision quality and graduated fail-safe", False, _sec_15),
    ("Part IV — Reference", "16", "Symbol glossary", False, _sec_16),
]

# ----------------------------------------------------- search infrastructure
class _Recorder:
    """Fake streamlit that records every string a section emits."""

    def __init__(self):
        self.buf = []

    def _add(self, *a):
        for x in a:
            if isinstance(x, str):
                self.buf.append(x)

    def markdown(self, t="", **k): self._add(t)
    def caption(self, t="", **k): self._add(t)
    def latex(self, t="", **k): self._add(t)
    def subheader(self, t="", **k): self._add(t)
    def image(self, *a, **k): pass

    def text(self):
        return " ".join(self.buf)


def _norm(t: str) -> str:
    return t.replace("\\", "").replace("{", "").replace("}", "").lower()


def _section_text(fn, ctx) -> str:
    """Run a section against the recorder and return its full plain text."""
    global st
    real, rec = st, _Recorder()
    st = rec
    try:
        fn(ctx)
    finally:
        st = real
    return rec.text()


# ------------------------------------------------------------------- render
def render():
    cfg = _cfg()
    ctx = (cfg, cfg.spread, cfg.suppression, cfg.intensity,
           cfg.value_weights, cfg.cost)

    st.subheader("System Description")
    st.markdown(
        "DisasterAware couples a **grid based, discrete time wildfire "
        "simulator** (the *system*) with a **Decision Support System** (DSS) "
        "that observes the fire and allocates suppression resources. This page "
        "describes both, plus the **cost function** that scores every decision. "
        "All equations are given in explicit, hand computable form; the "
        "default parameter tables are in (thresholds), (Table "
        "A.1 + B.1), (Table B.2), (Table C.1) and "
        "(cost model), and every value is adjustable in the Parameters page.")

    # --- search bar + expand/collapse controls
    c_search, c_e, c_c = st.columns([3, 1, 1])
    query = c_search.text_input(
        "Search", key="sd_search", placeholder=
        "Search all sections… e.g. suppression, theta_ign, cost, Table A.1",
        label_visibility="collapsed")
    if c_e.button("Expand all", use_container_width=True):
        st.session_state.sd_exp = True
    if c_c.button("Collapse all", use_container_width=True):
        st.session_state.sd_exp = False
    flag = st.session_state.get("sd_exp")

    def _x(default=False):
        return default if flag is None else flag

    q = _norm(query.strip()) if query else ""
    terms = q.split()

    # --- contents tree
    if not terms:
        with st.expander("Contents", expanded=(False if flag is None else flag)):
            tree, last_part = [], None
            for part, num, title, _d, _f in SECTIONS:
                if part != last_part:
                    tree.append(f"- **{part}**")
                    last_part = part
                tree.append(f"    - {num} · {title}")
            st.markdown("\n".join(tree))

    # --- sections (filtered by search)
    shown, last_part = 0, None
    for part, num, title, dflt, fn in SECTIONS:
        if terms:
            hay = _norm(title) + " " + _norm(_section_text(fn, ctx))
            if not all(t in hay for t in terms):
                continue
        if part != last_part:
            st.markdown(f"## {part}")
            last_part = part
        with st.expander(f"{num} · {title}",
                         expanded=(True if terms else _x(num in ("0", "1")))):
            fn(ctx)
        shown += 1
    if terms:
        if shown:
            st.caption(f"{shown} section(s) match '{query}'. Clear the box to "
                       "see everything.")
        else:
            st.info(f"No section matches '{query}'.")
