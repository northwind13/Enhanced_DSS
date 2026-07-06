"""System Description page of the DisasterAware dashboard.

A self contained, hand computable description of the whole framework in three
parts: the simulator (thesis Chapter 4, Appendices A-C), the decision support
system, and the cost function that scores decisions. Every function is written
out explicitly; no closed form is left opaque.
"""

from __future__ import annotations

import streamlit as st

from disasteraware import FUEL_MODELS, SimConfig


# ------------------------------------------------------- architecture figure
_ARCH_SVG = """<svg viewBox="0 0 880 780" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>
    </marker>
  </defs>
  <style>
    .box{stroke-width:1.4;} .lbl{font-size:12px;fill:#111;} .sm{font-size:11px;fill:#111;}
    .eq{font-size:12.5px;font-style:italic;fill:#111;} .fn{font-size:11.5px;font-style:italic;fill:#111;}
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
  <text x="600" y="68"  class="eq">t_sim = k &#183; &#916;t_sim ,&#160; k &#8712; &#8469;</text>
  <text x="600" y="90"  class="eq">t_k+1 = t_k + &#916;t_sim</text>
  <text x="600" y="112" class="eq">S_k(x,y) = { s_c,k }_c&#8712;G</text>
  <text x="600" y="134" class="eq">c &#8801; (x,y)</text>

  <!-- ===================== External Sources ===================== -->
  <rect x="560" y="215" width="240" height="26" fill="#ffb570" stroke="#d79b00" class="box"/>
  <text x="680" y="232" text-anchor="middle" class="hdr">External Sources</text>
  <rect x="560" y="241" width="240" height="30" fill="#ffe6cc" stroke="#d79b00" class="box"/>
  <text x="680" y="254" text-anchor="middle" class="lbl" font-weight="bold">Fire Spread Model</text>
  <text x="680" y="267" text-anchor="middle" class="lbl" font-weight="bold">External Sources</text>
  <rect x="560" y="271" width="240" height="30" fill="#fff2cc" stroke="#d6b656" class="box"/>
  <text x="572" y="290" class="eq">Meteorology&#160; U_Meteo,k (x,y)</text>
  <rect x="560" y="301" width="240" height="30" fill="#fff2cc" stroke="#d6b656" class="box"/>
  <text x="572" y="320" class="eq">Topography&#160; U_Geo (x,y)</text>
  <rect x="560" y="331" width="240" height="30" fill="#fff2cc" stroke="#d6b656" class="box"/>
  <text x="572" y="350" class="eq">Fuel&#160; U_Fuel,k (x,y)</text>
  <rect x="560" y="361" width="240" height="26" fill="#ffe6cc" stroke="#d79b00" class="box"/>
  <text x="680" y="378" text-anchor="middle" class="lbl" font-weight="bold">Decisional External Sources</text>
  <rect x="560" y="387" width="240" height="30" fill="#fff2cc" stroke="#d6b656" class="box"/>
  <text x="572" y="406" class="eq">Values&#160; U_Val,k (x,y)</text>
  <rect x="560" y="417" width="240" height="30" fill="#fff2cc" stroke="#d6b656" class="box"/>
  <text x="572" y="436" class="eq">Resource&#160; U_Res,k (x,y)</text>

  <!-- ===================== Simulation Core ===================== -->
  <rect x="30" y="230" width="440" height="235" fill="none" stroke="#b85450" stroke-width="1.6"/>
  <rect x="30" y="230" width="22" height="235" fill="#f08705" opacity="0.75"/>
  <text x="45" y="348" text-anchor="middle" class="hdr" transform="rotate(-90 45 348)">Simulation Core</text>
  <rect x="100" y="252" width="290" height="52" rx="8" fill="#f8cecc" stroke="#b85450" class="box"/>
  <text x="245" y="273" text-anchor="middle" class="lbl" font-weight="bold">Hybrid Fire Spread Model</text>
  <text x="245" y="291" text-anchor="middle" class="sm">(Deterministic + Stochastic)</text>
  <rect x="100" y="390" width="290" height="52" rx="8" fill="#f8cecc" stroke="#b85450" class="box"/>
  <text x="245" y="410" text-anchor="middle" class="lbl" font-weight="bold">Grid State Manager</text>
  <text x="245" y="429" text-anchor="middle" class="eq">S_k+1(x,y) = &#934;( S_k(x,y), F_in,k )</text>
  <path class="edge" d="M 170 390 L 170 304"/>
  <text x="128" y="352" class="eq">S_k(x,y)</text>
  <path class="edge" d="M 320 304 L 320 390"/>
  <text x="330" y="352" class="eq">S_k+1(x,y)</text>

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
  <!-- UI -> External Sources -->
  <path class="edge" d="M 400 100 L 640 100 L 640 215"/>
  <text x="420" y="93" class="eq">&#920;_UI &#8594; ( F_in,k , F_DSS,k )</text>
  <!-- External -> Hybrid -->
  <path class="edge" d="M 560 278 L 390 278"/>
  <text x="475" y="271" class="eq">F_in,k</text>
  <text x="510" y="266" class="fn">*(1)</text>
  <!-- External -> Local DSS -->
  <path class="edge" d="M 640 447 L 640 695 L 445 695"/>
  <text x="648" y="480" class="eq">F_DSS,k</text>
  <text x="700" y="475" class="fn">*(2)</text>
  <!-- GSM -> UI (feedback up the left side) -->
  <path class="edge" d="M 100 416 L 12 416 L 12 60 L 120 60"/>
  <text x="18" y="50" class="eq">S_k(x,y)</text>
  <!-- GSM -> Observations -->
  <path class="edge" d="M 355 442 L 355 545"/>
  <text x="363" y="500" class="eq">S_k(x,y)</text>
  <!-- Observations -> Local DSS -->
  <path class="edge" d="M 365 605 L 365 665"/>
  <text x="373" y="640" class="eq">O_k(x,y)</text>
  <text x="428" y="635" class="fn">*(3)</text>
  <!-- Local DSS -> Actions -->
  <path class="edge" d="M 145 665 L 145 635"/>
  <!-- Actions -> GSM -->
  <path class="edge" d="M 145 545 L 145 442"/>
  <text x="152" y="500" class="eq">U_DSS,k(x,y)</text>
  <text x="230" y="495" class="fn">*(4)</text>

  <!-- ===================== footnotes ===================== -->
  <text x="480" y="560" class="fn">*(1)&#160; F_in,k = { U_Meteo,k , U_Geo , U_Fuel,k , U_Ign,k , U_DSS,k }</text>
  <text x="480" y="586" class="fn">*(2)&#160; F_DSS,k = { U_Val,k , U_Res,k }</text>
  <text x="480" y="612" class="fn">*(3)&#160; O_k(x,y) = h( S_k(x,y), &#949;_k )</text>
  <text x="480" y="638" class="fn">*(4)&#160; U_DSS,k(x,y) = &#960;_DSS( O_k(x,y), F_DSS,k )</text>
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


# ------------------------------------------------------------------- render
def render():
    cfg = _cfg()
    sp, su, ip, vw, cp = (cfg.spread, cfg.suppression, cfg.intensity,
                          cfg.value_weights, cfg.cost)

    st.subheader("System Description")
    st.markdown(
        "DisasterAware couples a **grid based, discrete time wildfire "
        "simulator** (the *system*) with a **Decision Support System** (DSS) "
        "that observes the fire and allocates suppression resources. This page "
        "describes both, plus the **cost function** that scores every decision. "
        "All equations are given in explicit, hand computable form: each symbol "
        "is defined once, used consistently, and every function is fully "
        "written out.")

    # expand / collapse all
    b1, b2, _sp = st.columns([1, 1, 4])
    if b1.button("Expand all", use_container_width=True):
        st.session_state.sd_exp = True
    if b2.button("Collapse all", use_container_width=True):
        st.session_state.sd_exp = False
    flag = st.session_state.get("sd_exp")

    def _x(default=False):
        return default if flag is None else flag

    # ################################################################
    st.markdown("## Part I — Architecture")
    # ################################################################

    with st.expander("0 · The closed simulation–decision loop", expanded=_x(True)):
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
             r"transition operator $\Phi$ (Part II, Sec. 2)"])
        _eq(r"\mathcal{F}_{DSS,k}=\{\,U_{Val,k},\;U_{Res,k}\,\}",
            [r"$\mathcal{F}_{DSS,k}$ — the decisional context: values at risk "
             r"and the available resource pool. It informs the DSS but never "
             r"enters the fire physics"])
        _eq(r"\mathcal{O}_k(x,y)=h\big(S_k(x,y),\,\epsilon_k\big)",
            [r"$\mathcal{O}_k$ — the observation the DSS reasons on (Part III, "
             r"Sec. 11); $h$ projects the state, $\epsilon_k$ is bounded "
             r"observation noise"])
        _eq(r"U_{DSS,k}(x,y)=\pi_{DSS}\big(\mathcal{O}_k(x,y),\,"
            r"\mathcal{F}_{DSS,k}\big)",
            [r"$\pi_{DSS}$ — the decision policy (Part III, Sec. 12): it maps "
             r"observations and decisional context to the resource allocation "
             r"$U_{DSS,k}$, which re-enters the simulator as an input"])
        st.caption(
            "Time advances as $t_{sim}=k\\,\\Delta t_{sim}$; the global state "
            "is the collection $S_k(x,y)=\\{s_{c,k}\\}_{c\\in G}$ with "
            "$c\\equiv(x,y)$ (Sec. 3).")

    # ################################################################
    st.markdown("## Part II — The Simulator (the system)")
    # ################################################################

    # ================================================== 1. domain
    with st.expander("1 · Grid space and discrete time", expanded=_x(True)):
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
            f"| Time step | $\\Delta t$ | {cfg.dt:g} model time unit |")

    # ================================================== 2. external inputs
    with st.expander("2 · External input sources (the data tables the simulation reads)", expanded=_x()):
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
            "| $W_{ws,k}$ | Wind speed | spatio-temporal | m/s | Amplifies rate of spread (Sec. 5) and fire intensity (Sec. 7) |\n"
            "| $W_{wd,k}$ | Wind direction | spatio-temporal | rad | Direction the wind blows **toward** (math convention, $0$ = +x, counter-clockwise positive); sets the axis of anisotropic spread (Sec. 4) |\n"
            "| $W_{gust,k}$ | Wind gust speed | spatio-temporal | m/s | Exogenous stochastic forcing channel; not used by the deterministic core equations |\n"
            "| $W_{prec,k}$ | Precipitation | spatio-temporal | mm/h | Exogenous moistening channel; raises fuel moisture when the moisture dynamics option is active |")
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
            "| $G_{slope}$ | Terrain slope | static | rad | Accelerates uphill spread (Sec. 5) and raises intensity (Sec. 7) |\n"
            "| $G_{aspect}$ | Slope orientation (aspect) | static | rad | Interacts with wind direction in the spread rate (Sec. 5) |\n"
            "| $G_{access}$ | Accessibility index | static | – , $[0,1]$ | Gates suppression reachability (Sec. 6); $1$ = fully accessible, $0$ = unreachable |")

        # ---- 2.3 fuel
        st.markdown("#### 2.3 Fuel — $U_{Fuel,k}$")
        _eq(r"U_{Fuel,k}(x,y)=\begin{bmatrix}"
            r"F_{type}(x,y)\\ F_{load,0}(x,y)\\ F_{moist,k}(x,y)\end{bmatrix}")
        _table(
            "| Symbol | Field | Type | Unit | Role in the model |\n"
            "|---|---|---|---|---|\n"
            "| $F_{type}$ | Fuel class | static | – (categorical id) | Selects the per class parameters $r_{base}, m_{ext}, a_w, a_s, a_{asp}, b_{base}$ (Tables in Sec. 5–6) |\n"
            "| $F_{load,0}$ | Initial fuel load | initial condition | – , normalized $[0,1]$ | Initializes the dynamic state $F_{load,k}$; $1$ corresponds to a reference dry biomass of about 2 kg/m² |\n"
            "| $F_{moist,k}$ | Surface fuel moisture | spatio-temporal | – , mass fraction $[0,1]$ | Damps spread rate (Sec. 5) and combustion fraction (Sec. 6) |")
        st.caption(
            "The fuel load itself is part of the *state* (Sec. 3); only its "
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
            "the injection has no lasting effect (Sec. 4).")

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
            "$U_{DSS,k} \\rightarrow F_{red,k}$ (Sec. 6) and never overwrites "
            "the fire state directly.")
        st.caption(
            "Distinguish $U_{Res,k}$ from $U_{DSS,k}$: $U_{Res,k}$ is the "
            "*external resource pool* (what exists — part of "
            "$\\mathcal{F}_{DSS,k}$), while $U_{DSS,k}$ is the *allocation* the "
            "DSS derives from it (what is deployed where). With no DSS active, "
            "a static $U_{Res,k}$ can be applied directly as $U_{DSS,k}$ for "
            "what-if studies.")

    # ================================================== 3. state
    with st.expander("3 · System state — what the simulation remembers", expanded=_x()):
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
             r"$\mathcal{F}_{in,k}$ — the external input set of Section 2"],
            note="Φ itself is deterministic. Randomness can only enter through "
                 "the exogenous inputs (e.g. gust fluctuations), never through "
                 "the update rules.")

    # ================================================== 4. burning status
    with st.expander("4 · State transition I — burning status $B_k \\rightarrow B_{k+1}$", expanded=_x()):
        st.markdown(
            "A cell burns at the next step if **any** of three mechanisms fires: "
            "it keeps burning (persistence), it is ignited by its neighbours "
            "(propagation), or it is ignited externally (injection). The maximum "
            "realizes the logical OR of the three binary signals:")
        _eq(r"B_{k+1}(x,y)=\max\Big\{\,B_{k+1}^{pers}(x,y),\;B_{k+1}^{prop}(x,y),"
            r"\;I_{Ign,k}(x,y)\cdot H(x,y)\,\Big\}",
            [r"$B_{k+1}^{pers}$ — persistence term, defined below",
             r"$B_{k+1}^{prop}$ — neighbour propagation term, defined below",
             r"$I_{Ign,k}$ — external ignition injection (Sec. 2.4)",
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
        _eq(r"B_{k+1}^{prop}(x,y)=\mathbb{1}\big[\Psi_k(x,y)>\Theta_{ign}\big]"
            r"\cdot\mathbb{1}\big[F_{load,k}(x,y)>\epsilon_{fuel}\big]",
            [r"$\Psi_k(x,y)$ — total propagation influence received by the cell, "
             r"defined next",
             rf"$\Theta_{{ign}}$ — ignition threshold, currently "
             rf"${sp.theta_ign:g}$; the minimum accumulated influence (same "
             r"normalized units as $\Psi_k$) required to activate combustion in "
             r"a new cell"])
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
             r"(cells per step, Sec. 5): a fast-burning neighbour pushes more "
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
             r"low wind, see Sec. 9)"])
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

    # ================================================== 5. rate of spread
    with st.expander("5 · Rate of spread $R_{spread,k}$ — the Rothermel-type kernel", expanded=_x()):
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
            [r"$F_{moist,k}$ — surface fuel moisture (mass fraction, Sec. 2.3)",
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
            [r"$G_{slope}$ — terrain slope (rad, Sec. 2.2); in the code the "
             r"slope is clipped to $\pm1.4$ rad so $\tan(\cdot)$ stays finite, "
             r"and the whole factor is floored at $0$",
             r"$a_s(F_{type})$ — slope sensitivity of the fuel class "
             r"(dimensionless, table below)"])
        st.markdown("**(d) Aspect–wind alignment** — terrain orientation interacts "
                    "with wind direction:")
        _eq(r"g_{aspect,k}(x,y)=1+a_{asp}\big(F_{type}(x,y)\big)\cdot"
            r"\cos\!\big(G_{aspect}(x,y)-W_{wd,k}(x,y)\big)",
            [r"$G_{aspect}$ — slope orientation (rad, Sec. 2.2)",
             r"$W_{wd,k}$ — wind direction (rad)",
             r"$a_{asp}(F_{type})$ — aspect sensitivity of the fuel class "
             r"(dimensionless, table below). Aligned wind and aspect "
             r"($\cos=1$) boost spread; opposed ($\cos=-1$) damp it; the factor "
             r"is floored at $0$"])
        st.markdown("#### Fuel class parameters (rate of spread and combustion)")
        st.caption(
            "$r_{base}$ in cells/step (interpret relative to $\\Delta x$ and "
            "$\\Delta t$); $m_{ext}$ as mass fraction; $a_w$, $a_s$, $a_{asp}$ "
            "dimensionless; $b_{base}$ as fraction of fuel consumed per step "
            "(used in Sec. 6); $e$ as economic value per cell unit (used in "
            "Sec. 14).")
        rows = ["| id | Fuel class | $r_{base}$ | $m_{ext}$ | $a_w$ | $a_s$ | "
                "$a_{asp}$ | $b_{base}$ | forest | $e$ |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for fid, m in FUEL_MODELS.items():
            rows.append(f"| {fid} | {m.name} | {m.r_base:g} | {m.m_ext:g} | "
                        f"{m.a_w:g} | {m.a_s:g} | {m.a_asp:g} | {m.b_base:g} | "
                        f"{'yes' if m.is_forest else 'no'} | {m.economic_value:g} |")
        _table("\n".join(rows))
        st.caption(
            "Classes with $r_{base}=0$ (non_fuel, water) can never propagate "
            "fire; they act as natural firebreaks.")

    # ================================================== 6. fuel mass
    with st.expander("6 · State transition II — fuel mass $F_{load,k} \\rightarrow F_{load,k+1}$", expanded=_x()):
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
             r"fuel class, in fraction of fuel per step (Table in Sec. 5): "
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
             r"per model time unit (unit: 1/time). E.g. with $\beta_t=0.2$ and "
             r"$R_{time}=5$: $e^{-1}\approx0.368$",
             r"$G_{access}\in[0,1]$ — static accessibility (Sec. 2.2)"])
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
        _table(
            "| Parameter | Symbol | Typical range | Current | Effect when increased |\n"
            "|---|---|---|---|---|\n"
            f"| Global suppression gain | $\\alpha_s$ | 0.01 – 0.30 | {su.alpha_s:g} | Stronger overall mitigation per step |\n"
            f"| Travel-time decay | $\\beta_t$ | 0.05 – 0.50 | {su.beta_t:g} | Effectiveness drops faster with delay |\n"
            f"| Intensity resistance | $\\gamma_I$ | 0.5 – 5.0 | {su.gamma_I:g} | Intense fires become harder to suppress |")
        st.caption(
            "Operational reading: no resources ⇒ zero effect; unreachable cell "
            "⇒ effect decays smoothly to zero; poor access ⇒ proportional "
            "reduction; intense fire ⇒ diminishing returns. Suppression removes "
            "fuel — it never switches $B_k$ off directly. Extinction then "
            "follows physically through the persistence condition of Sec. 4.")

    # ================================================== 7. intensity
    with st.expander("7 · State transition III — fire intensity $I_k \\rightarrow I_{k+1}$", expanded=_x()):
        st.markdown(
            "The intensity proxy is a bounded indicator of combustion strength "
            "in $[0,1]$. It does not model temperature or heat flux; it exists "
            "to (i) make intense fires resist suppression (Sec. 6), (ii) rank "
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

    # ================================================== 8. ignition time
    with st.expander("8 · State transition IV — ignition time $\\tau_k \\rightarrow \\tau_{k+1}$", expanded=_x()):
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

    # ================================================== 9. implementation notes
    with st.expander("9 · Implementation notes — exact behaviour of the code", expanded=_x()):
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
            "3. **Low-wind isotropic blend.** The directional weight is applied "
            "as $g_{dir}^{eff}=(1-a)+a\\cdot g_{dir}$ with "
            "$a=\\min\\{W_{ws}/w_{aniso},1\\}$ and "
            f"$w_{{aniso}}={sp.aniso_wind_full:g}$ m/s. At zero wind the spread "
            "is isotropic (driven by fuel and slope alone); at "
            f"$W_{{ws}}\\ge{sp.aniso_wind_full:g}$ m/s the pure wind-aligned "
            "weight of Sec. 4 applies.\n"
            "4. **Slope clipping.** $G_{slope}$ is clipped to "
            f"$\\pm{sp.slope_clip_rad:g}$ rad before $\\tan(\\cdot)$ to avoid "
            "numerical blow-up near vertical terrain.\n"
            "5. **Optional realism modes (off by default).** An elliptical "
            "(Cell2Fire/FARSITE style) directional kernel and stochastic ember "
            "spotting can be enabled in the parameters; when off, the model is "
            "exactly the formulation above "
            f"(elliptical: {'on' if sp.elliptical else 'off'}, "
            f"spotting: {'on' if sp.spotting else 'off'}).\n"
            "6. **Optional moisture dynamics.** By default $F_{moist}$ is a "
            "static exogenous field; a toggle recomputes it each step from "
            "temperature and humidity with the Simard (1968) equilibrium "
            "moisture model.\n"
            "7. **Synchronous update.** All cells are updated from the same "
            "$S_k$; no cell sees a neighbour's $k{+}1$ value within a step.")

    # ================================================== 10. worked example
    with st.expander("10 · Worked example — one update step by hand", expanded=_x()):
        st.markdown(
            "Setting: a **grass** cell $C_{src}$ is burning; the wind blows due "
            "east ($W_{wd}=0$) at $W_{ws}=6$ m/s. The target cell $C_{tgt}$ "
            "lies directly east of $C_{src}$ (so $C_{src}$ is its western "
            "neighbour, $\\theta=0$). Both cells: $F_{moist}=0.08$, "
            "$G_{slope}=0.1$ rad, $G_{aspect}=0$, $F_{load}=1.0$. Default "
            "parameters ($w_0=10$, $\\Theta_{ign}=0.08$, $\\alpha_s=0.2$, "
            "$\\beta_t=0.2$, $\\gamma_I=2$, $\\beta=2$, $\\gamma_W=0.5$, "
            "$\\gamma_S=0.3$, $W_{ref}=20$, $S_{max}=\\pi/4$). Note "
            "$W_{ws}=6\\ge w_{aniso}=6$, so the pure directional weight applies.")
        st.markdown("**Step 1 — rate of spread at the burning source (Sec. 5):**")
        st.latex(r"g_{moist}=1-\tfrac{0.08}{0.25}=0.680,\qquad "
                 r"g_{wind}=1+2.0\tanh(\tfrac{6}{10})=1+2.0\times0.5370=2.0741")
        st.latex(r"g_{slope}=1+0.8\tan(0.1)=1+0.8\times0.1003=1.0803,\qquad "
                 r"g_{aspect}=1+0.3\cos(0-0)=1.300")
        st.latex(r"R_{spread}=1.20\times0.680\times2.0741\times1.0803"
                 r"\times1.300=2.3768\ \text{cells/step}")
        st.markdown("**Step 2 — propagation influence at the target (Sec. 4):** "
                    "only $C_{src}$ burns; it sits west, so $\\theta=0$ and "
                    "$g_{dir}=\\max\\{0,\\cos(0-0)\\}=1$ (non-diagonal, no "
                    "$\\sqrt2$ correction):")
        st.latex(r"\Psi=\tfrac{1}{8}\times1\times2.3768\times1=0.2971"
                 r"\;>\;\Theta_{ign}=0.08\;\Rightarrow\;B^{prop}=1")
        st.markdown("The target has fuel ($1.0>\\epsilon_{fuel}$), so "
                    "$B_{k+1}(C_{tgt})=1$: **the fire spreads east**, as the "
                    "wind dictates. (A cell east of the target with the wind "
                    "*behind* it would receive $g_{dir}=\\cos(\\pi)\\to0$: no "
                    "upwind ignition.)")
        st.markdown("**Step 3 — fuel update at the burning source (Sec. 6),** "
                    "with a suppression assignment $R_{cap}=0.5$, "
                    "$R_{avail}=1$, $R_{eff}=0.8$, $R_{time}=5$, "
                    "$G_{access}=1$, and current intensity $I_k=0.4$:")
        st.latex(r"F_{burn}=\min\{1,\,0.25(1-0.08)\}=0.2300")
        st.latex(r"\eta_{cap}=\tfrac{0.5}{1.0}=0.500,\quad \eta_{avail}=1,\quad "
                 r"\eta_{reach}=e^{-0.2\times5}\times1=e^{-1}=0.3679,\quad "
                 r"\eta_{eff}=\tfrac{0.8}{1+2\times0.4}=0.4444")
        st.latex(r"F_{red}=0.2\times0.500\times1\times0.3679\times0.4444=0.0164")
        st.latex(r"F_{load,k+1}=\max\{0,\;1.0-1\times0.2300\times1.0-0.0164\}"
                 r"=0.7536")
        st.markdown("**Step 4 — intensity of the newly ignited target (Sec. 7),** "
                    "using its current fuel $F_{load,k}=1.0$:")
        st.latex(r"\tilde F=1.0,\quad \tilde W=\tfrac{6}{20}=0.300,\quad "
                 r"\tilde S=\tfrac{\tan 0.1}{\tan(\pi/4)}=0.1003")
        st.latex(r"I_{k+1}=1\times\tanh\!\big(2\,(1.0+0.5\times0.300"
                 r"+0.3\times0.1003)\big)=\tanh(2.3602)=0.9823")
        st.markdown("**Step 5 — ignition clocks (Sec. 8):** the target is newly "
                    "ignited ($B_k=0\\to B_{k+1}=1$) so $\\tau_{k+1}(C_{tgt})=0$; "
                    "the source keeps burning so "
                    "$\\tau_{k+1}(C_{src})=\\tau_k+\\Delta t$.")
        st.caption("Every number above can be reproduced with a pocket "
                   "calculator; this is the complete arithmetic of one cell "
                   "update.")

    # ################################################################
    st.markdown("## Part III — The Decision Support System")
    # ################################################################

    # ================================================== 11. observation
    with st.expander("11 · Observation interface — $\\mathcal{O}_k=h(S_k,\\epsilon_k)$", expanded=_x()):
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

    # ================================================== 12. decision policy
    with st.expander("12 · Decision policy — how $U_{DSS,k}$ is produced and how it acts", expanded=_x()):
        st.markdown("The decision policy maps observations and decisional "
                    "context to the resource allocation:")
        _eq(r"U_{DSS,k}(x,y)=\pi_{DSS}\big(\mathcal{O}_k(x,y),\,"
            r"\mathcal{F}_{DSS,k}\big),\qquad "
            r"\mathcal{F}_{DSS,k}=\{U_{Val,k},\,U_{Res,k}\}",
            [r"$\pi_{DSS}$ — the decision policy (rule based, optimization "
             r"based, or learned); its output is the four-field allocation of "
             r"Sec. 2.6",
             r"$U_{Val,k}$ — values at risk (Sec. 2.5): *what to protect*",
             r"$U_{Res,k}$ — the external resource pool: *what exists to "
             r"deploy* (capacities, positions, availabilities)"])
        st.markdown("The full causal chain of an intervention is:")
        _eq(r"U_{DSS,k}=\big[R_{cap,k},R_{avail,k},R_{eff,k},R_{time,k}\big]"
            r"\;\xrightarrow{\text{Sec. 6}}\;F_{red,k}"
            r"\;\xrightarrow{\text{Sec. 6}}\;F_{load,k+1}"
            r"\;\xrightarrow{\text{Sec. 4}}\;B_{k+m}"
            r"\;\xrightarrow{\text{Sec. 7}}\;I_{k+m}")
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

    # ================================================== 13. interventions
    with st.expander("13 · Intervention vocabulary — what a decision can order", expanded=_x()):
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
            "interpretable: every rule consequent and every cost term (Sec. 14) "
            "has a concrete referent. The first four types act physically "
            "through $U_{DSS,k}$; the last two act on exposure and timing and "
            "are scored by the cost function.")

    # ================================================== 14. cost function
    with st.expander("14 · Cost function — how a decision is scored", expanded=_x()):
        st.markdown(
            "**Mathematical basis.** The cost follows the *cost-plus-loss* "
            "(cost plus net value change) principle of wildfire economics: the "
            "value of a response equals the money spent on suppression plus "
            "the net value lost to the fire. The cost of a decision is "
            "therefore an additive, weighted sum of the suppression effort it "
            "spends and the losses it fails to prevent. Each term is a "
            "distinct value that a decision trades against the others; the "
            "**weights**, not the list, encode operational priority.")

        st.markdown("#### 14.1 Bookkeeping quantities (how the terms are measured)")
        st.markdown(
            "All terms are computed from three cumulative fields the simulator "
            "maintains, plus the static value layers. Nothing else is needed:")
        _eq(r"A_k(x,y)=\max_{0\le\kappa\le k} B_\kappa(x,y)\;\in\{0,1\}",
            [r"$A_k$ — the **burned mask**: $1$ if the cell has burned at any "
             r"step up to $k$ (it never resets, unlike $B_k$)"])
        _eq(r"M_{cons,k}(x,y)=\sum_{\kappa=0}^{k-1}"
            r"B_\kappa(x,y)\,F_{burn,\kappa}(x,y)\,F_{load,\kappa}(x,y)",
            [r"$M_{cons,k}$ — cumulative fuel consumed by combustion "
             r"(normalized fuel units): the sum of the combustion terms of "
             r"Sec. 6 over all past steps"])
        _eq(r"M_{supp,k}(x,y)=\sum_{\kappa=0}^{k-1}F_{red,\kappa}(x,y)",
            [r"$M_{supp,k}$ — cumulative fuel removed by suppression "
             r"(normalized fuel units): the sum of the applied $F_{red}$ of "
             r"Sec. 6 over all past steps"])
        _eq(r"a_{ha}=\frac{\Delta x^2}{10^4}\ \text{[ha/cell]},\qquad "
            r"a_{km^2}=\frac{\Delta x^2}{10^6}\ \text{[km}^2\text{/cell]}",
            [rf"cell area conversions; with $\Delta x={cfg.cell_size_m:g}$ m: "
             rf"$a_{{ha}}={cfg.cell_area_ha:g}$ ha, "
             rf"$a_{{km^2}}={cfg.cell_area_ha/100.0:g}$ km²"])

        st.markdown("#### 14.2 The total decision cost")
        _eq(r"J_k=w_1 J_k^{burn}+w_2 J_k^{val}+w_3 J_k^{inf}"
            r"+w_4 J_k^{pop}+w_5 J_k^{sup}+w_6 J_k^{del}",
            [r"$J_k$ — total cost at step $k$ (currency units)",
             r"$w_1,\dots,w_6\ge0$ — non-negative priority weights; the "
             r"dashboard cost report uses $w_1=\dots=w_5=1$ with all terms "
             r"already expressed in a common currency, and $w_6$ is used by "
             r"the decision evaluation",
             r"each term $J^{(\cdot)}$ is defined explicitly below"])

        st.markdown("**Term 1 — land and vegetation loss $J^{burn}$** "
                    "(burned area rehabilitation + forest stand value):")
        _eq(r"J_k^{burn}=c_{ha}\cdot a_{ha}\sum_{(x,y)\in G}A_k(x,y)"
            r"\;+\;\lambda_{for}\sum_{(x,y)\in G}"
            r"e\big(F_{type}(x,y)\big)\,M_{cons,k}(x,y)\,"
            r"\mathbb{1}\big[F_{type}(x,y)\in\mathcal{T}_{for}\big]",
            [rf"$c_{{ha}}$ — land rehabilitation cost per burned hectare, "
             rf"currently {cp.cost_per_burned_ha:,.0f} currency/ha, applied to "
             r"the whole burned area",
             r"$e(F_{type})$ — economic value per cell unit of the fuel class "
             r"(Table in Sec. 5: shrub 6, pine litter 12, hardwood 18, …)",
             r"$\mathcal{T}_{for}$ — the set of forest fuel classes (flagged "
             r"*forest* in the same table); stand value loss scales with the "
             r"fuel **actually consumed** in forest cells, not with area alone",
             rf"$\lambda_{{for}}$ — forest value multiplier, currently "
             rf"{cp.forest_value_multiplier:g}"])
        st.markdown("**Term 2 — structure loss $J^{val}$** (wildland–urban interface):")
        _eq(r"J_k^{val}=c_{bld}\,\lambda_{loss}\sum_{(x,y)\in G}"
            r"A_k(x,y)\,V_{bld}(x,y)",
            [rf"$c_{{bld}}$ — value of one fully built cell "
             rf"($V_{{bld}}=1$), currently {cp.building_unit_value:,.0f} currency",
             rf"$\lambda_{{loss}}$ — fraction of asset value lost when the "
             rf"cell burns, currently {cp.value_loss_on_burn:g}",
             r"$V_{bld}$ — building footprint density (Sec. 2.5); the sum "
             r"runs only over burned cells because of the $A_k$ factor"])
        st.markdown("**Term 3 — critical infrastructure loss $J^{inf}$:**")
        _eq(r"J_k^{inf}=c_{crit}\,\lambda_{loss}\sum_{(x,y)\in G}"
            r"A_k(x,y)\,V_{crit}(x,y)",
            [rf"$c_{{crit}}$ — value of one fully critical cell "
             rf"($V_{{crit}}=1$), currently {cp.critical_unit_value:,.0f} "
             r"currency: an order of magnitude above buildings, because the "
             r"failure of a substation or hospital cascades beyond the fire",
             r"$V_{crit}$ — critical facility index (Sec. 2.5)"])
        st.markdown("**Term 4 — population exposure and human cost $J^{pop}$**, "
                    "measured in three explicit stages:")
        _eq(r"P_k^{exp}=a_{km^2}\sum_{(x,y)\in G}A_k(x,y)\,V_{pop}(x,y)"
            r"\quad\text{[persons]}",
            [r"$P_k^{exp}$ — population living inside the burned footprint: "
             r"density (person/km²) times burned cell area"])
        _eq(r"N_k^{cas}=\rho_{risk}\cdot P_k^{exp},\qquad "
            r"J_k^{pop}=v_L\cdot N_k^{cas}",
            [rf"$\rho_{{risk}}$ — fraction of the exposed population assumed "
             rf"at casualty risk, currently {cp.population_at_risk_fraction:g}",
             rf"$v_L$ — value of statistical life used for monetization, "
             rf"currently {cp.statistical_life_value:,.0f} currency",
             r"an effective evacuation decision (Sec. 13) lowers the realized "
             r"$P^{exp}$, which is how evacuation is scored by the cost"])
        st.markdown("**Term 5 — suppression cost $J^{sup}$** (the resources committed):")
        _eq(r"J_k^{sup}=c_{sup}\sum_{(x,y)\in G}M_{supp,k}(x,y)",
            [rf"$c_{{sup}}$ — cost per unit of fuel actually removed by "
             rf"intervention, currently {cp.suppression_unit_cost:,.0f} "
             r"currency/unit: paying for effort delivered, not for effort "
             r"ordered (an unavailable or unreachable allocation produces no "
             r"$F_{red}$ and therefore no cost)"])
        st.markdown("**Term 6 — response delay $J^{del}$** (decision-layer term):")
        _eq(r"J_k^{del}=\bar t_k^{\,resp}="
            r"\frac{\sum_{(x,y)}R_{cap,k}(x,y)\,R_{time,k}(x,y)}"
            r"{\sum_{(x,y)}R_{cap,k}(x,y)}\quad\text{(0 if no allocation)}",
            [r"$\bar t^{\,resp}$ — capacity-weighted mean travel time of the "
             r"allocated resources (model time units): late-arriving "
             r"allocations are penalized even before their physical effect "
             r"decays through $\eta_{reach}$ (Sec. 6)",
             r"this term penalizes slow action, since timeliness changes every "
             r"other loss; it is evaluated on the candidate allocation and "
             r"multiplied by the weight $w_6$ (currency per time unit)"])
        st.caption(
            "Terms 1–5 are implemented in the dashboard cost report and are "
            "recomputed at any step from the cumulative fields of Sec. 14.1. "
            "Term 6 is evaluated by the decision layer when comparing "
            "candidate allocations.")

        st.markdown("#### 14.3 Using the cost to compare decisions (rollout evaluation)")
        st.markdown(
            "The cost is not differentiable in the decision, so candidates are "
            "compared by **forward rollout**: a candidate allocation $U$ is "
            "applied to a copy of the simulator for a short horizon of $H$ "
            "steps, and its cost is read at the end:")
        _eq(r"J^{(H)}(U)=J_{k+H}\ \text{after simulating}\ "
            r"S_{k+1},\dots,S_{k+H}\ \text{under}\ \Phi\ \text{with}\ "
            r"U_{DSS}=U,\qquad \Delta J = J^{(H)}(U)-J^{(H)}(U_{base})",
            [r"$H$ — evaluation horizon (a few steps: long enough to see the "
             r"effect, short enough to evaluate online)",
             r"$U_{base}$ — a conservative baseline allocation (e.g. current "
             r"doctrine or no change)",
             r"$\Delta J<0$ — the candidate is an improvement; every "
             r"adaptation of the decision layer is accepted **only** when it "
             r"lowers the cost"])

        st.markdown("#### 14.4 Default cost parameters")
        _table(
            "| Parameter | Symbol | Current value | Unit |\n"
            "|---|---|---|---|\n"
            f"| Rehabilitation cost per burned ha | $c_{{ha}}$ | {cp.cost_per_burned_ha:,.0f} | currency/ha |\n"
            f"| Forest value multiplier | $\\lambda_{{for}}$ | {cp.forest_value_multiplier:g} | – |\n"
            f"| Building unit value | $c_{{bld}}$ | {cp.building_unit_value:,.0f} | currency/cell |\n"
            f"| Critical facility unit value | $c_{{crit}}$ | {cp.critical_unit_value:,.0f} | currency/cell |\n"
            f"| Value loss fraction on burn | $\\lambda_{{loss}}$ | {cp.value_loss_on_burn:g} | – |\n"
            f"| Value of statistical life | $v_L$ | {cp.statistical_life_value:,.0f} | currency/person |\n"
            f"| Population at risk fraction | $\\rho_{{risk}}$ | {cp.population_at_risk_fraction:g} | – |\n"
            f"| Suppression unit cost | $c_{{sup}}$ | {cp.suppression_unit_cost:,.0f} | currency/fuel unit |")
        st.caption(
            "Monetary figures are abstract currency units so the model can be "
            "calibrated to a real case study without changing its structure.")

        st.markdown("#### 14.5 Mini example — cost by hand")
        st.markdown(
            "$\\Delta x=30$ m ⇒ $a_{ha}=0.09$ ha, $a_{km^2}=0.0009$ km². "
            "Suppose by step $k$: 100 cells burned, of which 20 are pine "
            "litter forest ($e=12$) with $M_{cons}=0.8$ each; the burned "
            "footprint contains $\\sum A_k V_{bld}=3.0$, no critical cells, "
            "$\\sum A_k V_{pop}=1200$ person/km²; total suppressed fuel "
            "$\\sum M_{supp}=2.5$. Then, with default parameters:")
        st.latex(r"J^{burn}=1000\times0.09\times100+1\times(12\times0.8\times20)"
                 r"=9000+192=9192")
        st.latex(r"J^{val}=250{,}000\times1\times3.0=750{,}000,\qquad J^{inf}=0")
        st.latex(r"P^{exp}=0.0009\times1200=1.08\ \text{persons},\quad "
                 r"N^{cas}=0.02\times1.08=0.0216,\quad "
                 r"J^{pop}=1{,}500{,}000\times0.0216=32{,}400")
        st.latex(r"J^{sup}=5000\times2.5=12{,}500")
        st.latex(r"J=9192+750{,}000+0+32{,}400+12{,}500=804{,}092"
                 r"\ \text{currency units}")
        st.caption(
            "The structure term dominates — which is exactly the signal the "
            "DSS needs: protecting the three built cells is worth far more "
            "than saving grass, and the weights $w_m$ let an operator shift "
            "that balance.")

    # ================================================== 15. quality / fail-safe
    with st.expander("15 · Decision quality and graduated fail-safe", expanded=_x()):
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
            "Together, Sec. 14 and Sec. 15 close the loop: the cost drives the "
            "*search* for better decisions (accept only $\\Delta J<0$), the "
            "quality gate governs the *application* of the chosen one.")

    # ################################################################
    st.markdown("## Part IV — Reference")
    # ################################################################

    # ================================================== 16. glossary
    with st.expander("16 · Symbol glossary", expanded=_x()):
        _table(
            "| Symbol | Meaning | Unit / Range | Defined in |\n"
            "|---|---|---|---|\n"
            "| $G,\\ (x,y)$ | Grid domain, cell index | – | Sec. 1 |\n"
            "| $k,\\ \\Delta t,\\ \\Delta x$ | Step index, step length, cell size | –, time, m | Sec. 1 |\n"
            "| $\\Theta_{UI}$ | User interaction operator | – | Sec. 0 |\n"
            "| $\\mathcal{F}_{in,k}$ | Physical input set | – | Sec. 0, 2 |\n"
            "| $\\mathcal{F}_{DSS,k}$ | Decisional context set | – | Sec. 0, 12 |\n"
            "| $W_{temp,k}, W_{rh,k}$ | Air temperature, relative humidity | °C, % | Sec. 2.1 |\n"
            "| $W_{ws,k}, W_{wd,k}$ | Wind speed, wind direction | m/s, rad | Sec. 2.1 |\n"
            "| $W_{gust,k}, W_{prec,k}$ | Wind gust, precipitation | m/s, mm/h | Sec. 2.1 |\n"
            "| $G_{elev}, G_{slope}, G_{aspect}$ | Elevation, slope, aspect | m, rad, rad | Sec. 2.2 |\n"
            "| $G_{access}$ | Accessibility index | $[0,1]$ | Sec. 2.2 |\n"
            "| $F_{type}$ | Fuel class id | categorical | Sec. 2.3 |\n"
            "| $F_{load,k}$ | Fuel load (state) | $[0,1]$ norm. | Sec. 2.3, 3, 6 |\n"
            "| $F_{moist,k}$ | Fuel moisture | mass fraction | Sec. 2.3 |\n"
            "| $I_{Ign,k}$ | Ignition injection | $\\{0,1\\}$ | Sec. 2.4 |\n"
            "| $V_{bld}, V_{crit}, V_{pop}, V_{evac}, V_{prio}$ | Values at risk, priority score | see Sec. 2.5 | Sec. 2.5 |\n"
            "| $w_{bld}, w_{crit}, w_{pop}, w_{evac}$ | Priority weights (sum = 1) | – | Sec. 2.5 |\n"
            "| $R_{cap,k}, R_{avail,k}, R_{eff,k}, R_{time,k}$ | DSS resource fields | see Sec. 2.6 | Sec. 2.6 |\n"
            "| $U_{Res,k}$ | External resource pool | – | Sec. 2.6, 12 |\n"
            "| $s_k, S_k$ | Local / global state | – | Sec. 3 |\n"
            "| $B_k$ | Burning status (state) | $\\{0,1\\}$ | Sec. 3, 4 |\n"
            "| $I_k$ | Fire intensity proxy (state) | $[0,1]$ | Sec. 3, 7 |\n"
            "| $\\tau_k$ | Time since ignition (state) | time | Sec. 3, 8 |\n"
            "| $\\Phi$ | Transition operator | – | Sec. 3 |\n"
            "| $\\mathbb{1}[\\cdot]$ | Indicator function | $\\{0,1\\}$ | Sec. 4 |\n"
            "| $\\epsilon_{fuel}$ | Extinction fuel threshold | norm. fuel | Sec. 4 |\n"
            "| $\\Psi_k$ | Propagation influence | cells/step | Sec. 4 |\n"
            "| $\\Theta_{ign}$ | Ignition threshold | cells/step | Sec. 4 |\n"
            "| $N^8(x,y)$ | 8-connected neighbourhood | – | Sec. 4 |\n"
            "| $g_{dir}, \\theta_{(i,j)\\to(x,y)}$ | Directional weight, geometry angle | $[0,1]$, rad | Sec. 4 |\n"
            "| $R_{spread,k}$ | Rate of spread | cells/step | Sec. 5 |\n"
            "| $r_{base}$ | Base spread rate (per fuel class) | cells/step | Sec. 5 |\n"
            "| $g_{moist}, g_{wind}, g_{slope}, g_{aspect}$ | Spread modifiers | – | Sec. 5 |\n"
            "| $m_{ext}$ | Extinction moisture (per fuel class) | mass fraction | Sec. 5 |\n"
            "| $a_w, a_s, a_{asp}$ | Wind / slope / aspect sensitivity | – | Sec. 5 |\n"
            "| $w_0$ | Wind saturation scale | m/s | Sec. 5 |\n"
            "| $e(F_{type})$ | Economic value of fuel class | currency/cell unit | Sec. 5, 14 |\n"
            "| $F_{burn,k}$ | Combustion fraction per step | $[0,1]$ | Sec. 6 |\n"
            "| $b_{base}$ | Baseline combustion coefficient | fraction/step | Sec. 6 |\n"
            "| $F_{red,k}$ | Suppression fuel reduction | $[0,1]$ | Sec. 6 |\n"
            "| $\\alpha_s$ | Global suppression gain | fraction/step | Sec. 6 |\n"
            "| $\\eta_{cap}, \\eta_{avail}, \\eta_{reach}, \\eta_{eff}$ | Suppression factors | $[0,1]$ | Sec. 6 |\n"
            "| $R_{cap,max}$ | Reference max capacity | as $R_{cap}$ | Sec. 6 |\n"
            "| $\\beta_t$ | Travel-time decay rate | 1/time | Sec. 6 |\n"
            "| $\\gamma_I$ | Intensity resistance | – | Sec. 6 |\n"
            "| $\\tilde F, \\tilde W, \\tilde S$ | Normalized fuel / wind / slope | $[0,1]$ | Sec. 7 |\n"
            "| $F_{max}, W_{ref}, S_{max}$ | Normalization references | norm., m/s, rad | Sec. 7 |\n"
            "| $\\beta, \\gamma_W, \\gamma_S$ | Intensity gain and weights | – | Sec. 7 |\n"
            "| $\\mathcal{O}_k,\\ h,\\ \\epsilon_k$ | Observation, obs. function, obs. noise | – | Sec. 11 |\n"
            "| $\\pi_{DSS}$ | Decision policy | – | Sec. 12 |\n"
            "| $A_k$ | Cumulative burned mask | $\\{0,1\\}$ | Sec. 14 |\n"
            "| $M_{cons,k}, M_{supp,k}$ | Cumulative consumed / suppressed fuel | norm. fuel | Sec. 14 |\n"
            "| $a_{ha}, a_{km^2}$ | Cell area conversions | ha, km² | Sec. 14 |\n"
            "| $J_k,\\ J^{burn},J^{val},J^{inf},J^{pop},J^{sup},J^{del}$ | Total cost and its six terms | currency | Sec. 14 |\n"
            "| $w_1,\\dots,w_6$ | Cost term weights | – | Sec. 14 |\n"
            "| $c_{ha}, c_{bld}, c_{crit}, c_{sup}$ | Unit costs | currency | Sec. 14 |\n"
            "| $\\lambda_{for}, \\lambda_{loss}$ | Forest multiplier, loss fraction | – | Sec. 14 |\n"
            "| $P^{exp}, N^{cas}, \\rho_{risk}, v_L$ | Exposure, casualties, risk fraction, VSL | persons, persons, –, currency | Sec. 14 |\n"
            "| $\\bar t^{\\,resp}$ | Capacity-weighted mean response time | time | Sec. 14 |\n"
            "| $H,\\ U_{base},\\ \\Delta J$ | Rollout horizon, baseline, cost gain | steps, –, currency | Sec. 14 |\n"
            "| $Q_k, q_j, \\omega_j, \\eta$ | Quality score, components, weights, threshold | $[0,1]$ | Sec. 15 |")
