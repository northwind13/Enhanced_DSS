# -*- coding: utf-8 -*-
"""Section III of v27. Notation: family letter + one-letter quantity index, k only on time-varying fields.
W wind (s speed, d direction), G terrain (s slope, a aspect, c access), F fuel (t type, m moisture, x moisture of
extinction, b combustion fraction, r suppression reduction; F_k alone is the available fuel mass of the state),
R resources (c capacity, a availability, e effectiveness, t travel time), nu rate of spread."""

BLOCKS = [
("H1", "III. System Architecture"),
("H2", "A. Two Cores and the Interface"),
("P", "In Fig. 1, the architecture consists of four blocks, namely simulation core, decision core, external sources, and operator dashboard. The simulation core owns the hazard state $S_k$ and advances it with the transition $\\Phi$, while the decision core reads an observation $O_k$ of that state and returns the intervention field $U_{DSS,k}$. An operator edits the inputs through the dashboard $\\Theta_{UI}$ and never the state. The two cores meet at the input sets of (EQ:fin) and run the loop of (EQ:loop) once per decision cycle, which spans several simulation steps with the field held constant in between."),
("EQ", [("fin", [r"\mathcal{F}_{in,k}=\{U_{Meteo,k},\,U_{Geo},\,U_{Fuel,k},\,U_{Ign,k},\,U_{DSS,k}\}", r"\mathcal{F}_{DSS,k}=\{U_{Val,k},\,U_{Res,k}\}"]),
        ("loop", [r"S_k \to O_k=h(S_k,\epsilon_k) \to U_{DSS,k}=\pi_{DSS}(O_k,\mathcal{F}_{DSS,k})", r"\to U_{DSS,k}\subset \mathcal{F}_{in,k} \to S_{k+1}=\Phi(S_k,\mathcal{F}_{in,k})"]),
        ("obs", r"|\epsilon_k|\le(1-\gamma_k)\,\epsilon_{max}")]),
("P", "In all equations, $k$ is the simulation step at $t_k = k\\,\\Delta t_{sim}$, every field is defined per cell of the grid $\\mathcal{G}$, and a symbol names its family by a capital letter and its quantity by one index letter. $U_{Meteo,k}$ holds the wind speed $W_{s,k}$ and direction $W_{d,k}$, air temperature, relative humidity, and precipitation. $U_{Geo}$ holds elevation, slope $G_s$, aspect $G_a$, and road access $G_c$. $U_{Fuel,k}$ holds the fuel type $F_t$ of a standard fuel model {42}, the initial fuel load, and the dead fuel moisture $F_{m,k}$. $U_{Ign,k}\\in\\{0,1\\}$ injects detected and scenario ignitions, and $U_{DSS,k}$ is the only field the decision core writes. The decisional fields never enter $\\Phi$. $U_{Val,k}$ aggregates buildings, critical facilities, population, and evacuation access into a value at risk, and $U_{Res,k}$ holds the crews and water sources through the capacity $R_{c,k}$ that an order can draw on a cell, its availability $R_{a,k}$, its effectiveness $R_{e,k}$, and its travel time $R_{t,k}$. The observation function $h$ injects the epistemic uncertainty $\\epsilon_k$ at this point only, bounded by the cell confidence $\\gamma_k$ of Section IV as in (EQ:obs), so a confident cell is observed exactly and an unobserved one admits the worst case $\\epsilon_{max}$; $\\pi_{DSS}$ is the decision map of Section IV."),
("P", "The simulation core is replaceable by any hazard model that exposes its state through $h$ and accepts interventions through $\\mathcal{F}_{in,k}$, since the decision core never reads the state directly. A purpose-built core is used for one reason. It is seeded, snapshotted with its random state, and replayed exactly, so the same incident runs with and without a decision mechanism and the difference is attributed to the mechanism alone. This is the basis of every paired comparison in Section V."),
("H2", "B. State, Transition, and Spread"),
("P", "The state is the collection of per-cell states (EQ:state), with the burning status $B_k\\in\\{0,1\\}$, the available fuel mass $F_k\\in[0,1]$, the intensity proxy $I_k$, and the time since ignition $\\tau_k$. One step of $\\Phi$ updates the four components by (EQ:burn) to (EQ:tau), where $[\\cdot]$ is the Iverson bracket, equal to one when its condition holds. Burning (EQ:burn) persists while fuel exceeds the extinction load $\\epsilon_F=0.03$ and the moisture stays below the moisture of extinction $F_x$ of the fuel type {6}, propagates when the neighbor influence $\\Psi_k$ of (EQ:psi) exceeds the threshold $\\Theta_{ign}=0.125$, the cellular rule of {23}, starts on a seeded ember that lands on fuel, or starts on $U_{Ign,k}$, as in (EQ:pers). Fuel (EQ:fuel) falls through combustion by the fraction $F_{b,k}$ in burning cells and through the suppression term $F_{r,k}$ of (EQ:supp). The map $f_{supp}$ follows the initial-attack productivity model of {43}, in which the fuel or fireline a crew removes per step grows with the capacity it can field, $R_{c,k} R_{a,k}$, and is discounted by its effectiveness $R_{e,k}$, which falls with the local intensity $I_k$, and by the travel time $R_{t,k}$ and the access $G_c$ of the cell; hence an order changes what the fire can burn and never the burning state. The intensity proxy (EQ:inten) is a bounded map $f_{int}$ of fuel, wind, and slope, the three drivers of fireline intensity in the formulation of Byram {44} as parameterized in {6}, and the ignition clock (EQ:tau) counts active burning."),
("EQ", [("state", r"S_k=\{s_k(x,y)\}_{(x,y)\in\mathcal{G}},\quad s_k=(B_k,\,F_k,\,I_k,\,\tau_k)"),
        ("burn", r"B_{k+1}=\max\{B^{pers},\,B^{prop},\,B^{spot},\,U_{Ign,k}\}"),
        ("pers", [r"B^{pers}=B_k\,[F_k>\epsilon_F]\,[F_{m,k}<F_x]", r"B^{prop}=[\Psi_k>\Theta_{ign}],\quad B^{spot}=[\text{ember lands}]\,[F_k>\epsilon_F]"]),
        ("fuel", r"F_{k+1}=\max\{0,\;F_k-B_k\,F_k\,F_{b,k}-F_{r,k}\}"),
        ("supp", r"F_{r,k}=f_{supp}(R_{c,k},\,R_{a,k},\,R_{e,k},\,R_{t,k},\,G_c)"),
        ("inten", r"I_{k+1}=B_{k+1}\,f_{int}(F_k,\,W_{s,k},\,G_s)"),
        ("tau", r"\tau_{k+1}=\begin{cases}0 & B_{k+1}=1\wedge B_k=0\\ \tau_k+\Delta t_{sim} & B_{k+1}=1\wedge B_k=1\\ 0 & B_{k+1}=0\end{cases}")]),
("P", "Propagation (EQ:psi) is anisotropic. Each burning neighbor $(i,j)$ in the eight-neighborhood $\\mathcal{N}_8$ contributes its rate of spread $\\nu_k$ weighted by the cosine of the angle between the wind direction $W_{d,k}$ and the direction $\\varphi_{ij}$ to the target cell, and $\\nu_k$ is the Rothermel-type map $f_{ros}$ of (EQ:ros) over wind, slope, aspect, moisture, and fuel type {6}. The front advances in substeps, so spread never exceeds one cell per substep. Spotting is the only stochastic element and is driven by the seeded random stream. Precipitation raises $F_{m,k}$ toward $F_x$ with a time constant of about thirty minutes, so a front dies out gradually."),
("EQ", [("psi", [r"\Psi_k=\frac{1}{8}\sum_{(i,j)\in\mathcal{N}_8} B_k(i,j)\,\nu_k(i,j)\,g_k(i,j)", r"g_k(i,j)=\max\{0,\,\cos(W_{d,k}(i,j)-\varphi_{ij})\}"]),
        ("ros", r"\nu_k=f_{ros}(W_{s,k},\,W_{d,k},\,G_s,\,G_a,\,F_{m,k},\,F_t)")]),
]

# new references, in order of first appearance (numbers continue after the last reference of Section II)
NEW_SOURCES = [
 (42, "ScottBurgan05", {"type": "Report", "authors": [("Scott", "J.", "H."), ("Burgan", "R.", "E.")],
       "title": "Standard fire behavior fuel models: A comprehensive set for use with Rothermel's surface fire spread model",
       "year": "2005", "city": "Gen. Tech. Rep. RMRS-GTR-153"},
  'J. H. Scott and R. E. Burgan, "Standard fire behavior fuel models: A comprehensive set for use with Rothermel\'s surface fire spread model," Gen. Tech. Rep. RMRS-GTR-153, 2005. '),
 (43, "HirschMartell96", {"type": "JournalArticle", "authors": [("Hirsch", "K.", "G."), ("Martell", "D.", "L.")],
       "title": "A review of initial attack fire crew productivity and effectiveness", "journal": "Int. J. Wildland Fire",
       "year": "1996", "volume": "6", "pages": "199-215"},
  'K. G. Hirsch and D. L. Martell, "A review of initial attack fire crew productivity and effectiveness," |Int. J. Wildland Fire, |vol. 6, pp. 199-215, 1996. '),
 (44, "Byram59", {"type": "BookSection", "authors": [("Byram", "G.", "M.")], "title": "Combustion of forest fuels",
       "book": "Forest Fire: Control and Use", "year": "1959", "pages": "61-89"},
  'G. M. Byram, "Combustion of forest fuels," in |Forest Fire: Control and Use, |1959. '),
]
