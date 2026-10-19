"""v23 figures: Fig. 1 simulation core and Fig. 2 decision core, single column, top to bottom.
Canvas width 270 units = 3.375 in at scale 1/80, so fonts render at their true size. All symbols are math ($...$),
so the draw.io export renders them with MathJax and the PNG with mathtext."""
import os
from diag import Diagram, drawio_file, PAL

OUT = "/home/claude/v23/fig"; os.makedirs(OUT, exist_ok=True)
pages = []
W = 270; M = 14; IW = W - 2 * M
GREY = "#595959"; BLUE = "#2F5597"; GREEN = "#548235"; DBLUE = "#1F3864"; RED = "#C00000"
PAL["ui"] = ("#DEEBF7", "#1F4E79"); PAL["src"] = ("#FFF2CC", "#7F6000"); PAL["sim"] = ("#E2F0D9", "#385723"); PAL["dss"] = ("#FBE5D6", "#843C0C")
PAL["cell"] = ("#FFFFFF", "#7F7F7F"); PAL["burn"] = ("#F4B183", "#C55A11"); PAL["fire"] = ("#C00000", "#C00000"); PAL["target"] = ("#FFF2CC", "#BF9000")

def chips(d, x0, y0, items, cols, cw, ch, gap=4, kind="plain", fs=5.2):
    out = []
    for i, t in enumerate(items):
        r, c = divmod(i, cols)
        out.append(d.box(x0 + c * (cw + gap), y0 + r * (ch + gap), cw, ch, t, kind=kind, fs=fs))
    return out

# =====================================================================
# Fig. 1  Simulation core and interfaces (after thesis Fig. 3.1, single column, compact, no equation numbers)
# =====================================================================
G = 6
d = Diagram(W, 282, font=6.6)
y = 3
ui = d.box(M, y, IW, 28, "", kind="ui"); d.text(W / 2, y + 7, "Operator dashboard $\\Theta_{UI}$", bold=True, color="#1F4E79", ha="center")
cw = (IW - 8 - 3 * 4) / 4
chips(d, M + 4, y + 13, ["map view", "map and data editor", "scenario manager", "simulation settings"], 4, cw, 12, fs=5.0)
y += 28 + G + 4
ex = d.box(M, y, IW, 64, "", kind="src"); d.text(W / 2, y + 7, "External sources", bold=True, color="#7F6000", ha="center")
d.text(M + 6, y + 15, "fire spread model sources  $\\rightarrow \\mathcal{F}_{in,k}$", fs=5.2, color="#7F6000", ha="left")
items = ["meteorology\n$U_{Meteo,k}$", "topography\n$U_{Geo}$", "fuel\n$U_{Fuel,k}$", "ignition\n$U_{Ign,k}$"]
chips(d, M + 4, y + 20, items, 4, cw, 16, fs=5.1)
d.text(M + 6, y + 43, "decisional sources  $\\rightarrow \\mathcal{F}_{DSS,k}$", fs=5.2, color="#7F6000", ha="left")
cw2 = (IW - 8 - 4) / 2
chips(d, M + 4, y + 48, ["values at risk  $U_{Val,k}$", "resources  $U_{Res,k}$"], 2, cw2, 12, fs=5.1)
yex = y; y += 64 + G + 5
sim = d.box(M, y, IW, 78, "", kind="sim"); d.text(W / 2, y + 7, "Simulation core", bold=True, color="#385723", ha="center")
fm = d.box(M + 6, y + 13, IW - 12, 20, "hybrid fire spread model\ndeterministic spread, seeded spotting", kind="plain", fs=5.3)
gm = d.box(M + 6, y + 46, IW - 12, 22, "grid state manager\n$S_{k+1} = \\Phi(S_k,\\ \\mathcal{F}_{in,k})$", kind="plain", fs=5.3)
d.edge(gm, fm, p0=(M + 6 + (IW - 12) * 0.3, y + 46), p1=(M + 6 + (IW - 12) * 0.3, y + 33), label="$S_k$", loff=(-12, 0))
d.edge(fm, gm, p0=(M + 6 + (IW - 12) * 0.7, y + 33), p1=(M + 6 + (IW - 12) * 0.7, y + 46), label="$S_{k+1}$", loff=(16, 0))
d.text(W / 2, y + 73, "$t_k = k\\,\\Delta t_{sim},\\ k \\in \\mathbb{N}$", fs=5.0, italic=True, color="#385723", ha="center")
ysim = y; y += 78 + G + 5
dc = d.box(M, y, IW, 70, "", kind="dss"); d.text(W / 2, y + 7, "Decision core", bold=True, color="#843C0C", ha="center")
ob = d.box(M + 6, y + 13, IW - 12, 20, "observation\n$O_k = h(S_k,\\ \\epsilon_k)$", kind="plain", fs=5.3)
ac = d.box(M + 6, y + 46, IW - 12, 20, "actions\n$U_{DSS,k} = \\pi_{DSS}(O_k,\\ \\mathcal{F}_{DSS,k})$", kind="plain", fs=5.3)
d.edge(ob, ac, src="s", dst="n", label="reasoning", loff=(26, 0))
ydc = y
d.edge(ui, ex, p0=(W / 2, 31), p1=(W / 2, yex), label="$\\Theta_{UI} \\rightarrow (\\mathcal{F}_{in,k}, \\mathcal{F}_{DSS,k})$", loff=(54, 0))
d.edge(ex, sim, p0=(W / 2, yex + 64), p1=(W / 2, ysim), label="$\\mathcal{F}_{in,k}$", loff=(16, 0))
d.edge(sim, dc, p0=(W / 2, ysim + 78), p1=(W / 2, ydc), label="$S_k$", loff=(12, 0))
d.edge(ex, dc, p0=(M + IW, yex + 54), p1=(M + IW, ydc + 56), via=[(W - 9, yex + 54), (W - 9, ydc + 56)], lw=0.8); d.text(W - 3.5, (yex + 54 + ydc + 56) / 2, "$\\mathcal{F}_{DSS,k}$", fs=5.4, rot=90, ha="center")
d.edge(dc, sim, p0=(M, ydc + 56), p1=(M, ysim + 57), via=[(8, ydc + 56), (8, ysim + 57)], lw=0.8); d.text(3.5, (ydc + 56 + ysim + 57) / 2, "$U_{DSS,k} \\in \\mathcal{F}_{in,k}$", fs=5.2, rot=90, ha="center")
d.edge(sim, ui, p0=(M, ysim + 23), p1=(M, 22), via=[(8, ysim + 23), (8, 22)], lw=0.6, dashed=True); d.text(3.5, (ysim + 23 + 22) / 2, "$S_k$", fs=5.0, rot=90, ha="center", color="#555")
d.render(f"{OUT}/fig1_simcore.png", scale=1 / 80); pages.append(("Fig1 simulation core", d))

# =====================================================================
# Fig. 2  Decision core (one local agent), compact
# =====================================================================
d = Diagram(W, 440, font=6.0)
y = 3
h1 = d.box(M, y, IW, 48, "", kind="ext"); d.text(W / 2, y + 7, "Heterogeneous sources", bold=True, color=GREY, ha="center")
items = ["weather stations,\nforecasts", "satellite, UAV,\ncamera imagery", "terrain, fuel,\nvegetation maps", "roads, access\nnetwork",
         "crews, capacity,\nsuppression sources", "assets, population,\nvalue at risk", "field reports,\nhot spots", "operator inputs,\nscenario settings"]
cw = (IW - 8 - 3 * 4) / 4
chips(d, M + 4, y + 13, items, 4, cw, 14, gap=3, fs=4.5)
yh = y; y += 48 + G
f1 = d.box(M, y, IW, 66, "", kind="ext"); d.text(W / 2, y + 7, "Observation $O_k^{i}$ and features $z^{i}_{f,k}$", bold=True, color=GREY, ha="center")
d.text(W / 2, y + 16, "confidence $\\gamma_k = \\min\\{\\gamma_{obs}, \\gamma_{cov}, \\gamma_{fre}, \\gamma_{rel}\\}$, missing data carried by persistence", fs=4.7, italic=True, color="#555", ha="center")
feats = ["$z_1$ fire\nintensity", "$z_2$ spread\npotential", "$z_3$ weather\nseverity", "$z_4$ ignition\nproximity", "$z_5$ fuel\nload",
         "$z_6$ asset\nexposure", "$z_7$ resource\naccessibility", "$z_8$ access,\nroad status", "$z_9$ suppression\navailability", "$z_{10}$ temporal\nurgency"]
cw5 = (IW - 8 - 4 * 3) / 5
chips(d, M + 4, y + 21, feats, 5, cw5, 17, gap=3, fs=4.4)
d.text(W / 2, y + 61, "$z^{i}_{f,k} \\in [0,1]$, fuzzified over five terms $\\mu_{A}(z)$", fs=4.7, italic=True, color="#555", ha="center")
yf = y; y += 66 + G
c0 = d.box(M, y, IW, 78, "", kind="ext"); d.text(W / 2, y + 7, "Concept space: weighted aggregation with confidence gating", bold=True, color=GREY, ha="center")
base = ["$c^{(1)}_1$ fire\nseverity", "$c^{(1)}_2$ spread\nhazard", "$c^{(1)}_3$ fuel\nhazard", "$c^{(1)}_4$ asset\nvalue", "$c^{(1)}_5$ crew\nreachability", "$c^{(1)}_6$ logistics\nsupport"]
cw6 = (IW - 8 - 5 * 3) / 6
B = chips(d, M + 4, y + 13, base, 6, cw6, 16, gap=3, fs=4.3)
d.text(M + IW - 4, y + 35, "6 base concepts $a^{(1)}_{n,k}$", fs=4.6, color="#555", ha="right")
decc = ["$c^{(2)}_1$ fire\nthreat level", "$c^{(2)}_3$ suppression\nfeasibility", "$c^{(2)}_2$ asset\nexposure risk", "$c^{(3)}_1$ intervention\nurgency", "$c^{(3)}_2$ evacuation\npressure"]
D = chips(d, M + 4, y + 41, decc, 5, cw5, 18, gap=3, kind="agent", fs=4.4)
d.text(M + 4, y + 69, "5 decision concepts $a^{eff}_{n,k}$", fs=4.6, color=GREEN, ha="left")
coord = d.box(M + IW - 122, y + 63, 118, 11, "$c^{(4)}_1$ operational priority $p_i$, to the coordinator", kind="plain", fs=4.1)
for xx in (M + IW * 0.2, M + IW * 0.4, M + IW * 0.6):
    d.edge(B[0], D[0], p0=(xx, y + 29), p1=(xx, y + 41), lw=0.7)
yc = y; y += 78 + G
L1 = d.box(M, y, IW, 80, "", kind="closed"); d.text(W / 2, y + 7, "Decision layer 1, closed space: self-tuning TS doctrine, Algorithm 1", bold=True, color=DBLUE, ha="center")
r1 = d.box(M + 4, y + 13, IW / 2 - 6, 28, "sparse rule base $R$\nIF $a^{eff}_{n,k}$ is $A_{r,n}$ ... THEN $u_k = v_r$\nfive seed rules, constant consequents", kind="plain", fs=4.6)
r2 = d.box(M + IW / 2 + 2, y + 13, IW / 2 - 6, 28, "candidate $u^{i}_{m,k}$ over six families\nsuppression, deployment, containment,\nasset protection, evacuation, warning", kind="plain", fs=4.6)
r3 = d.box(M + 4, y + 47, IW / 2 - 6, 30, "shadow forecast $\\hat{J}_k$ on the core\nsatisficing bound, quality $Q_k \\geq \\eta$,\nelse fail-safe factor $\\lambda_k$", kind="plain", fs=4.6)
r4 = d.box(M + IW / 2 + 2, y + 47, IW / 2 - 6, 30, "parametric adaptation\nconsequent step, boundary shift,\nrule instantiation, term insertion", kind="plain", fs=4.6)
d.edge(r1, r2, src="e", dst="w"); d.edge(r2, r4, p0=(M + IW / 2 + 2 + (IW / 2 - 6) / 2, y + 41), p1=(M + IW / 2 + 2 + (IW / 2 - 6) / 2, y + 47))
d.edge(r1, r3, p0=(M + 4 + (IW / 2 - 6) / 2, y + 41), p1=(M + 4 + (IW / 2 - 6) / 2, y + 47))
d.edge(r3, r4, src="e", dst="w", dashed=True, lw=0.7)
y1 = y; y += 80 + G + 5
L2 = d.box(M, y, IW, 76, "", kind="open"); d.text(W / 2, y + 7, "Decision layer 2, open space: generative extension, Algorithm 2", bold=True, color=RED, ha="center")
g1 = d.box(M + 4, y + 13, IW / 2 - 6, 28, "proposer: pretrained LLM, in context only\nreceives $a^{eff}_{n,k}$, firing rules, deficit,\neffect and sector vocabulary", kind="plain", fs=4.6)
g2 = d.box(M + IW / 2 + 2, y + 13, IW / 2 - 6, 28, "one object $p$ per call\nrule, macro $\\beta$, or composed\nintervention $A$ from verified effects", kind="plain", fs=4.6)
g3 = d.box(M + 4, y + 47, IW - 8, 14, "gates G1 to G4 algebraic, G5 to G8 paired forecast on seeds $\\xi_1, \\xi_2$", kind="gate", fs=4.8)
d.text(M + 4, y + 68, "admitted $p$ joins $R$; a rejected $p$ returns with the gate name, one revision", fs=4.6, italic=True, color=RED, ha="left")
d.edge(g1, g2, src="e", dst="w"); d.edge(g2, g3, p0=(M + IW / 2 + 2 + (IW / 2 - 6) / 2, y + 41), p1=(M + IW / 2 + 2 + (IW / 2 - 6) / 2, y + 47))
y2 = y; y += 76 + G + 4
co = d.box(M, y, IW, 20, "Coordinator, no inference: attention share $\\sigma_i$ by $p_i$,\npool allocation, acceptance thresholds, no-harm guard", kind="agent", fs=4.8)
yco = y; y += 20 + G
out = d.box(M, y, IW, 15, "$U_{DSS,k} = \\mathcal{A}_{DSS}(\\{\\sigma_i U^{i}_{DSS,k}\\})$ → simulation core → $O_{k+1}^{i}$", kind="core", fs=5.0, bold=True)
d.edge(h1, f1, src="s", dst="n"); d.edge(f1, c0, src="s", dst="n"); d.edge(c0, L1, src="s", dst="n", label="$a^{eff}_{n,k}$", loff=(16, 0))
d.edge(L1, L2, src="s", dst="n", dashed=True, label="standing decision falls short", loff=(50, 0))
d.edge(L2, L1, p0=(M + IW, y2 + 54), p1=(M + IW, y1 + 27), via=[(W - 8, y2 + 54), (W - 8, y1 + 27)], lw=0.8); d.text(W - 2.5, (y2 + 54 + y1 + 27) / 2, "admitted $p$", fs=4.8, rot=90, ha="center", color=RED)
d.edge(L1, co, p0=(M, y1 + 62), p1=(M, yco + 10), via=[(7, y1 + 62), (7, yco + 10)], lw=0.8); d.text(2.5, (y1 + 62 + yco + 10) / 2, "candidate $u^{i}_{k}$", fs=4.8, rot=90, ha="center", color=DBLUE)
d.edge(co, out, src="s", dst="n")
d.render(f"{OUT}/fig2_decisioncore.png", scale=1 / 80); pages.append(("Fig2 decision core", d))

if __name__ == "__main__":
    drawio_file(pages, f"{OUT}/v23_blocks.drawio")
    print("done")
