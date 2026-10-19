"""Block diagrams for the v6 article. Renders PNGs and one .drawio with four pages."""
import os
from diag import Diagram, drawio_file

OUT = "/home/claude/v6/fig"
os.makedirs(OUT, exist_ok=True)
pages = []

# =====================================================================
# Fig. 1  System of systems: two cores, participants, interfaces (double column)
# =====================================================================
d = Diagram(690, 300, font=6.6)
# external providers
ext = d.box(10, 70, 110, 160, "", kind="ext")
d.text(65, 82, "External providers", bold=True, color="#595959", ha="center")
for i, t in enumerate(["weather stations,\nforecasts (ERA5)", "satellite and\nairborne imagery", "terrain, fuel, assets,\nroads (GIS)", "resource status,\nfield reports"]):
    d.box(18, 94 + i * 33, 94, 28, t, kind="plain", fs=5.8)
# simulation core
sim = d.box(140, 40, 190, 220, "", kind="core")
d.text(235, 52, "Simulation core (authoritative)", bold=True, color="#2F5597", ha="center")
s1 = d.box(150, 64, 170, 34, "Grid state $S_k$: burning status, fuel mass,\nintensity proxy, time since ignition", kind="plain", fs=5.8)
s2 = d.box(150, 106, 170, 40, "Hybrid spread $\\Phi$: persistence, 8-neighbour\npropagation (Rothermel rate), seeded\nember spotting, moisture and precipitation", kind="plain", fs=5.8)
s3 = d.box(150, 154, 170, 34, "Suppression and effect fields: wet, clear,\nignite, coat, draft, evacuate, prime", kind="plain", fs=5.8)
s4 = d.box(150, 196, 170, 26, "Observation function $h$: confidence-rated\nobservations only", kind="plain", fs=5.8)
s5 = d.box(150, 228, 170, 24, "Snapshots and counterfactual replay", kind="plain", fs=5.8)
# decision core
dec = d.box(360, 40, 250, 220, "", kind="agent")
d.text(485, 52, "Decision core (N regional agents)", bold=True, color="#548235", ha="center")
a1 = d.box(370, 64, 230, 24, "Perception: 10 features, each with a confidence", kind="plain", fs=5.8)
a2 = d.box(370, 94, 230, 24, "Concept space: 6 base, 5 decision, 1 coordination concept", kind="plain", fs=5.8)
l1 = d.box(370, 124, 230, 40, "Decision layer 1 (closed): TS rule base, satisficing\nacceptance, quality gate, fail-safe;\nparametric self-tuning of rules and terms", kind="closed", fs=5.8)
l2 = d.box(370, 170, 230, 40, "Decision layer 2 (open): language-model proposer\nwrites rules, blended and composed\ninterventions; admission by gates G1 to G8", kind="open", fs=5.8)
a5 = d.box(370, 216, 230, 36, "Coordinator (no inference): ranks regions, shares\nthe pool, sets acceptance thresholds; no-harm guard", kind="plain", fs=5.8)
# commander
cmd = d.box(630, 90, 52, 120, "Operator\n\ndashboard,\nreversible\ninputs:\npause,\nrollback,\nresources", kind="cmd", fs=5.6)
# ledger
led = d.box(360, 268, 250, 24, "Traceability ledger: concepts, rules, strengths, confidences, gate verdicts, proposals", kind="gate", fs=5.6)
# edges
d.edge(ext, sim, label="inputs $F_{in}$", src="e", dst="w")
d.edge(sim, dec, label="observations $O_k,\\ \\gamma$", p0=(330, 100), p1=(360, 100), loff=(0, -9))
d.edge(dec, sim, label="field $U_{DSS}$", p0=(360, 210), p1=(330, 210), loff=(0, 9))
d.edge(cmd, dec, p0=(630, 150), p1=(610, 150), dashed=True, label="supervision", loff=(0, -9))
d.edge(dec, led, src="s", dst="n", dashed=True)
d.text(165, 280, "Only the simulation core writes the hazard state; the decision core\nreads it through $h$ and acts only through $U_{DSS}$.", fs=5.4, italic=True, color="#555", ha="center")
d.render(f"{OUT}/fig1_architecture.png"); pages.append(("Fig1 architecture", d))

# =====================================================================
# Fig. 2  Local agent pipeline with the two decision layers (single column, tall)
# =====================================================================
d = Diagram(362, 560, font=6.4)
p1 = d.box(10, 8, 316, 44, "", kind="ext"); d.text(168, 18, "Input and perception", bold=True, color="#595959", ha="center")
d.box(18, 26, 300, 20, "10 bounded features $z_f \\in [0,1]$, confidence $\\gamma_f = \\min(\\mathrm{obs}, \\mathrm{cover}, \\mathrm{fresh}, \\mathrm{reliab})$", kind="plain", fs=5.6)
p2 = d.box(10, 62, 316, 74, "", kind="ext"); d.text(168, 72, "Concept space", bold=True, color="#595959", ha="center")
c1 = d.box(18, 82, 92, 44, "6 base concepts\nfire severity, spread hazard,\nfuel hazard, asset value,\ncrew reachability, logistics", kind="plain", fs=5.2)
c2 = d.box(122, 82, 110, 44, "5 decision concepts\nfire threat level, suppression\nfeasibility, asset exposure risk,\nintervention urgency, evacuation pressure", kind="plain", fs=5.2)
c3 = d.box(244, 82, 74, 44, "coordination\nconcept:\noperational priority\n(not in rules)", kind="plain", fs=5.2)
d.edge(c1, c2, src="e", dst="w"); d.edge(c2, c3, src="e", dst="w")
d.text(168, 131, "gating: $a_{eff} = \\gamma\\, a_{obs} + (1-\\gamma)\\,\\rho\\, a_{eff}(k-1)$", fs=5.4, italic=True, color="#555", ha="center")
# layer 1
L1 = d.box(10, 146, 316, 150, "", kind="closed"); d.text(168, 156, "Decision layer 1: closed decision space (self-tuning doctrine)", bold=True, color="#1F3864", ha="center")
r1 = d.box(18, 166, 140, 40, "Sparse TS rule base $R$\nseed: 5 rules over 5 concepts\n$\\mu_r$ = min of clause memberships", kind="plain", fs=5.4)
r2 = d.box(176, 166, 142, 40, "Candidate $u_f = \\sum_r \\mu_r v_{rf} / \\max(1, \\sum_r \\mu_r)$\nover 6 families; inaction = 0", kind="plain", fs=5.4)
r3 = d.box(18, 214, 140, 36, "Shadow forecast on the core:\n$\\hat J \\leq \\min(J_{TH}, (1-\\delta)\\,\\hat J(0))$  (satisficing)\n$Q \\geq \\eta$ (quality), else fail-safe", kind="plain", fs=5.4)
r4 = d.box(176, 214, 142, 36, "Parametric adaptation (evolving fuzzy):\nconsequent step $\\pm 0.05$, boundary shift,\nresolution rule on an uncovered cell", kind="plain", fs=5.4)
d.text(168, 262, "trigger: forecast deficit $\\hat J > J_{TH}$, coverage gap $\\sum_r \\mu_r < \\eta_c$, or renewed spread", fs=5.4, italic=True, color="#1F3864", ha="center")
d.text(168, 280, "every trial on a working copy, kept only if $\\Delta \\hat J_{phys} < 0$", fs=5.4, italic=True, color="#1F3864", ha="center")
d.edge(r1, r2, src="e", dst="w"); d.edge(r1, r3, src="s", dst="n"); d.edge(r2, r4, src="s", dst="n", dashed=True, label="tuned", lw=0.7, loff=(14, 0)); d.edge(r3, r4, src="e", dst="w", dashed=True, label="deficit")
# layer 2
L2 = d.box(10, 306, 316, 170, "", kind="open"); d.text(168, 316, "Decision layer 2: open decision space (generative extension)", bold=True, color="#C00000", ha="center")
g1 = d.box(18, 326, 140, 46, "Proposer: pretrained language model,\nin context only; receives concept\nactivations, firing rules, deficit,\nsector and band vocabulary", kind="plain", fs=5.4)
g2 = d.box(176, 326, 142, 46, "One object per call:\nrule $(T, v)$\nmacro $\\beta$ over families\nactuator $\\{(e, s, [\\rho_{in}, \\rho_{out}], \\alpha)\\}$", kind="plain", fs=5.4)
g3 = d.box(18, 382, 300, 30, "Gates: G1 form · G2 vocabulary, duplicate · G3 relevance · G4 pool\nG5 forecast $\\xi_1$ · G6 forecast $\\xi_2$ · G7 package margin $\\delta$ · G8 attribution", kind="gate", fs=5.4)
g4 = d.box(18, 420, 140, 26, "admitted: $R \\leftarrow R \\cup \\{p\\}$, joins layer 1\nlogged with gate values", kind="plain", fs=5.4)
g5 = d.box(176, 420, 142, 26, "rejected: gate name and both\nforecasts returned, one revision", kind="plain", fs=5.4)
d.edge(g1, g2, src="e", dst="w"); d.edge(g2, g3, src="s", dst="n"); d.edge(g3, g4, src="s", dst="n", via=[(88, 416)]); d.edge(g3, g5, src="s", dst="n", via=[(247, 416)])
d.edge(g5, g1, p0=(247, 446), p1=(18, 349), dashed=True, via=[(247, 456), (13, 456), (13, 349)], label="revise once", lpos=0.3, loff=(0, 6))
d.text(168, 470, "closed at effects, open at interventions: the basis is never proposable", fs=5.4, italic=True, color="#C00000", ha="center")
# bottom
b1 = d.box(10, 486, 316, 30, "Coordinator: attention share by operational priority, pool allocation, no-harm guard;\nlife-safety families (evacuation, warning) never attenuated", kind="agent", fs=5.4)
b2 = d.box(10, 524, 316, 26, "Fielded intervention field $U_{DSS}$ → simulation core → next observation", kind="core", fs=5.6, bold=True)
d.edge(p1, p2, src="s", dst="n"); d.edge(p2, L1, src="s", dst="n"); d.edge(L1, L2, src="s", dst="n", label="standing decision falls short", dashed=True)
d.edge(L1, b1, p0=(326, 230), p1=(326, 486), via=[(346, 230), (346, 480), (326, 480)], label="candidate", lpos=0.5, loff=(0, 0))
d.edge(b1, b2, src="s", dst="n")
d.render(f"{OUT}/fig2_pipeline.png"); pages.append(("Fig2 agent pipeline", d))

# =====================================================================
# Fig. 3  Gate chain as an acceptance function (single column)
# =====================================================================
d = Diagram(336, 250, font=6.2)
pr = d.box(10, 12, 70, 36, "proposal $p$\nfrom layer 2", kind="open", bold=True)
alg = d.box(92, 6, 234, 86, "", kind="gate", dashed=True); d.text(209, 14, "algebraic gates (no simulation step)", bold=True, color="#444", ha="center")
G = []
labels = [("G1", "canonical\nform"), ("G2", "vocabulary,\nranges, no\nduplicate"), ("G3", "fires now\n$\\mu_p(a) > 0$"), ("G4", "pool\neconomy")]
for i, (g, t) in enumerate(labels):
    G.append(d.box(98 + i * 57, 24, 52, 60, f"{g}\n{t}", kind="plain", fs=5.2))
sim = d.box(92, 104, 234, 92, "", kind="core", dashed=True); d.text(209, 112, "simulation gates (shadow forecast, horizon $H$)", bold=True, color="#2F5597", ha="center")
S = []
labels2 = [("G5", "$\\hat J_{\\xi_1}(R \\cup p)$\n$< \\hat J_{\\xi_1}(R)$"), ("G6", "$\\hat J_{\\xi_2}(R \\cup p)$\n$< \\hat J_{\\xi_2}(R)$\nsecond seed"), ("G7", "package:\n$\\Delta \\hat J \\leq -\\delta$"), ("G8", "package beats\nitself without\nthe new object")]
for i, (g, t) in enumerate(labels2):
    S.append(d.box(98 + i * 56, 124, 50, 64, f"{g}\n{t}", kind="plain", fs=5.2))
adm = d.box(230, 208, 96, 32, "admitted\n$R \\leftarrow R \\cup \\{p\\}$, ledger entry", kind="agent", bold=True, fs=5.6)
rej = d.box(10, 208, 150, 32, "rejected: gate name, $\\hat J$ values →\nproposer, one revision, then drop", kind="open", fs=5.4)
d.edge(pr, G[0], src="e", dst="w")
for a, b in zip(G[:-1], G[1:]): d.edge(a, b, src="e", dst="w")
d.edge(G[-1], S[0], src="s", dst="n", via=[(295, 96), (123, 96)])
for a, b in zip(S[:-1], S[1:]): d.edge(a, b, src="e", dst="w")
d.edge(S[-1], adm, src="s", dst="n")
d.edge(alg, rej, p0=(100, 92), p1=(60, 208), dashed=True, lw=0.7, via=[(100, 200), (60, 200)])
d.edge(sim, rej, p0=(100, 196), p1=(100, 208), dashed=True, lw=0.7)
d.text(150, 202, "any gate fails", fs=5.2, color="#C00000", ha="center")
d.render(f"{OUT}/fig3_gates.png"); pages.append(("Fig3 gate chain", d))

# =====================================================================
# Fig. 4  Effect grammar: clause -> geometry -> operator (single column)
# =====================================================================
d = Diagram(352, 236, font=6.0)
cl = d.box(10, 10, 160, 56, "clause $(e, s, [\\rho_{in}, \\rho_{out}], \\alpha)$\n$e$: wet, clear, ignite, coat,\n    draft, evacuate, prime\n$s$: head, flank, rear, ring,\n    at_fire, assets, populated", kind="open", fs=5.0, align="left")
geo = d.box(182, 10, 160, 56, "geometry\nband $B = D(\\rho_{out}) \\setminus D(\\rho_{in}-1)$ of core $C$\n$C$: burning cells, threatened\n    assets, or populated cells\nhead / flank / rear by $\\cos$ to wind", kind="core", fs=5.0, align="left")
msk = d.box(10, 80, 160, 46, "admissibility masks\nagent region; reach (wet, clear, ignite)\ndiggable, not built-up (clear, ignite)\npopulated (evacuate, prime)", kind="gate", fs=5.0, align="left")
op = d.box(182, 80, 160, 46, "operator on engine fields, $a = \\alpha u$\ncoat: $r_{ret} \\leftarrow \\max(r_{ret}, a)$\nwet, clear: $r_{cap}, r_{avail} \\leftarrow \\max(\\cdot, a)$\ndraft: $r_{cap} \\leftarrow r_{cap}(1 + \\beta(d_{water}))$", kind="core", fs=5.0, align="left")
pk = d.box(10, 140, 332, 28, "package = rule $r$ + actuator $A$ (up to 3 clauses), fired at the intensity $u$ of $r$;\nG7 margin and G8 attribution decide admission", kind="open", fs=5.2)
ex = d.box(10, 178, 332, 50, "admitted in S3, world 201, minute 2:\nIF asset exposure risk is VH AND suppression feasibility is VH THEN preemptive asset collar 0.9\n= {coat assets [0,3] 1.0; coat populated [0,2] 1.0; prime populated [0,2] 0.8}\n$\\hat J$ 0.1351 → 0.1284 on seed $\\xi_1$, same sign on $\\xi_2$, margin cleared", kind="plain", fs=5.0)
d.edge(cl, geo, src="e", dst="w"); d.edge(geo, msk, src="s", dst="n", via=[(262, 73), (90, 73)]); d.edge(msk, op, src="e", dst="w"); d.edge(op, pk, src="s", dst="n"); d.edge(pk, ex, src="s", dst="n", dashed=True)
d.render(f"{OUT}/fig4_grammar.png"); pages.append(("Fig4 effect grammar", d))

if __name__ == "__main__":
    drawio_file(pages, f"{OUT}/DisasterAware_Article_v6_Blocks.drawio")
print("done")
