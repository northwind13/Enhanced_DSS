import csv, glob, math, os, sys
import numpy as np

B = "/home/claude/enhanced_dss/03_Codes/experiments/benchmark"
F = os.path.join(B, "final"); R = os.path.join(B, "revision")
sys.path.insert(0, R)
from revision_tables import load, paired  # noqa

rng = np.random.default_rng(0)
NPERM = 1_000_000


def signflip(d):
    d = np.asarray(d, float); n = len(d); obs = abs(d.mean())
    cnt = 0; B_ = 20000
    for s in range(0, NPERM, B_):
        m = min(B_, NPERM - s)
        signs = rng.choice([-1.0, 1.0], size=(m, n))
        cnt += int((np.abs(signs @ d) / n >= obs - 1e-12).sum())
    return (cnt + 1) / (NPERM + 1)


def P(path, **f): return load(path, **f)

closed = P(os.path.join(F, "t1_reference", "runs.csv"), arm="F5Ev")
voda = P(os.path.join(F, "t6_voda", "runs.csv"), arm="LLMScreen")
k9 = os.path.join(R, "k9_novelty", "runs.csv")
k2b = P(os.path.join(R, "k2b_generalization", "runs.csv"), mode="canon")
sub = lambda rows, ks: {k: v for k, v in rows.items() if k in ks}
cl75 = sub(closed, set(k2b))

rows = []  # (table, label, printed, cfg, ref, src_cfg, src_ref)
def add(t, lab, pr, cfg, ref, sc, sr): rows.append((t, lab, pr, cfg, ref, sc, sr))

T7 = [
 ("Random, one", -9, "t4_random_single", "RandBlind"),
 ("Random, one, ungated", -10, "rev:k15_random_ungated", "RandBlind-ungated"),
 ("LLM, one", -28, "t7_llm_single", "LLM"),
 ("LLM, one, ungated", -25, "rev:k12u_ungated", "LLM-ungated"),
 ("Random, small", -32, "t5b_random_screened_small", "Random"),
 ("LLM, candidates only", -45, "t6b_voda_model_only", "LLMScreen"),
 ("Random, full", -46, "t5_random_screened", "Random"),
 ("Random, families only", -41, "rev:k13_closed_vocab", "Random-fam"),
 ("Random, with variations", -34, "rev:k14_random_var", "RandomVar"),
 ("Hand-written seeds, with variations", -31, "rev:k12_doctrine_seeded", "ExpertVar"),
 ("Evolutionary search", -49, "rev:k16_evo", "Evo"),
 ("GODA", -61, "t6_voda", "LLMScreen"),
 ("GODA repeat", -57, "t9_repeat", "LLMScreen"),
 ("Independent gates", -54, "rev:k8_gate_seeds", "LLMScreen"),
]
for lab, pr, fold, arm in T7:
    pth = os.path.join(R, fold[4:], "runs.csv") if fold.startswith("rev:") else os.path.join(F, fold, "runs.csv")
    add("VII", lab, pr, P(pth, arm=arm), closed, f"{fold}:{arm}", "t1_reference:F5Ev")

k3 = os.path.join(R, "k3_model_error*", "runs.csv")
for lab, pr, pert in [("Suppression intensity error (replay)", -56, "gammaI_x1.5"),
                      ("Travel-effect decay error (replay)", -54, "betat_x1.5"),
                      ("Knockdown threshold error (replay)", -57, "knock_x1.5")]:
    add("VII", lab, pr, P(k3, perturb=pert, arm="voda"), P(k3, perturb=pert, arm="closed"),
        f"k3_model_error:{pert}:voda", f"k3_model_error:{pert}:closed")
k6 = os.path.join(R, "k6_coating", "runs.csv"); k6nc = os.path.join(R, "k6_coating_nc", "runs.csv")
for lab, pr, pert in [("Weakened coating", -54, "coat_half"), ("Ineffective coating", -33, "coat_off"),
                      ("Coating price x2", -60, "coat_price_x2"), ("Coating stock 20 ha", -53, "coat_stock_20ha"),
                      ("Coating delay 12 min", -50, "coat_delay_12min")]:
    add("VII", lab, pr, P(k6nc, perturb=pert, arm="live"), P(k6, perturb=pert, arm="closed"),
        f"k6_coating_nc:{pert}:live", f"k6_coating:{pert}:closed")
k7 = os.path.join(R, "k7_spread", "runs.csv"); k7nc = os.path.join(R, "k7_spread_nc", "runs.csv")
for lab, pr, pert in [("Wind speed x1.5", -44, "wind_x1.5"), ("Wind direction +30 deg", -58, "wind_rot_p30"),
                      ("Wind direction -30 deg", -63, "wind_rot_m30")]:
    add("VII", lab, pr, P(k7nc, perturb=pert, arm="live"), P(k7, perturb=pert, arm="closed"),
        f"k7_spread_nc:{pert}:live", f"k7_spread:{pert}:closed")
k17 = os.path.join(R, "k17_kernel", "runs.csv")
add("VII", "Spread kernel and ember model", -56, P(k17, arm="live"), P(k17, arm="closed"),
    "k17_kernel:live", "k17_kernel:closed")

# revision round 2 rows (Table VII)
k21 = os.path.join(R, "k21_budget")
add("VII", "Random, 12 forecasts", -36, P(os.path.join(k21, "random_b12", "runs.csv"), arm="Random"), closed, "k21_budget/random_b12:Random", "t1_reference:F5Ev")
add("VII", "Random, 24 forecasts", -49, P(os.path.join(k21, "random_b24", "runs.csv"), arm="Random"), closed, "k21_budget/random_b24:Random", "t1_reference:F5Ev")
for lab, pr, sub_ in [("GODA, 12 forecasts", None, "goda_b12"), ("GODA, 24 forecasts", None, "goda_b24")]:
    pth = os.path.join(k21, sub_, "runs.csv")
    g = P(pth, arm="LLMScreen") if os.path.exists(pth) else {}
    if len(g) == 150:
        add("VII", lab, pr, g, closed, f"k21_budget/{sub_}:LLMScreen", "t1_reference:F5Ev")
k18 = os.path.join(R, "k18_combined", "runs.csv")
add("VII", "Kernel, wind and coating errors combined", -30, P(k18, arm="live"), P(k18, arm="closed"), "k18_combined:live", "k18_combined:closed")
k19 = os.path.join(R, "k19_perimeter", "runs.csv")
add("VII", "Perturbed fire state, 25% unseen", -60, P(k19, arm="live"), P(k19, arm="closed"), "k19_perimeter:live", "k19_perimeter:closed")
k22 = os.path.join(R, "k22_domain", "runs.csv")
add("VII", "Domain 3.6 x 2.7 km, 30 worlds", -57, P(k22, arm="live"), P(k22, arm="closed"), "k22_domain:live", "k22_domain:closed")

k23 = os.path.join(R, "k23_bref_live", "runs.csv")
if os.path.exists(k23) and len(P(k23, arm="live")) == 150:
    add("VII", "Reference burned fraction 0.25, live", -57, P(k23, arm="live"), P(k23, arm="closed"), "k23_bref_live:live", "k23_bref_live:closed")

k25 = os.path.join(R, "k25_one_future", "runs.csv")
if os.path.exists(k25) and len(P(k25, arm="live")) == 150:
    add("VII", "Gates on one future, G5 only", -55, P(k25, arm="live"), closed, "k25_one_future:live", "t1_reference:F5Ev")
k26 = os.path.join(R, "k26_parity", "runs.csv")
if os.path.exists(k26):
    if len(P(k26, arm="state")) == 150:
        add("VII", "Perturbed fire state, brief included", -55, P(k26, arm="state"), P(k19, arm="closed"), "k26_parity:state", "k19_perimeter:closed")
    if len(P(k26, arm="s5")) == 30:
        add("VII", "S5 with corrupted brief (30 worlds)", -53, P(k26, arm="s5"), sub(closed, set(P(k26, arm="s5"))), "k26_parity:s5", "t1_reference:F5Ev (S5)")
k27 = os.path.join(R, "k27_coat_parity", "runs.csv")
if os.path.exists(k27) and len(P(k27, arm="live")) == 150:
    add("VII", "Coating priced and reach-limited", -54, P(k27, arm="live"), P(k27, arm="closed"), "k27_coat_parity:live", "k27_coat_parity:closed")

k28 = os.path.join(R, "k28_no_persist", "runs.csv")
if os.path.exists(k28) and len(P(k28, arm="live")) == 150:
    add("VII", "Admitted objects withdrawn after one cycle", 2, P(k28, arm="live"), closed, "k28_no_persist:live", "t1_reference:F5Ev")

# Table VIII
add("VIII", "40 rules + Closed / Seeds + Closed", -11, P(os.path.join(F, "t13_full_doctrine", "runs.csv"), arm="F40Ev"),
    closed, "t13_full_doctrine:F40Ev", "t1_reference:F5Ev")
add("VIII", "Seeds + six hand-written tactics + Closed / Seeds + Closed", -9,
    P(os.path.join(R, "k11_expert_doctrine", "runs.csv"), mode="expert"), closed, "k11_expert_doctrine:expert", "t1_reference:F5Ev")
add("VIII", "Recorded + Closed / Seeds + Closed (75)", -50, k2b, cl75, "k2b_generalization:canon", "t1_reference:F5Ev (seeds 216-230)")
add("VIII", "Seeds (no deployment) + Closed / Seeds + Closed", 40,
    P(os.path.join(F, "t12_closed_resource_deployment", "runs.csv"), arm="F5Ev-nofam"), closed,
    "t12_closed_resource_deployment:F5Ev-nofam", "t1_reference:F5Ev")
add("VIII", "Seeds (no deployment) + Open / Seeds + Closed", -42,
    P(os.path.join(F, "t12_open_resource_deployment", "runs.csv"), arm="LLMScreen-nofam"), closed,
    "t12_open_resource_deployment:LLMScreen-nofam", "t1_reference:F5Ev")
A = P(k9, arm="A", set="novel"); Bn = P(k9, arm="B", set="novel"); Cn = P(k9, arm="C", set="novel")
Dn = P(k9, arm="D", set="novel"); Cc = P(k9, arm="C", set="control")
add("VIII", "S6-S8 Recorded + Closed / Seeds + Closed", -43, Bn, A, "k9_novelty:B:novel", "k9_novelty:A:novel")
add("VIII", "S6-S8 Seeds + Open / Recorded + Closed", -1, Dn, Bn, "k9_novelty:D:novel", "k9_novelty:B:novel")
add("VIII", "S6-S8 Recorded + Open / Recorded + Closed", -28, Cn, Bn, "k9_novelty:C:novel", "k9_novelty:B:novel")
add("VIII", "S1-S5 control Recorded + Open / Recorded + Closed", -35, Cc, k2b, "k9_novelty:C:control", "k2b_generalization:canon")
add("VIII", "S6-S8 Recorded + Open / Seeds + Open", -27, Cn, Dn, "k9_novelty:C:novel", "k9_novelty:D:novel")

out = []
for t, lab, pr, cfg, ref, sc, sr in rows:
    p = paired(ref, cfg)
    ks = sorted(set(ref) & set(cfg))
    d = [float(cfg[k]["burned_ha"]) - float(ref[k]["burned_ha"]) for k in ks]
    praw = signflip(d)
    out.append(dict(table=t, row_label=lab, n=p["n"], mean_ref_ha=round(p["mean_a"], 3), mean_cfg_ha=round(p["mean_b"], 3),
                    rel_change_pct=round(p["rel"], 2), hw=p["hw"], better=p["better"], tie=p["tie"], worse=p["worse"],
                    p_raw=praw, printed=pr, source_cfg=sc, source_ref=sr))
# Holm
m = len(out); order = sorted(range(m), key=lambda i: out[i]["p_raw"])
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * out[i]["p_raw"])); out[i]["p_holm"] = run
os.makedirs("/tmp/w4", exist_ok=True)
cols = ["table", "row_label", "n", "mean_ref_ha", "mean_cfg_ha", "rel_change_pct", "better", "tie", "worse", "p_raw", "p_holm", "source_cfg", "source_ref"]
with open(os.environ.get("HOLM_OUT", "/tmp/w4/holm.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for o in out: w.writerow({**o, "p_raw": f"{o['p_raw']:.6g}", "p_holm": f"{o['p_holm']:.6g}"})
print("| Table | Row | n | Rel. change (%) | b/t/w | p_raw | p_holm |")
print("|---|---|---|---|---|---|---|")
for o in out:
    flag = "" if o["printed"] is None or abs(round(o["rel_change_pct"]) - o["printed"]) <= 1 else f" (printed {o['printed']:+d})"
    print(f"| {o['table']} | {o['row_label']} | {o['n']} | {o['rel_change_pct']:+.0f} ± {o['hw']:.0f}{flag} | {o['better']}/{o['tie']}/{o['worse']} | {o['p_raw']:.2g} | {o['p_holm']:.2g} |")
print("\nMISMATCH:", [(o["row_label"], round(o["rel_change_pct"]), o["printed"]) for o in out if o["printed"] is not None and abs(round(o["rel_change_pct"]) - o["printed"]) > 1])
print("p_holm>=0.05:", [(o["row_label"], round(o["p_holm"], 3)) for o in out if o["p_holm"] >= 0.05])
