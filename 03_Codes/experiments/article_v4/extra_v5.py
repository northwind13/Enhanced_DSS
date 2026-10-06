"""Extra placeholders and Fig. 12 for the v5 article (language-model benchmark).

    python experiments/article_v4/extra_v5.py --bench experiments/benchmark/out \
        --results experiments/article_v4/out_llm

Writes <results>/extra_fill.json (stage-1/2/3 counts, gate verdicts, admission
by proposal kind, model calls and wall time) and <results>/figures/fig12_gates.png
(fate of the stage-3 proposals per scenario family, sized for the Fig. 12 frame).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import results_v4 as R

SCEN = ["S1", "S2", "S3", "S4", "S5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    runs = defaultdict(dict)
    with open(os.path.join(args.bench, "runs.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario"] in SCEN and r["arm"] in ("T0", "F5", "F5Ev", "LLM"):
                runs[(r["scenario"], r["arm"])][int(r["seed"])] = r
    seeds = {sc: sorted(set.intersection(*[set(runs[(sc, a)]) for a in ("T0", "F5", "F5Ev", "LLM")])) for sc in SCEN}
    rows = {a: [runs[(sc, a)][s] for sc in SCEN for s in seeds[sc]] for a in ("F5Ev", "LLM")}

    def tot(a, k):
        return int(round(sum(float(x[k]) for x in rows[a])))

    def mean(a, k):
        return float(np.mean([float(x[k]) for x in rows[a]]))

    F = {}
    for a, tag in (("F5Ev", "F5Ev"), ("LLM", "DA")):
        F[f"st1.{tag}"] = f"{tot(a,'acc_1')} of {tot(a,'tried_1')}"
        F[f"st2.{tag}"] = f"{tot(a,'acc_2')} of {tot(a,'tried_2')}"
        F[f"st1pct.{tag}"] = f"{100*tot(a,'acc_1')/max(1,tot(a,'tried_1')):.0f}"
        F[f"st2pct.{tag}"] = f"{100*tot(a,'acc_2')/max(1,tot(a,'tried_2')):.0f}"
        F[f"wall.{tag}"] = f"{mean(a,'wall_s'):.0f}"
        F[f"calls.{tag}"] = f"{mean(a,'prop_calls'):.1f}"
        F[f"sited.{tag}"] = f"{mean(a,'sited_orders'):.0f}"
        F[f"macros.{tag}"] = f"{mean(a,'n_macros'):.1f}"
        F[f"actuators.{tag}"] = f"{mean(a,'n_actuators'):.1f}"
    F["adm3pct.DA"] = f"{100*tot('LLM','acc_3')/max(1,tot('LLM','tried_3')):.0f}"
    F["wall_ratio"] = f"{mean('LLM','wall_s')/mean('F5Ev','wall_s'):.1f}"
    F["NWORLDS"] = str(sum(len(s) for s in seeds.values()))

    # proposals
    props = []
    with open(os.path.join(args.bench, "proposals.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("arm") == "LLM" and d.get("scenario") in SCEN and d.get("seed") in seeds[d["scenario"]]:
                props.append(d)
    cnt = {sc: defaultdict(int) for sc in SCEN}
    kinds = defaultdict(lambda: [0, 0])
    for d in props:
        g = d.get("gate") or ""
        k = "admitted" if d.get("accepted") else ("G5" if "G5" in g else "G4" if "G4" in g else "G3" if "G3" in g else "early")
        cnt[d["scenario"]][k] += 1
        kk = d.get("kind") or "rule"
        kinds[kk][1] += 1
        kinds[kk][0] += int(bool(d.get("accepted")))
    for k in ("admitted", "G5", "G4", "G3", "early"):
        F[f"rej.{k}"] = str(sum(cnt[sc][k] for sc in SCEN))
    for kk, (a, n) in kinds.items():
        F[f"adm.{kk}"] = f"{100*a/n:.0f}"
        F[f"n.{kk}"] = str(n)
        F[f"a.{kk}"] = str(a)
    F["nprops"] = str(len(props))

    os.makedirs(os.path.join(args.results, "figures"), exist_ok=True)
    json.dump(F, open(os.path.join(args.results, "extra_fill.json"), "w", encoding="utf-8"), indent=1)

    # Fig. 12: fate of the proposals per family, frame aspect 3.13 x 1.34 in
    cats = [("admitted", "admitted", R.COLOR["DA"]), ("G5", "rejected at G5 (package margin)", "#b5b5b5"),
            ("G4", "rejected at G4 (reseeded A/B)", "#8f8f8f"), ("G3", "rejected at G3 (no improvement)", "#656565"),
            ("early", "rejected at G1 or G2", "#3a3a3a")]
    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    x = np.arange(len(SCEN)); bottom = np.zeros(len(SCEN))
    for k, lab, col in cats:
        v = np.array([cnt[sc][k] for sc in SCEN], float)
        ax.bar(x, v, 0.62, bottom=bottom, color=col, edgecolor="white", linewidth=0.8, label=lab)
        bottom += v
    for xi, b, sc in zip(x, bottom, SCEN):
        ax.text(xi, b + 3, f"{cnt[sc]['admitted']}/{int(b)}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(SCEN, fontsize=7)
    ax.set_ylabel("stage-3 proposals", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylim(0, bottom.max() * 1.75)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=5.4, loc="upper left", ncol=2, frameon=False, handlelength=1.0, labelspacing=0.3, columnspacing=1.0, borderaxespad=0.2)
    fig.savefig(os.path.join(args.results, "figures", "fig12_gates.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(json.dumps(F, indent=1))


if __name__ == "__main__":
    main()
