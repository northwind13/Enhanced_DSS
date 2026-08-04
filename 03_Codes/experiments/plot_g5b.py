"""Two panels for the generative slice of Section 5.5.3.

(a) what stage ③ proposed and what the gates admitted, by object kind
(b) the gain each admitted clause actuator bought on the two reseeded
    rollouts that G5 measures

Both read geometry_campaign_g5bp8.csv and geometry_proposals_g5bp8.jsonl
and nothing else, so the figure cannot disagree with the table beside it.

    python experiments/plot_g5b.py
"""
from __future__ import annotations

import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                   "Enhanced_DSS", "01_Thesis", "figures")
if not os.path.isdir(FIG):
    FIG = os.path.join(os.path.dirname(HERE), "..", "01_Thesis",
                       "figures")
FIG = os.path.abspath(FIG)

SLICE = "g5bp8"
KINDS = [("rule", "Rule"), ("concept", "Intermediate\nconcept"),
         ("composite", "Macro\nintervention"),
         ("clause", "Clause\nactuator")]


def load():
    p = os.path.join(OUT, f"geometry_proposals_{SLICE}.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    props = load()
    prop = collections.Counter(r["kind"] for r in props)
    adm = collections.Counter(r["kind"] for r in props if r.get("accepted"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9))

    # ---- (a) proposed against admitted, by kind ---------------------
    xs = range(len(KINDS))
    w = 0.38
    pv = [prop.get(k, 0) for k, _ in KINDS]
    av = [adm.get(k, 0) for k, _ in KINDS]
    ax1.bar([x - w / 2 for x in xs], pv, w, label="proposed",
            color="#9fb8d4", edgecolor="#3c5a78")
    ax1.bar([x + w / 2 for x in xs], av, w, label="admitted",
            color="#3c5a78", edgecolor="#22384b")
    for x, (a, b) in enumerate(zip(pv, av)):
        ax1.text(x - w / 2, a + 0.2, str(a), ha="center", fontsize=9)
        ax1.text(x + w / 2, b + 0.2, str(b), ha="center", fontsize=9)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([n for _, n in KINDS], fontsize=9)
    ax1.set_ylabel("proposals")
    ax1.set_title("(a) Proposed and admitted, by object kind",
                  fontsize=10)
    ax1.legend(fontsize=9, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_ylim(0, max(pv) + 2)

    # ---- (b) the gain of each admitted actuator ---------------------
    rows = []
    for r in props:
        if r["kind"] != "clause" or not r.get("accepted"):
            continue
        ni = r["payload"]["new_intervention"]
        m = r["measured"]
        rows.append((ni["name"], int(r.get("step") or 0),
                     m["g3"]["j_without"] - m["g3"]["j_with"],
                     m["g4"]["j_without"] - m["g4"]["j_with"]))
    rows.sort(key=lambda t: t[1])
    ys = range(len(rows))
    h = 0.36
    ax2.barh([y + h / 2 for y in ys], [r[2] for r in rows], h,
             label="rollout 1 (seed a)", color="#9fb8d4",
             edgecolor="#3c5a78")
    ax2.barh([y - h / 2 for y in ys], [r[3] for r in rows], h,
             label="rollout 2 (seed b)", color="#3c5a78",
             edgecolor="#22384b")
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([f"{n}\n(t = {s} min)" for n, s, _a, _b in rows],
                        fontsize=8)
    ax2.axvline(1e-4, color="#b03a2e", lw=1.2, ls="--")
    ax2.text(1e-4, len(rows) - 0.4, "  G5 margin", color="#b03a2e",
             fontsize=8, va="top")
    ax2.set_xlabel("reduction in forecast physical cost")
    ax2.set_title("(b) What each admitted actuator bought", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig5_29_generative_products.png")
    fig.savefig(out, dpi=200)
    print("written:", out)
    print("  proposed:", dict(prop))
    print("  admitted:", dict(adm))
    for n, s, a, b in rows:
        print(f"  {n:28} t={s:>3}  {a:+.5f} / {b:+.5f}")


if __name__ == "__main__":
    main()
