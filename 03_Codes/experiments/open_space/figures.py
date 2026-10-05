"""Figures for the open intervention space article (from out/*.csv|jsonl).

  python experiments/open_space/figures.py --out experiments/open_space/out --fig <dir>
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
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False})
ARM_COLOR = {"T0": "#9a9a9a", "F5": "#6f6f6f", "F5Ev": "#1f1f1f",
             "Evo": "#1b6ca8", "Random": "#c0792a", "LLM": "#3a8f5c",
             "Evo-noprior": "#8fb8d9", "Random-noprior": "#e2b58a"}
ARM_LABEL = {"T0": "no action", "F5": "F5 static", "F5Ev": "F5+Ev (closed)",
             "Evo": "Open / evolutionary", "Random": "Open / random",
             "LLM": "Open / language model",
             "Evo-noprior": "Open / evo, no prior",
             "Random-noprior": "Open / random, no prior"}


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_rediscovery(out, figdir):
    runs = [r for r in _rows(os.path.join(out, "runs.csv"))
            if r["exp"] == "E1"]
    if not runs:
        return None
    hours = {"S6": 12, "S7": 12, "S8": 18}
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.8), sharey=True)
    for ax, sc in zip(axes, ("S6", "S7", "S8")):
        for arm in ("Evo", "Random"):
            rr = [r for r in runs if r["scenario"] == sc and r["arm"] == arm]
            n = len(rr)
            if not n:
                continue
            for tgt, col, ls in (("draft", "t_draft", "-"),
                                 ("coat", "t_coat", "--"),
                                 ("ignite", "t_ignite", ":")):
                ts = sorted(float(r[col]) for r in rr if r[col] not in ("", None))
                xs = [0.0] + ts + [hours[sc] * 60.0]
                ys = [0.0] + [100.0 * (i + 1) / n for i in range(len(ts))] \
                    + [100.0 * len(ts) / n]
                ax.step(np.asarray(xs) / 60.0, ys, where="post",
                        color=ARM_COLOR[arm], ls=ls, lw=1.1,
                        label=f"{ARM_LABEL[arm].split(' / ')[1]}, {tgt}")
        ax.text(0.02, 0.92, sc, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")
        ax.set_ylabel("runs with the tactic\nrediscovered (%)")
        ax.set_xlim(0, hours[sc])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
    axes[-1].set_xlabel("incident time (h)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06),
               fontsize=6.5, handlelength=2.2)
    fig.tight_layout()
    p = os.path.join(figdir, "fig09_rediscovery.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_paired(out, figdir, exp="E2", ref="F5Ev"):
    runs = [r for r in _rows(os.path.join(out, "runs.csv"))
            if r["exp"] == exp]
    if not runs:
        return None
    by = {(r["scenario"], r["arm"], int(r["seed"])): float(r["burned_ha"])
          for r in runs}
    scs = [s for s in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
           if any(k[0] == s for k in by)]
    arms = [a for a in ("Evo", "Random", "LLM") if any(k[1] == a for k in by)]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    w = 0.8 / max(len(arms), 1)
    for j, arm in enumerate(arms):
        means, hws = [], []
        for sc in scs:
            ds = [by[(sc, arm, sd)] - by[(sc, ref, sd)]
                  for (s_, a_, sd) in by if s_ == sc and a_ == arm
                  and (sc, ref, sd) in by]
            if ds:
                a = np.asarray(ds)
                means.append(a.mean())
                hws.append(1.96 * a.std(ddof=1) / np.sqrt(len(a))
                           if len(a) > 1 else 0.0)
            else:
                means.append(np.nan)
                hws.append(0.0)
        x = np.arange(len(scs)) + (j - (len(arms) - 1) / 2) * w
        ax.bar(x, means, w, yerr=hws, color=ARM_COLOR[arm],
               error_kw=dict(lw=0.6, capsize=1.5), label=ARM_LABEL[arm])
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(4.5, len(scs) - 0.5, color="#000000", alpha=0.05, lw=0)
    ax.set_xticks(np.arange(len(scs)))
    ax.set_xticklabels(scs)
    ax.set_ylabel("burned area, open minus closed (ha)")
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    p = os.path.join(figdir, "fig10_paired_burned.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()
    figdir = args.fig or os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    for fn in (fig_rediscovery, fig_paired):
        try:
            print(fn(args.out, figdir))
        except Exception as exc:
            print(fn.__name__, "failed:", exc)


if __name__ == "__main__":
    main()
