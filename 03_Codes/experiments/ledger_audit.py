"""The authoritative generative ledger, counted from the logs.

Three places in the thesis report what the generative stage produced and
they do not agree with each other. This script counts the same quantities
once, from the records themselves, so the figure, the chapter table and
the appendix can be made to tell one story.

TWO BODIES OF EVIDENCE, AND THEY ARE NOT THE SAME RUN. The Section 5.5
campaign wrote experiments/out/ladder_funnel.csv, and every one of its
stage-3 attempts carries source "template": the campaign ran on the
deterministic offline proposer, not on a live model. The named macro
interventions the chapter lists came from the dashboard runs recorded in
logs/dss_generated_state.json, which did call a live model. Mixing the
two is what produced the disagreement, so both are reported separately
and neither is presented as the other.

WHAT COUNTS AS A CREATED OBJECT. The three discoverable actuators, the
tactical burn, the water drafting and the aerial retardant drop, are
implemented in the physics and announced to the generative stage before
any run begins. A macro that recruits one of them has used a capability
the seed doctrine ignored; it has not created a physical effect. They are
therefore reported as recruited, never as created.

    python experiments/ledger_audit.py            print the audit
    python experiments/ledger_audit.py --figure   also redraw the cascade

Output: experiments/out/ledger_audit.json
        01_Thesis/figures/fig5_19_cascade.png  (with --figure)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGDIR = os.path.join(HERE, "..", "..", "01_Thesis", "figures")
STORE = os.path.join(HERE, "..", "logs", "dss_generated_state.json")

#: implemented before any run and announced to the stage; recruiting one
#: is not an act of creation
DISCOVERABLE = ("tactical_burn", "water_drafting", "retardant_drop")


def campaign_funnel():
    """The Section 5.5 campaign, from ladder_funnel.csv."""
    path = os.path.join(OUT, "ladder_funnel.csv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = dict(
        attempts=len(rows),
        sources=dict(collections.Counter(r["source"] for r in rows)),
        accepted=sum(1 for r in rows if r["accepted"] == "True"),
        rejected=sum(1 for r in rows if r["accepted"] != "True"),
        accepted_plain=sum(1 for r in rows if r["accepted"] == "True"
                           and r["package"] != "True"),
        accepted_package=sum(1 for r in rows if r["accepted"] == "True"
                             and r["package"] == "True"),
        by_gate=dict(collections.Counter(
            (r["gate"] or "(not recorded)") for r in rows
            if r["accepted"] != "True")),
        scenarios=dict(collections.Counter(r["scenario"] for r in rows)),
        seeds=len({r["seed"] for r in rows}))
    dj = [float(r["dJ"]) for r in rows
          if r["accepted"] == "True" and r["dJ"] not in ("", "None")]
    if dj:
        out["dJ_mean"] = round(sum(dj) / len(dj), 5)
        out["dJ_best"] = round(min(dj), 5)
    return out


def campaign_products():
    """What the campaign actually created, from ladder_products.jsonl."""
    path = os.path.join(OUT, "ladder_products.jsonl")
    if not os.path.exists(path):
        return None
    macros, rules = collections.Counter(), collections.Counter()
    rows = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows += 1
            r = json.loads(line)
            for m in (r.get("macros") or {}):
                macros[m] += 1
            for rl in (r.get("learned_rules") or []):
                rules[rl.get("name", "?")] += 1
    return dict(rows=rows, macros=sorted(macros),
                distinct_macros=len(macros),
                distinct_rule_names=len(rules),
                rule_entries=sum(rules.values()))


def live_store():
    """What the live model produced, from the learned store."""
    if not os.path.exists(STORE):
        return None
    with open(STORE, encoding="utf-8") as f:
        d = json.load(f)
    props = d.get("genai_proposals") or []
    ivs = d.get("genai_interventions") or []
    clause = [v for v in ivs if v.get("clauses")]
    composite = [v for v in ivs if not v.get("clauses")]
    recruited = sorted({c["channel"] for v in ivs
                        for c in (v.get("composition") or [])
                        if c["channel"] in DISCOVERABLE})
    return dict(
        proposals=len(props),
        accepted=sum(1 for p in props if p.get("accepted")),
        rejected=sum(1 for p in props if not p.get("accepted")),
        by_gate=dict(collections.Counter(
            str(p.get("gate") or "(not recorded)") for p in props
            if not p.get("accepted"))),
        rules=len(d.get("genai_rules") or []),
        concepts=[c.get("name") for c in (d.get("genai_concepts") or [])],
        interventions=[v["name"] for v in ivs],
        composite_interventions=len(composite),
        clause_actuators=len(clause),
        recruited_discoverable=recruited,
        evfis_modifications=len(d.get("evfis_rule_modifications") or []),
        created_objects=len(ivs) + len(d.get("genai_concepts") or []))


def name_check():
    """Whether a macro's name matches the effect its definition carries.

    A macro called a backburn that carries no igniting channel is a
    naming error, and one that reaches a table becomes a claim the logs
    do not support.
    """
    if not os.path.exists(STORE):
        return []
    with open(STORE, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for v in d.get("genai_interventions") or []:
        chans = [c["channel"] for c in (v.get("composition") or [])]
        clauses = [c.get("effect") for c in (v.get("clauses") or [])]
        burns = "tactical_burn" in chans or "ignite" in clauses
        claims = any(w in v["name"].lower()
                     for w in ("burn", "backburn", "counterfire", "fire"))
        if claims and not burns:
            out.append(dict(name=v["name"], composition=chans,
                            clauses=clauses, verdict="name claims fire, "
                            "definition carries none"))
    return out


def ordered_in_logs():
    """Macro names that runs actually ordered, whatever any table says."""
    import glob
    base = {"suppression_effort", "resource_deployment", "containment_line",
            "asset_protection", "evacuation", "public_warning",
            "tactical_burn", "water_drafting", "retardant_drop"}
    seen = collections.Counter()
    for f in sorted(glob.glob(os.path.join(HERE, "..", "logs",
                                           "DSS_*", "cycles.jsonl"))):
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    c = json.loads(line)
                except Exception:
                    continue
                for r in (c.get("regions") or {}).values():
                    for k, v in (r.get("orders_final") or {}).items():
                        if k not in base and float(v) > 0.05:
                            seen[k] += 1
    return dict(seen)


def figure(audit):
    """The generative cascade, drawn from the counts above."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    c = audit["campaign_funnel"]
    lv = audit["live_store"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    for ax, d, title in (
            (axes[0], c, "Section 5.5 campaign (offline proposer)"),
            (axes[1], lv, "recorded runs (live model)")):
        if not d:
            continue
        n = d.get("attempts", d.get("proposals", 0))
        acc = d["accepted"]
        stages = [("proposed", n), ("passed format and\nvocabulary",
                                    n - _gate_share(d, "G1", "G2")),
                  ("admitted", acc)]
        ys = np.arange(len(stages))[::-1]
        ax.barh(ys, [s[1] for s in stages], color="#2980b9", height=0.55)
        for y, s in zip(ys, stages):
            ax.text(s[1] + n * 0.015, y, str(s[1]), va="center",
                    fontsize=9, color="#333333")
        ax.set_yticks(ys)
        ax.set_yticklabels([s[0] for s in stages], fontsize=9)
        ax.set_xlim(0, n * 1.18)
        ax.set_xlabel("proposals", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, "fig5_19_cascade.png")
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("written:", os.path.abspath(p))


def _gate_share(d, *prefixes):
    return sum(v for k, v in (d.get("by_gate") or {}).items()
               if any(k.startswith(p) for p in prefixes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()

    audit = dict(campaign_funnel=campaign_funnel(),
                 campaign_products=campaign_products(),
                 live_store=live_store(),
                 naming_errors=name_check(),
                 ordered_in_logs=ordered_in_logs())
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ledger_audit.json"), "w",
              encoding="utf-8") as f:
        json.dump(audit, f, indent=1)

    for k, v in audit.items():
        print("=" * 68)
        print(k)
        print("=" * 68)
        print(json.dumps(v, indent=1)[:2000])
        print()
    if a.figure:
        figure(audit)


if __name__ == "__main__":
    main()
