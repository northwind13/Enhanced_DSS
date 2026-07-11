"""Persistent learned-rule store (survives fires, engines AND maps).

The store is a JSON file with two sections:
  "born"  - adaptation-born rules (A*/G*): full spec + strength
  "tuned" - evFIS consequent tunes of SEED rules: name -> consequents

An engine loads the store at construction: tunes are applied to the
seed copies it starts with, born rules whose antecedent cell is not
already covered are appended. Every accepted adaptation saves the
store, so the knowledge accumulates across scenarios; pruning keeps
only the strongest born rules (the convergence experiment: does the
minimal profile GROW toward the Table E.1 doctrine?)."""

from __future__ import annotations

import json
import os
from typing import List

from .rules import Rule


def _read_store(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {"profiles": {}}
    if "profiles" not in d:
        # legacy flat store: adopt it as the "full" lineage
        d = {"profiles": {"full": {"born": d.get("born", []),
                                   "tuned": d.get("tuned", {})}}}
    return d


def save_learned(rules: List[Rule], path: str,
                 profile: str = "full") -> None:
    """PER-PROFILE lineages: the selected seed profile is THE base;
    what evFIS/GenAI learn on top of it belongs to that lineage and
    never leaks into another profile's experiment."""
    born = [dict(name=r.name,
                 antecedents=[list(a) for a in r.antecedents],
                 consequents=[[i, float(v)] for i, v in r.consequents],
                 note=r.note, strength=float(getattr(r, "strength",
                                                     0.0)))
            for r in rules if r.name[:1] in "AG"]
    tuned = {r.name: [[i, float(v)] for i, v in r.consequents]
             for r in rules
             if r.name[:1] == "R" and "evFIS" in (r.note or "")}
    d = _read_store(path)
    d["profiles"][str(profile or "full")] = dict(born=born,
                                                 tuned=tuned)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)


def load_learned(path: str, profile: str = "full"):
    sec = _read_store(path)["profiles"].get(str(profile or "full"),
                                            {})
    return sec.get("born", []), sec.get("tuned", {})


def merge_learned(rules: List[Rule], path: str,
                  profile: str = "full") -> int:
    """Apply the PROFILE's lineage to a fresh seed base IN PLACE;
    returns how many born rules were appended."""
    born, tuned = load_learned(path, profile)
    for r in rules:
        if r.name in tuned:
            r.consequents = [(i, float(v)) for i, v in tuned[r.name]]
            r.note = (r.note + " | " if r.note else "") + \
                "evFIS: restored from the learned store"
    def covered(ants):
        aset = set(ants)
        return any(rr.active and aset.issubset(set(rr.antecedents))
                   for rr in rules)
    n = 0
    for b in sorted(born, key=lambda x: -float(x.get("strength", 0))):
        ants = [tuple(a) for a in b["antecedents"]]
        if covered(ants):
            continue
        rules.append(Rule(b["name"], ants,
                          [(i, float(v)) for i, v in b["consequents"]],
                          note=(b.get("note") or "")
                          + " | restored from the learned store",
                          strength=float(b.get("strength", 0.0))))
        n += 1
    return n


def prune_learned(path: str, keep: int = 10,
                  profile: str = "full") -> int:
    """Keep only the strongest `keep` born rules of the profile's
    lineage; returns survivors."""
    d = _read_store(path)
    sec = d["profiles"].get(str(profile or "full"), {})
    born = sorted(sec.get("born", []),
                  key=lambda x: -float(x.get("strength", 0.0)))
    sec["born"] = born[:max(0, int(keep))]
    d["profiles"][str(profile or "full")] = sec
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    return len(sec["born"])


def wipe_learned(path: str, profile: str | None = None) -> None:
    """profile=None wipes the whole file; otherwise only that
    lineage."""
    if profile is None:
        try:
            os.remove(path)
        except OSError:
            pass
        return
    d = _read_store(path)
    d["profiles"].pop(str(profile), None)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
    except OSError:
        pass
