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
                 profile: str = "full", engine=None) -> None:
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
    # partitions: every term that differs from the Table D.3 default
    # (evFIS moves AND stage-2 inserted terms = catalog growth)
    from .fuzzy import REGISTRY, default_partition
    dp = default_partition()
    parts = {}
    for var in REGISTRY.variables():
        diff = {}
        for t, abcd in REGISTRY.get(var).items():
            d0 = dp.get(t)
            if d0 is None or tuple(round(float(v), 4) for v in abcd) \
                    != tuple(round(float(v), 4) for v in d0):
                diff[t] = [float(v) for v in abcd]
        if diff:
            parts[var] = diff
    vocab = {}
    if engine is not None:
        from .concepts import HIERARCHY as _BH
        _new_c = {}
        for cn, (lvl, ins) in (engine.hierarchy or {}).items():
            if cn in _BH:
                continue
            _new_c[cn] = dict(
                level=int(lvl), inputs=[[a, float(b)] for a, b in ins],
                decision=cn in getattr(engine, "decision_concepts",
                                       []),
                family=list(getattr(engine, "concept_family",
                                    {}).get(cn, ())))
        if _new_c:
            vocab["concepts"] = _new_c
        if getattr(engine, "macros", None):
            vocab["macros"] = {
                k: dict(composition=[[a, float(b)] for a, b in
                                     v["composition"]])
                for k, v in engine.macros.items()}
    d = _read_store(path)
    d["profiles"][str(profile or "full")] = dict(born=born,
                                                 tuned=tuned,
                                                 parts=parts,
                                                 **vocab)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)


def load_learned(path: str, profile: str = "full"):
    sec = _read_store(path)["profiles"].get(str(profile or "full"),
                                            {})
    return sec.get("born", []), sec.get("tuned", {})


def load_parts(path: str, profile: str = "full") -> dict:
    sec = _read_store(path)["profiles"].get(str(profile or "full"),
                                            {})
    return sec.get("parts", {})


def merge_learned(rules: List[Rule], path: str,
                  profile: str = "full", engine=None) -> int:
    """Apply the PROFILE's lineage to a fresh seed base IN PLACE;
    returns how many born rules were appended."""
    born, tuned = load_learned(path, profile)
    if engine is not None:
        sec = _read_store(path)["profiles"].get(
            str(profile or "full"), {})
        for cn, cd in (sec.get("concepts") or {}).items():
            engine.hierarchy[cn] = (
                int(cd.get("level", 2)),
                [(a, float(b)) for a, b in cd.get("inputs", [])])
            if cd.get("decision"):
                if cn not in engine.decision_concepts:
                    engine.decision_concepts.append(cn)
                engine.concept_family[cn] = tuple(
                    cd.get("family") or ())
        for mn, md in (sec.get("macros") or {}).items():
            engine.macros[mn] = dict(
                composition=[(a, float(b)) for a, b in
                             md.get("composition", [])])
    from .fuzzy import REGISTRY
    for var, diff in (load_parts(path, profile) or {}).items():
        for t, abcd in diff.items():
            try:
                REGISTRY.set_term(var, t, abcd)
            except Exception:
                pass
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
