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
        # legacy flat store -> the single SHARED lineage
        d = {"profiles": {"shared": {"born": d.get("born", []),
                                     "tuned": d.get("tuned", {})}}}
    # learned rules are now SHARED across seed profiles: if an older
    # per-profile store exists without a "shared" section, adopt the
    # "full" lineage (the main one) so prior learning is not lost.
    if "shared" not in d["profiles"] and d["profiles"]:
        d["profiles"]["shared"] = d["profiles"].get(
            "full", next(iter(d["profiles"].values())))
    return d


def save_learned(rules: List[Rule], path: str,
                 profile: str = "full", engine=None,
                 use_evfis: bool = True, use_genai: bool = True) -> None:
    """PER-PROFILE lineages: the selected seed profile is THE base;
    what evFIS/GenAI learn on top of it belongs to that lineage and
    never leaks into another profile's experiment.

    use_evfis / use_genai say which stages the engine is actually managing.
    A stage whose LOAD flag is off is NOT in the engine, so its data must be
    PRESERVED from the store here instead of being overwritten with the empty
    engine state (that was silently deleting persisted rules / concepts /
    macros when a run was done with a use-stage toggle off)."""
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
                                     v.get("composition", [])],
                        **({"clauses": list(v["clauses"])}
                           if v.get("clauses") else {}))
                for k, v in engine.macros.items()}
    d = _read_store(path)
    _prev = d["profiles"].get("shared", {}) or {}
    # keep each stage's stored data when that stage's LOAD flag is off (the
    # engine is not managing it, so overwriting would delete it)
    _bornA = [b for b in born if str(b.get("name", ""))[:1] == "A"]
    _bornG = [b for b in born if str(b.get("name", ""))[:1] == "G"]
    _prevA = [b for b in _prev.get("born", [])
              if str(b.get("name", ""))[:1] == "A"]
    _prevG = [b for b in _prev.get("born", [])
              if str(b.get("name", ""))[:1] == "G"]

    def _keep(engine_born, prev_born):
        # union by rule name so a stored rule the engine did not re-load (a
        # seed already covers its cell, so merge_learned skipped it) is NOT
        # dropped on the next save; the engine's version wins on a name
        # clash because its strength is the freshest.
        by = {b.get("name"): b for b in prev_born}
        for b in engine_born:
            by[b.get("name")] = b
        return list(by.values())
    _shared = dict(
        born=(_keep(_bornA, _prevA) if use_evfis else _prevA)
        + (_keep(_bornG, _prevG) if use_genai else _prevG),
        tuned=(tuned if use_evfis else _prev.get("tuned", {})),
        parts=(parts if use_evfis else _prev.get("parts", {})))
    # only an engine can report the vocabulary; a save without one (or with
    # stage 3 not loaded) must KEEP what the store already holds instead of
    # writing an empty section over it
    _manage_vocab = bool(use_genai and engine is not None)
    _concepts = (vocab.get("concepts") if _manage_vocab
                 else _prev.get("concepts"))
    _macros = (vocab.get("macros") if _manage_vocab
               else _prev.get("macros"))
    if _concepts:
        _shared["concepts"] = _concepts
    if _macros:
        _shared["macros"] = _macros
    d["profiles"]["shared"] = _shared
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)


def load_learned(path: str, profile: str = "full"):
    sec = _read_store(path)["profiles"].get("shared",
                                            {})
    return sec.get("born", []), sec.get("tuned", {})


def load_vocab(path: str, profile: str = "full"):
    """The GENERATED vocabulary sitting in the store: (concepts, macros).

    Lets a view show what has been learned BEFORE any engine exists (a fresh
    app start, no run yet): the store is the ground truth, the engine is only
    its current instance."""
    sec = _read_store(path)["profiles"].get("shared", {})
    return (sec.get("concepts") or {}), (sec.get("macros") or {})


def load_parts(path: str, profile: str = "full") -> dict:
    sec = _read_store(path)["profiles"].get("shared",
                                            {})
    return sec.get("parts", {})


def merge_learned(rules: List[Rule], path: str,
                  profile: str = "full", engine=None,
                  use_evfis: bool = True, use_genai: bool = True) -> int:
    """Apply the PROFILE's lineage to a fresh seed base IN PLACE;
    returns how many born rules were appended.

    use_evfis / use_genai gate WHICH pre-learned rules are actually loaded:
      stage 1/2 (evFIS): tuned seed consequents, term inserts, A# rules
      stage 3 (GenAI):   G# rules, generated concepts, macro interventions
    Turning a stage off means the DSS runs WITHOUT those learned rules (the
    generated concepts / interventions simply are not brought in)."""
    born, tuned = load_learned(path, profile)
    if engine is not None and use_genai:
        # generated vocabulary (concepts + macro interventions) is stage 3
        sec = _read_store(path)["profiles"].get(
            "shared", {})
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
            if md.get("clauses"):
                engine.macros[mn]["clauses"] = list(md["clauses"])
    from .fuzzy import REGISTRY
    if use_evfis:
        # term inserts (stage 2 resolution) and consequent tunes (stage 1)
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
        _st = str(b.get("name", ""))[:1]
        # A# rules are evFIS stage-2 resolution; G# rules are GenAI stage 3
        if _st == "A" and not use_evfis:
            continue
        if _st == "G" and not use_genai:
            continue
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
    sec = d["profiles"].get("shared", {})
    born = sorted(sec.get("born", []),
                  key=lambda x: -float(x.get("strength", 0.0)))
    sec["born"] = born[:max(0, int(keep))]
    d["profiles"]["shared"] = sec
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    return len(sec["born"])


def wipe_learned(path: str, profile: str | None = None) -> None:
    """Learned rules are now SHARED across profiles; this wipes the whole
    store (the profile argument is kept for call compatibility)."""
    try:
        os.remove(path)
    except OSError:
        pass
    return
    d = _read_store(path)
    d["profiles"].pop("shared", None)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
    except OSError:
        pass
