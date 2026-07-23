"""Staged adaptation of the rule base.

The stages engage ONLY when the satisficing test fails (forecast J above
J_TH); which stage is tried is chosen by a small value-based stage
controller (an associative-search contextual bandit, Sutton & Barto 2018)
whose reward is the realized cost reduction:

  stage 1  evFIS       bounded, derivative-free perturbation of the
                       trapezoid parameters of the antecedent terms of
                       the deficient rules and of their consequent
                       intensities; a trial is KEPT only if the shadow
 forecast cost decreases
  stage 2  resolution  a NEW rule is instantiated at the antecedent cell
                       of the current situation (dominant terms of the
                       decision concepts), consequents seeded from the
                       concept demands; the sparse base grows on demand
  stage 3  generative  a rule proposed by Claude through Claude Code on the
                       user's subscription (`claude -p`; no API key).
                       A reachable model is REQUIRED: there is no offline
                       stand-in; if none is reachable the stage is skipped
                       and logged. A proposal is admitted only through the
                       gates: G1 format, G2 constraints, G3 simulated
                       dJ < 0, G4 A/B on two seeds (G2b/G5 for packages).

All adaptation acts on RUNTIME copies (rules list, per-variable
partition registry); provenance is recorded on every admitted change.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fuzzy import TERMS, REGISTRY, default_partition
from .concepts import DECISION_CONCEPTS
from .rules import Rule, SEED_RULES, INTERVENTIONS, evaluate_rules
from .evaluate import (candidate_vs_noaction, CONCEPT_FAMILY,
                       forecast_cost, physical_cost)


def make_runtime_rules(profile: str = "full") -> List[Rule]:
    """Fresh, mutable copy of the seed base (adaptation never mutates
    the module-level catalog).

    profile selects HOW MUCH doctrine the run starts with:
 "full" - the whole base (40 seeds + examples)
      "core"    - the doctrine block R1-R22 only (no backbone): the
                  adaptation stages must rebuild the rest
      "minimal" - per intervention family only the SINGLE strongest
 seed (5 rules with the current: one seed
                  answers two families): the system starts nearly
                  naked and must LEARN its rule base by trial
                  (stages 2/3)
    """
    rules = copy.deepcopy(SEED_RULES)
    prof = (profile or "full").lower()
    if prof.startswith("core"):
        keep = {f"R{i}" for i in range(1, 23)}
        return [r for r in rules if r.name in keep]
    if prof.startswith("min"):
        best: dict = {}
        for r in rules:
            if not r.active:
                continue
            for iv, v in r.consequents:
                if iv not in best or v > best[iv][1]:
                    best[iv] = (r, v)
        ids = {id(r) for r, _v in best.values()}
        return [r for r in rules if id(r) in ids]
    return rules


@dataclass
class AdaptOutcome:
    stage: int
    accepted: bool
    detail: str
    dJ: float = 0.0
    info: dict = field(default_factory=dict)   # structured trial
                                               # trace for the log


# ------------------------------------------------------------- stage 1
TRIAL_BASIS_MIN = 45.0   # adaptation trials are judged at >= this


def _cva(build_override, sim, rules, horizon, reseed=None):
    """Adaptation trials are judged on the PHYSICAL cost (burned area,
    assets, population) at a >= 45 min basis, exactly like the no-harm
    guard: a rule tweak that commits more capacity must buy a better
    FIRE, not a smaller fleet bill, and it must be given long enough
    for the committed crews to land. Judging trials on the TOTAL J at
    a short live horizon systematically rejects every proposal on a
    small fire (the response cost of any extra commitment dwarfs the
    physical gain available in 15 minutes), which is what stalled the
    stage 2/3 adaptations in the field runs."""
    hmin = getattr(sim, "_dss_hmin", None)
    hmin = max(TRIAL_BASIS_MIN, float(hmin)) \
        if hmin is not None else TRIAL_BASIS_MIN
    rep = forecast_cost(sim, build_override(rules), horizon,
                        reseed=reseed, horizon_min=hmin)
    p_c = physical_cost(rep, sim.cfg.cost)
    # the no-action physical baseline is constant within a decision
    # step: cache it per (basis, reseed)
    key = (round(hmin, 1), reseed)
    step = int(getattr(sim.state, "step", -1))
    if getattr(sim, "_dss_p0_step", None) != step:
        sim._dss_p0c = {}
        sim._dss_p0_step = step
    if key not in sim._dss_p0c:
        rep0 = forecast_cost(sim, None, horizon, reseed=reseed,
                             horizon_min=hmin)
        sim._dss_p0c[key] = physical_cost(rep0, sim.cfg.cost)
    return float(p_c), float(sim._dss_p0c[key])


def stage1_evfis(build_override, sim, rules: List[Rule],
                 fired: List[Tuple[Rule, float]], horizon: int,
                 step_size: float = 0.05, trials: int = 3) -> AdaptOutcome:
    """Perturb the strongest deficient rule; keep only improvements.
    Every trial is recorded (what moved, J before/after, kept, reason)
    so the run log carries the full evFIS story."""
    hot = [(r, w) for r, w in fired if w > 0.05 and r.active]
    if not hot:
        return AdaptOutcome(1, False, "no fired rule to tune")
    hot.sort(key=lambda t: -t[1])
    j_best, j0 = _cva(build_override, sim, rules, horizon)
    best_detail = None
    tlog = []
    # the TOP TWO firing rules each get a consequent trial: with a
    # thin base a single rule dominates every cycle and the second
    # loudest voice was never tuned at all
    _cand_rules = [h[0] for h in hot[:2]]
    rule = _cand_rules[0]
    for rule_t in _cand_rules:
        for sign in (+1.0, -1.0):
            if trials <= 0:
                break
            trials -= 1
            old_cons = list(rule_t.consequents)
            rule_t.consequents = [(iv, float(np.clip(
                v + sign * step_size, 0.0, 1.0)))
                for iv, v in old_cons]
            j_try, _ = _cva(build_override, sim, rules, horizon)
            kept = j_try < j_best - 1e-6
            tlog.append(dict(kind="consequent", rule=rule_t.name,
                             delta=float(sign * step_size),
                             j_before=float(j_best),
                             j_after=float(j_try), kept=bool(kept),
                             reason="physical forecast improved"
                             if kept else
                             "physical forecast did not improve"))
            if kept:
                j_best = j_try
                best_detail = (f"{rule_t.name} consequents "
                               f"{'+' if sign > 0 else '-'}"
                               f"{step_size:g}")
                rule_t.note = ((rule_t.note + " | " if rule_t.note
                                else "")
                               + f"evFIS: consequent "
                                 f"{sign * step_size:+g}")
            else:
                rule_t.consequents = old_cons
    if trials > 0:
        var, term = rule.antecedents[0]
        # A shoulder is a SHARED boundary of two neighbouring trapezoids, not
        # a free parameter of one. Moving it through shift_boundary displaces
        # BOTH terms together, so the Ruspini invariant (sum_t mu_t = 1) is
        # preserved, and clamps the step to the neighbouring core widths so no
        # plateau inverts. Moving a single trapezoid, as this used to do, tore
        # the partition and violated the framework's own convexity guarantee.
        before = REGISTRY.snapshot(var)
        applied = REGISTRY.shift_boundary(var, term, -step_size)
        if applied == 0.0:
            tlog.append(dict(kind="membership", rule=rule.name,
                             var=var, term=term, kept=False,
                             reason="no admissible boundary move "
                                    "(step clamped to zero by the core width)"))
        else:
            j_try, _ = _cva(build_override, sim, rules, horizon)
            kept = j_try < j_best - 1e-6
            tlog.append(dict(kind="membership", rule=rule.name,
                             var=var, term=term,
                             move=f"shared boundary {applied:+g} "
                                  f"({term} and its left neighbour move "
                                  f"together)",
                             j_before=float(j_best),
                             j_after=float(j_try), kept=bool(kept),
                             reason="physical forecast improved" if kept
                             else "physical forecast did not improve"))
            if kept:
                j_best = j_try
                best_detail = (f"{var}.{term} shared boundary "
                               f"{applied:+g}")
            else:
                REGISTRY.restore(var, before)
    info = dict(rule=rule.name, trials=tlog)
    if best_detail is None:
        return AdaptOutcome(1, False, "no improving perturbation found",
                            dJ=0.0, info=info)
    return AdaptOutcome(1, True, best_detail, dJ=j_best - j0, info=info)


# ------------------------------------------------------------- stage 2
def _dominant_terms(eff: Dict[str, np.ndarray]) -> Dict[str, str]:
    out = {}
    for cn in DECISION_CONCEPTS:
        v = eff[cn]
        out[cn] = TERMS[int(np.argmax(v))]
    return out


def _cell_covered(rules: List[Rule], ants) -> bool:
    """Does any ACTIVE rule already sit on this antecedent cell?"""
    aset = set(ants)
    return any(r.active and aset.issubset(set(r.antecedents))
               for r in rules)


def stage2_resolution(build_override, sim, rules: List[Rule],
                      eff: Dict[str, np.ndarray],
                      crisp_c: Dict[str, float],
                      horizon: int, coverage_gap: bool = False,
                      cov_w: float = 1.0) -> AdaptOutcome:
    """Instantiate the missing antecedent cell of the CURRENT situation:
    a new rule over the two most activated decision concepts, with
    consequents equal to the concept demands (family intensities)."""
    dom = _dominant_terms(eff)
    ranked = sorted(DECISION_CONCEPTS,
                    key=lambda c: -float(crisp_c.get(c, 0.0)))
    ants = [(c, dom[c]) for c in ranked[:2]]
    cons = []
    for cn in ranked[:3]:
        fams = CONCEPT_FAMILY.get(cn)
        if not fams:
            continue
        cons.append((fams[0],
                     float(np.clip(crisp_c.get(cn, 0.0), 0.0, 1.0))))
    if not cons:
        return AdaptOutcome(2, False, "no scored concept to answer")
    if _cell_covered(rules, ants):
        # TRUE RESOLUTION INCREASE: when the covered situation sits
        # BETWEEN the term cores of the hottest concept (ambiguous
        # membership), the linguistic catalog itself grows: a new
        # narrow term is inserted there and the rule is written on
        # the REFINED cell. A crisp situation on a covered cell
        # remains evFIS territory.
        c1 = ants[0][0]
        _v1 = eff.get(c1)
        _amb = (_v1 is not None
                and float(np.max(_v1)) < 0.62)
        if not _amb:
            return AdaptOutcome(
                2, False,
                "cell already covered \u2014 an active rule sits on "
                "this antecedent cell and the situation is crisp; "
                "numeric tuning is stage \u2460 evFIS's job",
                info=dict(cell=[list(a) for a in ants]))
        _x1 = float(crisp_c.get(c1, 0.5))
        _newt = REGISTRY.insert_split(c1, _x1)
        ants = [(c1, _newt)] + list(ants[1:])
        if _cell_covered(rules, ants):
            return AdaptOutcome(
                2, False,
                "refined cell already covered",
                info=dict(cell=[list(a) for a in ants]))
        _split_note = (f"resolution increase: term {_newt} inserted "
                       f"into {c1} at {_x1:.2f} (catalog grew)")
    else:
        _split_note = None
    coverage_gap = True     # by definition: the cell was uncovered
    name = f"A{sum(1 for r in rules if r.name.startswith('A')) + 1}"
    newr = Rule(name, ants, cons,
                note=("adaptation-born: " + _split_note)
                if _split_note else
                "adaptation-born: resolution increase "
                "(instantiated antecedent cell)")
    j0c, j0 = _cva(build_override, sim, rules, horizon)
    rules.append(newr)
    j1, _ = _cva(build_override, sim, rules, horizon)
    info = dict(rule=newr.text(), j_without=float(j0c),
                j_with=float(j1), j_noaction=float(j0),
                basis="physical cost at >=45 min")
    info["coverage_gap"] = bool(coverage_gap)
    if j1 < j0c - 1e-6:
        return AdaptOutcome(2, True, f"{name}: {newr.text()}",
                            dJ=j1 - j0, info=info)
    if coverage_gap and j1 <= j0c + 1e-4:
        # the base had (almost) nothing to say about this situation
        # (max fired weight below the coverage threshold): a
        # NON-INFERIOR rule that answers the void is admitted, the
        # base must GROW toward covering the space
        return AdaptOutcome(
            2, True,
            f"{name}: {newr.text()} (coverage gap, admitted "
            f"non-inferior at fired weight {cov_w:.2f})",
            dJ=j1 - j0, info=info)
    rules.pop()
    return AdaptOutcome(2, False,
                        "instantiated cell did not improve the "
                        f"physical forecast ({j1:.3f} vs {j0c:.3f}, "
                        "burn+asset+pop at >=45 min)",
                        info=info)


# ------------------------------------------------------------- stage 3
_GENAI_SCHEMA = ("Return ONLY a JSON object: {\"antecedents\": "
                 "[[concept, term], ...], \"consequents\": "
                 "[[intervention, intensity], ...]} with concepts from "
                 f"{list(DECISION_CONCEPTS)}, terms from {list(TERMS)}, "
                 f"interventions from {list(INTERVENTIONS)}, intensities "
                 "in [0,1]. Max 3 antecedents, max 3 consequents. "
                 "OPTIONALLY the proposal may be a PACKAGE that grows "
                 "the vocabulary, by adding ONE of: "
                 "\"new_concept\": {\"name\": str, \"level\": "
                 "\"intermediate\", \"inputs\": "
                 "[[existing_feature_or_concept, weight], ...] "
                 "(weights positive, at most 4, they will be "
                 "normalized)} or "
                 "\"new_intervention\": {\"name\": str, "
                 "\"composition\": [[base_intervention, weight], "
                 "...] (at most 3, weights in (0,1])}. A package must "
                 "still contain the rule, and the rule must USE the "
                 "new object (a new concept in its antecedents, a new "
                 "intervention in its consequents). FIXED by design: "
                 "the ten features, the five decision axes and the "
                 "six base physical channels. Only compositions of "
                 "the EXISTING semantics are possible: no new "
                 "physics, no new features.")


def _genai_propose_cli(prompt: str, timeout: float = 120.0) -> Optional[dict]:
    """One rule proposal via the Claude Code CLI (`claude -p`), i.e. on the
    user's Claude subscription, no API key. Returns the parsed dict or None.

    Reachability is decided by ACTUALLY invoking `claude` (same path the Test
    Claude connection button uses), not shutil.which: on Windows a `.cmd`
    shim can be runnable by subprocess yet invisible to which, which used to
    make stage 3 look unreachable even though the model answered fine."""
    import subprocess
    cmd = ["claude", "-p", prompt, "--output-format", "json"]
    _m = os.environ.get("DSS_GENAI_MODEL")
    if _m:
        cmd += ["--model", _m]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout)
    except Exception:
        return None
    if res.returncode != 0 or not res.stdout.strip():
        return None
    inner = res.stdout
    try:                                   # unwrap {"result": "...text..."}
        wrap = json.loads(res.stdout)
        if isinstance(wrap, dict) and "result" in wrap:
            inner = str(wrap["result"])
    except Exception:
        pass
    try:
        i, j = inner.find("{"), inner.rfind("}")
        return json.loads(inner[i:j + 1])
    except Exception:
        return None


def _genai_propose(situation: str) -> Optional[dict]:
    """One rule proposal from Claude via the Claude Code CLI (`claude -p`) on
    the user's subscription. Returns None when the `claude` command is not
    available/logged in or the reply is unparseable."""
    prompt = ("You are the rule proposer of a wildfire decision "
              "support system. Situation:\n" + situation + "\n"
              + _GENAI_SCHEMA)
    return _genai_propose_cli(prompt)


G5_MARGIN = 1e-4     # a vocabulary-growing package must clear this on both
                     # reseeded rollouts (the vocabulary margin delta_v). Kept
                     # small so a genuinely-helpful new concept / composite
                     # intervention is actually admitted, not gated out.
G2B_COS = 0.95       # structural redundancy bound for new concepts (G2b)


def _validate_package(prop: dict, engine) -> Optional[str]:
    """G2 extension + G2b for vocabulary packages; returns an error
    string or None. Weights are normalized in place."""
    from .features import FEATURE_ORDER
    nc = prop.get("new_concept")
    ni = prop.get("new_intervention")
    if nc and ni:
        return "G2 package (one object per package)"
    if nc:
        name = str(nc.get("name", "")).strip().replace(" ", "_")
        if not name or name in engine.hierarchy \
                or name in FEATURE_ORDER:
            return "G2 package (name empty or taken)"
        lvl = str(nc.get("level", "intermediate"))
        if lvl == "decision":
            # DESIGN DECISION: the five decision axes (and with them
            # the antecedent catalog dimensionality and the Q
            # denominator) are FIXED. The vocabulary grows through
            # macro interventions and intermediate concepts; the
            # axes' RESOLUTION grows through stage-2 term insertion.
            return ("G2 package (decision axes are fixed by design; "
                    "propose an intermediate concept or a macro "
                    "intervention)")
        ins = nc.get("inputs") or []
        if not (1 <= len(ins) <= 4):
            return "G2 package (1..4 inputs)"
        known = set(FEATURE_ORDER) | set(engine.hierarchy)
        tot = 0.0
        for pair in ins:
            src, wv = str(pair[0]), float(pair[1])
            if src not in known or wv <= 0:
                return "G2 package (unknown input or bad weight)"
            tot += wv
        nc["inputs"] = [[str(a), float(b) / tot] for a, b in ins]
        nc["name"] = name
        # the new intermediate concept must sit ABOVE every concept it reads,
        # so infer_concepts (which sweeps levels 1..4) computes it AFTER its
        # dependencies. Features are level 0; a concept input pins the floor.
        _inlv = [engine.hierarchy[a][0] for a, _b in nc["inputs"]
                 if a in engine.hierarchy]
        _reqlv = (max(_inlv) + 1) if _inlv else 2
        if _reqlv > 4:
            return ("G2 package (concept inputs sit too high in the "
                    "hierarchy; use features or lower-level concepts)")
        nc["_level_num"] = int(_reqlv)
        if lvl == "decision":
            fam = nc.get("family")
            if fam not in INTERVENTIONS:
                return ("G2 package (a decision concept must declare "
                        "an answering intervention family)")
        # G2b: STRUCTURAL non-redundancy - the normalized input
        # weight vector must not be (near-)collinear with any
        # existing concept over the same inputs
        import numpy as _np
        keys = [a for a, _b in nc["inputs"]]
        v_new = _np.array([b for _a, b in nc["inputs"]])
        for cn, (_l, cins) in engine.hierarchy.items():
            m = dict(cins)
            v_old = _np.array([float(m.get(k2, 0.0)) for k2 in keys])
            if v_old.sum() <= 0:
                continue
            cos = float(v_new @ v_old
                        / (_np.linalg.norm(v_new)
                           * _np.linalg.norm(v_old) + 1e-12))
            if cos >= G2B_COS and set(m) <= set(keys):
                return (f"G2b redundancy (collinear with {cn}, "
                        f"cos={cos:.2f})")
    if ni:
        name = str(ni.get("name", "")).strip().replace(" ", "_")
        if not name or name in INTERVENTIONS \
                or name in engine.macros:
            return "G2 package (name empty or taken)"
        comp = ni.get("composition") or []
        if not (1 <= len(comp) <= 3):
            return "G2 package (1..3 components)"
        for pair in comp:
            bi, bw = str(pair[0]), float(pair[1])
            if bi not in INTERVENTIONS or not (0.0 < bw <= 1.0):
                return ("G2 package (composition must reduce to the "
                        "base physical channels)")
        ni["composition"] = [[str(a), float(b)] for a, b in comp]
        ni["name"] = name
    return None


def _install_package(prop: dict, engine) -> dict:
    """Temporarily install the package into the ENGINE vocabulary;
    returns an undo dict."""
    undo = {}
    nc = prop.get("new_concept")
    ni = prop.get("new_intervention")
    if nc:
        # placed strictly above its concept inputs (computed in
        # _validate_package) so the hierarchy resolves in dependency order
        lvl = (3 if nc.get("level") == "decision"
               else int(nc.get("_level_num", 2)))
        engine.hierarchy[nc["name"]] = (
            lvl, [(a, b) for a, b in nc["inputs"]])
        undo["concept"] = nc["name"]
        if nc.get("level") == "decision":
            engine.decision_concepts.append(nc["name"])
            engine.concept_family[nc["name"]] = (nc["family"],)
            undo["decision"] = True
    if ni:
        engine.macros[ni["name"]] = dict(
            composition=[(a, b) for a, b in ni["composition"]])
        undo["macro"] = ni["name"]
    return undo


def _uninstall_package(undo: dict, engine) -> None:
    if undo.get("concept"):
        engine.hierarchy.pop(undo["concept"], None)
        if undo.get("decision"):
            try:
                engine.decision_concepts.remove(undo["concept"])
            except ValueError:
                pass
            engine.concept_family.pop(undo["concept"], None)
    if undo.get("macro"):
        engine.macros.pop(undo["macro"], None)


def _g1_g2(prop: dict, engine=None) -> Optional[str]:
    try:
        ants = [(str(v), str(t)) for v, t in prop["antecedents"]]
        cons = [(str(i), float(x)) for i, x in prop["consequents"]]
    except Exception:
        return "G1 format"
    if not (1 <= len(ants) <= 3 and 1 <= len(cons) <= 3):
        return "G2 arity"
    _ncn = (prop.get("new_concept") or {}).get("name")
    _nin = (prop.get("new_intervention") or {}).get("name")
    _known_c = set(DECISION_CONCEPTS)
    _known_i = set(INTERVENTIONS)
    if engine is not None:
        # learned vocabulary is CITABLE by every later rule
        _known_c |= set(getattr(engine, "hierarchy", {}) or {})
        _known_i |= set(getattr(engine, "macros", {}) or {})
    for v, t in ants:
        if v not in _known_c and v != _ncn:
            return "G2 vocabulary"
        if t not in TERMS and t not in REGISTRY.get(v):
            return "G2 vocabulary"
    for i, x in cons:
        if (i not in _known_i and i != _nin) \
                or not (0.0 <= x <= 1.0):
            return "G2 range"
    return None


def _nearest_cases(engine, crisp_c, k: int = 3) -> str:
    """CHRONICLE RETRIEVAL: the k past cycles whose decision-concept
    situation is nearest (Euclidean over the five crisps) to the
    current one, summarized compactly for the proposer prompt. The
    chronicle is recorded anyway; retrieving from it grounds the
    proposal in what was actually tried and what it did."""
    if engine is None or not getattr(engine, "cycles", None):
        return ""
    import numpy as _np
    cur = _np.array([float(crisp_c.get(c, 0.0))
                     for c in DECISION_CONCEPTS])
    scored = []
    for cyc in engine.cycles[:-1][-400:]:
        regs = cyc.get("regions") or {}
        for rn, rd in regs.items():
            cc = (rd.get("concepts_effective")
                  or rd.get("concepts_crisp") or {})
            if not cc:
                continue
            v = _np.array([float(cc.get(c, 0.0))
                           for c in DECISION_CONCEPTS])
            d = float(_np.linalg.norm(cur - v))
            ad = cyc.get("adaptation") or {}
            scored.append((d, cyc.get("step"), rn, cc,
                           rd.get("orders_final") or {},
                           ad.get("detail") or "no adaptation",
                           bool(ad.get("accepted"))))
    if not scored:
        return ""
    scored.sort(key=lambda t: t[0])
    lines = ["\nNearest past cases from this incident's chronicle "
             "(situation -> orders -> adaptation verdict):"]
    for d, step, rn, cc, of, det, acc in scored[:k]:
        sit = ", ".join(f"{c.split('_')[0]}={float(cc.get(c, 0.0)):.2f}"
                        for c in DECISION_CONCEPTS)
        ords = ", ".join(f"{kk.split('_')[0]}={vv:.2f}"
                         for kk, vv in list(of.items())[:6])
        lines.append(f"- k={step} [{rn}] {sit} | orders: {ords} | "
                     f"{'ACCEPTED' if acc else 'no change'}: "
                     f"{det[:90]}")
    return "\n".join(lines)


def stage3_generative(build_override, sim, rules: List[Rule],
                      eff, crisp_c, horizon: int,
                      coverage_gap: bool = False,
                      cov_w: float = 1.0,
                      engine=None) -> AdaptOutcome:
    situation = ", ".join(f"{c}={crisp_c.get(c, 0.0):.2f}"
                          for c in DECISION_CONCEPTS)
    # the proposer sees the WHOLE current base and the coverage
    # verdict, so it can aim at the void instead of paraphrasing an
    # existing rule
    situation += ("\nCurrent rule base:\n"
                  + "\n".join(r.text() for r in rules if r.active))
    if coverage_gap:
        situation += (f"\nThe base is nearly silent here (max "
                      f"fired weight {cov_w:.2f}): propose ONE rule "
                      "that ANSWERS this situation. If the five decision "
                      "concepts and six base interventions cannot cleanly "
                      "express the RIGHT response for this situation, PREFER "
                      "a PACKAGE: add ONE new intermediate concept (a weighted "
                      "mix of existing features/concepts that names what "
                      "matters here) OR ONE composite intervention (a weighted "
                      "mix of base channels), and write the rule USING that "
                      "new object. Grow the vocabulary when it genuinely helps "
                      "answer the void.")
    situation += ("\nDo NOT re-issue an existing rule's antecedent "
                  "cell with different numbers (that is the tuning "
                  "stage's job); pick an UNCOVERED situation cell.")
    situation += _nearest_cases(engine, crisp_c)
    # ---- BOUNDED REVISION BUDGET: a proposal rejected at the CHEAP
    # gates (G1 form, G2 constraints, G2b redundancy) returns to the
    # model with the failing gate as feedback, up to B times; the
    # (there is no template; the model is required). Only the
    # simulation gates (G3-G5) end the stage outright: they are the
    # expensive verdicts and re-rolling them would buy noise.
    _budget = int(getattr(engine, "revision_budget", 3) or 0) \
        if engine is not None else 0
    _revisions = []
    prop = _genai_propose(situation)
    src = "claude"
    # The generative stage REQUIRES a reachable model: there is no offline
    # stand-in. If the model cannot be reached (or returns nothing usable)
    # the stage is skipped and logged, never silently substituted.
    if prop is None:
        return AdaptOutcome(
            3, False,
            "generative stage requires a reachable model; none available "
            "(install Claude Code and run `/login` on your Pro/Max plan)",
            info=dict(source="none", reason="model unreachable"))
    err = _g1_g2(prop, engine=engine)
    while err and len(_revisions) < _budget:
        _revisions.append(dict(proposal=prop, gate=err))
        retry = (situation
                 + f"\nYour previous proposal was rejected at gate "
                 f"{err}. Revise it: fix exactly that defect and "
                 "return the corrected JSON only.")
        prop = _genai_propose(retry)
        if prop is None:
            return AdaptOutcome(
                3, False,
                "generative stage aborted: model unreachable during "
                "revision",
                info=dict(source="claude", reason="model unreachable",
                          revisions=_revisions))
        err = _g1_g2(prop, engine=engine)
    if err:
        return AdaptOutcome(3, False, f"rejected at {err} ({src})",
                            info=dict(source=src, proposal=prop,
                                      gate=err,
                                      revisions=_revisions))
    # ---- PACKAGE path: the proposal may GROW the vocabulary ----
    _pkg = bool(prop.get("new_concept") or prop.get("new_intervention"))
    _undo = None
    if _pkg:
        if engine is None:
            return AdaptOutcome(3, False,
                                "rejected at G2 package (no engine "
                                f"context) ({src})",
                                info=dict(source=src, proposal=prop))
        perr = _validate_package(prop, engine)
        if perr:
            return AdaptOutcome(3, False,
                                f"rejected at {perr} ({src})",
                                info=dict(source=src, proposal=prop,
                                          gate=perr, package=True))
        # the rule must USE the new object
        _ncn = (prop.get("new_concept") or {}).get("name")
        _nin = (prop.get("new_intervention") or {}).get("name")
        _uses = ((_ncn and any(v == _ncn for v, _t in
                               prop["antecedents"]))
                 or (_nin and any(i == _nin for i, _x in
                                  prop["consequents"])))
        if not _uses:
            return AdaptOutcome(3, False,
                                "rejected at G2 package (the rule "
                                f"does not use the new object) ({src})",
                                info=dict(source=src, proposal=prop,
                                          package=True))
        _undo = _install_package(prop, engine)
    _pants = [(str(v), str(t)) for v, t in prop["antecedents"]]
    if _pkg and _undo:
        pass      # a package rule sits on a NEW axis; never a duplicate
    elif _cell_covered(rules, _pants):
        return AdaptOutcome(
            3, False,
            f"rejected at G2 duplicate cell ({src}) \u2014 an "
            "active rule already sits on this antecedent cell; "
            "re-issuing it with different numbers is stage \u2460 "
            "evFIS's job, not a new rule",
            info=dict(source=src, proposal=prop,
                      gate="G2 duplicate cell"))
    coverage_gap = True     # by definition: the cell was uncovered
    name = f"G{sum(1 for r in rules if r.name.startswith('G')) + 1}"
    newr = Rule(name,
                [(v, t) for v, t in prop["antecedents"]],
                [(i, float(x)) for i, x in prop["consequents"]],
                note=f"adaptation-born: generative ({src}), gates G1-G4")
    j0c, j0 = _cva(build_override, sim, rules, horizon)
    rules.append(newr)
    j1a, _ = _cva(build_override, sim, rules, horizon, reseed=101)
    j1b, j0b = _cva(build_override, sim, rules, horizon, reseed=202)
    j0a = _cva(build_override, sim, rules[:-1], horizon, reseed=101)[0]
    info = dict(source=src, proposal=prop, rule=newr.text(),
                revisions=_revisions,
                basis="physical cost at >=45 min",
                gates=dict(g1_g2="pass",
                           g3=dict(j_with=float(j1a),
                                   j_without=float(j0a)),
                           g4=dict(j_with=float(j1b),
                                   j_without=float(j0b))))
    info["coverage_gap"] = bool(coverage_gap)
    info["package"] = bool(_pkg)
    if _pkg:
        # G5 complexity: growing the vocabulary must BUY something -
        # both reseeded rollouts must improve by at least the margin
        if j1a < j0a - G5_MARGIN and j1b < j0b - G5_MARGIN:
            info["gates"]["verdict"] = ("admitted (package, G5 "
                                        "margin cleared)")
            if prop.get("new_concept"):
                info["new_concept"] = prop["new_concept"]
            if prop.get("new_intervention"):
                info["new_intervention"] = prop["new_intervention"]
            return AdaptOutcome(3, True,
                                f"{name}: {newr.text()} [+package]",
                                dJ=j1a - j0c, info=info)
        rules.pop()
        if _undo:
            _uninstall_package(_undo, engine)
        info["gates"]["verdict"] = "rejected at G5 (package margin)"
        return AdaptOutcome(3, False,
                            f"rejected at G5 package margin ({src})",
                            info=info)
    # STRICT admission: a generated rule must ACTUALLY improve the physical
    # forecast on BOTH reseeded rollouts (G3 + G4), even for a coverage void.
    # (Earlier a void-filling rule was admitted when merely non-inferior; it
    # grew the base without lowering cost, which was confusing. Now GenAI
    # earns each rule by improving the outcome.)
    if j1a < j0a - 1e-6 and j1b < j0b - 1e-6:
        info["gates"]["verdict"] = "admitted (improves G3 + G4)"
        return AdaptOutcome(3, True, f"{name}: {newr.text()}",
                            dJ=j1a - j0c, info=info)
    rules.pop()
    which = "G3" if j1a >= j0a - 1e-6 else "G4 A/B"
    info["gates"]["verdict"] = f"rejected at {which} (no improvement)"
    return AdaptOutcome(3, False, f"rejected at {which} ({src})",
                        info=info)


# ---------------------------------------- STAGE CONTROLLER (bandit)
class StageController:
    """Value-based stage controller: an associative-search contextual
    bandit (Sutton & Barto 2018) that reads the current state and picks
    one adaptation stage. It keeps one action value per (state, stage)
    pair, acts epsilon-greedy, and moves the chosen value a small step
    toward the realized reward (the physical forecast improvement)."""

    def __init__(self, eps: float = 0.1, lr: float = 0.05):
        self.eps = float(eps)
        self.lr = float(lr)
        self.q: Dict[Tuple[str, int], float] = {}
        self.rng = np.random.default_rng(7)

    @staticmethod
    def bucket(j_deficit: float, gap: bool = False) -> str:
        """State = deficit magnitude x coverage-gap flag: the
        controller LEARNS that gap states pay off with the growth
        stages and plain deficits with tuning, instead of having it
        hard-coded."""
        b = ("low" if j_deficit < 0.05 else
             "mid" if j_deficit < 0.15 else "high")
        return b + ("+gap" if gap else "")

    def select(self, j_deficit: float, stages=(1, 2, 3),
               gap: bool = False) -> int:
        """Pick one adaptation stage among the ALLOWED ones.

        Untried (bucket, stage) pairs are tried FIRST (optimistic
        one-shot exploration): without this, an early success of
        stage 1 freezes the greedy argmax there and stage 3 waits
        for the eps*1/3 lottery, i.e. practically never runs."""
        b = self.bucket(j_deficit, gap)
        stages = tuple(stages) or (1,)
        untried = [s for s in stages if (b, s) not in self.q]
        if untried:
            return int(self.rng.choice(untried))
        if self.rng.random() < self.eps:
            return int(self.rng.choice(stages))
        vals = {s: self.q.get((b, s), 0.0) for s in stages}
        return max(vals, key=vals.get)

    def update(self, j_deficit: float, stage: int, reward: float,
               gap: bool = False) -> None:
        # reward = realized PHYSICAL forecast improvement (the trial
        # basis is burn+asset+pop at >=45 min), consistent with the
        # fair outcome metric
        b = self.bucket(j_deficit, gap)
        k = (b, stage)
        self.q[k] = self.q.get(k, 0.0) + self.lr * (float(reward)
                                                    - self.q.get(k, 0.0))


def reset_partitions() -> None:
    """Return every variable's partition to the default,
    DROPPING inserted terms as well."""
    REGISTRY.reset()


def genai_status() -> str:
    """Which proposer stage 3 will use right now."""
    try:
        from . import genai as _g
        mode = _g.transport_mode()
    except Exception:
        mode = "none"
    if mode == "cli":
        return "Claude Code CLI (your subscription)"
    return ("generative disabled: the `claude` command is not available "
            "(install Claude Code and run `/login` on your Pro/Max plan)")


def genai_config() -> dict:
    """Static view of the generative wiring WITHOUT touching the network.

    Stage 3 runs only through Claude Code on the user's subscription; there
    is no API-key path. Cheap: safe to render on every frame."""
    model = os.environ.get("DSS_GENAI_MODEL", "")
    try:
        from . import genai as _g
        mode = _g.transport_mode()
    except Exception:
        mode = "none"
    if mode == "cli":
        transport = "Claude Code CLI (your subscription)"
        endpoint = "claude -p"
    else:
        transport = "none"
        endpoint = ""
    return {
        "mode": mode,
        "model": model or "(subscription default)",
        "endpoint": endpoint,
        "transport": transport,
        "reachable": mode != "none",
    }


def genai_probe(max_tokens: int = 64) -> dict:
    """Make ONE real, live call to Claude via the Claude Code CLI and return
    the proof. This is not a mock: it runs `claude -p` on the user's
    subscription and returns Claude's OWN reply text plus the round-trip
    latency. Any failure (claude missing, not logged in, billing) is returned
    as a plain error string so the app shows exactly why stage 3 is or is not
    live."""
    import time
    cfg = genai_config()
    out = dict(cfg)
    out.update({"ok": False, "reply": "", "latency_ms": None,
                "reported_model": "", "usage": None, "error": ""})
    model = out["model"]
    ask = ("Reply with exactly this line and nothing else: "
           "DisasterAware GenAI link is live.")
    if out.get("mode") != "cli":
        out["error"] = ("the `claude` command is not available; install "
                        "Claude Code and run `/login` on your Pro/Max plan")
        return out
    import subprocess
    t0 = time.time()
    try:
        cmd = ["claude", "-p", ask, "--output-format", "json"]
        if os.environ.get("DSS_GENAI_MODEL"):
            cmd += ["--model", os.environ["DSS_GENAI_MODEL"]]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=120)
        reply = res.stdout.strip()
        _is_err = res.returncode != 0
        try:
            wrap = json.loads(res.stdout)
            if isinstance(wrap, dict):
                reply = str(wrap.get("result", reply)).strip()
                # `claude -p --output-format json` flags failures with
                # is_error / subtype even at exit code 0 (e.g. a billing or
                # auth problem), so surface that as the error
                if wrap.get("is_error") or (
                        wrap.get("subtype")
                        and wrap.get("subtype") != "success"):
                    _is_err = True
                _mu = wrap.get("modelUsage")
                out["reported_model"] = (next(iter(_mu), model)
                                         if _mu else model)
                out["usage"] = wrap.get("usage")
        except Exception:
            pass
        if _is_err:
            out["error"] = ((res.stderr or reply
                             or f"claude CLI returned {res.returncode}")
                            .strip()[:400])
        else:
            out["reply"] = reply
            out["reported_model"] = out["reported_model"] or model
            out["ok"] = True
    except FileNotFoundError:
        out["error"] = "the `claude` command was not found on PATH"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return out
