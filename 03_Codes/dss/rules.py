"""Layer 4 rule base: the representative decision rules of Appendix D.

Antecedents are written over the gated activations of the five decision
concepts (five linguistic terms each). Consequents are NUMERIC
intervention intensities in [0, 1] (zero-order Takagi-Sugeno singletons,
the form the adaptation loop tunes); some rules additionally carry an
EFFECT (a cap, a scaling, a raise or the withdrawal of offensive
intensities) applied in proportion to the rule's firing strength.

Combination per intervention: activation-weighted average with a unit
floor on the denominator,

    u_i = sum_r w_r v_r / max(sum_r w_r, 1)

so a weakly supported conclusion yields a weak order and a fully fired
rule delivers its full intensity. A situation whose STRONGEST firing
stays below the coverage floor alpha_min is flagged as a coverage gap
(the adaptation trigger) and falls back to the watchful posture.

The base is SPARSE and evolving: the antecedent space holds 5^5 = 3125
combinations, only situation-activated rules are ever instantiated.
R13 and R14 of Appendix D illustrate ADAPTATION-BORN rules (evFIS-tuned
shoulder, GenAI-admitted rule); they arrive with the evolution loop and
are catalogued here with their provenance, inactive until then.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .fuzzy import TERMS, REGISTRY, term_vector

ALPHA_MIN = 0.05    # coverage floor (thesis Table E.3)

INTERVENTIONS = ("suppression_effort", "resource_deployment",
                 "containment_line", "asset_protection",
                 "evacuation", "public_warning")
INTERVENTION_LABEL = {
    "suppression_effort": "suppression effort",
    "resource_deployment": "resource deployment",
    "containment_line": "containment line",
    "asset_protection": "asset protection",
    "evacuation": "evacuation",
    "public_warning": "public warning",
}
# withdrawn on life-safety rules: the DIRECT-ATTACK families only;
# deployment and protection are support roles and remain
OFFENSIVE = ("suppression_effort", "containment_line")


@dataclass
class Rule:
    name: str
    antecedents: List[Tuple[str, str]]          # (concept, term)
    consequents: List[Tuple[str, float]]        # (intervention, intensity)
    effects: List[tuple] = field(default_factory=list)
    note: str = ""
    active: bool = True                          # provenance placeholders
    strength: float = 0.0    # accumulated fired weight of APPLIED
                             # decisions: the survival metric of the
                             # persistent learned-rule store

    def text(self) -> str:
        a = " AND ".join(f"{v.replace('_', ' ')} is {t}"
                         for v, t in self.antecedents)
        c = ", ".join(f"{i.replace('_', ' ')} {v:.1f}"
                      for i, v in self.consequents)
        fx = {"cap": lambda e: f"cap {e[1].replace('_', ' ')} at {e[2]:.1f}",
              "raise": lambda e: f"raise {e[1].replace('_', ' ')} by "
                                 f"{e[2]:.2f}",
              "scale_active": lambda e: f"scale active intensities x{e[1]}",
              "withdraw_offensive":
                  lambda e: "withdraw offensive intensities"}
        parts = [c] + [fx[e[0]](e) for e in self.effects]
        out = f"{self.name}: IF {a} THEN " + "; ".join(p for p in parts if p)
        if self.note:
            out += f" ({self.note})"
        return out


def _R(name, ants, cons, effects=None, note="", active=True):
    return Rule(name, ants, cons, list(effects or []), note, active)


SEED_RULES: List[Rule] = [
    # nominal suppression regime
    _R("R1", [("fire_threat_level", "H"), ("suppression_feasibility", "H")],
       [("suppression_effort", 0.8), ("resource_deployment", 0.6)]),
    _R("R2", [("fire_threat_level", "M"), ("suppression_feasibility", "M")],
       [("suppression_effort", 0.6)]),
    _R("R3", [("fire_threat_level", "VL")],
       [("suppression_effort", 0.1)], note="watchful posture"),
    _R("R4", [("fire_threat_level", "H"), ("suppression_feasibility", "H"),
              ("intervention_urgency", "H")],
       [("suppression_effort", 0.9), ("resource_deployment", 0.8)]),
    _R("R5", [("fire_threat_level", "M"), ("suppression_feasibility", "H"),
              ("asset_exposure_risk", "L")],
       [("suppression_effort", 0.6), ("resource_deployment", 0.4)]),
    # asset protection regime
    _R("R6", [("asset_exposure_risk", "H"), ("fire_threat_level", "M")],
       [("asset_protection", 0.7), ("suppression_effort", 0.4)]),
    _R("R7", [("asset_exposure_risk", "VH"),
              ("suppression_feasibility", "L")],
       [("asset_protection", 0.9), ("containment_line", 0.6)]),
    _R("R8", [("asset_exposure_risk", "H"), ("fire_threat_level", "H"),
              ("suppression_feasibility", "M")],
       [("asset_protection", 0.8), ("suppression_effort", 0.6),
        ("containment_line", 0.4)]),
    _R("R9", [("asset_exposure_risk", "M"), ("fire_threat_level", "H")],
       [("asset_protection", 0.5), ("suppression_effort", 0.6)]),
    # evacuation and warning regime
    _R("R10", [("evacuation_pressure", "M"), ("intervention_urgency", "H")],
       [("public_warning", 0.7), ("evacuation", 0.4)]),
    _R("R11", [("evacuation_pressure", "VH"),
               ("suppression_feasibility", "VL")],
       [("evacuation", 1.0), ("public_warning", 1.0),
        ("resource_deployment", 0.7)],
       effects=[("withdraw_offensive",)],
       note="deployment supports the evacuation; offensive intensities "
            "withdrawn"),
    _R("R12", [("evacuation_pressure", "H"), ("asset_exposure_risk", "VH"),
               ("intervention_urgency", "H")],
       [("evacuation", 0.9), ("public_warning", 0.9),
        ("asset_protection", 0.5)]),
    _R("R13", [("evacuation_pressure", "M"), ("fire_threat_level", "VH")],
       [("public_warning", 0.8), ("evacuation", 0.5)]),
    # feasibility-limited regime
    _R("R14", [("fire_threat_level", "H"), ("suppression_feasibility", "L")],
       [("containment_line", 0.7), ("resource_deployment", 0.8)]),
    _R("R15", [("fire_threat_level", "VH"),
               ("suppression_feasibility", "VL")],
       [("containment_line", 0.9), ("evacuation", 0.6),
        ("public_warning", 0.8)]),
    _R("R16", [("fire_threat_level", "H"), ("suppression_feasibility", "L"),
               ("asset_exposure_risk", "H")],
       [("containment_line", 0.8), ("asset_protection", 0.7),
        ("resource_deployment", 0.6)]),
    _R("R17", [("fire_threat_level", "M"),
               ("suppression_feasibility", "VL"),
               ("intervention_urgency", "M")],
       [("containment_line", 0.5), ("resource_deployment", 0.3)]),
    _R("R18", [("suppression_feasibility", "M"),
               ("intervention_urgency", "H")],
       [("resource_deployment", 0.6), ("containment_line", 0.4)]),
    # life-safety boundary
    _R("R19", [("fire_threat_level", "H"), ("asset_exposure_risk", "VH")],
       [("evacuation", 0.9)],
       note="regardless of cost ranking, hard ceiling active"),
    _R("R20", [("intervention_urgency", "VH")],
       [("resource_deployment", 1.0)],
       note="priority preemption across regions"),
    _R("R21", [("fire_threat_level", "VH"), ("evacuation_pressure", "H"),
               ("suppression_feasibility", "L")],
       [("evacuation", 1.0), ("public_warning", 1.0),
        ("containment_line", 0.6)]),
    _R("R22", [("asset_exposure_risk", "VH"),
               ("intervention_urgency", "VH")],
       [("asset_protection", 0.9), ("evacuation", 0.8),
        ("resource_deployment", 0.9)]),
    # backbone: fire threat level -> suppression effort
    _R("R23", [("fire_threat_level", "L")], [("suppression_effort", 0.2)]),
    _R("R24", [("fire_threat_level", "M")], [("suppression_effort", 0.5)]),
    _R("R25", [("fire_threat_level", "H")], [("suppression_effort", 0.8)]),
    _R("R26", [("fire_threat_level", "VH")],
       [("suppression_effort", 1.0), ("resource_deployment", 0.8)]),
    # backbone: asset exposure risk -> asset protection
    _R("R27", [("asset_exposure_risk", "L")], [("asset_protection", 0.1)]),
    _R("R28", [("asset_exposure_risk", "M")], [("asset_protection", 0.4)]),
    _R("R29", [("asset_exposure_risk", "H")], [("asset_protection", 0.7)]),
    _R("R30", [("asset_exposure_risk", "VH")],
       [("asset_protection", 0.9), ("containment_line", 0.5)]),
    # backbone: suppression feasibility (gatekeeper)
    _R("R31", [("suppression_feasibility", "VL")],
       [("containment_line", 0.6)],
       effects=[("cap", "suppression_effort", 0.2),
                ("raise", "evacuation", 0.25)],
       note="suppression capped, evacuation raised one term"),
    _R("R32", [("suppression_feasibility", "L")],
       [("containment_line", 0.5)],
       effects=[("cap", "suppression_effort", 0.4)],
       note="suppression capped"),
    _R("R33", [("suppression_feasibility", "H")],
       [("resource_deployment", 0.6)], note="toward the active front"),
    _R("R34", [("suppression_feasibility", "VH")],
       [("resource_deployment", 0.8)],
       effects=[("raise", "suppression_effort", 0.25)],
       note="suppression raised one term; deployment toward direct "
            "attack"),
    # backbone: intervention urgency -> deployment tempo
    _R("R35", [("intervention_urgency", "M")],
       [("resource_deployment", 0.4)]),
    _R("R36", [("intervention_urgency", "H")],
       [("resource_deployment", 0.7)],
       effects=[("scale_active", 1.2)],
       note="active intensities scaled within bounds"),
    # backbone: evacuation pressure -> evacuation
    _R("R37", [("evacuation_pressure", "L")], [("public_warning", 0.3)]),
    _R("R38", [("evacuation_pressure", "M")],
       [("public_warning", 0.6), ("evacuation", 0.3)]),
    _R("R39", [("evacuation_pressure", "H")],
       [("evacuation", 0.7), ("public_warning", 0.8)]),
    _R("R40", [("evacuation_pressure", "VH")],
       [("evacuation", 1.0), ("public_warning", 1.0)],
       effects=[("withdraw_offensive",)],
       note="offensive intensities withdrawn"),
    # adaptation-born examples (provenance; arrive with the evolution loop)
    _R("R41", [("fire_threat_level", "H")],
       [("suppression_effort", 0.85)],
       note="adaptation-born: evFIS shoulder 0.55 to 0.50, consequent "
            "retuned from 0.80", active=False),
    _R("R42", [("evacuation_pressure", "H"),
               ("suppression_feasibility", "VL")],
       [("evacuation", 0.8), ("public_warning", 0.9)],
       effects=[("withdraw_offensive",)],
       note="adaptation-born: GenAI-admitted through gates G1-G4",
       active=False),
]


def _membership(var: str, term: str, concepts: Dict[str, np.ndarray],
                features: Dict[str, float]) -> float:
    if var in concepts:
        return float(concepts[var][TERMS.index(term)])
    if var in features:
        return float(term_vector(float(features[var]),
                                 var=var)[TERMS.index(term)])
    return 0.0


def evaluate_rules(concepts: Dict[str, np.ndarray],
                   features: Dict[str, float],
                   rules: List[Rule] | None = None):
    """Zero-order TS pass over the active rules.

    Returns (intensities dict, trace rows [(rule, firing strength)]).
    A coverage gap (strongest firing < ALPHA_MIN) is reported through
    the trace: all rows carry ~zero strength and the intensities fall
    back to the watchful posture."""
    rules = SEED_RULES if rules is None else rules
    num = {i: 0.0 for i in INTERVENTIONS}
    den = {i: 0.0 for i in INTERVENTIONS}
    fx: List[Tuple[tuple, float]] = []
    trace = []
    w_max = 0.0
    for r in rules:
        if not r.active:
            trace.append((r, 0.0))
            continue
        w = min(_membership(v, t, concepts, features)
                for v, t in r.antecedents)
        trace.append((r, float(w)))
        w_max = max(w_max, w)
        if w <= 1e-9:
            continue
        for interv, v in r.consequents:
            num[interv] += w * float(v)
            den[interv] += w
        for e in r.effects:
            fx.append((e, w))
    if w_max < ALPHA_MIN:
        return {i: 0.0 for i in INTERVENTIONS}, trace
    out = {i: (num[i] / max(den[i], 1.0)) for i in INTERVENTIONS}
    # effects, blended by firing strength
    for e, w in fx:
        if e[0] == "cap":
            _, iv, cap = e
            out[iv] = min(out[iv], cap * w + out[iv] * (1.0 - w))
        elif e[0] == "raise":
            _, iv, dv = e
            out[iv] = min(1.0, out[iv] + dv * w)
        elif e[0] == "scale_active":
            k = 1.0 + (float(e[1]) - 1.0) * w
            for iv in INTERVENTIONS:
                if out[iv] > 0.0:
                    out[iv] = min(1.0, out[iv] * k)
        elif e[0] == "withdraw_offensive":
            for iv in OFFENSIVE:
                out[iv] = out[iv] * (1.0 - w)
    return {i: float(np.clip(v, 0.0, 1.0)) for i, v in out.items()}, trace
