"""Layer 4 rule base: representative decision rules (Appendix D).

Rules are stated in the canonical conjunctive form over the gated
activations of the five decision concepts; two rules additionally read a
gated feature (access status, temporal urgency) as auxiliary antecedent.
The base is SPARSE and evolving: only situation-activated combinations are
ever instantiated, seeded here with the representative subset R1-R8.

Inference is Mamdani: firing strength = min over antecedent memberships,
consequents clipped at the firing strength, aggregated by pointwise max
per intervention, defuzzified by centroid over the five-term output
partition on [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .fuzzy import TERMS, default_partition, trapmf, term_vector

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


@dataclass
class Rule:
    name: str
    antecedents: List[Tuple[str, str]]        # (variable, term)
    consequents: List[Tuple[str, str]]        # (intervention, term)
    note: str = ""

    def text(self) -> str:
        a = " AND ".join(f"{v.replace('_', ' ')} is {t}"
                         for v, t in self.antecedents)
        c = ", ".join(f"{i.replace('_', ' ')} is {t}"
                      for i, t in self.consequents)
        return f"{self.name}: IF {a} THEN {c}" + \
            (f" ({self.note})" if self.note else "")


# Appendix D seeds (R4 and R7 read gated features as auxiliary antecedents)
SEED_RULES: List[Rule] = [
    Rule("R1", [("fire_threat_level", "H"),
                ("suppression_feasibility", "H")],
         [("suppression_effort", "H")]),
    Rule("R2", [("fire_threat_level", "VL")],
         [("suppression_effort", "VL")], "watchful posture"),
    Rule("R3", [("asset_exposure_risk", "H"),
                ("fire_threat_level", "M")],
         [("asset_protection", "H")]),
    Rule("R4", [("evacuation_pressure", "H"),
                ("access_road_status", "L")],
         [("evacuation", "VH"), ("public_warning", "VH")]),
    Rule("R5", [("suppression_feasibility", "L"),
                ("fire_threat_level", "H")],
         [("containment_line", "H"), ("resource_deployment", "H")]),
    Rule("R6", [("intervention_urgency", "VH")],
         [("resource_deployment", "VH")], "priority preemption"),
    Rule("R7", [("evacuation_pressure", "M"),
                ("temporal_urgency", "H")],
         [("public_warning", "H")]),
    Rule("R8", [("fire_threat_level", "H"),
                ("asset_exposure_risk", "VH")],
         [("evacuation", "H")], "life-safety ceiling, regardless of cost"),
]

_X = np.linspace(0.0, 1.0, 101)
_PART = default_partition()
_OUT_MF = {t: trapmf(_X, _PART[t]) for t in TERMS}


def _membership(var: str, term: str, concepts: Dict[str, np.ndarray],
                features: Dict[str, float]) -> float:
    """Antecedent membership: concept term vectors first, gated features
    (fuzzified on demand) as the auxiliary path."""
    if var in concepts:
        return float(concepts[var][TERMS.index(term)])
    if var in features:
        return float(term_vector(float(features[var]))[TERMS.index(term)])
    return 0.0


def evaluate_rules(concepts: Dict[str, np.ndarray],
                   features: Dict[str, float],
                   rules: List[Rule] | None = None):
    """Mamdani pass. Returns (intensities per intervention in [0,1],
    trace rows [(rule, firing strength)])."""
    rules = SEED_RULES if rules is None else rules
    agg = {i: np.zeros_like(_X) for i in INTERVENTIONS}
    trace = []
    for r in rules:
        w = min(_membership(v, t, concepts, features)
                for v, t in r.antecedents)
        trace.append((r, float(w)))
        if w <= 1e-9:
            continue
        for interv, term in r.consequents:
            agg[interv] = np.maximum(agg[interv],
                                     np.minimum(_OUT_MF[term], w))
    out = {}
    for i in INTERVENTIONS:
        m = agg[i]
        s = float(m.sum())
        out[i] = float((m * _X).sum() / s) if s > 1e-9 else 0.0
    return out, trace
