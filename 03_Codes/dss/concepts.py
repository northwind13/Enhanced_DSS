"""Layer 3: the four-level concept hierarchy (Eqs. 40-41).

Concept activations are five-term vectors formed by weighted aggregation:
at the base level over the term memberships of the assigned features, at
every higher level over the activations of the contributing concepts one
level below, with nonnegative weights summing to one per concept (Eq. 40).
Raw activations are never consumed directly: every activation is gated by
its concept-level confidence (the weakest feature confidence among its
inputs) and blended with a persistence-carried prior (Eq. 41):

    a_eff = gamma * a_obs + (1 - gamma) * rho * a_eff_prev

so a reliably observed concept passes through unchanged while an
uninformative one falls back to the decayed previous estimate.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .fuzzy import TERMS, term_vector, expected_value
from .features import FEATURE_ORDER

RHO_PERSIST = 0.9      # decay of the persistence prior per decision cycle

# concept -> (level, [(input_name, weight)]), inputs are features (z keys)
# for level 1 and concepts for levels 2-4; weights sum to one per concept
HIERARCHY: Dict[str, Tuple[int, List[Tuple[str, float]]]] = {
    # level 1 - base concepts
    "fire_severity":       (1, [("fire_intensity", 0.60),
                                ("spread_potential", 0.40)]),
    "spread_hazard":       (1, [("spread_potential", 0.45),
                                ("weather_severity", 0.35),
                                ("ignition_proximity", 0.20)]),
    "fuel_hazard":         (1, [("weather_severity", 0.40),
                                ("fuel_load", 0.60)]),
    "asset_value":         (1, [("asset_exposure", 1.00)]),
    "crew_reachability":   (1, [("resource_accessibility", 0.55),
                                ("access_road_status", 0.45)]),
    "logistics_support":   (1, [("suppression_availability", 1.00)]),
    # level 2 - mid decision concepts
    "fire_threat_level":   (2, [("fire_severity", 0.40),
                                ("spread_hazard", 0.40),
                                ("fuel_hazard", 0.20)]),
    "asset_exposure_risk": (2, [("spread_hazard", 0.45),
                                ("asset_value", 0.55)]),
    "suppression_feasibility": (2, [("crew_reachability", 0.60),
                                    ("logistics_support", 0.40)]),
    # level 3 - higher decision concepts
    "intervention_urgency": (3, [("fire_threat_level", 0.45),
                                 ("asset_exposure_risk", 0.35),
                                 ("temporal_urgency", 0.20)]),
    "evacuation_pressure":  (3, [("asset_exposure_risk", 0.40),
                                 ("access_road_status_inv", 0.30),
                                 ("temporal_urgency", 0.30)]),
    # level 4 - coordination concept (fires no local rule)
    "operational_priority": (4, [("fire_threat_level", 0.50),
                                 ("asset_exposure_risk", 0.50)]),
}

DECISION_CONCEPTS = ("fire_threat_level", "asset_exposure_risk",
                     "suppression_feasibility", "intervention_urgency",
                     "evacuation_pressure")
CONCEPT_LABEL = {
    "fire_severity": "fire severity", "spread_hazard": "spread hazard",
    "fuel_hazard": "fuel hazard", "asset_value": "asset value",
    "crew_reachability": "crew reachability",
    "logistics_support": "logistics support",
    "fire_threat_level": "fire threat level",
    "asset_exposure_risk": "asset exposure risk",
    "suppression_feasibility": "suppression feasibility",
    "intervention_urgency": "intervention urgency",
    "evacuation_pressure": "evacuation pressure",
    "operational_priority": "operational priority",
}


def _feature_inputs(features: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Five-term vectors of the features (plus the inverted road status
    used by evacuation pressure: blocked egress raises the pressure)."""
    vecs = {k: term_vector(float(features[k])) for k in FEATURE_ORDER}
    vecs["access_road_status_inv"] = term_vector(
        1.0 - float(features["access_road_status"]))
    return vecs


def infer_concepts(features: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Observed activations a_obs (Eq. 40): five-term vector per concept."""
    vecs = _feature_inputs(features)
    act: Dict[str, np.ndarray] = {}
    for lvl in (1, 2, 3, 4):
        for name, (l, inputs) in HIERARCHY.items():
            if l != lvl:
                continue
            v = np.zeros(len(TERMS))
            for src, w in inputs:
                v += w * (act[src] if src in act else vecs[src])
            act[name] = np.clip(v, 0.0, 1.0)
    return act


def concept_conf(features_conf: float) -> float:
    """Concept-level gate: weakest feature confidence among the inputs.
    Phase note: the network exposes one fused region confidence, so it is
    the common lower bound for every concept in the region."""
    return float(np.clip(features_conf, 0.0, 1.0))


class GatedConcepts:
    """Keeps the persistence prior and applies the gate of Eq. 41."""

    def __init__(self, rho: float = RHO_PERSIST):
        self.rho = float(rho)
        self.prev: Dict[str, np.ndarray] = {}
        self.step = None

    def gate(self, act_obs: Dict[str, np.ndarray], gamma: float,
             step: int | None = None) -> Dict[str, np.ndarray]:
        if step is not None and self.step == step and self.prev:
            base = self.prev          # same step: do not re-decay
        else:
            base = self.prev
        g = float(np.clip(gamma, 0.0, 1.0))
        eff: Dict[str, np.ndarray] = {}
        for name, a in act_obs.items():
            p = base.get(name, a)
            eff[name] = np.clip(g * a + (1.0 - g) * self.rho * p, 0.0, 1.0)
        if step is None or self.step != step:
            self.prev = {k: v.copy() for k, v in eff.items()}
            self.step = step
        return eff


def crisp(act: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {k: expected_value(v) for k, v in act.items()}
