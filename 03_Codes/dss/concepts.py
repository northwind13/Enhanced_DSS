"""Layer 3 concept space: four-level hierarchy with confidence gating.

Implements the concept construction of the article (Eq. 5, TABLE I) and the
confidence gating with a persistence-based prior (Eq. 6). The hierarchy runs:

    Level 1 (base, from features):
        fire_severity, spread_hazard, fuel_hazard, asset_value,
        crew_reachability, logistics
    Level 2 (decision): fire_threat_level, asset_risk, suppression_feasibility
    Level 3 (decision): intervention_urgency, evacuation_pressure
    Level 4 (coordination): operational_priority

Every concept is a convex combination of its sources, so activations stay in
[0, 1] by construction. A source can be inverted (weight applied to 1 - x),
which is how poor road access raises evacuation pressure (TABLE I, inverted).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# each source is (name, weight, invert); weights are normalized at build time
Source = Tuple[str, float, bool]

BASE_CONCEPTS: Dict[str, List[Source]] = {
    # TABLE I, level 1: built from features
    "fire_severity": [("fire_intensity", 0.6, False),
                      ("spread_potential", 0.4, False)],
    "spread_hazard": [("spread_potential", 0.5, False),
                      ("weather_severity", 0.3, False),
                      ("ignition_proximity", 0.2, False)],
    "fuel_hazard": [("fuel_load", 0.6, False),
                    ("weather_severity", 0.4, False)],
    "asset_value": [("asset_exposure", 1.0, False)],
    "crew_reachability": [("resource_accessibility", 0.6, False),
                          ("access_road_status", 0.4, False)],
    "logistics": [("suppression_availability", 1.0, False)],
}

MID_CONCEPTS: Dict[str, List[Source]] = {
    # TABLE I, level 2
    "fire_threat_level": [("fire_severity", 0.40, False),
                          ("spread_hazard", 0.35, False),
                          ("fuel_hazard", 0.25, False)],
    "asset_risk": [("spread_hazard", 0.45, False),
                   ("asset_value", 0.55, False)],
    "suppression_feasibility": [("crew_reachability", 0.60, False),
                                ("logistics", 0.40, False)],
}

HIGHER_CONCEPTS: Dict[str, List[Source]] = {
    # TABLE I, level 3; temporal_urgency and access_road_status are features
    "intervention_urgency": [("fire_threat_level", 0.40, False),
                             ("asset_risk", 0.30, False),
                             ("temporal_urgency", 0.30, False)],
    "evacuation_pressure": [("asset_risk", 0.40, False),
                            ("access_road_status", 0.25, True),
                            ("temporal_urgency", 0.35, False)],
}

TOP_CONCEPTS: Dict[str, List[Source]] = {
    # TABLE I, level 4: coordination only, never fires decision rules
    "operational_priority": [("fire_threat_level", 0.5, False),
                             ("asset_risk", 0.5, False)],
}

DECISION_CONCEPTS = ("fire_threat_level", "asset_risk",
                     "suppression_feasibility", "intervention_urgency",
                     "evacuation_pressure")

CONCEPT_LEVELS = (BASE_CONCEPTS, MID_CONCEPTS, HIGHER_CONCEPTS, TOP_CONCEPTS)

ALL_CONCEPTS = tuple(name for level in CONCEPT_LEVELS for name in level)


def _aggregate(sources: List[Source], signals: Dict[str, np.ndarray]) -> np.ndarray:
    total_w = sum(w for _, w, _ in sources)
    if total_w <= 0:
        raise ValueError("concept must have positive total weight")
    acc = None
    for name, w, invert in sources:
        x = signals[name]
        x = 1.0 - x if invert else x
        acc = (w / total_w) * x if acc is None else acc + (w / total_w) * x
    return np.clip(acc, 0.0, 1.0)


def compute_concepts(features) -> Dict[str, np.ndarray]:
    """Raw concept activations c_n per cell (Eq. 5), level by level.

    features : a FeatureSet or a plain dict of feature fields.
    """
    signals: Dict[str, np.ndarray] = dict(
        features.values if hasattr(features, "values")
        and isinstance(getattr(features, "values"), dict) else features)
    concepts: Dict[str, np.ndarray] = {}
    for level in CONCEPT_LEVELS:
        for name, sources in level.items():
            concepts[name] = _aggregate(sources, signals)
        signals.update(concepts)
    return concepts


def gate_concepts(raw: Dict[str, np.ndarray],
                  prior: Optional[Dict[str, np.ndarray]],
                  kappa: np.ndarray) -> Dict[str, np.ndarray]:
    """Confidence gating with persistence prior (Eq. 6).

        c_gated = kappa * c_raw + (1 - kappa) * c_prior

    A fully reliable observation (kappa = 1) passes through unchanged; an
    uninformative one (kappa = 0) falls back to the previous estimate. On the
    first step (no prior) the raw activation is used as its own prior, which
    reduces the gate to the identity.
    """
    gated: Dict[str, np.ndarray] = {}
    for name, c_raw in raw.items():
        c_prior = raw[name] if prior is None or name not in prior else prior[name]
        gated[name] = np.clip(kappa * c_raw + (1.0 - kappa) * c_prior, 0.0, 1.0)
    return gated
