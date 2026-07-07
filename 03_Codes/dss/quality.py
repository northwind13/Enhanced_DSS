"""Decision quality score (article Eq. 10).

The quality score measures how closely a candidate intervention matches the
decision concepts it should serve. Each intervention type is driven by one
decision concept (article Section II.G); the per-component score is one minus
the mean absolute mismatch between the component intensity and its driving
concept over the cells where either is active. The total quality is the
weighted mean of the component scores.

This score gates the graduated fail-safe (Eq. 14) and, together with the
rollout cost (Eq. 9), drives the rule-base evolution in Phase 2.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# intervention type -> driving decision concept (article Section II.G)
DRIVER_MAP = {
    "suppression_effort": "fire_threat_level",
    "resource_deployment": "suppression_feasibility",
    "containment_line": "fire_threat_level",
    "asset_protection": "asset_risk",
    "evacuation": "evacuation_pressure",
    "public_warning": "intervention_urgency",
}

DEFAULT_COMPONENT_WEIGHTS = {
    "suppression_effort": 0.25,
    "resource_deployment": 0.15,
    "containment_line": 0.15,
    "asset_protection": 0.15,
    "evacuation": 0.20,
    "public_warning": 0.10,
}

ACTIVITY_EPS = 0.02


def decision_quality(intervention: Dict[str, np.ndarray],
                     concepts: Dict[str, np.ndarray],
                     region_mask: Optional[np.ndarray] = None,
                     weights: Optional[Dict[str, float]] = None) -> float:
    """Quality Q in [0, 1] of a candidate intervention (Eq. 10)."""
    weights = weights or DEFAULT_COMPONENT_WEIGHTS
    total_w = sum(weights.values())
    q_total = 0.0
    for itype, u in intervention.items():
        driver = concepts[DRIVER_MAP[itype]]
        active = (u > ACTIVITY_EPS) | (driver > ACTIVITY_EPS)
        if region_mask is not None:
            active &= region_mask
        if active.any():
            q_j = 1.0 - float(np.mean(np.abs(u[active] - driver[active])))
        else:
            q_j = 1.0  # nothing demanded, nothing applied: perfect match
        q_total += weights.get(itype, 0.0) * q_j
    return float(np.clip(q_total / total_w, 0.0, 1.0))
