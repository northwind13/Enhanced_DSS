"""Fuzzy inference over the gated concept vector (article Eq. 15-16).

Rule activation uses the min t-norm over the antecedent memberships of the
gated decision concepts (Eq. 15). The intervention component per type is the
strength-weighted convex combination of the singleton consequents of the
rules that mention that type (Eq. 16), so every output intensity stays inside
the range spanned by the fired consequents. Everything is vectorized per
grid cell.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .rules import RuleBase, INTERVENTION_TYPES

EPS = 1e-9


def fire_rules(rule_base: RuleBase, signals: Dict[str, np.ndarray]
               ) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, np.ndarray]]]:
    """Fire the rule base on per-cell signal fields.

    signals : gated decision concepts plus any feature referenced by an
        antecedent (e.g. access_road_status in R4).

    Returns (intervention, firings):
        intervention : dict intervention_type -> intensity field in [0, 1]
        firings      : list of (rule_id, strength field) for traceability
    """
    part = rule_base.partition
    shape = next(iter(signals.values())).shape

    firings: List[Tuple[str, np.ndarray]] = []
    num = {j: np.zeros(shape) for j in INTERVENTION_TYPES}
    den = {j: np.zeros(shape) for j in INTERVENTION_TYPES}

    for rule in rule_base.rules:
        strength = None
        for name, term in rule.antecedent.items():
            if name not in signals:
                raise KeyError(f"rule {rule.rule_id} references unknown "
                               f"signal {name!r}")
            mu = part.membership(signals[name], term)
            strength = mu if strength is None else np.minimum(strength, mu)
        w = rule.weight * strength
        firings.append((rule.rule_id, w))
        for itype, y in rule.consequent.items():
            num[itype] += w * y
            den[itype] += w

    intervention = {}
    for itype in INTERVENTION_TYPES:
        out = np.where(den[itype] > EPS, num[itype] / (den[itype] + EPS), 0.0)
        intervention[itype] = np.clip(out, 0.0, 1.0)
    return intervention, firings
