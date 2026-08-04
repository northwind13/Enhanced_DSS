"""The closed rule catalogue, enumerated.

A CLOSED configuration is one that cannot write a rule at run time, so
every situation it will ever meet has to be answered in advance. Over the
five decision concepts with five linguistic terms each that is 5^5 = 3125
antecedent cells, the number the thesis quotes when it argues that a
closed catalogue is not maintainable by hand. This module builds it, so
that the closed baseline of the H1 experiment is closed in the sense the
claim uses: complete, and therefore large.

THE ALTERNATIVE WOULD BE A STRAW MAN IN THE WRONG DIRECTION. Running the
centralized baseline on the 42 seed rules makes it cheaper than the
distributed system rather than dearer, because it is then not covering
the decision space at all, only the corner of it the seed rules were
written for. That configuration answers a different question, and it is
available here as `seed_gap` so the coverage failure can be shown for
what it is.

The consequents are generated from a monotone policy rather than written
one by one. This is GENEROUS TO THE CLOSED BASELINE: a catalogue that a
team actually maintained by hand would be less consistent than one
produced by a formula, and would take longer to evaluate only if it were
worse. Nothing in the comparison depends on the policy being good, since
the complexity claim is about the size of the base and the cost of
sweeping it.
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from dss.concepts import DECISION_CONCEPTS   # noqa: E402
from dss.fuzzy import TERMS                  # noqa: E402
from dss.rules import Rule                   # noqa: E402

#: the linguistic terms as positions on the unit interval, so a cell of
#: the catalogue can be answered by arithmetic instead of by hand
_LEVEL = {t: i / (len(TERMS) - 1.0) for i, t in enumerate(TERMS)}


def build() -> list:
    """Every cell of the antecedent space, with an answer attached.

    The policy mirrors what the seed rules say in the corner they cover:
    suppression follows threat where suppression is feasible, a line is
    built where it is not, deployment follows threat and urgency, asset
    protection follows exposure, and evacuation follows evacuation
    pressure. Read across the whole space this is a complete controller,
    which is exactly what a closed configuration has to be.
    """
    out = []
    for n, combo in enumerate(itertools.product(TERMS,
                                                repeat=len(
                                                    DECISION_CONCEPTS))):
        d = dict(zip(DECISION_CONCEPTS, combo))
        thr = _LEVEL[d["fire_threat_level"]]
        ae = _LEVEL[d["asset_exposure_risk"]]
        sf = _LEVEL[d["suppression_feasibility"]]
        iu = _LEVEL[d["intervention_urgency"]]
        ep = _LEVEL[d["evacuation_pressure"]]
        cons = [("suppression_effort", round(min(1.0, thr * sf), 3)),
                ("resource_deployment",
                 round(min(1.0, 0.5 * thr + 0.5 * iu), 3)),
                ("containment_line", round(min(1.0, thr * (1.0 - sf)), 3)),
                ("asset_protection", round(ae, 3)),
                ("evacuation", round(ep, 3))]
        out.append(Rule(
            name=f"K{n + 1}",
            antecedents=[(c, d[c]) for c in DECISION_CONCEPTS],
            consequents=[(i, v) for i, v in cons if v > 0.0],
            note="closed catalogue, enumerated"))
    return out


def size():
    """(rules, antecedent evaluations per pass) without building it."""
    n = len(TERMS) ** len(DECISION_CONCEPTS)
    return n, n * len(DECISION_CONCEPTS)


if __name__ == "__main__":
    rs = build()
    print(f"{len(rs)} rules, "
          f"{sum(len(r.antecedents) for r in rs)} antecedent evaluations "
          f"per pass")
