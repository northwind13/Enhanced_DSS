"""Five-term trapezoidal fuzzification (Layer 3 entry).

The partition follows the design established in the background chapter
(Figure 2.3): five terms VL, L, M, H, VH on a normalized [0, 1] universe,
with the exact (a, b, c, d) parameters of thesis Table E.4. Adjacent terms
overlap so that every reading activates AT MOST TWO terms and the two
memberships sum to one on the shoulders. Worked check: a reading of 0.62
gives medium 0.53 and high 0.47, exactly the example of the text.

Each membership function is stored as an editable trapezoid (a, b, c, d);
the evolving-FIS stage perturbs precisely these four parameters.
"""

from __future__ import annotations

import numpy as np

TERMS = ("VL", "L", "M", "H", "VH")
TERM_CENTER = {"VL": 0.0, "L": 0.25, "M": 0.5, "H": 0.75, "VH": 1.0}


def default_partition() -> dict:
    """{term: (a, b, c, d)} trapezoids of the five-term partition.

    Values are Table E.4 of the thesis appendices, verbatim. The
    partition reproduces the worked example of the background chapter
    (a reading of 0.62 gives medium 0.533 and high 0.467) and adjacent
    terms overlap so at most two terms activate anywhere."""
    return {"VL": (0.00, 0.00, 0.15, 0.30),
            "L":  (0.15, 0.30, 0.35, 0.50),
            "M":  (0.35, 0.50, 0.55, 0.70),
            "H":  (0.55, 0.70, 0.75, 0.90),
            "VH": (0.70, 0.85, 1.00, 1.00)}


class PartitionRegistry:
    """Per-variable membership partitions.

    Every linguistic variable (the ten features, the twelve concepts and
    the six intervention outputs) owns its OWN editable copy of the
    five-term partition, initialized to the shared default. This is the
    surface the evolving-FIS stage operates on: it perturbs the (a, b,
    c, d) parameters of the terms involved in deficient rules for one
    variable WITHOUT touching any other variable's semantics."""

    def __init__(self):
        self._parts: dict = {}

    def get(self, var: str | None) -> dict:
        if var is None:
            return default_partition()
        if var not in self._parts:
            self._parts[var] = default_partition()
        return self._parts[var]

    def set_term(self, var: str, term: str, abcd) -> None:
        part = dict(self.get(var))
        a, b, c, d = (float(v) for v in abcd)
        assert a <= b <= c <= d, "trapezoid must satisfy a <= b <= c <= d"
        part[term] = (a, b, c, d)
        self._parts[var] = part

    def variables(self):
        return sorted(self._parts)


REGISTRY = PartitionRegistry()


def trapmf(x, abcd):
    """Trapezoid membership; degenerate shoulders (a == b or c == d, the
    saturated ends of Table E.4) read as hard steps so the extremes of
    the universe carry full membership."""
    a, b, c, d = abcd
    x = np.asarray(x, dtype=float)
    up = ((x - a) / (b - a)) if b > a else (x >= a).astype(float) * 2.0
    dn = ((d - x) / (d - c)) if d > c else (x <= d).astype(float) * 2.0
    return np.clip(np.minimum(np.minimum(up, 1.0), dn), 0.0, 1.0)


def fuzzify(z: float, partition: dict | None = None,
            var: str | None = None) -> dict:
    """{term: membership} of a normalized reading z in [0, 1].
    With var given, the variable's OWN partition from the registry is
    used (falls back to the shared default until it is ever edited)."""
    part = partition or REGISTRY.get(var)
    z = float(np.clip(z, 0.0, 1.0))
    return {t: float(trapmf(z, part[t])) for t in TERMS}


def term_vector(z: float, partition: dict | None = None,
                var: str | None = None) -> np.ndarray:
    """Memberships as a 5-vector ordered as TERMS."""
    mu = fuzzify(z, partition, var=var)
    return np.array([mu[t] for t in TERMS], dtype=float)


def expected_value(vec) -> float:
    """Crisp readout of a five-term activation vector (center average).
    The divisor is capped at one so a decayed (persistence-faded) vector
    reads lower: fading is visible, a fresh full-mass reading unchanged."""
    vec = np.asarray(vec, dtype=float)
    w = vec.sum()
    if w <= 1e-12:
        return 0.0
    centers = np.array([TERM_CENTER[t] for t in TERMS])
    return float((vec * centers).sum() / max(w, 1.0))
