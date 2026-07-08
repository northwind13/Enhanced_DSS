"""Five-term trapezoidal fuzzification (Layer 3 entry).

The partition follows the design established in the background chapter
(Figure 2.3): five terms VL, L, M, H, VH on a normalized [0, 1] universe,
uniform centers at 0, 0.25, 0.5, 0.75, 1, trapezoidal cores of half-width
0.05 and supports of half-width 0.20, with saturated ends. Adjacent terms
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
    """{term: (a, b, c, d)} trapezoids of the five-term partition."""
    part = {}
    for t in TERMS:
        c = TERM_CENTER[t]
        a, b = c - 0.20, c - 0.05
        cc, d = c + 0.05, c + 0.20
        if t == "VL":
            a, b = -1.0, 0.0          # saturated left end
        if t == "VH":
            cc, d = 1.0, 2.0          # saturated right end
        part[t] = (a, b, cc, d)
    return part


def trapmf(x, abcd):
    a, b, c, d = abcd
    x = np.asarray(x, dtype=float)
    up = (x - a) / max(b - a, 1e-9)
    dn = (d - x) / max(d - c, 1e-9)
    return np.clip(np.minimum(np.minimum(up, 1.0), dn), 0.0, 1.0)


def fuzzify(z: float, partition: dict | None = None) -> dict:
    """{term: membership} of a normalized reading z in [0, 1]."""
    part = partition or default_partition()
    z = float(np.clip(z, 0.0, 1.0))
    return {t: float(trapmf(z, part[t])) for t in TERMS}


def term_vector(z: float, partition: dict | None = None) -> np.ndarray:
    """Memberships as a 5-vector ordered as TERMS."""
    mu = fuzzify(z, partition)
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
