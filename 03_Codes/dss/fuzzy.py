"""Five-term trapezoidal fuzzification (article Section III.D, Eq. 4).

Each bounded feature or concept in [0, 1] is fuzzified into five linguistic
terms (Very Low, Low, Medium, High, Very High). The membership functions are
trapezoids constructed so that they form a partition of unity: at every point
of the universe the memberships sum exactly to one. This keeps the concept
aggregation and the rule inference normalized by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

TERMS: Tuple[str, ...] = ("VL", "L", "M", "H", "VH")
CENTERS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

TERM_ALIASES = {
    "VL": "VL", "VERY LOW": "VL", "VERYLOW": "VL",
    "L": "L", "LOW": "L",
    "M": "M", "MEDIUM": "M", "MODERATE": "M",
    "H": "H", "HIGH": "H",
    "VH": "VH", "VERY HIGH": "VH", "VERYHIGH": "VH", "MAXIMUM": "VH",
}


def canonical_term(name: str) -> str:
    key = name.strip().upper()
    if key not in TERM_ALIASES:
        raise ValueError(f"unknown linguistic term: {name!r}")
    return TERM_ALIASES[key]


@dataclass(frozen=True)
class Trapezoid:
    """Trapezoidal membership function with knots a <= b <= c <= d."""

    a: float
    b: float
    c: float
    d: float

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        out = np.where((x >= self.b) & (x <= self.c), 1.0, out)
        if self.b > self.a:
            rise = (x - self.a) / (self.b - self.a)
            out = np.where((x >= self.a) & (x < self.b), rise, out)
        if self.d > self.c:
            fall = (self.d - x) / (self.d - self.c)
            out = np.where((x > self.c) & (x <= self.d), fall, out)
        return np.clip(out, 0.0, 1.0)


class FivePartition:
    """Five trapezoids over [0, 1] forming a partition of unity.

    Each term i has a plateau of half-width `plateau` around its center
    c_i in (0, 0.25, 0.5, 0.75, 1). Between the plateau of term i and the
    plateau of term i+1 the two memberships cross linearly and sum to one.
    The outer terms carry shoulders so the partition covers the full
    universe.
    """

    def __init__(self, plateau: float = 0.05):
        if not 0.0 <= plateau < 0.125:
            raise ValueError("plateau half-width must be in [0, 0.125)")
        self.plateau = float(plateau)
        w = self.plateau
        c = CENTERS
        mfs = []
        for i, ci in enumerate(c):
            a = c[i - 1] + w if i > 0 else -1.0
            b = max(ci - w, 0.0) if i > 0 else -1.0
            cc = min(ci + w, 1.0) if i < len(c) - 1 else 2.0
            d = c[i + 1] - w if i < len(c) - 1 else 2.0
            mfs.append(Trapezoid(a, b, cc, d))
        self.mfs: Sequence[Trapezoid] = tuple(mfs)

    def fuzzify(self, x) -> np.ndarray:
        """Return memberships stacked on the last axis, shape (..., 5)."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        return np.stack([mf(x) for mf in self.mfs], axis=-1)

    def membership(self, x, term: str) -> np.ndarray:
        """Membership of x in a single named term."""
        idx = TERMS.index(canonical_term(term))
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        return self.mfs[idx](x)

    @staticmethod
    def singleton(term: str) -> float:
        """Center value used as the singleton consequent of a term."""
        return CENTERS[TERMS.index(canonical_term(term))]
