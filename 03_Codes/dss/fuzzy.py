"""Five-term trapezoidal fuzzification (Layer 3 entry).

The partition follows the design established in the background chapter
(Figure 2.3): five terms VL, L, M, H, VH on a normalized [0, 1] universe,
with the (a, b, c, d) parameters of thesis Table D.3, the
single place the document states them. Adjacent terms
overlap so that every reading activates AT MOST TWO terms and the two
memberships sum to one on the shoulders. Worked check: a reading of 0.62
gives medium 0.53 and high 0.47, exactly the example of the text.

RUSPINI INVARIANT. The partition is a Ruspini (strong) partition:
sum_t mu_t(x) = 1 for every x in [0, 1]. This holds because neighbouring
trapezoids SHARE their transition interval: the descending edge (c, d) of
term i is the ascending edge (a, b) of term i+1. The invariant is what makes
the inference output a convex combination of the consequents, so it is a
hard constraint, not a cosmetic property.

Consequently a shoulder is NOT a free parameter of one term: it is a SHARED
boundary of two terms. The evolving-FIS stage must therefore move it through
``PartitionRegistry.shift_boundary``, which displaces both trapezoids at once
and bounds the step by the neighbouring core widths (a larger step would
collapse a plateau and then invert the trapezoid). Moving a single trapezoid
in isolation tears the partition (sum_t mu_t != 1) and is a bug.
"""

from __future__ import annotations

import numpy as np

TERMS = ("VL", "L", "M", "H", "VH")
TERM_CENTER = {"VL": 0.0, "L": 0.25, "M": 0.5, "H": 0.75, "VH": 1.0}


def default_partition() -> dict:
    """{term: (a, b, c, d)} trapezoids of the five-term partition.

    Values follow Table D.3 with ONE correction. The
    published "very high" row reads (0.70, 0.85, 1.00, 1.00), which does
    not share "high"'s descending edge (0.75, 0.90) and therefore breaks
    the Ruspini invariant: sum_t mu_t climbs to 1.333 on x in [0.70,
    0.90], and the concept activation vectors inherit that defect. The
    row is corrected here to (0.75, 0.90, 1.00, 1.00), which restores
    sum_t mu_t = 1 everywhere and leaves every other row untouched. The
    partition still reproduces the worked example of the background
    chapter (a reading of 0.62 gives medium 0.533 and high 0.467) and
    adjacent terms overlap so at most two terms activate anywhere.

    Table D.3 in the document must be corrected on that one row."""
    return {"VL": (0.00, 0.00, 0.15, 0.30),
            "L":  (0.15, 0.30, 0.35, 0.50),
            "M":  (0.35, 0.50, 0.55, 0.70),
            "H":  (0.55, 0.70, 0.75, 0.90),
            # CORRECTED ROW (Table D.3): VH must ascend on H's descending
            # edge (0.75, 0.90). The published (0.70, 0.85) does NOT share
            # that boundary and breaks the Ruspini invariant: sum_t mu_t
            # reaches 1.333 on x in [0.70, 0.90].
            "VH": (0.75, 0.90, 1.00, 1.00)}


class PartitionRegistry:
    """Per-variable membership partitions.

    Every linguistic variable (the ten features, the twelve concepts and
    the six intervention outputs) owns its OWN editable copy of the
    five-term partition, initialized to the shared default. This is the
    surface the evolving-FIS stage operates on: it perturbs the (a, b,
    c, d) parameters of the terms involved in deficient rules for one
    variable WITHOUT touching any other variable's semantics."""

    def __init__(self):
        """Start with every variable on the default five-term partition.

        A variable only gets an entry of its own once something has
        changed it, so an untouched run carries no per-variable state
        and two engines cannot drift apart by accident.
        """
        self._parts: dict = {}

    def get(self, var: str | None) -> dict:
        """The partition in force for a variable.

        Asking for a variable that has never been reshaped installs the
        default for it, so a caller never has to know whether the
        resolution stage has been here.
        """
        if var is None:
            return default_partition()
        if var not in self._parts:
            self._parts[var] = default_partition()
        return self._parts[var]

    def set_term(self, var: str, term: str, abcd) -> None:
        """Replace one trapezoid, keeping the partition well formed.

        The assertion is not decoration: a term whose corners are out of
        order produces a membership that is negative somewhere, and the
        activation that comes out of it would be silently wrong rather
        than obviously broken.
        """
        part = dict(self.get(var))
        a, b, c, d = (float(v) for v in abcd)
        assert a <= b <= c <= d, "trapezoid must satisfy a <= b <= c <= d"
        part[term] = (a, b, c, d)
        self._parts[var] = part

    def variables(self):
        """Every variable whose partition has been reshaped, in order.

        Sorted because this drives the before-and-after diff of a
        resolution step, and a diff that depends on dictionary order is
        not a diff.
        """
        return sorted(self._parts)

    def snapshot(self, var: str) -> dict:
        """Copy of a variable's partition, for restoring a rejected trial."""
        return dict(self.get(var))

    def restore(self, var: str, part: dict) -> None:
        """Put a partition back after a trial is rejected.

        Every adaptation trial is evaluated on a shadow forecast and
        most are refused, so the ability to undo cleanly is what makes
        trying cheap.
        """
        self._parts[var] = dict(part)

    def shift_boundary(self, var: str, term: str, delta: float) -> float:
        """Move the SHARED boundary between ``term`` and its left neighbour.

        The ascending edge (a, b) of ``term`` IS the descending edge (c, d) of
        the previous term: one boundary, two trapezoids. Displacing it keeps
        the Ruspini invariant only if both are moved together, which is what
        this does. ``delta`` is clamped so that neither plateau inverts:

            -(c_left - b_left)  <=  delta  <=  (c_term - b_term)

        i.e. the step may at most collapse the neighbouring core to a point
        (trapezoid -> triangle), never past it. Returns the delta actually
        applied (0.0 when no admissible move exists, e.g. for the first term,
        which has no left neighbour)."""
        if term not in TERMS:
            return 0.0
        i = TERMS.index(term)
        if i == 0:
            return 0.0                       # no left neighbour, no boundary
        left = TERMS[i - 1]
        part = dict(self.get(var))
        aL, bL, cL, dL = (float(v) for v in part[left])
        aT, bT, cT, dT = (float(v) for v in part[term])
        lo = -(cL - bL)                      # cannot invert the left core
        hi = (cT - bT)                       # cannot invert this term's core
        d = float(np.clip(float(delta), lo, hi))
        if abs(d) < 1e-12:
            return 0.0
        part[left] = (aL, bL, cL + d, dL + d)
        part[term] = (aT + d, bT + d, cT, dT)
        for t in (left, term):               # keep a <= b <= c <= d
            p = part[t]
            if not (p[0] <= p[1] <= p[2] <= p[3]):
                return 0.0
        self._parts[var] = part
        return d

    def insert_split(self, var: str, x: float) -> str:
        """TRUE resolution increase: insert a NEW narrow term into
        var's partition, centered on the reading x that fell between
        the existing term cores. Grows the linguistic catalog of this
        variable (5 -> 6 -> ...); returns the new term's name."""
        part = dict(self.get(var))
        x = float(np.clip(x, 0.0, 1.0))
        # REUSE an existing inserted term when the reading already falls in its
        # support: without this the catalog fills with identical duplicate
        # terms (X1, X2, ... all centered on the same value)
        for t, abcd in part.items():
            if t.startswith("X") and abcd[0] <= x <= abcd[3]:
                return t
        n = sum(1 for t in part if t.startswith("X")) + 1
        name = f"X{n}"
        a = max(0.0, x - 0.12)
        b = max(a, x - 0.05)
        c = min(1.0, x + 0.05)
        d = min(1.0, max(c, x + 0.12))
        part[name] = (a, b, c, d)
        self._parts[var] = part
        return name

    def reset(self) -> None:
        """Drop every modification INCLUDING inserted terms: every
        variable returns to the exact Table D.3 default."""
        self._parts = {}


REGISTRY = PartitionRegistry()


def partition_defect(partition: dict | None = None, var: str | None = None,
                     n: int = 1001) -> float:
    """max |sum_t mu_t(x) - 1| over the universe: 0 for a Ruspini partition.

    The five base terms carry the partition; terms inserted by the resolution
    stage are refinements evaluated outside this algebra and are excluded."""
    part = partition or REGISTRY.get(var)
    xs = np.linspace(0.0, 1.0, int(n))
    tot = np.zeros_like(xs)
    for t in TERMS:
        tot = tot + trapmf(xs, part[t])
    return float(np.max(np.abs(tot - 1.0)))


def trapmf(x, abcd):
    """Trapezoid membership; degenerate shoulders (a == b or c == d, the
    saturated ends of Table D.3) read as hard steps so the extremes of
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
