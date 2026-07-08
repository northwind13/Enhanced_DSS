"""Spatial partition of the domain into Local DSS regions.

Each Local DSS is a REGIONAL decision agent responsible for a block of
cells; resources/firefighters are NOT agents, they are
what the agents allocate. One Global DSS coordinates the regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Region:
    """Rectangular responsibility area of one Local DSS agent."""

    name: str
    x0: int
    y0: int
    x1: int          # exclusive
    y1: int          # exclusive

    @property
    def box(self):
        return (self.x0, self.y0, self.x1, self.y1)

    def slices(self):
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    @property
    def n_cells(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


def partition_n(nx: int, ny: int, n: int) -> List[Region]:
    """Split the grid into EXACTLY n regions that cover every cell.

    Rows are chosen near sqrt(n); columns are distributed over the rows so
    the block counts sum to n (e.g. n=5 -> a 3-block row and a 2-block
    row). Agent_1 starts at the north-west, numbering is row-major."""
    n = max(1, int(n))
    rows = max(1, int(round(n ** 0.5)))
    base, extra = divmod(n, rows)
    per_row = [base + (1 if j < extra else 0) for j in range(rows)]
    per_row = [c for c in per_row if c > 0]
    rows = len(per_row)
    ys = [round(j * ny / rows) for j in range(rows + 1)]
    out = []
    k = 1
    for j, cols in enumerate(per_row):
        xs = [round(i * nx / cols) for i in range(cols + 1)]
        for i in range(cols):
            out.append(Region(f"Agent_{k}", xs[i], ys[j], xs[i + 1],
                              ys[j + 1]))
            k += 1
    return out


def partition(nx: int, ny: int, rows: int, cols: int) -> List[Region]:
    """Split the nx x ny grid into rows x cols agent regions (row-major:
    Agent_1 is the north-west block)."""
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    xs = [round(i * nx / cols) for i in range(cols + 1)]
    ys = [round(j * ny / rows) for j in range(rows + 1)]
    out = []
    k = 1
    for j in range(rows):
        for i in range(cols):
            out.append(Region(f"Agent_{k}", xs[i], ys[j], xs[i + 1],
                              ys[j + 1]))
            k += 1
    return out
