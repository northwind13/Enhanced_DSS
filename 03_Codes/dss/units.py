"""Resource units: suppression resources as first-class agents.

The article treats U_Res as four per-cell fields (capacity, availability,
effectiveness, travel time; Table A.II) and names the resources crews,
engines, and aircraft (Section II.G). Here every resource is a unit with a
staging position, an operating radius, a capacity, an efficiency, a speed,
and an availability. The coordinator tasks each unit to the highest priority
demand inside its radius, and the tasked units are rasterized into exactly
the four U_Res fields the simulation core consumes. The unit fleet is the
real resource constraint: demand outside every radius simply goes unserved.

Units are droppable assets like sensors and will be placeable on the
dashboard map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from disasteraware.layers import ResourceLayer

UNIT_PRESETS = {
    # kind: (radius, work_radius, capacity, efficiency, speed)
    "crew": (6, 1, 0.6, 0.8, 2.0),
    "engine": (10, 1, 1.0, 0.7, 3.0),
    "helicopter": (None, 2, 0.8, 0.6, 8.0),
    "airtanker": (None, 3, 1.5, 0.5, 12.0),
    "dozer": (8, 1, 1.2, 0.9, 1.0),
}

DEMAND_EPS = 0.05
STACK_DISCOUNT = 0.3      # discourage piling every unit on the same target


@dataclass
class ResourceUnit:
    """A single suppression resource staged on the map."""

    kind: str
    x: int = 0
    y: int = 0
    radius: Optional[int] = None   # tasking radius from staging; None = all
    work_radius: int = 1           # footprint radius around the tasked cell
    capacity: float = 1.0          # rcap contribution (Table A.II)
    efficiency: float = 0.7        # reff contribution
    speed: float = 2.0             # cells per step, sets rtime
    availability: float = 1.0      # ravail contribution
    unit_id: str = ""

    @classmethod
    def preset(cls, kind: str, x: int = 0, y: int = 0,
               unit_id: str = "", **overrides) -> "ResourceUnit":
        if kind not in UNIT_PRESETS:
            raise ValueError(f"unknown unit kind: {kind!r}")
        radius, work_r, cap, eff, speed = UNIT_PRESETS[kind]
        params = dict(kind=kind, x=x, y=y, radius=radius, work_radius=work_r,
                      capacity=cap, efficiency=eff, speed=speed,
                      unit_id=unit_id or f"{kind}_{x}_{y}")
        params.update(overrides)
        return cls(**params)

    def reach(self, shape) -> np.ndarray:
        if self.radius is None:
            return np.ones(shape, dtype=bool)
        ny, nx = shape
        yy, xx = np.ogrid[:ny, :nx]
        return (xx - self.x) ** 2 + (yy - self.y) ** 2 <= self.radius ** 2


@dataclass
class Assignment:
    """One tasking decision, kept for the audit trail."""

    unit_id: str
    kind: str
    target: Tuple[int, int]
    travel_time: float


def _disk(shape, x, y, r) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.ogrid[:ny, :nx]
    return (xx - x) ** 2 + (yy - y) ** 2 <= max(r, 0) ** 2


def assign_units(units: List[ResourceUnit], demand: np.ndarray,
                 priority: Optional[np.ndarray] = None,
                 world=None) -> Tuple[ResourceLayer, List[Assignment]]:
    """Task every available unit and rasterize the fleet into U_Res fields.

    demand   : combined physical suppression demand field in [0, 1]
    priority : operational-priority concept field for conflict resolution
    """
    shape = demand.shape
    layer = ResourceLayer.none(*shape)
    log: List[Assignment] = []
    if priority is None:
        priority = np.zeros(shape)
    score = np.clip(demand, 0, 1) * (0.5 + 0.5 * np.clip(priority, 0, 1))

    for u in sorted(units, key=lambda u: -u.capacity):
        if u.availability <= 0:
            continue
        s = np.where(u.reach(shape), score, 0.0)
        if float(s.max()) <= DEMAND_EPS:
            continue
        ty, tx = np.unravel_index(int(np.argmax(s)), shape)
        foot = _disk(shape, tx, ty, u.work_radius)
        travel = float(np.hypot(tx - u.x, ty - u.y)) / max(u.speed, 0.1)

        layer.rcap[foot] += u.capacity * np.clip(demand[foot], 0, 1)
        layer.reff[foot] = np.maximum(layer.reff[foot], u.efficiency)
        layer.ravail[foot] = np.maximum(layer.ravail[foot], u.availability)
        prev = layer.rtime[foot]
        layer.rtime[foot] = np.where(prev > 0, np.minimum(prev, travel),
                                     travel)
        score[foot] *= STACK_DISCOUNT
        log.append(Assignment(u.unit_id, u.kind, (int(tx), int(ty)), travel))

    return layer, log


def default_fleet(world) -> List[ResourceUnit]:
    """A modest reference fleet staged at the lower grid edge."""
    ny, nx = world.shape
    return [
        ResourceUnit.preset("engine", x=2, y=ny - 3, unit_id="engine_1"),
        ResourceUnit.preset("engine", x=nx - 3, y=ny - 3, unit_id="engine_2"),
        ResourceUnit.preset("crew", x=2, y=ny - 3, unit_id="crew_1"),
        ResourceUnit.preset("crew", x=nx - 3, y=ny - 3, unit_id="crew_2"),
        ResourceUnit.preset("helicopter", x=nx // 2, y=ny - 2,
                            unit_id="heli_1"),
        ResourceUnit.preset("dozer", x=nx // 2, y=ny - 3, unit_id="dozer_1"),
    ]
