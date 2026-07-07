"""Global coordination: non-inferential assembly, normalization, fail-safe.

The coordinator G assembles the regional intervention fields into a globally
consistent field without performing any inference of its own (article
Section III.B). It applies the graduated fail-safe of Eq. 14 per agent, a
per-component doctrinal floor, and resolves resource conflicts. When a fleet
of ResourceUnits is supplied, the physical suppression demand is served by
tasking units (dss.units.assign_units) so the fleet itself is the resource
constraint; otherwise the demand is mapped onto U_Res fields directly, with
an optional shared budget allocated by operational priority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from disasteraware.layers import ResourceLayer

from .rules import INTERVENTION_TYPES
from .agent import LocalDecision
from .units import ResourceUnit, Assignment, assign_units

# doctrinal minimum protection: most exposed cells keep at least this level
FLOOR_EXPOSURE_THRESHOLD = 0.6
FLOOR_LEVELS = {"asset_protection": 0.5, "evacuation": 0.25}


@dataclass
class GlobalDecision:
    """Merged, normalized, fail-safed intervention ready for injection."""

    step: int
    intervention: Dict[str, np.ndarray]
    resource_layer: ResourceLayer
    per_agent_quality: Dict[str, float]
    fail_safe_applied: Dict[str, bool]
    suppression_scale: float = 1.0            # budget trim factor (field path)
    assignments: List[Assignment] = field(default_factory=list)


def baseline_intervention(shape, asset_exposure: np.ndarray,
                          burning: np.ndarray) -> Dict[str, np.ndarray]:
    """Fixed, policy-defined conservative baseline U_base (article III.F)."""
    exposed = asset_exposure >= FLOOR_EXPOSURE_THRESHOLD
    base = {j: np.zeros(shape) for j in INTERVENTION_TYPES}
    base["asset_protection"][exposed] = 1.0
    base["evacuation"][exposed] = 0.75
    base["public_warning"][exposed] = 0.5
    base["suppression_effort"][burning > 0.5] = 0.25
    return base


class Coordinator:
    """Non-inferential coordinator G over the regional agents."""

    def __init__(self, world, quality_threshold: float = 0.6,
                 units: Optional[List[ResourceUnit]] = None,
                 suppression_budget: Optional[float] = None,
                 base_efficiency: float = 0.8,
                 rtime_scale: float = 5.0):
        """units : resource fleet; when given, deployment happens by unit
        tasking and suppression_budget is ignored."""
        self.world = world
        self.eta = float(quality_threshold)
        self.units = units
        self.budget = suppression_budget
        self.base_efficiency = float(base_efficiency)
        self.rtime_scale = float(rtime_scale)

    # ------------------------------------------------------------- assembly
    def merge(self, decisions: List[LocalDecision], obs) -> GlobalDecision:
        shape = self.world.shape
        merged = {j: np.zeros(shape) for j in INTERVENTION_TYPES}
        quality: Dict[str, float] = {}
        failsafe: Dict[str, bool] = {}

        exposure = np.clip(
            self.world.value.priority(self.world.config.value_weights), 0, 1)
        base = baseline_intervention(shape, exposure, obs.burning)

        for dec in decisions:
            quality[dec.agent_id] = dec.quality
            for j in INTERVENTION_TYPES:
                u = dec.intervention[j]
                if dec.quality >= self.eta:
                    applied = u
                    failsafe.setdefault(dec.agent_id, False)
                else:
                    # graduated fail-safe (Eq. 14), inside the agent's region
                    lam = min(1.0, dec.quality / self.eta) if self.eta > 0 \
                        else 0.0
                    blended = base[j] + lam * (u - base[j])
                    applied = np.where(dec.region_mask, blended, u) \
                        if dec.region_mask is not None else blended
                    failsafe[dec.agent_id] = True
                merged[j] = merged[j] + applied     # disjoint regions

        exposed = exposure >= FLOOR_EXPOSURE_THRESHOLD
        for j, lvl in FLOOR_LEVELS.items():
            merged[j] = np.where(exposed, np.maximum(merged[j], lvl), merged[j])
        for j in INTERVENTION_TYPES:
            merged[j] = np.clip(merged[j], 0.0, 1.0)

        # operational priority for conflict resolution (Level 4 concept)
        priority = np.zeros(shape)
        for dec in decisions:
            priority += dec.concepts["operational_priority"]
        priority = np.clip(priority, 0.0, 1.0)

        scale = 1.0
        assignments: List[Assignment] = []
        demand = self._suppression_field(merged)

        if self.units is not None:
            # fleet path: units are the real resource constraint
            layer, assignments = assign_units(self.units, demand, priority,
                                              self.world)
        else:
            # field path with optional shared budget, allocated by priority
            if self.budget is not None:
                rcap_max = self.world.config.suppression.rcap_max
                demand_field = demand * rcap_max
                total = float(demand_field.sum())
                if total > self.budget > 0:
                    order = np.argsort(priority, axis=None)[::-1]
                    flat = demand_field.ravel()[order]
                    keep_n = int(np.searchsorted(np.cumsum(flat),
                                                 self.budget) + 1)
                    keep = np.zeros(demand_field.size, dtype=bool)
                    keep[order[:keep_n]] = True
                    keep = keep.reshape(shape)
                    for j in ("suppression_effort", "containment_line",
                              "resource_deployment"):
                        merged[j] = np.where(keep, merged[j], 0.0)
                    demand = self._suppression_field(merged)
                    kept = float((demand * rcap_max).sum())
                    if kept > self.budget:
                        scale = self.budget / kept
                        for j in ("suppression_effort", "containment_line",
                                  "resource_deployment"):
                            merged[j] *= scale
                        demand = self._suppression_field(merged)
            layer = self._to_resource_layer(merged, demand)

        return GlobalDecision(step=obs.step, intervention=merged,
                              resource_layer=layer, per_agent_quality=quality,
                              fail_safe_applied=failsafe,
                              suppression_scale=scale,
                              assignments=assignments)

    # ----------------------------------------------------- physical coupling
    @staticmethod
    def _suppression_field(intervention: Dict[str, np.ndarray]) -> np.ndarray:
        """Combined physical suppression demand: direct attack plus the
        preventive fuel reduction of containment lines (both act on fuel mass
        through the same Eq. 130-135 mapping)."""
        return np.clip(intervention["suppression_effort"]
                       + 0.5 * intervention["containment_line"], 0.0, 1.0)

    def _to_resource_layer(self, intervention: Dict[str, np.ndarray],
                           supp: np.ndarray) -> ResourceLayer:
        """Field path: map intervention components onto U_DSS (Table A.II)."""
        cfg = self.world.config
        deploy = intervention["resource_deployment"]
        rcap = supp * cfg.suppression.rcap_max
        ravail = np.clip(np.where(supp > 0, 0.3 + 0.7 * deploy, 0.0), 0.0, 1.0)
        reff = np.where(supp > 0, self.base_efficiency, 0.0)
        access = np.clip(self.world.topo.access, 0.0, 1.0)
        rtime = np.where(supp > 0,
                         self.rtime_scale * (1.0 - access) * (1.0 - 0.5 * deploy),
                         0.0)
        return ResourceLayer(rcap=rcap, ravail=ravail, reff=reff, rtime=rtime)
