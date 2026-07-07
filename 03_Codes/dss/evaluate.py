"""Decision cost J of Eq. (9): regional, normalized, rollout-based.

    J_k^i = w1 J_burn + w2 J_val + w3 J_inf + w4 J_pop + w5 J_sup + w6 J_del

Every term is normalized to [0, 1] by the reference scale of the agent's
region (total exposed value, population, and area within the evaluation
horizon), so the weights are dimensionless and the weighted sum stays in
[0, 1]. The burned-area and value terms are disjoint: the area term counts
land loss only, the value terms count the assets standing on it.

Term definitions:
    J_burn : newly burned region cells within the horizon / region area.
    J_val  : protection-discounted building value burned / total building
             value in the region.
    J_inf  : the same for critical infrastructure value.
    J_pop  : evacuation-discounted population in newly burned cells / total
             region population.
    J_sup  : response cost. Every applied intervention intensity is priced,
             suppression and deployment at full weight, containment and
             protection at intermediate weight, evacuation and public
             warning at reduced weight, so no action is free and
             over-response is penalized.
    J_del  : response delay. The number of steps between the moment a cell
             becomes threatened (the front reaches its 8-neighborhood) and
             the first suppression applied to that cell, averaged over
             threatened cells and capped by the horizon. For a candidate
             held constant over the rollout this reduces to the fraction of
             threatened cells with no suppression coverage; the live
             closed-loop metric uses actual application times.

The evaluation clones the simulator, injects the candidate through the same
coordinator mapping (units when a fleet is given), rolls the clone forward H
steps, and reads the terms from the difference. The authoritative simulator
is never touched.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .rules import INTERVENTION_TYPES
from .mitigation import RHO_EVAC, RHO_PROT

# response cost coefficients per intervention type (calibration defaults)
RESPONSE_COST_COEFF = {
    "suppression_effort": 1.0,
    "resource_deployment": 0.6,
    "containment_line": 0.8,
    "asset_protection": 0.4,
    "evacuation": 0.5,
    "public_warning": 0.1,
}

# Eq. 9 weights w1..w6 (calibration defaults; sensitivity swept in Section IV)
DEFAULT_J_WEIGHTS = {
    "burn": 0.25, "val": 0.15, "inf": 0.15,
    "pop": 0.25, "sup": 0.10, "del": 0.10,
}

COVERAGE_EPS = 0.05      # rcap fraction that counts as suppression coverage
EPS = 1e-9


@dataclass
class CostBreakdown:
    """J with its six normalized components."""

    total: float
    terms: Dict[str, float]
    horizon: int
    region_cells: int

    def to_dict(self) -> dict:
        return {"total": self.total, "horizon": self.horizon,
                "region_cells": self.region_cells,
                **{f"j_{k}": v for k, v in self.terms.items()}}


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def evaluate_intervention(sim, intervention: Dict[str, np.ndarray],
                          region_mask: Optional[np.ndarray] = None,
                          horizon: int = 10,
                          weights: Optional[Dict[str, float]] = None,
                          units: Optional[List] = None) -> CostBreakdown:
    """Roll out a candidate intervention on a clone and score it (Eq. 9)."""
    from .coordinator import Coordinator  # local import avoids cycle
    from .units import assign_units

    weights = weights or DEFAULT_J_WEIGHTS
    clone = copy.deepcopy(sim)
    world = clone.world
    shape = world.shape
    region = region_mask if region_mask is not None \
        else np.ones(shape, dtype=bool)
    n_region = int(region.sum())
    if n_region == 0:
        raise ValueError("empty region")

    co = Coordinator(world)
    demand = co._suppression_field(intervention)
    if units:
        layer, _ = assign_units(units, demand)
    else:
        layer = co._to_resource_layer(intervention, demand)
    covered = layer.rcap > COVERAGE_EPS * world.config.suppression.rcap_max

    burned0 = clone.ever_burned.copy()
    threatened = np.zeros(shape, dtype=bool)

    for _ in range(int(horizon)):
        burning = clone.state.burning > 0.5
        threatened |= _dilate8(burning) & ~clone.ever_burned
        clone.step(resource_override=layer)
        if clone.is_quiescent() and clone.ever_burned.any():
            break

    newly = clone.ever_burned & ~burned0 & region
    threatened &= region

    # --- loss terms, discounted by the protective effect model --------------
    evac = np.clip(intervention["evacuation"], 0, 1)
    prot = np.clip(intervention["asset_protection"], 0, 1)

    j_burn = float(newly.sum()) / n_region

    total_bld = float(world.value.vbld[region].sum())
    j_val = float((world.value.vbld * (1 - RHO_PROT * prot))[newly].sum()) \
        / max(total_bld, EPS) if total_bld > 0 else 0.0

    total_crit = float(world.value.vcrit[region].sum())
    j_inf = float((world.value.vcrit * (1 - RHO_PROT * prot))[newly].sum()) \
        / max(total_crit, EPS) if total_crit > 0 else 0.0

    total_pop = float(world.value.vpop[region].sum())
    j_pop = float((world.value.vpop * (1 - RHO_EVAC * evac))[newly].sum()) \
        / max(total_pop, EPS) if total_pop > 0 else 0.0

    # --- response cost: every applied intensity is priced -------------------
    coeff_sum = sum(RESPONSE_COST_COEFF.values())
    j_sup = sum(RESPONSE_COST_COEFF[j]
                * float(np.clip(intervention[j], 0, 1)[region].mean())
                for j in INTERVENTION_TYPES) / coeff_sum

    # --- response delay ------------------------------------------------------
    n_thr = int(threatened.sum())
    j_del = float((threatened & ~covered).sum()) / n_thr if n_thr else 0.0

    terms = {"burn": j_burn, "val": j_val, "inf": j_inf,
             "pop": j_pop, "sup": j_sup, "del": j_del}
    w_sum = sum(weights.values())
    total = sum(weights[k] * terms[k] for k in terms) / max(w_sum, EPS)
    return CostBreakdown(total=float(np.clip(total, 0, 1)),
                         terms={k: round(v, 6) for k, v in terms.items()},
                         horizon=int(horizon), region_cells=n_region)
