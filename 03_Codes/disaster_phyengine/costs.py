"""Decision cost model (Sec. 2.5.2, Table 2.4).

Five terms, each an ACTUAL quantity divided by a scenario reference so
it lands in [0, 1]; J is their weighted mean. The formulas:

  J_burn  = x / (1 + x),  x = A_burned / A_ref
            A_burned = burned cells (any fuel; forest reported
            separately), A_ref = burn_reference_fraction (5%) of the
            burnable cells = a "major fire". RATIONAL saturation, not
            exponential: 1 - exp(-x) is numerically DEAD past x ~ 5
            (a fire growing 2900 -> 3500 cells moved J_phys only in
            the 5th decimal, so satisficing, the adaptation trials
            and the no-harm guard all lost their gradient exactly
            when the fight mattered most). x/(1+x) is linear near the
            origin, reads 0.5 at the reference fire, approaches one
            without reaching it, and its slope decays only
            quadratically: every saved cell keeps a MEASURABLE worth
            at any fire size.
  J_asset = value lost / total value at risk
            (buildings + critical infrastructure indices over burned
            cells vs the whole map).
  J_pop   = person-steps in burning cells / (total population x H)
            accumulated persons inside actively burning cells per
            step, H = cost.horizon_steps.
  J_resp  = committed capacity-steps / (capacity_reference x H)
            time-integrated fielded capacity: an army kept in the
            field for hours costs more than a quick strike. Zero when
            nothing is ever fielded; the small weight w_resp and the
            expensive J_burn keep "never intervene" from winning.
  J_del   = capacity-weighted mean dispatch/travel time of the
            fielded resources / delay_reference (60 min).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .config import FUEL_MODELS, CostParams


@dataclass
class CostReport:
    step: int
    burned_area_ha: float
    burned_forest_ha: float
    active_fire_cells: int
    burned_cells: int
    burnable_cells: int
    asset_value_lost: float
    asset_value_total: float
    population_exposed: float
    population_person_steps: float
    population_evacuated: float
    population_at_risk_total: float
    committed_capacity: float
    available_capacity: float
    mean_response_delay: float
    horizon_steps: float
    j_burn: float
    j_asset: float
    j_pop: float
    j_resp: float
    j_delay: float
    j_total: float
    # J_outcome: the PHYSICAL loss only (burn + asset + pop, weights
    # renormalized). The FAIR cross-run comparison metric: a no-DSS
    # run pays no response cost by definition, so comparing on
    # j_total would reward doing nothing. The DSS keeps optimizing
    # j_total (a commander DOES economize the fleet); every
    # DSS-vs-no-DSS verdict reads j_physical.
    j_physical: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _is_forest_mask(ftype: np.ndarray) -> np.ndarray:
    out = np.zeros(ftype.shape, dtype=bool)
    for fid, m in FUEL_MODELS.items():
        if m.is_forest:
            out |= (ftype == fid)
    return out


def compute_costs(sim, cost: CostParams | None = None) -> CostReport:
    """Compute the normalized decision cost for the current simulator state."""
    world = sim.world
    cfg = sim.cfg
    cost = cost or cfg.cost
    eps = cost.min_reference

    burned = sim.ever_burned
    ftype = world.fuel.ftype
    fload0 = world.fuel.fload0
    cell_ha = cfg.cell_area_ha
    cell_km2 = cell_ha / 100.0

    forest = _is_forest_mask(ftype)
    burnable = fload0 > eps
    n_burned = int(burned.sum())
    n_burnable = int(burnable.sum())
    burned_area_ha = n_burned * cell_ha
    burned_forest_ha = float((burned & forest).sum()) * cell_ha

    _bref = float(getattr(cost, "burn_reference_fraction", 0.05))
    _xb = n_burned / max(_bref * n_burnable, 1.0)
    j_burn = float(_xb / (1.0 + _xb))

    asset_field = np.clip(world.value.vbld, 0.0, 1.0) \
        + np.clip(world.value.vcrit, 0.0, 1.0)
    asset_total = float(asset_field.sum())
    asset_lost = float(asset_field[burned].sum())
    j_asset = asset_lost / max(asset_total, eps)

    pop_field = world.value.vpop * cell_km2
    population_exposed = float(pop_field[burned].sum())
    pop_total = float(pop_field.sum())
    pop_at_risk_total = cost.population_at_risk_fraction * pop_total
    person_steps = float(getattr(sim, "exposure_person_steps", 0.0))
    pop_evac = float(getattr(sim, "population_evacuated", 0.0))
    denom_pop = max(pop_total * cost.horizon_steps, eps)
    j_pop = min(1.0, person_steps / denom_pop)

    # the response cost is charged on the resources ACTUALLY applied in
    # the last step (a DSS/user override), falling back to the world's
    # passive resource field when nothing was ordered
    res = getattr(sim, "last_applied_resource", None)
    if res is None:
        res = world.resource
    committed = float((res.rcap * np.clip(res.ravail, 0.0, 1.0)).sum())
    available = max(cost.capacity_reference, eps)
    _resp_steps = getattr(sim, "response_capacity_steps", None)
    if _resp_steps is not None:
        # time-integrated effort: fielded capacity summed over steps
        j_resp = min(1.0, float(_resp_steps)
                     / max(available * cost.horizon_steps, eps))
    else:
        j_resp = min(1.0, committed / available)

    cap_sum = float(res.rcap.sum())
    mean_delay = float((res.rcap * res.rtime).sum()) / cap_sum \
        if cap_sum > eps else 0.0
    j_delay = min(1.0, mean_delay / max(cost.delay_reference, eps))

    w_sum = (cost.w_burn + cost.w_asset + cost.w_pop
             + cost.w_resp + cost.w_delay)
    w_sum = max(w_sum, eps)
    j_total = (cost.w_burn * j_burn + cost.w_asset * j_asset
               + cost.w_pop * j_pop + cost.w_resp * j_resp
               + cost.w_delay * j_delay) / w_sum
    w_phys = max(cost.w_burn + cost.w_asset + cost.w_pop, eps)
    j_physical = (cost.w_burn * j_burn + cost.w_asset * j_asset
                  + cost.w_pop * j_pop) / w_phys

    return CostReport(
        step=sim.state.step,
        burned_area_ha=burned_area_ha,
        burned_forest_ha=burned_forest_ha,
        active_fire_cells=int((sim.state.burning > 0.5).sum()),
        burned_cells=n_burned,
        burnable_cells=n_burnable,
        asset_value_lost=asset_lost,
        asset_value_total=asset_total,
        population_exposed=population_exposed,
        population_evacuated=pop_evac,
        population_person_steps=person_steps,
        population_at_risk_total=pop_at_risk_total,
        committed_capacity=committed,
        available_capacity=available,
        mean_response_delay=mean_delay,
        horizon_steps=float(cost.horizon_steps),
        j_burn=j_burn,
        j_asset=j_asset,
        j_pop=j_pop,
        j_resp=j_resp,
        j_delay=j_delay,
        j_physical=j_physical,
        j_total=j_total,
    )
