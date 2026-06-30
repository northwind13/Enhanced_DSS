"""Cost and impact model.

Turns the physical simulation outcome into operational and economic metrics:

    burned area and burned forest area
    asset and value loss (buildings, critical facilities)
    human exposure and monetized casualty risk
    infrastructure loss
    suppression cost

All monetary figures use the abstract currency units defined in
config.CostParams so that they can be calibrated to a real case study. The
report can be produced at any point during a run, which makes it suitable for
building time series in the dashboard.
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
    # physical loss
    fuel_consumed_total: float
    # value loss (currency units)
    forest_value_loss: float
    building_loss: float
    critical_infrastructure_loss: float
    # human impact
    population_exposed: float
    expected_casualties: float
    human_cost: float
    # operations
    suppression_cost: float
    # totals
    total_economic_cost: float

    def to_dict(self) -> dict:
        return asdict(self)


def _is_forest_mask(ftype: np.ndarray) -> np.ndarray:
    out = np.zeros(ftype.shape, dtype=bool)
    for fid, m in FUEL_MODELS.items():
        if m.is_forest:
            out |= (ftype == fid)
    return out


def _economic_value_field(ftype: np.ndarray) -> np.ndarray:
    out = np.zeros(ftype.shape, dtype=float)
    for fid, m in FUEL_MODELS.items():
        out[ftype == fid] = m.economic_value
    return out


def compute_costs(sim, cost: CostParams | None = None) -> CostReport:
    """Compute the full cost report for the current simulator state."""
    world = sim.world
    cfg = sim.cfg
    cost = cost or cfg.cost

    burned = sim.ever_burned
    ftype = world.fuel.ftype
    cell_ha = cfg.cell_area_ha
    cell_km2 = cell_ha / 100.0

    forest = _is_forest_mask(ftype)
    burned_area_ha = float(burned.sum()) * cell_ha
    burned_forest_ha = float((burned & forest).sum()) * cell_ha

    # forest value loss scales with the fuel actually consumed in forest cells
    econ = _economic_value_field(ftype)
    consumed = sim.fuel_consumed_total
    forest_value_loss = float(np.sum(econ * consumed * forest)) \
        * cost.forest_value_multiplier
    # land rehabilitation cost on all burned area
    forest_value_loss += burned_area_ha * cost.cost_per_burned_ha

    # building and critical facility loss over burned footprint
    building_loss = float(np.sum(world.value.vbld[burned])) \
        * cost.building_unit_value * cost.value_loss_on_burn
    critical_loss = float(np.sum(world.value.vcrit[burned])) \
        * cost.critical_unit_value * cost.value_loss_on_burn

    # human exposure: population living in burned cells
    population_exposed = float(np.sum(world.value.vpop[burned])) * cell_km2
    expected_casualties = population_exposed * cost.population_at_risk_fraction
    human_cost = expected_casualties * cost.statistical_life_value

    # suppression cost proportional to fuel actually removed by intervention
    suppression_cost = float(sim.fuel_suppressed_total.sum()) \
        * cost.suppression_unit_cost

    total = (forest_value_loss + building_loss + critical_loss
             + human_cost + suppression_cost)

    return CostReport(
        step=sim.state.step,
        burned_area_ha=burned_area_ha,
        burned_forest_ha=burned_forest_ha,
        active_fire_cells=int((sim.state.burning > 0.5).sum()),
        fuel_consumed_total=float(consumed.sum()),
        forest_value_loss=forest_value_loss,
        building_loss=building_loss,
        critical_infrastructure_loss=critical_loss,
        population_exposed=population_exposed,
        expected_casualties=expected_casualties,
        human_cost=human_cost,
        suppression_cost=suppression_cost,
        total_economic_cost=total,
    )
