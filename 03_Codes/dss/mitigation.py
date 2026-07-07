"""Effect model of the protective interventions on the loss accounting.

Three intervention types act on people and assets rather than on fire
physics: evacuation, asset protection, and public warning. The simulation
core is untouched; instead, this module tracks the protective intensity that
was applied to each cell at the moment it first ignited and discounts the
corresponding loss terms of the cost report. The discounts are proportional
to the applied intensity with calibrated effectiveness ceilings:

    population exposure  *= 1 - rho_evac * evacuation
    casualty fraction    *= 1 - rho_warn * public_warning   (on the remainder)
    building loss        *= 1 - rho_prot * asset_protection
    critical loss        *= 1 - rho_prot * asset_protection

A cell evacuated at full intensity before the fire arrives therefore keeps
at most (1 - rho_evac) of its exposure. Intensities applied after ignition
have no effect, which encodes the operational fact that late protective
action cannot recover a burned cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from disasteraware.costs import compute_costs, CostReport

# effectiveness ceilings (calibration defaults, article Section III.E)
RHO_EVAC = 0.9      # max fraction of exposure removed by full evacuation
RHO_PROT = 0.7      # max fraction of structure loss removed by protection
RHO_WARN = 0.5      # max casualty-fraction reduction by full public warning

PROTECTIVE_TYPES = ("evacuation", "asset_protection", "public_warning")


@dataclass
class MitigationTracker:
    """Records protective intensities at each cell's ignition moment."""

    shape: tuple
    at_burn: Dict[str, np.ndarray] = field(default_factory=dict)
    _burned_seen: Optional[np.ndarray] = None

    def __post_init__(self):
        for j in PROTECTIVE_TYPES:
            self.at_burn[j] = np.zeros(self.shape)
        self._burned_seen = np.zeros(self.shape, dtype=bool)
        self._applied_max = {j: np.zeros(self.shape) for j in PROTECTIVE_TYPES}

    def update(self, intervention: Dict[str, np.ndarray],
               ever_burned: np.ndarray) -> None:
        """Call once per step with the applied global intervention fields
        and the simulator's cumulative burned mask AFTER the step."""
        for j in PROTECTIVE_TYPES:
            self._applied_max[j] = np.maximum(self._applied_max[j],
                                              np.clip(intervention[j], 0, 1))
        newly = ever_burned & ~self._burned_seen
        if newly.any():
            for j in PROTECTIVE_TYPES:
                self.at_burn[j][newly] = self._applied_max[j][newly]
        self._burned_seen |= ever_burned

    def reset(self) -> None:
        self.__post_init__()


def mitigated_costs(sim, tracker: MitigationTracker,
                    cost=None) -> CostReport:
    """Cost report with the protective-intervention discounts applied."""
    base = compute_costs(sim, cost)
    world = sim.world
    cfg = sim.cfg
    params = cost or cfg.cost
    burned = sim.ever_burned
    cell_km2 = cfg.cell_area_ha / 100.0

    evac = tracker.at_burn["evacuation"]
    prot = tracker.at_burn["asset_protection"]
    warn = tracker.at_burn["public_warning"]

    pop_factor = 1.0 - RHO_EVAC * evac
    exposed = float(np.sum(world.value.vpop * pop_factor * burned)) * cell_km2
    casualty_fraction = params.population_at_risk_fraction
    casualties = float(np.sum(
        world.value.vpop * pop_factor * (1.0 - RHO_WARN * warn) * burned)) \
        * cell_km2 * casualty_fraction
    human_cost = casualties * params.statistical_life_value

    prot_factor = 1.0 - RHO_PROT * prot
    building_loss = float(np.sum(world.value.vbld * prot_factor * burned)) \
        * params.building_unit_value * params.value_loss_on_burn
    critical_loss = float(np.sum(world.value.vcrit * prot_factor * burned)) \
        * params.critical_unit_value * params.value_loss_on_burn

    total = (base.forest_value_loss + building_loss + critical_loss
             + human_cost + base.suppression_cost)

    return CostReport(
        step=base.step,
        burned_area_ha=base.burned_area_ha,
        burned_forest_ha=base.burned_forest_ha,
        active_fire_cells=base.active_fire_cells,
        fuel_consumed_total=base.fuel_consumed_total,
        forest_value_loss=base.forest_value_loss,
        building_loss=building_loss,
        critical_infrastructure_loss=critical_loss,
        population_exposed=exposed,
        expected_casualties=casualties,
        human_cost=human_cost,
        suppression_cost=base.suppression_cost,
        total_economic_cost=total,
    )
