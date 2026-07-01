"""Parameter definitions for the DisasterAware simulation core.

All physical and numerical parameters of the simulator are collected here as
dataclasses so that a scenario can be configured, serialized and tuned from a
single place. The values follow the formulation given in Chapter 4 and the
appendices of the DisasterAware PhD thesis.

Symbol map (thesis -> code):
    r_base   = r_base(Ftype)        Appendix A, Table A.1 (Eq. 123)
    m_ext    = m_ext(Ftype)         extinction moisture (Eq. 124)
    a_w      = a_w(Ftype)           wind sensitivity (Eq. 126)
    a_s      = a_s(Ftype)           slope sensitivity (Eq. 127)
    a_asp    = a_asp(Ftype)         aspect sensitivity (Eq. 128)
    b_base   = b_base(Ftype)        baseline combustion coefficient (Eq. 129)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict


@dataclass
class FuelModel:
    """Static combustion properties of a single fuel class.

    Values for the four reference classes are taken from thesis Table A.1
    (rate of spread parameters) and Table A.2 (combustion coefficient).
    """

    name: str
    r_base: float       # base rate of spread multiplier
    m_ext: float        # extinction moisture fraction
    a_w: float          # wind sensitivity coefficient
    a_s: float          # slope sensitivity coefficient
    a_asp: float        # aspect sensitivity coefficient
    b_base: float       # baseline combustion coefficient (fraction per step)
    is_forest: bool = False   # used by the cost model to separate forest loss
    economic_value: float = 0.0   # nominal value per cell unit (forest stand value)


# Reference fuel library. Index 0 is reserved for non burnable cells so that
# water, bare ground and urban footprints never propagate fire on their own.
FUEL_MODELS: Dict[int, FuelModel] = {
    0: FuelModel("non_fuel",    r_base=0.00, m_ext=1.00, a_w=0.0, a_s=0.0, a_asp=0.0, b_base=0.00, is_forest=False, economic_value=0.0),
    1: FuelModel("grass",       r_base=1.20, m_ext=0.25, a_w=2.0, a_s=0.8, a_asp=0.30, b_base=0.25, is_forest=False, economic_value=2.0),
    2: FuelModel("shrub",       r_base=0.80, m_ext=0.35, a_w=1.5, a_s=1.0, a_asp=0.25, b_base=0.15, is_forest=True,  economic_value=6.0),
    3: FuelModel("pine_litter", r_base=0.60, m_ext=0.45, a_w=1.2, a_s=1.2, a_asp=0.20, b_base=0.10, is_forest=True,  economic_value=12.0),
    4: FuelModel("hardwood",    r_base=0.40, m_ext=0.50, a_w=0.8, a_s=0.6, a_asp=0.15, b_base=0.05, is_forest=True,  economic_value=18.0),
    5: FuelModel("water",       r_base=0.00, m_ext=1.00, a_w=0.0, a_s=0.0, a_asp=0.00, b_base=0.00, is_forest=False, economic_value=0.0),
}

FUEL_NAME_TO_ID: Dict[str, int] = {m.name: i for i, m in FUEL_MODELS.items()}


@dataclass
class SpreadParams:
    """Parameters of the Rothermel type rate of spread and propagation rule."""

    w0: float = 10.0                 # reference wind speed in the tanh saturation (Eq. 126)
    theta_ign: float = 0.08         # ignition threshold on accumulated influence (Eq. 45)
    eps_fuel: float = 1.0e-4        # extinction fuel threshold (Eq. 44)
    diagonal_distance_weighting: bool = True   # divide diagonal influence by sqrt(2)
    slope_clip_rad: float = 1.40    # clip terrain slope to avoid tan() blow up near 90 deg


@dataclass
class SuppressionParams:
    """Suppression to fuel reduction mapping (Appendix B, Eq. 130 to 135)."""

    alpha_s: float = 0.20           # global suppression gain (Eq. 130, Table B range 0.01 to 0.30)
    beta_t: float = 0.20            # travel time decay coefficient (Eq. 133)
    gamma_I: float = 2.0            # intensity resistance factor (Eq. 134)
    rcap_max: float = 1.0           # reference maximum suppression capacity (Eq. 131)


@dataclass
class IntensityParams:
    """Fire intensity proxy parameters (Appendix C, Eq. 136 to 137)."""

    beta: float = 2.0               # global intensity gain (Table C)
    gamma_w: float = 0.5            # wind weighting coefficient
    gamma_s: float = 0.3            # slope weighting coefficient
    fload_max: float = 1.0          # reference maximum fuel load for normalization
    wws_max: float = 20.0           # reference maximum wind speed (m/s)
    slope_max_rad: float = 0.7854   # reference maximum slope (45 deg) for normalization


@dataclass
class ValueWeights:
    """Protection priority aggregation weights (Eq. 55)."""

    w_bld: float = 0.20
    w_crit: float = 0.40
    w_pop: float = 0.25
    w_evac: float = 0.15

    def normalized(self) -> "ValueWeights":
        total = self.w_bld + self.w_crit + self.w_pop + self.w_evac
        if total <= 0:
            return ValueWeights(0.25, 0.25, 0.25, 0.25)
        return ValueWeights(self.w_bld / total, self.w_crit / total,
                            self.w_pop / total, self.w_evac / total)


@dataclass
class CostParams:
    """Unit costs used by the cost model. Units are intentionally abstract
    (currency units) so that they can be calibrated to a real case study.
    """

    cost_per_burned_ha: float = 1_000.0       # base land rehabilitation cost
    forest_value_multiplier: float = 1.0      # scales FuelModel.economic_value
    building_unit_value: float = 250_000.0    # value of a built footprint cell at Vbld = 1
    critical_unit_value: float = 5_000_000.0  # value of a critical facility cell at Vcrit = 1
    value_loss_on_burn: float = 1.0           # fraction of asset value lost when a cell burns
    statistical_life_value: float = 1_500_000.0  # value used to monetize exposed population
    population_at_risk_fraction: float = 0.02    # fraction of exposed population assumed casualty risk
    suppression_unit_cost: float = 5_000.0    # cost per unit of suppression capacity applied per step


@dataclass
class SimConfig:
    """Top level simulation configuration."""

    nx: int = 80                    # grid cells along x
    ny: int = 60                    # grid cells along y
    cell_size_m: float = 30.0       # edge length of one square cell in meters
    dt: float = 1.0                 # time step length (model time units)
    max_steps: int = 500            # safety cap for headless runs
    rng_seed: int = 42              # seed for any stochastic forcing

    spread: SpreadParams = field(default_factory=SpreadParams)
    suppression: SuppressionParams = field(default_factory=SuppressionParams)
    intensity: IntensityParams = field(default_factory=IntensityParams)
    value_weights: ValueWeights = field(default_factory=ValueWeights)
    cost: CostParams = field(default_factory=CostParams)

    @property
    def cell_area_ha(self) -> float:
        """Area of a single cell in hectares."""
        return (self.cell_size_m ** 2) / 10_000.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimConfig":
        data = dict(data)
        sub = {
            "spread": SpreadParams,
            "suppression": SuppressionParams,
            "intensity": IntensityParams,
            "value_weights": ValueWeights,
            "cost": CostParams,
        }
        for key, klass in sub.items():
            if key in data and isinstance(data[key], dict):
                data[key] = klass(**data[key])
        return cls(**data)
