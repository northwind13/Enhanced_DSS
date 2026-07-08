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
    r_base: float
    m_ext: float
    a_w: float
    a_s: float
    a_asp: float
    b_base: float
    is_forest: bool = False
    economic_value: float = 0.0


FUEL_MODELS: Dict[int, FuelModel] = {
    0: FuelModel("non_fuel",    r_base=0.00, m_ext=1.00, a_w=0.0,  a_s=0.0, a_asp=0.00, b_base=0.00, is_forest=False, economic_value=0.0),
    1: FuelModel("grass",       r_base=2.00, m_ext=0.15, a_w=25.0, a_s=2.0, a_asp=0.30, b_base=0.85, is_forest=False, economic_value=2.0),
    2: FuelModel("shrub",       r_base=0.80, m_ext=0.20, a_w=15.0, a_s=2.0, a_asp=0.25, b_base=0.35, is_forest=True,  economic_value=6.0),
    3: FuelModel("pine_litter", r_base=0.80, m_ext=0.25, a_w=6.0,  a_s=2.2, a_asp=0.20, b_base=0.15, is_forest=True,  economic_value=12.0),
    4: FuelModel("hardwood",    r_base=0.30, m_ext=0.30, a_w=4.0,  a_s=1.8, a_asp=0.15, b_base=0.08, is_forest=True,  economic_value=18.0),
    5: FuelModel("water",       r_base=0.00, m_ext=1.00, a_w=0.0,  a_s=0.0, a_asp=0.00, b_base=0.00, is_forest=False, economic_value=0.0),
    6: FuelModel("urban",       r_base=0.40, m_ext=0.20, a_w=3.0,  a_s=1.0, a_asp=0.20, b_base=0.10, is_forest=False, economic_value=0.0),
}

FUEL_NAME_TO_ID: Dict[str, int] = {m.name: i for i, m in FUEL_MODELS.items()}


@dataclass
class SpreadParams:
    """Parameters of the Rothermel type rate of spread and propagation rule."""

    w0: float = 10.0
    theta_ign: float = 0.125
    buildup_leak: float = 0.05
    eps_fuel: float = 1.0e-4
    diagonal_distance_weighting: bool = True
    slope_clip_rad: float = 1.40
    aniso_wind_full: float = 6.0
    slope_wind_equiv: float = 10.0
    slope_gain_max: float = 3.0
    back_frac: float = 0.10
    elliptical: bool = True
    lb_ratio_base: float = 1.0
    lb_ratio_wind: float = 0.06
    spotting: bool = True
    spot_prob: float = 0.02
    spot_distance: int = 6
    spot_intensity_min: float = 0.6


@dataclass
class SuppressionParams:
    """Suppression to fuel reduction mapping (Appendix B, Eq. 130 to 135)."""

    alpha_s: float = 0.20
    beta_t: float = 0.03
    gamma_I: float = 2.0
    rcap_max: float = 1.0


@dataclass
class IntensityParams:
    """Fire intensity proxy parameters (Appendix C, Eq. 136 to 137)."""

    beta: float = 2.0
    gamma_w: float = 0.5
    gamma_s: float = 0.3
    fload_max: float = 1.0
    wws_max: float = 20.0
    slope_max_rad: float = 0.7854
    crown_fire_threshold: float = 0.6
    heat_content: float = 18000.0
    biomass_ref: float = 2.0


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
    """Decision cost parameters (Sec. 2.5.2, Table 2.4)."""

    w_burn: float = 1.0
    w_asset: float = 1.0
    w_pop: float = 1.0
    w_resp: float = 1.0
    w_delay: float = 1.0

    population_at_risk_fraction: float = 0.02
    horizon_steps: float = 200.0
    capacity_reference: float = 100.0
    delay_reference: float = 60.0
    acceptance_fraction: float = 0.35
    population_ceiling: float = 1.0
    min_reference: float = 1e-9


@dataclass
class SimConfig:
    """Top level simulation configuration."""

    nx: int = 80
    ny: int = 60
    cell_size_m: float = 30.0
    dt: float = 1.0
    step_minutes: float = 1.0
    max_steps: int = 500
    rng_seed: int = 42

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
