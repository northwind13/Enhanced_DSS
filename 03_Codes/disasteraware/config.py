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
#
# Calibration is literature based (not thesis specific) at the reference grid
# of 30 m cells and 30 min steps, where 1 cell/step = 1 m/min:
#   r_base  no-wind, dry, flat rate of spread (m/min); with the wind factor
#           1 + a_w tanh(Wws/w0) the benchmark ROS of the Anderson (1982)
#           fuel models are approximated: grass FM1/FM3 ~ 15-50 m/min,
#           shrub FM5/FM6 ~ 5-10, long-needle litter FM9 ~ 2-3, compact
#           hardwood litter FM8 ~ 0.5-1 (see also Scott & Burgan 2005).
#   m_ext   dead fuel moisture of extinction (Anderson 1982): FM1 0.12-0.15,
#           FM5/6 0.20, FM9 0.25, FM8 0.30.
#   b_base  fraction of available fuel consumed per 30 min of flaming front
#           residence (fast in cured grass, slow in compact litter).
#   a_s     slope response (Rothermel 1972 slope factor, tan-linearized).
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

    w0: float = 10.0                # reference wind speed in the tanh saturation (Eq. 126)
    theta_ign: float = 0.125        # ignition threshold on the accumulated influence
                                    # buildup (Eq. 45). 1/8 cancels the neighbourhood
                                    # normalization, so a front driven by one aligned
                                    # neighbour advances at exactly R_spread.
    buildup_leak: float = 0.05      # influence buildup leak per reference step:
                                    # heating dissipates when the source disappears
    eps_fuel: float = 1.0e-4        # extinction fuel threshold (Eq. 44)
    diagonal_distance_weighting: bool = True   # divide diagonal influence by sqrt(2)
    slope_clip_rad: float = 1.40    # clip terrain slope to avoid tan() blow up near 90 deg
    aniso_wind_full: float = 6.0    # wind speed (m/s) for fully directional spread;
                                    # at zero wind spread is isotropic (fuel/slope driven)
    slope_wind_equiv: float = 10.0  # slope-equivalent wind (FARSITE style):
                                    # upslope acts like an extra wind of
                                    # slope_wind_equiv * tan(slope) m/s blowing
                                    # uphill; steers fire up mountainsides even
                                    # against a light gradient wind
    slope_gain_max: float = 3.0     # saturation of the slope factor: real fires
                                    # roughly double per ~15 deg but the effect
                                    # levels off on very steep ground; unbounded
                                    # tan() explodes on real DEM cliffs
    back_frac: float = 0.10         # flank/backing floor of the directional weight:
                                    # backing fires run at ~5-15% of the head fire
                                    # rate (Rothermel 1972; Cheney & Sullivan 2008)
    # --- realism modes (on by default; off = plain cosine kernel, no embers) ---
    elliptical: bool = True         # Cell2Fire/FARSITE style wind elongated ellipse
    lb_ratio_base: float = 1.0      # length-to-breadth ratio at zero wind
    lb_ratio_wind: float = 0.06     # extra length-to-breadth per m/s of wind
    spotting: bool = True           # ember spotting ahead of the front
    spot_prob: float = 0.02         # per hot cell probability per REFERENCE step
                                    # (compounded consistently over substeps)
    spot_distance: int = 6          # spotting distance downwind (cells)
    spot_intensity_min: float = 0.6 # only intense cells throw embers


@dataclass
class SuppressionParams:
    """Suppression to fuel reduction mapping (Appendix B, Eq. 130 to 135)."""

    alpha_s: float = 0.20           # global suppression gain (Eq. 130, range 0.01 to 0.30)
    beta_t: float = 0.03            # travel time decay per minute (Eq. 133):
                                    # effectiveness halves after ~23 min delay,
                                    # matching initial-attack response curves
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
    crown_fire_threshold: float = 0.6   # forest cells above this intensity -> crown fire
    heat_content: float = 18000.0   # heat of combustion H (kJ/kg) for Byram intensity
    biomass_ref: float = 2.0        # kg/m^2 represented by fuel load = 1 (Byram w)


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
    step_minutes: float = 1.0       # real time represented by one step, in
                                    # minutes. The dynamics are calibrated at a
                                    # 30 min reference and rescaled to this
                                    # length (core.py, SD Sec. 9 note 8).
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
