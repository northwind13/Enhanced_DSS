"""Parameter definitions for the DisasterAware simulation core.

All physical and numerical parameters of the simulator are collected here as
dataclasses so that a scenario can be configured, serialized and tuned from a
single place. The values follow the formulation given in and the
DisasterAware spread model.

Symbol map (math -> code):
    r_base = r_base(Ftype), Table A.1 
    m_ext = m_ext(Ftype) extinction moisture 
    a_w = a_w(Ftype) wind sensitivity 
    a_s = a_s(Ftype) slope sensitivity 
    a_asp = a_asp(Ftype) aspect sensitivity 
    b_base = b_base(Ftype) baseline combustion coefficient 
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict


@dataclass
class FuelModel:
    """Static combustion properties of a single fuel class.

    Values for the four reference classes follow the standard fuel-model
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
# Calibration is literature based at the reference grid
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
    # URBAN. A built-up block is a broken fuel bed (masonry, streets,
    # gardens), so its base rate stays the lowest of the burnable covers.
    # But a_w=3.0 made it almost DEAF to the weather: measured, doubling
    # the wind from 7 to 15 m/s moved the burned count from 49 cells to 50,
    # and a fire lit beside a town went out before the people in it were
    # ever affected. Real WUI destruction is wind and ember driven, so the
    # wind response and the burning fierceness are raised while the base
    # rate is left alone. Measured on a grass-to-town interface, the town
    # burns 4% / 28% / 54% at 1, 2 and 3 hours in calm air and 14% / 48% /
    # 85% at 14 m/s, against 2% / 10% / 19% and 5% / 16% / 28% before.
    6: FuelModel("urban",       r_base=0.40, m_ext=0.20, a_w=15.0, a_s=1.0, a_asp=0.20, b_base=0.25, is_forest=False, economic_value=0.0),
}

#: THE CULTIVATED FUEL LOADS, and nothing else may sit on them.
#: A worked field carries about half the fine fuel of natural grass, and the
#: renderer needs to know WHICH cells are fields so it can draw each parcel
#: in its own colour. Reading that off a load RANGE was wrong: natural grass
#: on poor ground runs down to 0.37, so eighty-odd wild cells per map fell
#: inside the range and were each painted a different field colour, which
#: rendered as confetti. A parcel takes one value off this ladder, and a
#: continuous noise field never lands exactly on one.
CROP_FUEL_LOADS = (0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44)

FUEL_NAME_TO_ID: Dict[str, int] = {m.name: i for i, m in FUEL_MODELS.items()}


@dataclass
class SpreadParams:
    """Parameters of the Rothermel type rate of spread and propagation rule."""

    w0: float = 10.0 # reference wind speed in the tanh saturation 
    theta_ign: float = 0.125        # ignition threshold on the accumulated influence
                                    # buildup. 1/8 cancels the neighbourhood
                                    # normalization, so a front driven by one aligned
                                    # neighbour advances at exactly R_spread.
    buildup_leak: float = 0.05      # influence buildup leak per reference step:
                                    # heating dissipates when the source disappears
    eps_fuel: float = 0.03          # extinction fuel threshold: below a
                                    # few percent of the nominal load a
                                    # cell cannot sustain flaming
                                    # combustion. The fuel update is
                                    # GEOMETRIC (F <- F(1-f_burn)), so the
                                    # old 1e-4 made every ignited cell
                                    # burn for days (pine ~32 h, hardwood
                                    # ~97 h) and made containment lines
                                    # unreachable; 0.03 gives realistic
                                    # burnout (grass ~75 min) and lets a
                                    # pressed line become a true break
    diagonal_distance_weighting: bool = True   # divide diagonal influence by sqrt(2)
    slope_clip_rad: float = 1.40    # clip terrain slope to avoid tan() blow up near 90 deg
    aniso_wind_full: float = 6.0    # wind speed (m/s) for fully directional spread;
                                    # at zero wind spread is isotropic (fuel/slope driven)
    terrain_wind_gain: float = 0.35  # terrain modification of the wind field:
                                    # exposed ridges speed the wind up, valleys
                                    # shelter it (WindNinja-style mass-consistent
                                    # adjustment, linearized on normalized
                                    # elevation). 0 disables.
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
    """Suppression to fuel reduction mapping (,)."""

    alpha_s: float = 0.20 # global suppression gain (, range 0.01 to 0.30)
    beta_t: float = 0.03 # travel time decay per minute:
                                    # effectiveness halves after ~23 min delay,
                                    # matching initial-attack response curves
    gamma_I: float = 2.0 # intensity resistance factor 
    wet_gain: float = 2.0           # suppression wets the fuel: moisture
                                    # relaxes toward 0.35 at pressure x
                                    # this gain (0 disables wetting)
    knockdown_ratio: float = 0.15   # a burning cell is quenched when the
                                    # suppression pressure (eta product of
                                    # without alpha_s) exceeds this
                                    # threshold scaled by the cell's burn
                                    # fierceness (f_burn / 0.10 per step);
                                    # 0 disables knockdown
    rcap_max: float = 1.0 # reference maximum suppression capacity 


@dataclass
class IntensityParams:
    """Fire intensity proxy parameters (,)."""

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
    """Protection priority aggregation weights."""

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
class SelfEvacuationParams:
    """People leave on their own, without waiting for an order.

    Nobody stands in a burning street because no official told them to go.
    Residents who can see or smell the fire self-evacuate, and the closer
    and hotter it is the faster they move. The model had no such term at
    all: without an order the population sat where it was until the flame
    arrived, which made the ordered evacuation look like the only thing
    between a town and its casualty count.

    They also have to have somewhere to go. Flight is only counted when a
    neighbouring direction is NOT alight, so a settlement the fire has
    already surrounded does not quietly empty itself.
    """

    enabled: bool = True
    #: fraction leaving per minute when the fire is in the cell itself
    in_flame_per_min: float = 0.08
    #: and when it is in a neighbouring cell (seen, not yet arrived)
    adjacent_per_min: float = 0.03
    #: how far a cell can be from the fire and still notice it, in cells
    awareness_cells: int = 6
    #: the rate at that awareness range; it falls off linearly to it
    aware_per_min: float = 0.004
    #: nobody leaves once this share of a cell's people has gone: the last
    #: few are the ones who cannot or will not move
    max_share: float = 0.9


@dataclass
class DryingParams:
    """Dead fuel moisture DRYING, the counterpart of the wetting terms.

    Rain, a retardant coat and suppression all raise the moisture field and
    nothing lowered it, so moisture was monotonically non-decreasing over a
    run: fuel burned to ash kept its ambient value, the front never dried
    the cells it was about to reach, and a cell wetted once stayed wet for
    the rest of the scenario. That last one flattered the response, because
    a line held once went on holding itself for free.

    Three mechanisms, each switchable so the previous behaviour can still be
    reproduced for comparison.
    """

    enabled: bool = True

    # 1. TIMELAG RECOVERY. Dead fuel relaxes toward the equilibrium moisture
    # content of the ambient air with a response time; fine dead fuels
    # (grass, litter) are the classic 1-hour timelag class. Applied in the
    # DRYING direction only: absorption from humid air is slower and weaker
    # than the wetting terms already modelled, and adding it would silently
    # re-baseline every scenario that starts drier than its equilibrium.
    timelag_h: float = 1.0

    # 2. PREHEATING. A cell next to the flame front is radiantly heated
    # above air temperature, so it dries below the ambient level and it
    # dries faster. The depth is the fraction by which the target falls at
    # full neighbour intensity; the gain shortens the response time.
    #
    # CALIBRATED, NOT GUESSED. The first pass used 0.60 and 8.0, which dried
    # the fuel ahead of the front faster than the crews could work it: on the
    # end-to-end test the DSS stopped being able to put the fire out at all
    # (221 cells burned and still alight at the horizon, against 16 cells and
    # out by step 9 without any drying). At 0.20 and 2.0 the mechanism is
    # present and the fire is still extinguished on schedule.
    preheat_depth: float = 0.20
    preheat_gain: float = 2.0

    # 3. COMBUSTION. A burning cell drives its moisture off in minutes, so
    # the response time collapses and the target is the residual left in
    # char rather than anything the air dictates.
    burn_timelag_min: float = 5.0
    burn_floor: float = 0.02


@dataclass
class CostParams:
    """Decision cost parameters (,).

    The decision cost follows the cost-plus-loss principle as a weighted sum
    of five terms, each normalized to [0, 1] against a scenario reference
    scale so that the terms are dimensionless and mutually summable before
    the priority weights apply. The weights, not the term list, encode
    operational priority.
    """

    # non-negative priority weights of the five normalized terms
    # J_burn normalization: burned area is charged against a MAJOR-FIRE
    # reference (this fraction of the burnable area), not the whole map.
    # Against the whole map a 600-cell forest fire scored ~0.06 while a
    # committed response cost 0.2*j_resp — the optimizer read "letting
    # the forest burn is cheaper than fighting it", which is wrong.
    burn_reference_fraction: float = 0.05
    w_burn: float = 1.0     # burned area (land and ecological loss)
    w_asset: float = 1.0    # asset loss (structures + critical infrastructure)
    w_pop: float = 1.0      # population exposure (life safety)
    w_resp: float = 0.2     # response cost (committed capacity). It is
                            # a SECONDARY objective: fielding the whole
                            # fleet during a major fire is normal and
                            # must never outweigh saving the town, so
                            # its default weight is a tie-breaker scale
    w_delay: float = 0.2    # response delay (timeliness), secondary

    # normalization references and safeguards
    # AN ASSET THE FIRE REACHED IS LOST. A structure does not need its own
    # cell to carry fuel to be destroyed: radiant heat and embers from the
    # cell next door do it, which is the whole WUI mechanism. Charging only
    # the cells that burned themselves also put a ceiling under the term
    # that no fire could reach. Set False for the strict "this cell burned"
    # reading.
    asset_lost_on_contact: bool = True
    # WHAT A DISPLACED PERSON COSTS, against 1.0 for one exposed to flame
    # for the same time. An ordered evacuation is not free: people are
    # housed, fed and cut off from their work and their services. At zero,
    # which is what it was, moving a whole town cost nothing at all and the
    # cheapest answer to any fire was to empty the map.
    evacuation_weight: float = 0.05
    population_at_risk_fraction: float = 0.02  # rho_risk; casualty share of exposed
    horizon_steps: float = 200.0               # scenario horizon H (decision steps)
    capacity_reference: float = 100.0          # total available capacity pool
    delay_reference: float = 60.0              # reference travel time for the delay term
    acceptance_fraction: float = 0.35          # acceptance threshold as fraction of the do-nothing cost
    population_ceiling: float = 1.0            # hard ceiling on the normalized population term (acceptance gate)
    min_reference: float = 1e-9                # lower bound on any denominator


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
                                    # length (core.py, SD note 8).
    max_steps: int = 500            # safety cap for headless runs
    rng_seed: int = 42              # seed for any stochastic forcing

    spread: SpreadParams = field(default_factory=SpreadParams)
    drying: DryingParams = field(default_factory=DryingParams)
    self_evac: SelfEvacuationParams = field(
        default_factory=SelfEvacuationParams)
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
