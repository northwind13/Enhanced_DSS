"""Rate of spread and directional propagation (Appendix A, Eq. 123 to 128, 46, 48).

The rate of spread mapping is a semi empirical Rothermel type formulation:

    R_spread = r_base(Ftype) * g_moist * g_wind * g_slope * g_aspect      (123)
    g_moist  = max(0, 1 - Fmoist / m_ext(Ftype))                          (124)
    g_wind   = 1 + a_w(Ftype) * tanh(Wws / w0)                            (126)
    g_slope  = 1 + a_s(Ftype) * tan(Gslope)                               (127)
    g_aspect = 1 + a_asp(Ftype) * cos(Gaspect - Wwd)                      (128)

Propagation influence accumulated at a target cell (Eq. 46) is a wind aligned
weighted sum over the 8 connected neighbourhood, with directional weights
(Eq. 48):

    g_dir = max(0, cos(Wwd - theta_(i,j -> x,y)))
"""

from __future__ import annotations

import numpy as np

from .config import FUEL_MODELS, SpreadParams

# 8 connected neighbour offsets expressed as (drow, dcol) = (dy_array, dx).
# For each offset we also store the geometric direction theta of the vector
# pointing FROM the neighbour TO the target cell, using a north up convention
# where increasing row index means going south.
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]


def _neighbour_to_target_angle(drow: int, dcol: int) -> float:
    """Angle of the vector from the neighbour to the target cell.

    The neighbour sits at array offset (drow, dcol) relative to the target, so
    the vector neighbour -> target is (-dcol, +drow_north). Using a north up
    convention, the y component is +drow (a neighbour to the south, drow = +1,
    points north toward the target)."""
    dx = -dcol
    dy = drow  # north up: a southern neighbour (drow=+1) points north (+y)
    return float(np.arctan2(dy, dx))


def _shift(arr: np.ndarray, drow: int, dcol: int) -> np.ndarray:
    """Return B where B[y, x] = arr[y + drow, x + dcol], zero filled at edges."""
    out = np.zeros_like(arr)
    ny, nx = arr.shape
    ys_dst = slice(max(0, -drow), ny - max(0, drow))
    xs_dst = slice(max(0, -dcol), nx - max(0, dcol))
    ys_src = slice(max(0, drow), ny - max(0, -drow))
    xs_src = slice(max(0, dcol), nx - max(0, -dcol))
    out[ys_dst, xs_dst] = arr[ys_src, xs_src]
    return out


def _fuel_param(ftype: np.ndarray, attr: str) -> np.ndarray:
    """Vectorized lookup of a per cell fuel parameter."""
    out = np.zeros(ftype.shape, dtype=float)
    for fid, model in FUEL_MODELS.items():
        out[ftype == fid] = getattr(model, attr)
    return out


def rate_of_spread(fuel, topo, meteo, params: SpreadParams) -> np.ndarray:
    """Compute the per cell rate of spread field R_spread (Eq. 123)."""
    ftype = fuel.ftype
    r_base = _fuel_param(ftype, "r_base")
    m_ext = _fuel_param(ftype, "m_ext")
    a_w = _fuel_param(ftype, "a_w")
    a_s = _fuel_param(ftype, "a_s")
    a_asp = _fuel_param(ftype, "a_asp")

    # moisture damping (Eq. 124 to 125): zero spread at or above extinction moisture
    with np.errstate(divide="ignore", invalid="ignore"):
        g_moist = np.where(m_ext > 0, 1.0 - fuel.fmoist / m_ext, 0.0)
    g_moist = np.clip(g_moist, 0.0, 1.0)

    # wind enhancement (Eq. 126)
    g_wind = 1.0 + a_w * np.tanh(meteo.wws / max(params.w0, 1e-6))

    # slope enhancement (Eq. 127); slope clipped to keep tan() finite
    slope = np.clip(topo.slope, -params.slope_clip_rad, params.slope_clip_rad)
    g_slope = 1.0 + a_s * np.tan(slope)
    g_slope = np.maximum(g_slope, 0.0)

    # aspect alignment with wind (Eq. 128)
    g_aspect = 1.0 + a_asp * np.cos(topo.aspect - meteo.wwd)
    g_aspect = np.maximum(g_aspect, 0.0)

    ros = r_base * g_moist * g_wind * g_slope * g_aspect
    return np.maximum(ros, 0.0)


def propagation_influence(burning: np.ndarray, ros: np.ndarray,
                          wwd: np.ndarray, params: SpreadParams,
                          wws: np.ndarray = None) -> np.ndarray:
    """Accumulated wind aligned propagation influence Psi (Eq. 46, 48).

    Default uses the thesis directional weight g_dir = max(0, cos(Wwd - theta)).
    When params.elliptical is True, an optional Cell2Fire/FARSITE style wind
    elongated ellipse is used instead, with a length-to-breadth ratio that grows
    with wind speed. The thesis behaviour is unchanged unless elliptical is on."""
    source = burning.astype(float) * ros
    psi = np.zeros_like(source)
    ecc = None
    aniso = 1.0
    if params.elliptical and wws is not None:
        lb = params.lb_ratio_base + params.lb_ratio_wind * np.asarray(wws)
        lb = np.maximum(lb, 1.0)
        ecc = np.sqrt(np.clip(1.0 - 1.0 / (lb * lb), 0.0, 0.999))
    elif wws is not None:
        # blend the directional weight toward isotropic as wind drops to zero,
        # so with no wind the fire spreads symmetrically driven by fuel and slope
        aniso = np.clip(np.asarray(wws) / max(params.aniso_wind_full, 1e-6), 0.0, 1.0)
    for drow, dcol in _OFFSETS:
        theta = _neighbour_to_target_angle(drow, dcol)
        cosd = np.cos(wwd - theta)
        if ecc is not None:
            g_dir = np.clip((1.0 - ecc) / (1.0 - ecc * cosd), 0.0, None)
        else:
            # aniso in [0,1]: 0 -> isotropic (g_dir=1), 1 -> thesis max(0,cos)
            g_dir = (1.0 - aniso) + aniso * np.maximum(0.0, cosd)
        contrib = _shift(source, drow, dcol) * g_dir
        if params.diagonal_distance_weighting and drow != 0 and dcol != 0:
            contrib = contrib / np.sqrt(2.0)
        psi += contrib
    return psi / 8.0
