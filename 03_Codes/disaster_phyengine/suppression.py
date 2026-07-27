"""Suppression to fuel reduction mapping (,).

    F_red = alpha_s * eta_cap * eta_avail * eta_reach * eta_eff             (130)
    eta_cap   = R_cap / R_cap_max                                          (131)
    eta_avail = R_avail                                                    (132)
    eta_reach = exp(-beta_t * R_time) * G_access                           (133)
    eta_eff   = R_eff / (1 + gamma_I * I)                                  (134)

The combined raw reduction is given explicitly by. The mapping converts
operational capability and reachability into a normalized fuel reduction effect
that the transition operator subtracts from the available fuel mass.
"""

from __future__ import annotations

import numpy as np

from .config import SuppressionParams


def fuel_reduction(resource, topo, intensity: np.ndarray,
                   params: SuppressionParams,
                   air_scale=1.0) -> np.ndarray:
    """Compute the per cell suppression driven fuel reduction F_red.

    air_scale (scalar or field, [0, 1]) derates the AERIAL share for
    weather: strong wind grounds the aircraft. Where the resource layer
    carries an aerial share, that share replaces the road-access factor
    (helicopters do not need a road), so remote ground stays workable
    from the air."""
    # The clips write into the temporaries this function already owns
    # (out=), so the same numbers are produced without allocating a second
    # array for every bound. Nothing here clips a CALLER's array in place:
    # eta_cap and eta_eff are fresh division results, and ravail/access/
    # intensity are copied by their own clip.
    eta_cap = resource.rcap / max(params.rcap_max, 1e-6)
    np.clip(eta_cap, 0.0, 1.0, out=eta_cap)

    eta_avail = np.clip(resource.ravail, 0.0, 1.0)

    _acc = np.clip(topo.access, 0.0, 1.0)
    _ra = getattr(resource, "rair", None)
    if _ra is not None:
        _acc = np.maximum(_acc, np.clip(_ra, 0.0, 1.0) * air_scale,
                          out=_acc)
    eta_reach = np.exp(-params.beta_t * resource.rtime)
    eta_reach *= _acc

    eta_eff = resource.reff / (1.0 + params.gamma_I
                               * np.clip(intensity, 0.0, 1.0))

    f_red = params.alpha_s * eta_cap
    f_red *= eta_avail
    f_red *= eta_reach
    f_red *= eta_eff
    return np.clip(f_red, 0.0, 1.0, out=f_red)
