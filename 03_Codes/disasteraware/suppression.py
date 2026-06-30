"""Suppression to fuel reduction mapping (Appendix B, Eq. 130 to 135).

    F_red = alpha_s * eta_cap * eta_avail * eta_reach * eta_eff             (130)
    eta_cap   = R_cap / R_cap_max                                          (131)
    eta_avail = R_avail                                                    (132)
    eta_reach = exp(-beta_t * R_time) * G_access                           (133)
    eta_eff   = R_eff / (1 + gamma_I * I)                                  (134)

The combined raw reduction is given explicitly by Eq. 135. The mapping converts
operational capability and reachability into a normalized fuel reduction effect
that the transition operator subtracts from the available fuel mass.
"""

from __future__ import annotations

import numpy as np

from .config import SuppressionParams


def fuel_reduction(resource, topo, intensity: np.ndarray,
                   params: SuppressionParams) -> np.ndarray:
    """Compute the per cell suppression driven fuel reduction F_red (Eq. 135)."""
    eta_cap = resource.rcap / max(params.rcap_max, 1e-6)
    eta_cap = np.clip(eta_cap, 0.0, 1.0)

    eta_avail = np.clip(resource.ravail, 0.0, 1.0)

    eta_reach = np.exp(-params.beta_t * resource.rtime) * np.clip(topo.access, 0.0, 1.0)

    eta_eff = resource.reff / (1.0 + params.gamma_I * np.clip(intensity, 0.0, 1.0))

    f_red = params.alpha_s * eta_cap * eta_avail * eta_reach * eta_eff
    return np.clip(f_red, 0.0, 1.0)
