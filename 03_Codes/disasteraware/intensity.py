"""Fire intensity proxy evolution (Appendix C, Eq. 136 to 137).

Normalized drivers (Eq. 136):
    F = Fload / Fload_max,  W = Wws / Wws_max,  S = tan(Gslope) / tan(Gslope_max)

Intensity update (Eq. 137):
    I_{k+1} = B_{k+1} * tanh(beta * F + gamma_w * W + gamma_s * S)

The proxy is a bounded combustion strength indicator in [0, 1]. It does not feed
back into the spread activation condition; non burning cells are forced to zero
through the multiplication by B_{k+1}.
"""

from __future__ import annotations

import numpy as np

from .config import IntensityParams


def fire_intensity(burning_next: np.ndarray, fload: np.ndarray, topo, meteo,
                   params: IntensityParams) -> np.ndarray:
    """Compute the next intensity proxy field I_{k+1} (Eq. 137)."""
    f_norm = np.clip(fload / max(params.fload_max, 1e-6), 0.0, 1.0)
    w_norm = np.clip(meteo.wws / max(params.wws_max, 1e-6), 0.0, 1.0)
    denom = np.tan(params.slope_max_rad) if params.slope_max_rad > 0 else 1.0
    s_norm = np.clip(np.tan(np.clip(topo.slope, -1.4, 1.4)) / denom, 0.0, 1.0)

    arg = params.beta * (f_norm + params.gamma_w * w_norm + params.gamma_s * s_norm)
    return burning_next.astype(float) * np.tanh(arg)
