"""Fire intensity proxy evolution (,).

Normalized drivers:
    F = Fload / Fload_max,  W = Wws / Wws_max,  S = tan(Gslope) / tan(Gslope_max)

Intensity update:
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
    """Compute the next intensity proxy field I_{k+1}."""
    # in-place bounds on this function's OWN temporaries (see suppression)
    f_norm = fload / max(params.fload_max, 1e-6)
    np.clip(f_norm, 0.0, 1.0, out=f_norm)
    w_norm = meteo.wws / max(params.wws_max, 1e-6)
    np.clip(w_norm, 0.0, 1.0, out=w_norm)
    denom = np.tan(params.slope_max_rad) if params.slope_max_rad > 0 else 1.0
    s_norm = np.tan(np.clip(topo.slope, -1.4, 1.4))
    s_norm /= denom
    np.clip(s_norm, 0.0, 1.0, out=s_norm)

    arg = f_norm
    arg += params.gamma_w * w_norm
    arg += params.gamma_s * s_norm
    arg *= params.beta
    return burning_next.astype(float) * np.tanh(arg)
