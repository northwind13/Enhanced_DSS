"""Observation interface O_k = h(S_k, eps_k) (,).

The Decision Support System never reads the authoritative state directly. It
reasons on observable quantities produced by an observation function that
projects the state and injects epistemic uncertainty at the observation stage
only. This module provides that read only boundary: it copies the four state
fields and optionally adds bounded noise. It never mutates S_k, preserving the
state immutability principle of

This is the Simulation Core side of the interface. The full region specific
observation model with per component observability weights lives
in the DSS implementation (Chapters 5 to 6) and can build on this function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Observation:
    """Observable projection O_k of the wildfire state for the DSS."""

    burning: np.ndarray       # observed burning status
    fload: np.ndarray         # observed fuel load
    intensity: np.ndarray     # observed intensity proxy
    tau: np.ndarray           # observed ignition time
    step: int


def observe(source, epsilon: float = 0.0, seed: Optional[int] = None,
            region: Optional[Tuple[int, int, int, int]] = None) -> Observation:
    """Project the state onto observable quantities O_k = h(S_k, eps_k).

    source   : a Simulator or a SimulationState.
    epsilon  : magnitude of bounded epistemic disturbance added to the
               continuous fields (fload, intensity) at the observation stage
               only. eps = 0 yields a faithful (full observability) reading.
    region   : optional (x0, y0, x1, y1) window; cells outside are returned as
               zero, modelling a region specific observer with limited field of
               view.

    The state is never modified; all arrays are copies.
    """
    state = getattr(source, "state", source)
    B = state.burning.copy()
    F = state.fload.copy()
    I = state.intensity.copy()
    tau = state.tau.copy()

    if epsilon > 0:
        rng = np.random.default_rng(seed)
        F = np.clip(F + rng.uniform(-epsilon, epsilon, F.shape), 0.0, 1.0)
        I = np.clip(I + rng.uniform(-epsilon, epsilon, I.shape), 0.0, 1.0)

    if region is not None:
        x0, y0, x1, y1 = region
        mask = np.zeros(B.shape, dtype=bool)
        xa, xb = sorted((int(x0), int(x1)))
        ya, yb = sorted((int(y0), int(y1)))
        mask[ya:yb + 1, xa:xb + 1] = True
        B = np.where(mask, B, 0.0)
        F = np.where(mask, F, 0.0)
        I = np.where(mask, I, 0.0)
        tau = np.where(mask, tau, 0.0)

    return Observation(burning=B, fload=F, intensity=I, tau=tau,
                       step=getattr(state, "step", 0))
