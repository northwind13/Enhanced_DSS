"""Monte Carlo burn probability (FlamMap style risk map).

Runs the deterministic simulator many times under perturbed wind and ignition
locations and returns, for every cell, the fraction of runs in which it burned.
This turns the deterministic core into a probabilistic risk product without
changing the core itself.
"""

from __future__ import annotations

import numpy as np

from .core import Simulator
from .world import IgnitionEvent


def burn_probability(world, n_runs: int = 30, n_steps: int = 150,
                     wind_speed_jitter: float = 2.0,
                     wind_dir_jitter_deg: float = 25.0,
                     ignition_jitter: int = 2, seed: int = 0,
                     progress=None) -> np.ndarray:
    """Return a burn probability field in [0, 1] over the grid.

    world must already carry at least one ignition. Wind speed and direction are
    perturbed with Gaussian noise each run; ignition points are jittered by up to
    ignition_jitter cells."""
    rng = np.random.default_rng(seed)
    ny, nx = world.shape
    acc = np.zeros((ny, nx), dtype=float)

    base_ws = world.meteo.wws.copy()
    base_wd = world.meteo.wwd.copy()
    base_ign = [(e.x, e.y, e.step, e.radius) for e in world.ignitions]
    base_fload0 = world.fuel.fload0.copy()

    for r in range(int(n_runs)):
        world.meteo.wws[:] = np.clip(base_ws + rng.normal(0, wind_speed_jitter),
                                     0.0, 60.0)
        world.meteo.wwd[:] = base_wd + np.radians(rng.normal(0, wind_dir_jitter_deg))
        world.ignitions = [
            IgnitionEvent(
                x=int(np.clip(x + rng.integers(-ignition_jitter, ignition_jitter + 1), 0, nx - 1)),
                y=int(np.clip(y + rng.integers(-ignition_jitter, ignition_jitter + 1), 0, ny - 1)),
                step=stp, radius=rad)
            for (x, y, stp, rad) in base_ign]
        world.fuel.fload[:] = base_fload0          # fresh fuel each run
        sim = Simulator(world)
        sim.run(n_steps=n_steps, stop_when_quiescent=True)
        acc += sim.ever_burned
        if progress is not None:
            progress((r + 1) / n_runs)

    # restore the world to its original inputs
    world.meteo.wws[:] = base_ws
    world.meteo.wwd[:] = base_wd
    world.fuel.fload[:] = base_fload0
    world.ignitions = [IgnitionEvent(x=x, y=y, step=s, radius=rd)
                       for (x, y, s, rd) in base_ign]
    return acc / max(n_runs, 1)
