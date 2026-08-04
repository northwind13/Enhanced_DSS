"""The standard testbed world, and where a fire is started on it.

Any experiment that reports a number about the decision layer has to
hold the landscape fixed, otherwise the map moves under the measurement
and the result belongs to the terrain rather than to the thing being
studied. These two functions are that fixed ground, kept apart from any
one study so that a study can be retired without taking the testbed with
it.

    w = build_world(201)
    base, _ = dss.resource_suggestion(w)
    spots = pick_ignitions(w, base, 201, 4)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from disaster_phyengine import terrain            # noqa: E402
from disaster_phyengine.config import SimConfig   # noqa: E402


def build_world(seed: int):
    """One landscape family for every experiment that uses this testbed.

    A study varies one thing at a time, so the map has to be the
    constant. It is a mixed landscape with settlements on it, under
    critical fire weather, which is the condition the decision layer is
    meant for. The seed moves the terrain and the settlements, not the
    weather: the point of repeating a measurement over seeds is to
    average out the map, not to average out the scenario.
    """
    cfg = SimConfig(nx=80, ny=60, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=seed, preset="Rolling hills", n_settlements=5,
        population_per_settlement=15000)
    w.fuel.fmoist[:] = 0.08
    w.meteo.wws[:] = 8.0
    return w


def pick_ignitions(w, base, seed: int, n: int):
    """The n burnable spots farthest from the resource bases.

    A fire that starts next to a base is out in minutes whatever the
    parameters say, so the ignition points are chosen where response is
    slowest. The spacing relaxes as n grows, because twelve mutually
    distant points do not exist on a map this size; returning fewer than
    n would change the fire load the caller asked for, which is worse
    than placing two of them closer together than intended.
    """
    ok = ((w.fuel.fload > 0.4) & (w.fuel.ftype != 0)
          & (w.fuel.ftype != 5) & (w.fuel.ftype != 6))
    ys, xs = np.where(ok)
    order = np.argsort(-base.rtime[ys, xs])
    for gap in (20, 14, 10, 6, 3):
        spots = []
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            if all((x - a) ** 2 + (y - b) ** 2 > gap ** 2
                   for a, b in spots):
                spots.append((x, y))
            if len(spots) == n:
                return spots
    return spots
