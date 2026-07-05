"""Validation utilities: compare a simulated burn against observed data.

To trust the simulator you compare its output to a real fire. The standard way
is to overlay the simulated burned area (or fire perimeter / arrival time) on the
observed one and score the overlap. This module implements the common metrics
used in the wildfire literature (FARSITE/Cell2Fire validation studies):

    Jaccard / IoU   intersection over union of the two burned areas
    Sorensen-Dice   2|A n B| / (|A| + |B|)
    hit rate        fraction of observed burn correctly predicted
    false alarm     fraction of predicted burn that did not actually burn
    front position  mean distance from simulated perimeter to observed perimeter

All functions take boolean masks aligned on the same grid. Use the GIS import to
put real DEM / fuel / perimeter rasters on a common grid first.
"""

from __future__ import annotations

import numpy as np


def compare_masks(sim_mask: np.ndarray, obs_mask: np.ndarray) -> dict:
    """Overlap metrics between a simulated and an observed burned area."""
    a = np.asarray(sim_mask, dtype=bool)
    b = np.asarray(obs_mask, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"masks must share a grid; got {a.shape} vs {b.shape}")
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    sa, sb = int(a.sum()), int(b.sum())
    jaccard = inter / union if union else 1.0
    dice = 2 * inter / (sa + sb) if (sa + sb) else 1.0
    hit = inter / sb if sb else 0.0            # observed cells correctly burned
    false_alarm = (sa - inter) / sa if sa else 0.0
    return {"jaccard": jaccard, "dice": dice, "hit_rate": hit,
            "false_alarm": false_alarm, "sim_cells": sa, "obs_cells": sb,
            "intersection": inter, "union": union}


def _perimeter(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros_like(m)
    p = np.pad(m, 1)
    interior = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~interior


def front_position_error(sim_mask, obs_mask, cell_size_m: float = 30.0) -> dict:
    """Mean and 90th percentile distance from the simulated fire perimeter to the
    nearest observed perimeter cell, in metres. Lower is better."""
    sp = np.argwhere(_perimeter(sim_mask))
    op = np.argwhere(_perimeter(obs_mask))
    if sp.size == 0 or op.size == 0:
        return {"mean_m": float("nan"), "p90_m": float("nan")}
    # nearest observed perimeter cell for each simulated perimeter cell
    d = np.sqrt(((sp[:, None, :] - op[None, :, :]) ** 2).sum(-1)).min(axis=1)
    d = d * cell_size_m
    return {"mean_m": float(d.mean()), "p90_m": float(np.percentile(d, 90))}


def validate_run(sim, obs_mask: np.ndarray, cell_size_m: float = None) -> dict:
    """Score a finished Simulator against an observed burned mask."""
    cs = cell_size_m if cell_size_m is not None else sim.world.config.cell_size_m
    out = compare_masks(sim.ever_burned, obs_mask)
    out.update(front_position_error(sim.ever_burned, obs_mask, cs))
    return out
