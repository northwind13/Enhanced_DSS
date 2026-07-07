"""Event-level validation of the simulator against observed fire data.

The standard practice in the wildfire modelling literature (FARSITE,
Prometheus, Cell2Fire validation studies) is to simulate a documented
historical fire on its real landscape and weather, then score the simulated
burned area against the observed final perimeter with overlap metrics:

    Jaccard / IoU        |A n B| / |A u B|
    Sorensen-Dice        2|A n B| / (|A| + |B|)
    Hit rate (POD)       |A n B| / |B|        (B = observed)
    False alarm ratio    |A \\ B| / |A|        (A = simulated)
    Area bias            |A| / |B|
    Front position error distances between the two perimeters (m)

Reported reference values: Cell2Fire reproduces historical fires with
Sorensen-Dice around 0.7-0.9 on calibrated cases; 0.5-0.7 is typical for
uncalibrated semi-empirical models. Because ember spotting is stochastic,
validation should be run over several seeds and reported as mean +/- sd.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def compare_masks(sim_mask: np.ndarray, obs_mask: np.ndarray) -> Dict[str, float]:
    """Overlap metrics between the simulated and observed burned masks."""
    a = np.asarray(sim_mask, dtype=bool)
    b = np.asarray(obs_mask, dtype=bool)
    inter = float((a & b).sum())
    union = float((a | b).sum())
    na, nb = float(a.sum()), float(b.sum())
    return {
        "jaccard": inter / union if union else 1.0,
        "dice": 2.0 * inter / (na + nb) if (na + nb) else 1.0,
        "hit_rate": inter / nb if nb else 1.0,
        "false_alarm": (na - inter) / na if na else 0.0,
        "area_bias": na / nb if nb else float("inf"),
        "sim_cells": na, "obs_cells": nb, "overlap_cells": inter,
    }


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Cells of the mask that touch a non-mask cell (4-neighbourhood)."""
    m = np.asarray(mask, dtype=bool)
    er = np.ones_like(m)
    er[1:, :] &= m[:-1, :]; er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]; er[:, :-1] &= m[:, 1:]
    return m & ~(er & m)


def front_distance_errors(sim_mask, obs_mask, cell_size_m: float,
                          max_points: int = 4000) -> Dict[str, float]:
    """Distances from the simulated perimeter to the observed one (meters).

    Symmetric nearest-neighbour distances between the two boundaries; the
    mean and the 90th percentile are the front position errors reported in
    FARSITE-style validation studies.
    """
    pa = np.argwhere(_boundary(sim_mask))
    pb = np.argwhere(_boundary(obs_mask))
    if len(pa) == 0 or len(pb) == 0:
        return {"mean_m": float("nan"), "p90_m": float("nan")}
    rng = np.random.default_rng(0)
    if len(pa) > max_points:
        pa = pa[rng.choice(len(pa), max_points, replace=False)]
    if len(pb) > max_points:
        pb = pb[rng.choice(len(pb), max_points, replace=False)]

    def _nn(p, q):
        out = np.empty(len(p))
        step = max(1, 2_000_000 // max(len(q), 1))
        for i in range(0, len(p), step):
            d = np.sqrt(((p[i:i + step, None, :] - q[None, :, :]) ** 2)
                        .sum(-1)).min(axis=1)
            out[i:i + step] = d
        return out

    d = np.concatenate([_nn(pa, pb), _nn(pb, pa)]) * cell_size_m
    return {"mean_m": float(d.mean()), "p90_m": float(np.percentile(d, 90))}


def validate_run(sim, obs_mask: np.ndarray) -> Dict[str, float]:
    """Full report for a finished simulation against an observed burn mask."""
    rep = compare_masks(sim.ever_burned, obs_mask)
    rep.update(front_distance_errors(sim.ever_burned, obs_mask,
                                     sim.cfg.cell_size_m))
    return rep


# --------------------------------------------------------------------- CORINE
# CORINE Land Cover level-3 codes -> internal fuel classes, for Turkish /
# European case studies (CLC codes, see Copernicus land monitoring service).
CORINE_TO_FUEL: Dict[int, int] = {
    # artificial surfaces -> urban (6) or non fuel
    111: 6, 112: 6, 121: 6, 122: 0, 123: 0, 124: 0, 131: 0, 132: 0, 133: 0,
    141: 1, 142: 6,
    # agriculture -> grass-like fine fuels
    211: 1, 212: 1, 213: 1, 221: 2, 222: 2, 223: 2, 231: 1,
    241: 1, 242: 1, 243: 1, 244: 2,
    # forest and semi natural
    311: 4,   # broad-leaved forest      -> hardwood litter
    312: 3,   # coniferous forest        -> pine litter
    313: 3,   # mixed forest             -> pine litter (conservative)
    321: 1,   # natural grasslands
    322: 2,   # moors and heathland
    323: 2,   # sclerophyllous (maquis)
    324: 2,   # transitional woodland-shrub
    331: 0, 332: 0, 333: 0, 334: 0, 335: 0,
    # wetlands and water
    411: 0, 412: 0, 421: 0, 422: 0, 423: 0,
    511: 5, 512: 5, 521: 5, 522: 5, 523: 5,
}
