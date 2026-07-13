"""Candidate evaluation: shadow-simulation forecasting (Layer 4).

A candidate decision is judged by what it is EXPECTED to cost: the live
simulator is deep-copied, the candidate's resource allocation is held on
the copy for a short forecast horizon, and the normalized decision cost
J of the end state is compared against the no-action shadow run started
from the same state. Nothing touches the live simulation.

The same machinery serves the acceptance test (J <= J_TH), the G3 gate
of generative rules (dJ < 0), the A/B gate (two seeds), and the
counterfactual replay of the analysis view.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from .concepts import DECISION_CONCEPTS
from .rules import INTERVENTIONS


def clone_sim(sim, keep_snapshots: bool = False):
    """Isolated copy of the simulator (world included); the copy carries
    the rng state, so a replay without intervention changes reproduces
    the exact stochastic history."""
    s2 = copy.deepcopy(sim)
    if not keep_snapshots:
        s2.record_states = False
        s2._snapshots.clear()
    return s2


FORECAST_DT_MIN = 5.0    # shadow runs step at >= this (scaling law
                         # keeps the metric physics identical, so a
                         # 1-minute live tick still gets a real horizon)


def forecast_cost(sim, override, horizon: int, reseed: int | None = None,
                  horizon_min: float | None = None):
    """Run a shadow copy under a CONSTANT resource override (None = no
    action) and return the end-state cost report. With horizon_min the
    shadow steps at FORECAST_DT_MIN (coarse but physically equivalent
    through the reference-step scaling), so short live ticks do not
    starve the lookahead."""
    from disaster_phyengine.costs import compute_costs
    s2 = clone_sim(sim)
    s2._substep_cap = 10     # shadow fidelity cap (see core.step)
    if reseed is not None:
        s2._rng = np.random.default_rng(reseed)
    if horizon_min is not None:
        live_dt = float(getattr(s2.cfg, "step_minutes", 1.0))
        fdt = max(live_dt, FORECAST_DT_MIN)
        s2.cfg.step_minutes = fdt
        horizon = max(2, int(round(float(horizon_min) / fdt)))
    for _ in range(int(horizon)):
        s2.step(resource_override=override)
    return compute_costs(s2)


def physical_cost(rep, cost) -> float:
    """The PHYSICAL part of the decision cost (burned area, asset
    loss, population), excluding the response/delay bookkeeping.
    The no-harm fail-safe compares candidates on this basis: paying
    for the fleet is acceptable, buying a worse FIRE is not."""
    wsum = max(cost.w_burn + cost.w_asset + cost.w_pop, 1e-9)
    return float((cost.w_burn * rep.j_burn
                  + cost.w_asset * rep.j_asset
                  + cost.w_pop * rep.j_pop) / wsum)


def candidate_vs_noaction(sim, override, horizon: int,
                          reseed: int | None = None,
                          horizon_min: float | None = None,
                          j0: float | None = None):
    """(J_candidate, J_noaction) over the same horizon and start
    state. j0 short-circuits the no-action shadow run: within one
    decision cycle the no-action future does not change between
    adaptation trials, so the caller caches it (halves the cost
    of every trial)."""
    j_c = forecast_cost(sim, override, horizon, reseed=reseed,
                        horizon_min=horizon_min).j_total
    if j0 is None and reseed is None:
        j0 = getattr(sim, "_dss_j0", None)
    if j0 is None:
        j0 = forecast_cost(sim, None, horizon, reseed=reseed,
                           horizon_min=horizon_min).j_total
    return float(j_c), float(j0)


# which intervention family ANSWERS which decision concept (quality Q);
# suppression feasibility is the gatekeeper concept: it modulates the
# others through R23-R25 and is deliberately not scored against a family
CONCEPT_FAMILY = {
    "fire_threat_level": ("suppression_effort",),
    "asset_exposure_risk": ("asset_protection",),
    "intervention_urgency": ("resource_deployment",),
    "evacuation_pressure": ("evacuation", "public_warning"),
}


def quality_Q(crisp_concepts: Dict[str, float],
              intensities: Dict[str, float],
              family: Dict | None = None) -> float:
    """Decision quality: does the candidate serve the concepts that
    demanded it. Per scored concept the effective activation is compared
    with the strongest intensity of its answering family; Q = 1 - mean
    absolute mismatch."""
    fam_map = CONCEPT_FAMILY if family is None else family
    errs = []
    for cn, fams in fam_map.items():
        a = float(crisp_concepts.get(cn, 0.0))
        u = max(float(intensities.get(f, 0.0)) for f in fams)
        errs.append(abs(a - u))
    return float(1.0 - np.mean(errs)) if errs else 1.0


def graduated_failsafe(intensities: Dict[str, float], Q: float,
                       eta: float) -> Tuple[Dict[str, float], bool]:
    """Below the quality gate the OFFENSIVE intensities are attenuated
    toward the watchful baseline in proportion to the deficit; the
    life-safety orders (evacuation, public warning) are never reduced."""
    if Q >= eta or eta <= 0.0:
        return dict(intensities), False
    scale = max(0.0, Q / eta)
    out = dict(intensities)
    for k in ("suppression_effort", "resource_deployment",
              "containment_line", "asset_protection"):
        out[k] = float(intensities[k]) * scale
    return out, True
