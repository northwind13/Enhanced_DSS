"""Mission brief: the STANDING order the generative stage receives.

Built ONCE per simulation (map + doctrine + capabilities) and kept
verbatim for every stage-3 call of that run. Two reasons:

  1. The model should reason like an incident commander who has read
     the map: where the towns are, what is worth protecting, where
     the water is, what the terrain allows. Rebuilding a partial
     picture on every call produced blind, situation-only proposals.
  2. The brief is a byte-identical prompt PREFIX across calls, so
     provider-side prompt caching can reuse it; only the short
     per-cycle situation changes.

The brief is also written to the run's log directory as
mission_brief.md, so the operator can audit exactly what the model
was told about the world.
"""

from __future__ import annotations

import numpy as np

from .rules import (ACTUATOR_LIBRARY, DOCTRINE_INTERVENTIONS,
                    INTERVENTION_LABEL)

_WATER_DEPENDENT = ("water_drafting", "retardant_drop")


def _landcover(world) -> str:
    from disaster_phyengine.config import FUEL_MODELS
    ft = np.asarray(world.fuel.ftype)
    tot = ft.size
    bits = []
    for fid, m in FUEL_MODELS.items():
        n = int((ft == fid).sum())
        if n:
            bits.append(f"{m.name.replace('_', ' ')} "
                        f"{100.0 * n / tot:.0f}%")
    return ", ".join(bits)


def _water(world, cell: float) -> str:
    wat = np.asarray(world.fuel.ftype == 5)
    if not wat.any():
        return ("- water: NONE on this map. water_drafting and "
                "retardant_drop cannot be supplied here; do not "
                "order them, a rule citing them is rejected.")
    ys, xs = np.where(wat)
    return (f"- water: {int(wat.sum())} cells of lake/river/sea, "
            f"centred near ({xs.mean():.0f},{ys.mean():.0f}); "
            "engines and aircraft can draft/refill there, so "
            "sustained capacity near the water is real.")


def _assets(world, cell: float) -> str:
    lines = []
    for a in (getattr(world, "assets", None) or [])[:14]:
        bit = (f"  {a.kind} '{a.name}' at ({a.x},{a.y})"
               + (f", ~{a.population:.0f} persons" if a.population
                  else "")
               + (f", value {a.value:.1f}" if a.value else ""))
        lines.append(bit)
    return ("- assets on the map (protect by value and life):\n"
            + "\n".join(lines) if lines
            else "- assets: none registered on this map.")


def _resources(base) -> str:
    if base is None:
        return "- resources: no pool staged yet."
    cap = float((np.asarray(base.rcap)
                 * np.clip(np.asarray(base.ravail), 0, 1)).sum())
    tt = float(np.asarray(base.rtime).mean())
    air = "yes" if getattr(base, "rair", None) is not None else "no"
    return (f"- resources: staged capacity {cap:.0f} (rcap sum), "
            f"map-mean travel time {tt:.0f} min, air support: {air}.")


def _values(world) -> str:
    vw = world.config.value_weights
    cw = world.config.cost
    return (
        "- protection priority weights (WHERE effort goes): "
        f"critical {vw.w_crit:.2f}, population {vw.w_pop:.2f}, "
        f"building {vw.w_bld:.2f}, evacuation {vw.w_evac:.2f}\n"
        "- loss weights of the decision cost J (WHAT counts as a bad "
        f"outcome): burn {cw.w_burn:.2f}, asset {cw.w_asset:.2f}, "
        f"population {cw.w_pop:.2f}, response {cw.w_resp:.2f}, "
        f"delay {cw.w_delay:.2f}. Trials and cross-run comparisons "
        "are judged on the PHYSICAL part (burn + asset + population).")


def build_mission_brief(world, base_pool=None) -> str:
    cell = float(getattr(world.config, "cell_size_m", 30.0))
    has_water = bool(np.asarray(world.fuel.ftype == 5).any())
    L = []
    L.append("=== MISSION BRIEF (standing; read once, applies to "
             "every decision of this incident) ===")
    L.append(f"MAP: {world.config.nx} x {world.config.ny} cells at "
             f"{cell:.0f} m; land cover: {_landcover(world)}.")
    L.append(_water(world, cell))
    L.append(_assets(world, cell))
    L.append(_resources(base_pool))
    L.append(_values(world))
    L.append(
        "DOCTRINE FAMILIES (seed rules draw only on these): "
        + ", ".join(f"{n} ({INTERVENTION_LABEL[n]})"
                    for n in DOCTRINE_INTERVENTIONS))
    L.append("ACTUATOR LIBRARY (physics available beyond the "
             "doctrine; NO seed rule orders them, discovering a use "
             "is your job):")
    for n, d in ACTUATOR_LIBRARY.items():
        flag = ("  [UNAVAILABLE HERE: no water body to supply it]"
                if (n in _WATER_DEPENDENT and not has_water) else "")
        L.append(f"  {n}: {d}{flag}")
    L.append(
        "YOU MAY ALSO DEFINE A NEW ACTUATOR as data (a package with "
        "\"clauses\"): each clause = one verified effect (wet, clear, "
        "ignite, coat, evacuate, prime, draft) on a sector (head, "
        "flank, rear, ring, at_fire, assets, populated) at a cell "
        "range [rin, rout] from the front with an amount. That is a "
        "genuinely new tactic (WHERE and HOW); a mere re-weighting "
        "of channels is the tuning stage's job and fails the "
        "novelty gates.")
    L.append(
        "WHEN TO INVENT WHAT (think like an incident commander):\n"
        "  - a plain rule: the situation is expressible in the "
        "current concepts and an existing action answers it.\n"
        "  - a new CONCEPT: the same kind of situation keeps "
        "recurring and the five decision concepts cannot name it "
        "(for example ember pressure on a settlement edge, urban "
        "interface stress). Concepts change the architecture: they "
        "enter Layer 3 and later rules can cite them.\n"
        "  - a composite intervention: two channels must act AS ONE "
        "with a fixed ratio (hardening a town: protection + line).\n"
        "  - a clause actuator: the tactic needs its own geometry "
        "(a flank firing operation, a deep pre-wetted band). Use "
        "only what THIS map supports: no water means no drafting "
        "and no aerial drops; no population nearby means evacuate "
        "and prime are wasted orders.")
    L.append(
        "STANDING RULES OF THE GAME: every proposal must FIRE in "
        "the situation it is proposed for (use current dominant "
        "terms, or '>=' for a rising threat); orders must be strong "
        "enough to move a 45-minute physical forecast; every "
        "candidate is judged by simulation gates (form, vocabulary, "
        "relevance, availability, two reseeded A/B forecasts, and a "
        "growth margin for new vocabulary); a rejected proposal "
        "comes back to you once with the failing gate named, so fix "
        "exactly that. Always return ONLY the JSON.")
    L.append("=== END OF MISSION BRIEF ===")
    return "\n".join(L) + "\n"
