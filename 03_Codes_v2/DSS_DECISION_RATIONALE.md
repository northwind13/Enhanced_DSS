# How DisasterAware Makes Correct Decisions

This note explains, step by step, how the decision support system turns raw fire
observations into the right operational action, and how that maps to the thesis.
It is meant to be read alongside the running dashboard: every quantity named here
is visible in the interface.

## The operational questions (thesis 5.1.4)

At every cell the DSS answers four questions, exactly as a commander would:

1. How severe is the fire threat here?            -> concept c1 (fire threat level)
2. Can we intervene effectively here?             -> concept c2 (suppression feasibility)
3. What is at stake here?                          -> concept c3 (asset exposure risk)
4. How soon must we act?                           -> concept c4 (intervention urgency)

## From observation to concepts

Each step the DSS observes the fire through a noisy, bounded observation (controlled
by epsilon). From the observation it extracts six bounded features: fire intensity,
fuel load, spread potential, asset exposure, resource accessibility, and temporal
urgency. The features are fuzzified into five linguistic terms (very low to very high)
and aggregated analytically into the four concepts above. When observation confidence
is low, the concept is blended with its previous value (confidence gating), so missing
information attenuates the decision instead of corrupting it.

## From concepts to actions (three intervention types, thesis 5.4.2)

The concepts drive three operationally distinct actions, each a small fuzzy rule base:

- m1 Direct suppression (priority alpha=1.0): act where the fire is burning. Fires when
  threat and feasibility are both high. In the land-use map this is mostly the DSS
  putting out the forest that is actively burning.
- m2 Preventive fuel reduction (alpha=0.7): cut a fuel break just ahead of the front so
  the fire cannot cross. Fires when threat is real but direct suppression is less
  feasible (substitution logic). This is what stops the fire from reaching homes.
- m3 Asset protection (alpha=0.9): concentrate effort on high-value cells (residential,
  city, critical facility) when exposure and urgency are high, regardless of feasibility.
  This is the DSS defending the building, the hospital, the town.

The three degrees are merged into one executable effort by a priority-weighted average
(thesis eq. 70), U = (1.0*u1 + 0.7*u2 + 0.9*u3) / (sum of weights). The blend is
proportional, not winner-takes-all, so the system can suppress, build a break, and
defend assets at the same cell when needed. The dashboard colours each cell by the
dominant action (cyan direct, blue fuel reduction, magenta asset protection) and the
decision text reports how many cells of each type, broken down by land use, so you can
read "direct suppression on 45 forest cells, asset protection on 2 residential cells".

## Respecting resources (coordination)

The total requested effort is compared with the available capacity. If demand exceeds
capacity, every cell's effort is scaled down proportionally (resource normalisation) and
projected onto the feasible set. This is why a small crew (low capacity) contains a small
fire but is overwhelmed by a large one, exactly as in the field.

## Knowing when the plan is good enough (satisficing)

The DSS does not optimise; it satisfices. It scores its plan with a quality Q in [0,1]
built from four criteria with the thesis weights:

  Q = 0.35*fire-spread mitigation + 0.30*asset-risk reduction
    + 0.20*resource economy        + 0.15*timeliness

If Q is at least the acceptance threshold eta, the plan is applied as-is. If not, a
graduated fail-safe attenuates it toward a safe baseline rather than discarding it. The
dashboard shows Q, whether it was accepted, and the four sub-scores, so the operator can
see not just what was done but why it was judged adequate.

## Why this produces correct decisions

- It puts the fire out where it is burning (direct suppression on forest).
- It stops the fire before it reaches people (fuel break ahead of the front).
- It defends what matters most first (asset protection on homes and critical facilities,
  weighted by the thesis value priority: critical 0.40, population 0.25, building 0.20,
  evacuation 0.15).
- It never spends more than it has (resource normalisation).
- It acts decisively when confident and cautiously when not (confidence gating + eta).
- Every action is traceable back to a concept, a rule, and an observation, so the
  recommendation can be explained and audited rather than trusted blindly.

Run the dashboard, start a fire next to the town, and watch the right panel: the DSS
will cut a magenta/blue line between the fire and the houses while the left panel (no
intervention) lets the fire reach them. The burned-area and asset-loss numbers, broken
down by land-use category, quantify the difference.
