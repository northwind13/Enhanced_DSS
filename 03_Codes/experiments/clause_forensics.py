"""Why each proposed clause actuator was admitted or refused.

A campaign row says how many clause actuators were proposed and how many
survived. It does not say why, and "all rejected at G5" turned out to
cover at least three different reasons: a sector that resolved to no
cells, a coating below the extinction moisture of the fuel it was laid
on, and a tactic that acted but bought nothing the forecast could see.
Those are different findings and a table that merges them is useless.

This reads every recorded proposal and reports, per clause:

  effect, sector, range, amount   what was written
  effective strength              amount x the intensity the rule
                                  realises, which is what the physics
                                  receives
  threshold                       for a coating, the strength at which
                                  the coated cells stop burning on this
                                  map; blank for the effects that have
                                  no threshold
  verdict                         below threshold, or acted, or the
                                  sector was empty
  measured                        the two reseeded rollouts, when the
                                  campaign recorded them

    python experiments/clause_forensics.py
    python experiments/clause_forensics.py out/geometry_proposals_p8.jsonl
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.dirname(HERE))

#: the engine constant that turns a coating into a moisture floor
RETARD_TO_MOISTURE = 0.45

#: THE INTENSITY A RULE ACTUALLY REALISES. The aggregation divides the
#: weighted sum by max(sum of firing weights, 1), so a single rule firing
#: at w with a consequent c yields w*c rather than c. Half is the value
#: seen across the recorded slices and it is used only to ESTIMATE the
#: effective strength when the proposal itself was not measured.
TYPICAL_REALISED = 0.5

THRESHOLDED = ("coat",)


def _fuel_thresholds():
    from disaster_phyengine.config import FUEL_MODELS
    return {m.name: m.m_ext / RETARD_TO_MOISTURE
            for i, m in FUEL_MODELS.items() if i not in (0, 5)}


def rows(paths):
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") != "clause":
                    continue
                r["_file"] = os.path.basename(p)
                out.append(r)
    return out


def main():
    paths = (sys.argv[1:] or
             sorted(glob.glob(os.path.join(OUT, "geometry_proposals*.jsonl"))))
    thr = _fuel_thresholds()
    lightest = min(thr.values())
    heaviest = max(thr.values())
    data = rows(paths)
    print(f"clause actuators recorded: {len(data)}")
    print(f"coating threshold on the standard map: {lightest:.2f} of full "
          f"strength for the lightest fuel, {heaviest:.2f} for the "
          f"heaviest")
    print()
    tally = {"below threshold": 0, "acts": 0, "admitted": 0}
    for r in data:
        ni = (r.get("payload") or {}).get("new_intervention") or {}
        cons = dict((a, float(b)) for a, b in
                    ((r.get("payload") or {}).get("consequents") or []))
        ordered = cons.get(ni.get("name"), 0.0)
        realised = ordered * TYPICAL_REALISED
        print(f"{r['_file']}  {r.get('arm', '?'):3} seed {r.get('seed')}  "
              f"{ni.get('name')}")
        print(f"   ordered at {ordered:.2f}, realised near "
              f"{realised:.2f}, accepted={r.get('accepted')} "
              f"gate={r.get('gate')}")
        for c in (ni.get("clauses") or []):
            eff = str(c.get("effect"))
            amt = float(c.get("amount", 0.0))
            strength = amt * realised
            note = ""
            if eff in THRESHOLDED:
                note = (f"needs {lightest:.2f} (lightest fuel) .. "
                        f"{heaviest:.2f} (heaviest)")
                if strength < lightest:
                    note += "  -> BELOW every threshold, slows only"
                    tally["below threshold"] += 1
                else:
                    note += "  -> clears the lightest fuel"
                    tally["acts"] += 1
            else:
                note = "no threshold, acts in proportion"
                tally["acts"] += 1
            print(f"     {eff:9} {str(c.get('sector')):9} "
                  f"{str(c.get('range')):8} amount {amt:.2f} "
                  f"-> effective {strength:.3f}   {note}")
        m = r.get("measured") or {}
        if m:
            g3, g4 = m.get("g3") or {}, m.get("g4") or {}
            if g3:
                same = abs(float(g3.get("j_with", 0))
                           - float(g3.get("j_without", 0))) < 1e-12
                print(f"     rollout 1: with {g3.get('j_with'):.6f} "
                      f"without {g3.get('j_without'):.6f}"
                      + ("   IDENTICAL, the tactic touched nothing"
                         if same else ""))
            if g4:
                d = float(g4.get("j_with", 0)) - float(g4.get("j_without", 0))
                print(f"     rollout 2: with {g4.get('j_with'):.6f} "
                      f"without {g4.get('j_without'):.6f}   "
                      f"delta {d:+.6f}")
        if r.get("accepted"):
            tally["admitted"] += 1
        print()
    print("clause effects below every coating threshold:",
          tally["below threshold"])
    print("clause effects that act:", tally["acts"])
    print("actuators admitted:", tally["admitted"])


if __name__ == "__main__":
    main()
