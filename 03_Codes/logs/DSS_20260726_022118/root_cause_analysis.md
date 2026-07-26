## AFTER-ACTION REVIEW — Wildfire DSS Exercise

**VERDICT: Failure.** The fire was never contained. Burning cells rose from 170 to a peak of 715 and *ended* at 715 — no decline, fire-out NO, 845 cells (~12% of the 7,000-cell map) consumed. Only 3,534 of ~15,750 residents evacuated (22%). End response cost j_resp = 0.709. The suppression curve never bent.

**ROOT CAUSES, by weight.**

1. **Capacity latency (heaviest).** Allocated rcap ran 0.0 → 400.9 across all 42 steps; resources arrived only as the fire finished growing, which is why peak equals end. Staged capacity 303 at a 20-min mean travel time could not touch a burn that reached 715 early.
2. **Sterile adaptation / gate behaviour.** 9 proposals tried, 0 accepted, 0 no-harm vetoes — the learning layer contributed nothing. Restored rules G1–G4 all gate on suppression_feasibility VH, a term that never held in this fire, so tuned offence never fired.
3. **Doctrine mismatched to a low-feasibility fire.** Agent_1 ordered near-max suppression (0.92) start to finish with no effect on the burn; direct attack was infeasible, yet no aerial-only path (retardant, which ignores roads and feasibility) was forced.
4. **Sensing geometry.** Aerial coverage clustered east (x 47–78); the westward corridor from the (52,17) seed toward the Hospital/Power-plant/Town cluster at (0–8, 0–10) — the flank actually threatening the high-value assets — was thin.

**WHAT WOULD HAVE BEEN BETTER.** Force retardant on the head when feasibility is low: it needs no roads, and this map has 206 water cells for refill. Stage ground line-crews on the west flank to cut the 20-min travel to the town-ward front. Watch the (52,17)→town corridor. Loosen the adaptation gain gate so marginally-positive rules survive.

### JSON
{
 "verdict": "failure",
 "recommendations": [
  {
   "type": "setting",
   "key": "min_gain",
   "value": 0.02,
   "why": "0 of 9 adaptations accepted; loosen the growth-margin gate so marginally-positive rules survive."
  },
  {
   "type": "setting",
   "key": "eta",
   "value": 0.75,
   "why": "rcap ramped 0.0->400.9 over the whole run; raise allocation responsiveness to commit capacity before the fire peaks."
  },
  {
   "type": "rule",
   "antecedents": [
    [
     "fire_threat_level",
     ">=H"
    ],
    [
     "suppression_feasibility",
     "L"
    ]
   ],
   "consequents": [
    [
     "retardant_drop",
     0.9
    ],
    [
     "containment_line",
     0.8
    ]
   ],
   "why": "Direct attack was infeasible yet aerial retardant needs no roads and the map has water to refill, the only path to bend the head here."
  },
  {
   "type": "sensor",
   "kind": "aerial",
   "x": 30,
   "y": 10,
   "why": "Aerial sensing clustered east (47-78); the (52,17)->town corridor threatening the Hospital/Power-plant cluster was blind."
  },
  {
   "type": "depot",
   "kind": "depot",
   "x": 28,
   "y": 12,
   "why": "20-min mean travel let the fire grow uncontained; stage line-crews on the west flank between the seed and the high-value town assets."
  },
  {
   "type": "tune_rule",
   "name": "R26",
   "consequents": [
    [
     "suppression_effort",
     1.0
    ],
    [
     "retardant_drop",
     0.95
    ],
    [
     "resource_deployment",
     0.9
    ],
    [
     "containment_line",
     0.8
    ]
   ],
   "why": "On VH threat lean the tuned rule harder on aerial retardant since ground feasibility failed to reduce the burn."
  }
 ]
}