**VERDICT — Failure.** The fire grew from 170 to 605 burning cells and *ended* there: never knocked down, 805 cells burned, still active at step 74 (j_resp 0.635). The only win was evacuation — 14,252 of ~15,750 residents cleared.

**ROOT CAUSES (by weight).**
1. **Concept blindness.** Despite a 605-cell fire, `fire_threat_level` never rose above 0.30 (L) anywhere. The entire offensive battery — R26, R15, G1–G4 — is gated on `fire_threat_level` H/VH *and* `suppression_feasibility` VH. Those never co-occurred, so the attack rules essentially never fired. Evidence: Agent_3 fire_threat_level 0.14..0.30 while holding 614 footprint cells.
2. **Geometry / logistics.** The fire mass is in Agent_3, where `logistics_support`=0.25 and `suppression_feasibility` is stuck at 0.42. Water sits at (52,17) in fire-free Agent_2; both depots ((2,1),(6,69)) are far from centroid (19,45) at 36-min mean travel. Capacity was on the wrong side of the map.
3. **Effort mis-allocation.** Agent_1 (191 fire cells, feasibility 0.86) issued strong orders (suppression 0.9); Agent_3 (614 cells) issued weak ones (suppression 0.19→0.22, retardant 0.46). Effort went where it was easy, not where it burned.
4. **Gate behaviour.** 11 no-harm vetoes across steps 31–52 — the growth-to-peak window — blocked adaptation; only 5 of 17 tried changes were accepted.
5. **Rule-mix dilution.** Agent_1's `asset_protection` (0.83) and `evacuation` (0.77) dominate its mix, spending the well-supplied region's capacity on defense while offensive channels stayed thin.

**WHAT WOULD HAVE BEEN BETTER.** Gate the attack rules at M, not VH, so they fire at this map's real readings; put eyes and a helibase on the fire body near (19,45) to lift the threat concept and cut travel; extend the forecast to 45 min so retardant can lead the head into the Agent_3 fuel bed.

### JSON
{
 "verdict": "failure",
 "recommendations": [
  {
   "type": "rule",
   "antecedents": [
    [
     "intervention_urgency",
     ">=L"
    ],
    [
     "suppression_feasibility",
     "M"
    ]
   ],
   "consequents": [
    [
     "suppression_effort",
     0.9
    ],
    [
     "containment_line",
     0.8
    ],
    [
     "retardant_drop",
     0.8
    ]
   ],
   "why": "The VH-gated attack battery never fired; Agent_3 (614 fire cells) only reaches urgency ~0.36 and feasibility 0.42=M, so an M-level trigger is what actually orders offense in the fire region."
  },
  {
   "type": "rule",
   "antecedents": [
    [
     "asset_exposure_risk",
     ">=M"
    ]
   ],
   "consequents": [
    [
     "retardant_drop",
     0.8
    ],
    [
     "containment_line",
     0.7
    ]
   ],
   "why": "Ground logistics in Agent_3 are 0.25; asset_exposure_risk reaches H(0.75)/M(0.60), and air-delivered retardant coats the head without needing road access."
  },
  {
   "type": "sensor",
   "kind": "aerial",
   "x": 19,
   "y": 45,
   "why": "Sensors sit only at corners; the fire body at centroid (19,45) had no coverage, which is why fire_threat_level read L across a 605-cell fire."
  },
  {
   "type": "depot",
   "kind": "helibase",
   "x": 20,
   "y": 46,
   "why": "36-min travel and Agent_3 logistics 0.25 starved the fire region; a helibase on the body enables aerial retardant and raises feasibility where it burns."
  },
  {
   "type": "setting",
   "key": "horizon_min",
   "value": 45,
   "why": "A 30-min horizon is shorter than the 45-min physical forecast; extending it lets retardant lead a fast-spreading head."
  },
  {
   "type": "setting",
   "key": "min_gain",
   "value": 0.02,
   "why": "11 no-harm vetoes at steps 31–52 blocked adaptation during growth-to-peak; a lower gain floor admits incremental offensive gains."
  }
 ]
}