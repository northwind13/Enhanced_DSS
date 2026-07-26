=== AFTER-ACTION REVIEW ===

**VERDICT — failure.** The fire ignited at (43,114) and burned **1565 cells** (x0–48, y103–152), entirely inside assetless region Agent_3. It peaked at **424** burning cells and was still burning (**153**) at step 24 — **not out**. Allocated capacity was **0.0 at start and 0.0 at end**: across all 24 steps the DSS never delivered offensive capacity. The decline from 424 to 153 is fuel and terrain, not suppression.

**ROOT CAUSES (ranked).**
1. **Logistics/capacity (heaviest).** rcap 0.0 throughout; logistics_support 0.00 in every region; the nearest base (depot/helibase at 22,64) sits ~62 cells from centroid (19,126) at map-mean travel **522 min** — longer than a 60-min step. Staged capacity 130 never reached the fire.
2. **Gate/min_gain block.** Achieved response gain j_resp **0.01996** fell just under the **0.02** min_gain bar, so improving actions scored insufficient and were dropped.
3. **Priority geometry.** Agent_3 has asset_value **0.00**, so operational_priority stayed 0.02–0.39; the allocator's "focus on Agent_2, priority 0.02" starved the real fire while defensive rules served the assetless city/village regions.
4. **Concept calibration / sensing.** R26/G1–G4 need fire_threat_level H/VH and suppression_feasibility VH; in the fire region threat maxed at **0.56 (M)** and feasibility was pinned at **0.33 (L)**, so the offensive channel could never legally fire. No sensor lay within ~60 cells of the footprint.
5. **Wasted orders.** water_drafting 0.83 was ordered though the lake (126,187) is ~100 cells off — dead weight, dry map.

**WHAT WOULD HAVE BEEN BETTER.** Stage a helibase inside the footprint (~30,130) for retardant sorties (crew_reachability only 0.66 → aerial, not ground). Put a camera at the SW head (~10,145) to lift fire_threat_level out of M. Add an M-level offensive rule so the channel fires before VH ever comes. Drop min_gain to 0.01 so ~0.02 gains clear the bar.

### JSON
{
 "verdict": "failure",
 "recommendations": [
  {
   "type": "setting",
   "key": "min_gain",
   "value": 0.01,
   "why": "achieved gain 0.01996 fell just under the 0.02 bar, so every improving action was rejected as insufficient"
  },
  {
   "type": "depot",
   "kind": "helibase",
   "x": 30,
   "y": 130,
   "why": "rcap stayed 0.0 all run; nearest base is ~62 cells / 522 min away, so stage aerial retardant inside the footprint"
  },
  {
   "type": "sensor",
   "kind": "ground_camera",
   "x": 10,
   "y": 145,
   "why": "no sensor within ~60 cells of the footprint; fire_threat_level pinned at M at the SW head kept offensive gates shut"
  },
  {
   "type": "rule",
   "antecedents": [
    [
     "fire_threat_level",
     ">=M"
    ]
   ],
   "consequents": [
    [
     "suppression_effort",
     0.9
    ],
    [
     "retardant_drop",
     0.9
    ],
    [
     "containment_line",
     0.7
    ]
   ],
   "why": "threat only reached 0.56 (M); existing R26/G1-G4 need H/VH so nothing fired — needs an M-level offensive trigger with dry-map retardant"
  }
 ]
}