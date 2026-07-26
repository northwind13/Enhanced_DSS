VERDICT: **Partial, trending to failure.** The fire was never knocked down — 836 cells burning at step 50, unchanged from peak, 986 total burned (~14% of the grid), fire out: NO, j_resp 0.67. What the run got right: 3607 residents (Village 1) evacuated, zero no-harm vetoes, no reported asset loss, and the towns on the far-west edge stayed clear. It contained the people, not the fire.

**ROOT CAUSES.** *Capacity* is dominant. 836 burning cells against 239 staged / 313 allocated is a ~3:1 deficit; Agent_1 sat pinned at 0.92 suppression and 0.83 drafting the entire run — no headroom, demand far above supply. The fire settled into a capacity-limited plateau it could not be pushed below. *Cold-start logistics* second: alloc rcap began at 0.0 with 170 cells already alight, and the ramp to 313 lagged the climb to 836; 23-min mean travel and air-only access (road_corridor None) slowed force closure. *Doctrine/gate regime mismatch* third: generative rules G1–G4 all demand suppression_feasibility VH — a bar an 800-cell fire never clears — so on the actual H/low-feasibility regime only defensive containment fired. Zero vetoes confirms the gates weren't blocking; the antecedents were. *Sensing* fourth: aerials at (47,3) and (60,40) bracket the fire but leave the eastern spread thin, so no decisive containment-failure signal drove escalation. Geometry itself was favorable — water with a helibase on it, towns far west — not a cause of loss.

**WHAT WOULD HAVE BEEN BETTER.** Front-load allocation so rcap isn't zero while 170 burn. Open the retardant/drafting playbook at H threat and L feasibility — this map has real water-sustained air capacity to spend. Stage a forward helibase east of (52,17), and put eyes on the under-sensed eastern front.

### JSON
{
 "verdict": "partial",
 "recommendations": [
  {
   "type": "setting",
   "key": "eta",
   "value": 0.5,
   "why": "rcap cold-started at 0.0 against 170 already-burning cells; faster allocation closes force before the fire plateaus at 836."
  },
  {
   "type": "setting",
   "key": "j_threshold",
   "value": 0.2,
   "why": "lower the commit bar so the DSS acts decisively during the early climb instead of settling into a capacity-limited stalemate."
  },
  {
   "type": "rule",
   "antecedents": [
    [
     "fire_threat_level",
     "H"
    ],
    [
     "suppression_feasibility",
     "L"
    ]
   ],
   "consequents": [
    [
     "containment_line",
     0.9
    ],
    [
     "retardant_drop",
     0.8
    ]
   ],
   "why": "fills the pre-VH gap G1-G4 leave; this fire lived at H/low-feasibility where only defensive rules fired."
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
     "resource_deployment",
     0.8
    ],
    [
     "retardant_drop",
     0.8
    ],
    [
     "containment_line",
     0.7
    ]
   ],
   "why": "R26 is the VH-threat workhorse but orders no retardant/line; on this water-capable map it should spend aerial capacity to cap the head."
  },
  {
   "type": "depot",
   "kind": "helibase",
   "x": 62,
   "y": 26,
   "why": "forward staging east of the (52,17) water helibase shortens the 23-min sortie Agent_1 saturated on, raising sustained drop rate on the eastern front."
  },
  {
   "type": "sensor",
   "kind": "aerial",
   "x": 78,
   "y": 38,
   "why": "coverage stops at the (60,40) aerial; the eastern spread that held the fire at 836 was under-sensed, so escalation never triggered."
  }
 ]
}