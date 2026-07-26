## After-Action Review — Wildfire DSS Exercise

**VERDICT: PARTIAL.** Life safety held; the fire did not. 13,241 of ~15,750 residents were evacuated and no-harm vetoes stayed at zero, so the evacuation doctrine (R11/R15) worked. But the fire never went out: burning cells rose from 669 to a peak of 890 and *ended* at 890 — a plateau, not a knockdown — with 1,065 cells burned and response cost j_resp pinned at 0.844. Growth was arrested; extinguishment was not achieved.

**ROOT CAUSES, ranked.**

*1 — Capacity saturation (dominant).* Allocated rcap barely moved (307.6→313.7) while every region ran flat out: Agent_1 orders pinned at 0.9+, and the global statement shows Agent_1 and Agent_3 shares at 1.00. Fully committed, no reserve — the plateau is the signature of demand exceeding capacity.

*2 — Logistics geometry.* The three region bases sit at (6,69), (2,1) and (52,17); the eastern third (Agent_3, the stated priority at 0.27) has **no depot and no helibase**, and the lake at (52,17) is a region away — hence Agent_3's water_drafting collapses to 0.07 and its effort *falls* last-third (0.34→0.25). Drafting does nothing where there is no water.

*3 — Gate/doctrine mismatch.* The generative offensive rules G1–G4 all require suppression feasibility VH; under VH-threat/VL-feasibility they never fire. Adaptation accepted just 1 of 7 proposals. Agent_3 leaves aerial retardant (needs no water, no road) almost unused at 0.06 — the one tool that works in its dry, roadless sector.

*4 — Sensing.* East of x=60 is blind but for aerial (60,40); the priority fire ran partly unobserved.

**WHAT WOULD HAVE BEEN BETTER.** Give the eastern region its own base and eyes, and let it drop retardant instead of chasing water it cannot reach.

### JSON
{
 "verdict": "partial",
 "recommendations": [
  {
   "type": "depot",
   "kind": "helibase",
   "x": 78,
   "y": 35,
   "why": "Region 3 (east, Agent_3 priority) has no base; nearest is (52,17) a region away, starving the plateaued fight and cutting the 23-min travel."
  },
  {
   "type": "sensor",
   "kind": "aerial",
   "x": 80,
   "y": 35,
   "why": "East of x=60 is blind except (60,40); the priority Agent_3 fire ran there partly unobserved."
  },
  {
   "type": "rule",
   "antecedents": [
    [
     "fire threat level",
     "VH"
    ],
    [
     "suppression feasibility",
     "L"
    ]
   ],
   "consequents": [
    [
     "retardant drop",
     0.9
    ],
    [
     "containment line",
     0.6
    ]
   ],
   "why": "Gates G1-G4 need feasibility VH and never fired; retardant is aerial (no water/road) and usable in region 3 where drafting collapsed to 0.07."
  },
  {
   "type": "setting",
   "key": "min_gain",
   "value": 0.02,
   "why": "Only 1 of 7 adaptation proposals accepted; loosen the growth margin so feasibility-appropriate offensive rules survive the gate."
  },
  {
   "type": "setting",
   "key": "horizon_min",
   "value": 60,
   "why": "Retardant-coated fuel resists ignition after drying; a 45->60 min forecast lets that persistence register as gain instead of being discounted."
  },
  {
   "type": "setting",
   "key": "eta",
   "value": 45,
   "why": "Map-mean travel is 23 min but a full region had no base; a longer response ETA better models the real reach of the saturated fleet so allocation stops over-promising."
  }
 ]
}