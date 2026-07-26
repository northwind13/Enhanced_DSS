## AFTER-ACTION REVIEW — Wildfire DSS Exercise (100 steps, 100×70 @ 30 m)

**VERDICT: partial.** The run burned 1493 cells (~21% of the 7000-cell map) and ended with the fire *not* out — 971 cells still burning at step 100, down only marginally from a peak of 1128 that grew from 170 at ignition. Against that, zero population was lost (0 evacuated, and none was actually threatened) and `asset_protection` held steady at 0.48 throughout with no asset loss reported. So the physical cost was almost entirely *burn*, not asset or population: a containment failure wrapped around a protection success.

**ROOT CAUSES, by weight.**
1. **Logistics/capacity (dominant).** `alloc rcap start 0.0` with `travel 36 min`: there was zero deployed capacity at ignition, so the fire tripled to peak before resources arrived. The helibase (2,1) and depot (6,69) both sit ~50 cells from the interior burn *and* from the water.
2. **Sensing geometry.** Every sensor is low-x or on an edge — (0,0),(47,3),(2,1),(8,69). The interior band x30–70 / y20–60, where the 1493 cells actually burned, had no eyes; late detection compounded late arrival.
3. **Doctrine under-commitment.** First-third suppression 0.16 vs last-third 0.42 — the VH rule (R26) fired only after the fire was already large.
4. **Water underuse.** 206 water cells at (52,17), yet `water_drafting`/`retardant_drop` appear only in the last third at 0.25/0.22; `drafting_sustained_attack` was learned but rarely ordered — no seed rule invokes it and the helibase isn't on the water.
5. **Wasted evacuation.** Evacuation was ordered 0.22–0.25 every cycle for a non-problem.

**WHAT WOULD HAVE BEEN BETTER.** Stage a helibase *on* the lake at (52,17) so capacity is forward and drafting is real; put a sensor in the interior blind zone near (60,40); trip strong suppression earlier by lowering `j_threshold`; add a rule that orders `water_drafting` when threat is high and feasibility isn't VL; and drop the reflexive evacuation orders.

### JSON
{
 "verdict": "partial",
 "recommendations": [
  {
   "type": "depot",
   "kind": "helibase",
   "x": 52,
   "y": 17,
   "why": "rcap started at 0.0 and travel is 36 min; a helibase on the 206-cell lake makes drafting real and cuts arrival to the interior burn"
  },
  {
   "type": "sensor",
   "kind": "aerial",
   "x": 60,
   "y": 40,
   "why": "all sensors sit in low-x corners/edges; the 1493-cell burn grew unseen in the interior x30-70/y20-60 band"
  },
  {
   "type": "setting",
   "key": "j_threshold",
   "value": 0.3,
   "why": "suppression stayed at 0.16 while the fire ran 170->1128; lower the trip point so strong orders fire before peak"
  },
  {
   "type": "setting",
   "key": "horizon_min",
   "value": 90,
   "why": "extend the 45-min forecast so the loop anticipates the interior run and pre-commits capacity 36 min out"
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
     ">=M"
    ]
   ],
   "consequents": [
    [
     "water_drafting",
     0.8
    ]
   ],
   "why": "water at (52,17) gives real sustained capacity but drafting only came late and weak; order it whenever threat is high and feasibility isn't VL"
  },
  {
   "type": "setting",
   "key": "eta",
   "value": 15,
   "why": "with resources 36 min out and none staged, a lower eta gate dispatches sooner instead of waiting for an arrival window that never opens"
  }
 ]
}