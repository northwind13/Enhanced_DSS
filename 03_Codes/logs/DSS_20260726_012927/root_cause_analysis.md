# AFTER-ACTION REVIEW — Wildfire DSS Exercise (100×70 grid, 3-region, minimal seed)

## 1. VERDICT: PARTIAL

Life-safety succeeded; fire control failed, and the two outcomes are not close.

On the population axis this run is a clear win: 14,580 of the ~15,750 persons across Town residents (2,1) and Village 1 (8,69) were moved out — roughly 93% — with evacuation orders climbing through the run (Agent_1 evacuation 0.37 → 0.57) exactly as doctrine intended. No one was left in the offensive-collapse zone.

On the fire axis it is a loss dressed up as activity. Burning cells rose from 170 to a peak of 1,297 and **ended at 1,297** — the front never turned, it plateaued at its maximum and the exercise ran out of clock. Total burned reached 1,725 cells, about 24.6% of the 7,000-cell map, and the fire was **not out** at step 150. Critically, this happened with the response account fully saturated: `j_resp` ended at 1.00000 and allocated capacity ran to 229.5 against a staged 174 — a 32% surge above the planned resource base. We spent everything we had and more, and the burned area still only grew. That is the signature of a partial result: the people were saved by geometry (they were on the sensed, depot-side western edge), and the fire was lost by geometry (it burned where we could neither see it nor reach it in time).

## 2. ROOT CAUSES, ranked by weight

**First — logistics and geometry (dominant).** Every piece of hard infrastructure sits on the western margin: depot (6,69), helibase (2,1), with the road corridor unplaced (x,y = None). The map's real sustained-capacity advantage — 206 water cells centred at (52,17) — is an hour of travel from all of it, against a map-mean travel time of 36 min. The evidence line is the pairing of a *saturated* response (`j_resp` 1.00, rcap 229.5 > 174 staged) with a fire that *never declined*: the capacity existed and was committed, but it could not be delivered to the front fast enough to bend a 45-minute forecast. When maximum effort produces zero containment, the constraint is delivery, not doctrine.

**Second — sensing blindness east of the asset cluster.** The eyes are stacked on the west: satellite (0,0), two public reports at the settlements (2,1) and (8,69), and a single aerial at (47,3). Everything from mid-map east and south — including the water body and the ground where 1,725 cells actually burned — is dark. The consequence shows up in the doctrine trace: containment_line fell 0.85 → 0.62, retardant_drop 0.82 → 0.67, downwind_retardant_shield 0.60 → 0.30, while R11/R15 (fire threat VH **AND suppression feasibility VL** → withdraw offensive intensities) took over. Feasibility read "VL" not because suppression was impossible but because nothing was measuring it out there. The gates then did their job correctly on bad information and pulled the offense.

**Third — gate/adaptation behaviour starving a whole region.** Adaptation tried 39 candidates (region 1: 21, region 2: 11, region 3: 7) and accepted 10 (region 1: 8, region 3: 2). **Region 2 accepted zero of eleven.** With no-harm vetoes at 0, none of those rejections were safety kills — they failed the forecast/growth-margin gates. The generative rules G1–G4 tell the same story: born, rejected, then "restored (stage 3)." The learning loop was churning hard and converging on nothing for an entire region, which is why the last global statement is a flat "Agent_3 share 1.00, Agent_1 1.00, Agent_2 1.00" at priority 0.17 — no differentiation, no winner.

**Fourth — doctrine leaving the map's one real advantage on the table.** This is a *wet* map, and the system knew it: it invented `drafting_sustained_attack`, `wet_containment_line`, `head_knockdown`, `counterfire_strip`. Then it ordered them at 0.16, 0.15, 0.16 in the first third and let them decay to nothing by the last. Those strengths are below what moves a 45-minute physical forecast, so the invented water tactics could never earn their keep at the gates — the tactic was discovered and then never actually thrown. `downwind_backburn` (0.12 → 0.06) is the same waste.

Capacity and doctrine are downstream symptoms here; the primary faults are **where the depots are** and **where the eyes aren't**.

## 3. WHAT WOULD HAVE BEEN BETTER — this map, specifically

Put refill-capable air on the water. A helibase adjacent to the lake centre near (52,17) turns 206 water cells and the "air support: yes" line into a real drafting cycle at the front instead of an hour-away theory. Pair it with a forward ground depot near the same water so engines aren't launching from the far western corners into a 36-minute transit.

Open the eastern eye. One aerial or in-situ sensor in the dark band beyond x≈50 (around (60,35)) would have given the feasibility layer something to read, and R11/R15 would not have defaulted the whole eastern front to "VL feasibility → withdraw offense." The offense collapsed on missing data, not on genuine infeasibility.

Actually throw the water tactics. `drafting_sustained_attack` and `wet_containment_line` were the right inventions for this map and were ordered at a strength that guaranteed gate rejection. Order them hard where feasibility permits, and give sustained water tactics a longer forecast horizon to register their (inherently slower) gain — otherwise a map built for drafting fights it dry. And loosen the growth-margin just enough that region 2 stops going 0-for-11 on non-harmful candidates.

### JSON
{
 "verdict": "partial",
 "recommendations": [
  {
   "type": "depot",
   "kind": "helibase",
   "x": 52,
   "y": 17,
   "why": "All air/ground staging sat on the west edge while the fire burned 36 min away; a helibase on the water centre (206 cells) makes draft-and-refill at the front real and cuts delivery lag that left j_resp saturated yet uncontained."
  },
  {
   "type": "sensor",
   "kind": "aerial",
   "x": 60,
   "y": 35,
   "why": "The eastern two-thirds where 1,725 cells burned had one eye (aerial 47,3); feasibility read VL by default and R11/R15 withdrew offense — a forward sensor restores the feasibility signal that pulled the offense."
  },
  {
   "type": "depot",
   "kind": "depot",
   "x": 50,
   "y": 22,
   "why": "Ground crews launched from (6,69)/(2,1) into a 36-min transit; a forward engine depot beside the water body puts sustained capacity within reach of the central front instead of an hour behind it."
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
     "drafting_sustained_attack",
     0.85
    ]
   ],
   "why": "The map's water advantage was discovered then ordered at 0.16 — below forecast-moving strength; fire the learned water tactic hard where feasibility permits so the invention actually clears the gates."
  },
  {
   "type": "setting",
   "key": "horizon_min",
   "value": 90,
   "why": "Sustained water tactics build capacity slowly and never showed gain on the 45-min forecast; a longer horizon lets drafting_sustained_attack/wet_containment_line register their benefit and pass the A/B gates."
  },
  {
   "type": "setting",
   "key": "min_gain",
   "value": 0.03,
   "why": "Adaptation accepted 10 of 39 with region 2 at 0-of-11 and zero no-harm vetoes — the rejections were margin, not safety; loosening the growth margin admits the water-tactic candidates the wet map needs."
  }
 ]
}