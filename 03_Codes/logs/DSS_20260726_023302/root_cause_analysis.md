## After-Action Review — Wildfire DSS Exercise

**VERDICT: Failure.** 801 cells burned, and the fire was never contained — burning cells rose from 170 to a peak of 642 and ended at 628, with `j_resp` finishing at 0.38, above the 0.35 gate. The physical loss (burn + asset + population) was never arrested.

**ROOT CAUSES, by weight.**

*1 — Sensing and concept calibration.* `fire_severity` reached 0.60 and 801 cells burned, yet `fire_threat_level` never exceeded 0.30, `intervention_urgency` 0.38, `evacuation_pressure` 0.63 — none crossed H/VH. Every strong seed rule (R15, R26, R20, G1–G4) is gated on VH threat or urgency and therefore *never fired*. No sensor sits inside the 609-cell south-central burn; the nearest aerial is at (47,3). The fire core around centroid (20,44) was under-observed, so the FIS never escalated.

*2 — Capacity–geometry mismatch.* 609 of 801 burned cells lay in Agent_3 (logistics_support 0.25, reachability 0.58, feasibility 0.42), while effort concentrated on Agent_1 (feasibility 0.86, only 192 cells; "focus on Agent_1, priority 0.31"). The DSS resourced the region the fire had largely spared.

*3 — Doctrine drift.* Agent_1 suppression fell 0.88→0.38 as evacuation rose to 0.77 — offense disengaged while Agent_3 kept burning.

*4 — Gate/horizon.* Ten consecutive no-harm vetoes (steps 34–43), only 1 of 12 adaptations accepted; a 30-min horizon under a 45-min physics window made late orders look harmless.

Logistics note: water sits at (52,17) in fireless Agent_2 — drafting was worthless near the fire; aerial retardant was the only real reach and was underused.

**WHAT WOULD HAVE BEEN BETTER.** An aerial sensor over the (20,44) core to lift threat/urgency into firing range; a helibase inside Agent_3 so the south had capacity; an M-band suppression rule that actually fires this run; a 45-min horizon; a lower response gate.

### JSON
{
 "verdict": "failure",
 "recommendations": [
  {
   "type": "sensor",
   "kind": "aerial",
   "x": 20,
   "y": 48,
   "why": "No sensor sits in the 609-cell south-central burn near centroid (20,44); undersensing kept fire_threat_level <=0.30 so escalation rules never fired."
  },
  {
   "type": "depot",
   "kind": "helibase",
   "x": 20,
   "y": 52,
   "why": "Agent_3 held 609/801 burned cells with logistics_support 0.25 and reachability 0.58; an aerial base needs no roads and puts capacity where the fire actually was."
  },
  {
   "type": "setting",
   "key": "horizon_min",
   "value": 45,
   "why": "A 30-min forecast under a 45-min physics window drove 10 no-harm vetoes (steps 34-43); matching the horizon lets slow-building spread register."
  },
  {
   "type": "setting",
   "key": "j_threshold",
   "value": 0.25,
   "why": "Concepts never crossed H/VH while 801 cells burned; a lower response gate escalates effort before severity reaches 0.60."
  },
  {
   "type": "rule",
   "antecedents": [
    [
     "asset_exposure_risk",
     "M"
    ],
    [
     "intervention_urgency",
     ">=0.35"
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
     0.7
    ]
   ],
   "why": "Fires in this run's observed M-band (exposure to 0.75, urgency to 0.38) where the VH-gated seed rules never could, ordering strong suppression."
  }
 ]
}