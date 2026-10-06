# Benchmark figures and tables (No DSS / F5 / F5+EvFIS / DisasterAware)

Data: `experiments/benchmark/out/runs.csv`, `curves.csv`, `proposals.jsonl`. 600 paired runs, S1-S5, 30 worlds per scenario (seeds 201-230), episodes to extinction or 12 h. The DisasterAware column is the language-model proposer (`claude -p`) behind the full gate chain. Rebuild with `python experiments\benchmark\benchmark_figures.py --out experiments\benchmark\out`.

Figures are written as PNG (300 dpi) and PDF at IEEE column widths (3.5 in single, 7.16 in double). Palette: neutral grey for the no-action reference, blue / green / orange for the three decision layers (checked for colour-vision deficiency).

## Figures (`figures/`)

| File | Width | Suggested caption |
|---|---|---|
| `figA_outcomes` | single | Outcome at episode end per scenario family and configuration, 30 paired worlds per family: (a) burned area with 95 % confidence interval, (b) affected population on a logarithmic axis, (c) share of incidents extinguished within the episode. |
| `figB_paired` | double | Paired differences in burned area on the same worlds. Points give the mean paired difference with its 95 % confidence interval; the right column gives the value and the number of worlds in which the first configuration burned less. |
| `figC_scatter` | single | Burned area per world, DisasterAware against F5+EvFIS on the same world. Points below the diagonal are worlds in which the open decision space burned less: 128 of 150, with 13 ties on small fires that the doctrine already contained. |
| `figD_curves` | double | Mean burned area against time for the four configurations, one panel per scenario family. Worlds that reach extinction keep their final area, so the curve end equals the mean in Table 1. |
| `figE_cost` | single | Decision cost J_total averaged over S1-S5, decomposed into its five terms (weights 1, 1, 1, 0.2, 0.2, normalised by 3.4). |
| `figF_gates` | single | Fate of the 788 stage-3 proposals per scenario family: admitted, or rejected at G5 (package margin), G4 (reseeded A/B forecast), G3 (no improvement) or earlier gates. |
| `figG_gain_vs_admissions` | single | Burned area saved against F5+EvFIS per world as a function of the number of stage-3 admissions in that world; bars mark the median per count. Worlds with no admission show no difference, which is the no-harm property. |

## Tables (`tables/`, .csv and .md)

| File | Content |
|---|---|
| `T1_outcome` | burned area, affected population, extinguished share, time to extinction per scenario and configuration (mean ± 95 % half-width, n = 30; Mean row n = 150) |
| `T2_cost` | J_burn, J_asset, J_pop, J_resp, J_delay, J_phys, J_total per scenario and configuration |
| `T3_paired` | five paired comparisons per scenario: burned area saved, worlds better, ties, population saved, J_phys and J_total saved, extinguished points |
| `T4_adaptation` | rules at episode end, stage 1 / 2 / 3 admissions over attempts, model calls, macros, actuators, sited orders, wall time per run |
| `T5_proposals_gates` | stage-3 proposals per scenario by verdict (admitted, G5, G4, G3, G1/G2) |
| `T6_proposals_kind` | proposals and admissions by kind (actuator package, concept package, macro package, single rule) |
| `T7_admitted_clauses` | effect x sector counts of the clauses inside admitted packages |

## Headline numbers

Mean over S1-S5 (150 worlds): burned area 250.7 / 80.9 / 75.4 / 45.6 ha; extinguished 0 / 13 / 52 / 75 %; affected population 5793 / 494 / 391 / 117; J_total 0.528 / 0.187 / 0.164 / 0.110.

DisasterAware vs F5+EvFIS: 29.9 ± 7.2 ha saved, better in 128 of 150 worlds, +23 points extinguished, 274 ± 61 fewer affected persons, J_total 0.055 ± 0.011 lower. F5+EvFIS vs F5: 5.5 ± 7.0 ha (73 of 150, not significant on area) but +39 points extinguished. Cost of the open space: 10.3 model calls and 124 s per run against 22 s for F5+EvFIS; 788 proposals, 338 admitted (43 %); macro packages admitted most often (80 %), single rules least (28 %).
