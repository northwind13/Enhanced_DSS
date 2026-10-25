# Revision runs against the article campaign

Reference: closed layer F5Ev, 150 worlds, mean burned 75.4 ha, extinguished 52%, affected 391 persons.

## K1 Strong closed reference (whole written doctrine, 40 rules)

| Configuration | Burned (ha) | vs closed layer | better/tie/worse | Extinguished | Affected (persons) |
|---|---|---|---|---|---|
| No decision support | 250.7 | +232 ± 14% | 0/0/150 | 0% | 5793 |
| Closed layer, 5 seed rules (article) | 75.4 | +0 ± 0% | 0/150/0 | 52% | 391 |
| Closed layer, 40 rules (K1) | 66.9 | -11 ± 8% | 92/15/43 | 55% | 295 |
| VODA (article) | 29.1 | -61 ± 14% | 131/10/9 | 91% | 79 |

VODA against the 40-rule closed layer: -56 ± 14%, better in 130 of 150 worlds.

## K2 Discovered tactics given to the closed layer in advance

| Configuration | Burned (ha) | vs closed layer | better/tie/worse | Extinguished | Affected (persons) |
|---|---|---|---|---|---|
| Closed + 4 canonical tactics at minute 0 (K2 canon) | 34.0 | -55 ± 10% | 136/12/2 | 82% | 3 |
| Closed + own admitted objects at minute 0 (K2 own) | 25.7 | -66 ± 14% | 133/15/2 | 87% | 80 |
| VODA (article) | 29.1 | -61 ± 14% | 131/10/9 | 91% | 79 |

### K2b Tactics derived from seeds 201 to 215, frozen, evaluated on seeds 216 to 230 (unseen worlds)

| Configuration | Burned (ha) | vs closed layer | better/tie/worse | Extinguished | Affected (persons) |
|---|---|---|---|---|---|
| Closed layer (same 75 worlds) | 73.2 | +0 ± 0% | 0/75/0 | 52% | 367 |
| Closed + frozen tactics from the other 75 worlds (K2b) | 36.3 | -50 ± 15% | 68/6/1 | 83% | 76 |
| VODA (same 75 worlds) | 33.7 | -54 ± 18% | 66/4/5 | 91% | 104 |

VODA against the frozen library on unseen worlds: -7 ± 27%, better in 35 of 75 worlds.

VODA against closed + canonical tactics: -14 ± 20%, better in 59 of 150 worlds.

VODA against closed + own objects: +13 ± 11%, better in 20 of 150 worlds.

## K3 Planning on the nominal model, evaluated under model error

| Perturbation of the live world | Closed (ha) | VODA replay (ha) | VODA vs closed | better/tie/worse | Extinguished closed / VODA |
|---|---|---|---|---|---|
| none (nominal, article) | 75.4 | 33.8 | -55 ± 14% | 126/15/9 | 52% / 87% |
| gammaI_x1.5 | 75.8 | 33.0 | -56 ± 13% | 127/15/8 | 51% / 87% |
| betat_x1.5 | 91.1 | 41.5 | -54 ± 12% | 127/15/8 | 46% / 81% |
| knock_x1.5 | 89.5 | 38.2 | -57 ± 12% | 133/13/4 | 49% / 86% |

## K6 Coating weaker in the live world, planning on the nominal coating

| Live-world coating | Closed (ha) | Replay (ha) | Replay vs closed | b/t/w | Live VODA (ha) | Live vs closed | b/t/w | Coated ha closed / replay / live |
|---|---|---|---|---|---|---|---|---|
| nominal (article) | 75.4 | 33.8 | -55 ± 14% | 126/15/9 | 29.1 | -61 ± 14% | 131/10/9 | 0.0 / 42.2 / 42.8 |
| coat_half | 75.4 | 37.5 | -50 ± 13% | 129/15/6 | 40.7 | -46 ± 12% | 123/10/17 | 0.0 / 46.6 / 48.6 |
| coat_off | 75.4 | 52.0 | -31 ± 10% | 114/19/17 | 47.8 | -37 ± 11% | 125/13/12 | 0.0 / 56.7 / 59.3 |
| coat_price_x2 | 75.4 | 33.8 | -55 ± 14% | 126/15/9 | 34.5 | -54 ± 13% | 131/11/8 | 0.0 / 42.2 / 47.2 |
| coat_stock_20ha | 75.4 | 41.3 | -45 ± 13% | 120/17/13 | 40.9 | -46 ± 11% | 131/9/10 | 0.0 / 11.8 / 10.3 |
| coat_delay_12min | 75.4 | 38.4 | -49 ± 13% | 126/15/9 | 37.5 | -50 ± 12% | 131/10/9 | 0.0 / 46.7 / 49.9 |

## K7 Wind different in the live world, planning on the nominal wind

| Live-world wind | Closed (ha) | Replay (ha) | Replay vs closed | b/t/w | Live VODA (ha) | Live vs closed | b/t/w | Ext. closed / replay / live |
|---|---|---|---|---|---|---|---|---|

## K8 Gates on independent futures (seeds 303, 404), VODA run anew

no K8 runs yet (0 rows)

## K4 Hindcast, 50 replicates, stated stopping rule

Protocol window12, stop rule area

| Fire | Wind offset | Footprint (h) | Reached in (h) | POD | Front mean / p90 (m) | Arrival order rho | Arrival MAE (h) | Area sim/obs |
|---|---|---|---|---|---|---|---|---|
| Sc1 | +0 | 12 | 1.5 | 0.921 ± 0.002 | 315 ± 12 / 654 | 0.86 | 1.45 ± 0.02 | 1.37 |
| Sc2 | +0 | 12 | 2.0 | 0.866 ± 0.0 | 345 ± 0 / 849 | -0.22 | 1.03 ± 0.0 | 1.0 |
| Sc3 | +0 | 12 | 2.0 | 0.977 ± 0.0 | 109 ± 10 / 204 | 0.99 | 4.04 ± 0.0 | 1.24 |
| Sc4 | +0 | 12 | 7.0 | 0.822 ± 0.0 | 477 ± 0 / 900 | 1.0 | 1.54 ± 0.0 | 1.03 |

protocol thesis, stop rule area: no run yet
protocol clean, stop rule area: no run yet
protocol clean, stop rule none: no run yet