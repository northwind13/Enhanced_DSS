# DisasterAware benchmark: No DSS / F5 / F5+EvFIS / DisasterAware

Run from `03_Codes\experiments\benchmark\run_benchmark.bat` (double click, or from a shell in `03_Codes`). Before the first run: `03_Codes\claude-login.bat` (the batch uses the same config directory).

    run_benchmark.bat          30 worlds per scenario, S1-S5 (default)
    run_benchmark.bat 12       12 worlds per scenario (quick, about a third of the time)
    run_benchmark.bat 30 S6    S1-S8 (adds the three families where the doctrine is structurally short)

Resumable: if it is interrupted, run it again; finished runs are skipped.

## Configurations (same worlds, same ignitions, same random stream)

| Name | Decision layer | Stage 1 and 2 (evFIS) | Stage 3 (generative) |
|---|---|---|---|
| No DSS (T0) | none, free burn | - | - |
| F5 | five seed rules, static | off | off |
| F5+EvFIS | five seed rules | on | off |
| DisasterAware | five seed rules | on | on, language-model proposer (`claude -p`), full gate chain G1-G5b, two reseeded forecasts |

## Scenarios (thesis Table 5.7; 72 x 50 cells at 30 m, 2 min physical step, 12 min decision cycle, 24 min horizon, episodes to extinction or 12 h, critical fire weather: fuel moisture 0.08, wind 8 m/s)

| ID | Landscape | Settlements | Resource pool | Ignitions | Observation |
|---|---|---|---|---|---|
| S1 | Mountain forest | 2 x 4000 | full | 1, remote | full |
| S2 | Mountain forest | 2 x 4000 | 0.6 x | 2, remote | full |
| S3 | Rolling hills, wildland-urban interface | 12 x 8000 | full | 1, near the interface | full |
| S4 | Rolling hills, wildland-urban interface | 12 x 8000 | 0.6 x | 2, near the interface | full |
| S5 | as S4 | 12 x 8000 | 0.6 x | 2, near the interface | outage 0.25, noise 0.10 |

Optional S6 lakeside (pool 0.4, lake 8-11 cells downwind), S7 fast interface fire (12 m/s, fuel moisture 0.06), S8 long incident (18 h, pool 0.4).

## Monte Carlo

Each scenario is generated 30 times from seeds 201-230 (12 times for the quick run). Every configuration replays the same 30 worlds, so every comparison is paired. Default run: 5 scenarios x 30 worlds x 4 configurations = 600 runs (3 x 150 without the model, 150 with it). The LLM runs make about 5-10 model calls each.

## What is measured (episode end)

Outcome: burned area (ha), affected population (persons), share of incidents extinguished within the episode.
Cost: J_burn, J_asset, J_pop (physical terms), J_resp, J_delay (response terms), J_phys and J_total (weights 1, 1, 1, 0.2, 0.2).
Paired statistics: for each pair of configurations, the difference on the same world, mean with 95 % half-width and the number of worlds in which the first configuration is better.
Adaptation: rules at episode end, stage-3 attempts and admissions, wall time per run.

## Outputs (`experiments\benchmark\out\report\`)

`table1_outcome` (burned, population, extinguished per scenario and configuration), `table2_cost` (seven cost terms), `table3_paired` (DisasterAware vs No DSS / F5 / F5+EvFIS, F5+EvFIS vs F5, F5 vs No DSS), `table4_adaptation`, `fig_burned.png`, `fig_curves.png`, `summary.txt`. Each table as .csv and .md. Raw data: `out\runs.csv`, `proposals.jsonl` (every LLM proposal with its gate verdict), `admitted.jsonl`, `curves.csv`.

To rebuild the tables alone: `python experiments\benchmark\benchmark_tables.py --out experiments\benchmark\out`.
