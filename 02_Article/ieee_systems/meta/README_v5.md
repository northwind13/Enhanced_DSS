# DisasterAware, IEEE Systems Journal, v5 (language-model benchmark)

Date: 2026-10-06. v5 is built on `DisasterAware_IEEESystems_v2.docx` with every change tracked (author "Reviewer"); it supersedes v4, which carried the thesis campaign with the stand-in proposer. The structure of v4 is kept (abstract, introduction, related work II.B-D, V.C, V.D, V.E, V.G, VI, VII, references [35]-[52]); the numbers and the result prose now come from the benchmark of `03_Codes/experiments/benchmark/` (600 paired runs, S1-S5, 30 worlds per family, seeds 201-230, DisasterAware with the language-model proposer `claude -p` behind the full gate chain).

## Files

| Path | What it is |
|---|---|
| `DisasterAware_IEEESystems_v5.docx` | v2 with all v5 changes as tracked changes. |
| `DisasterAware_IEEESystems_v5_clean.docx` | The same with all changes accepted. |
| `DisasterAware_IEEESystems_v5_preview.pdf` | LibreOffice rendering, 13 pages there (Word sets tighter, expect about 12). Table VI, equation numbers and the old reference block render differently in LibreOffice; rendering artifacts. |
| `figures/fig11_ladder_burned_v5.png` | Fig. 11, burned area per family and configuration, 95 % CI. |
| `figures/fig12_gates_v5.png` | Fig. 12 (new): fate of the 788 stage-3 proposals per family. Replaces the v2 funnel figure; the frame keeps its width and takes the new height. |
| `tables/v5_outcomes.csv`, `v5_paired.csv`, `v5_fill.json` | Table V data, paired deltas, every placeholder value written into the text. |
| `meta/v5_manuscript.md` | The v5 text with `{{...}}` placeholders. |
| `03_Codes/experiments/article_v4/extra_v5.py` | Stage-1/2/3 counts, gate verdicts, admission by proposal kind, model calls, wall time; draws Fig. 12. |
| `03_Codes/experiments/benchmark/out/report/` | Seven further figures (figA-G) and seven tables (T1-T7) from the same campaign, with `FIGURES_TABLES.md` captions; candidates for the supplement. |

## What changed from v4

Abstract: proposals issued and admitted (788 / 338); extinction 75 % against 13 % (static) and 52 % (self-tuning); burned area 40 % below the self-tuning doctrine and 82 % below no action; 76 % fewer people affected than the static doctrine.

V.D: F5+Ev now described as changing the character of the response (extinction 13 to 52 %, delay term down) without a significant change in burned area (5.5 ± 7.0 ha, better in 73 of 150); DisasterAware 45.6 ha, 75 %, 117 persons, Jphys 0.081, Jtot 0.110; paired margin over F5+Ev 29.9 ± 7.2 ha, better in 128 of 150; largest gains in S2 (63.6 ha) and S4/S5 (30/30, 29/30, extinction 97 and 100 %); S5 loss 7 %; computational cost 10.3 model calls and 124 s per episode against 22 s. The proposer sentence states that the language-model proposer fed stage 3.

V.E: rewritten on the campaign itself: stage 1 kept 451 of 2093, stage 2 469 of 1411, stage 3 338 of 788 (43 %); rejections 276 at G5, 155 at G3, 12 at G4, 7 earlier; admission by kind: macro packages 80 %, actuator packages 46 %, concept packages 37 %, plain rules 28 %; dominant clauses; rule base 10.4 against 10.3 for F5+Ev; 39 sited orders, 1.6 macro and 1.2 actuator objects per episode. Fail-safe and no-harm counts kept from the recorded runs with a sentence placing them below the adaptation loop. New Fig. 12 and caption.

V.G: decision-quality sentence adds the margin over the self-tuning doctrine; adaptability counts updated; the limitation "noise and communication failure untested" corrected to "tested at one level only" (S5 has outage 0.25 and noise 0.10).

VI: "four proposals in five rejected" corrected to "more than half, most at the package margin"; limitations gain one sentence on the proposer setting the size of the margin, citing the separate 50-world campaign with the template writer (76 ha against 83 ha for F5+Ev).

VII: numbers synced; one clause on admitted proposals.

## Numbers written

Mean over S1-S5: burned area 250.7 / 80.9 / 75.4 / 45.6 ha; extinguished 0 / 13 / 52 / 75 %; affected population 5793 / 494 / 391 / 117; Jphys 0.598 / 0.155 / 0.139 / 0.081; Jtotal (0.528) / 0.187 / 0.164 / 0.110; Jresp 0.501 (F5) / 0.428 (F5+Ev) / 0.308 (DA); Jdelay 0.347 / 0.274 / 0.340.

## Rebuild

    cd 03_Codes
    python experiments/article_v4/results_v4.py --source open --da-arm LLM --runs experiments/benchmark/out/runs.csv --curves experiments/benchmark/out/curves.csv --out experiments/article_v4/out_llm
    python experiments/article_v4/extra_v5.py --bench experiments/benchmark/out --results experiments/article_v4/out_llm
    python experiments/article_v4/write_article_v4.py --v2 ../02_Article/ieee_systems/DisasterAware_IEEESystems_v2.docx --manuscript ../02_Article/ieee_systems/meta/v5_manuscript.md --results experiments/article_v4/out_llm --source open --out-dir ../02_Article/ieee_systems --name DisasterAware_IEEESystems_v5

## Still open

Word page count (target 12). Cover letter and title page (still TSMC-era). Submission PDF from Word. Whether to cite the thesis for the template-writer campaign sentence in VI. Supplement: figA-G and T1-T7 from the benchmark report.
