# DisasterAware, IEEE Systems Journal, v3 package

Date: 2026-10-04. v3 builds on `DisasterAware_IEEESystems_v2.docx` (clean v2) and adds the open intervention space study.

## Files

| Path | What it is |
|---|---|
| `DisasterAware_IEEESystems_v3.docx` | v2 with every v3 change as a tracked change (author "Reviewer"). Open in Word, review, accept or reject. |
| `DisasterAware_IEEESystems_v3_clean.docx` | The same with all changes accepted. |
| `DisasterAware_IEEESystems_v3_preview.pdf` | LibreOffice rendering of the clean file, for a quick look only (14 pages in LibreOffice; Word sets the same text tighter, check the count there). Equation numbers and the old reference block render differently in LibreOffice than in Word; that is a rendering artifact, not a document change. |
| `figures/fig09_rediscovery.png` | Fig. 9, cumulative rediscovery share against incident time (E1), column width, 300 dpi. |
| `figures/fig10_paired_burned.png` | Fig. 10, paired burned area, open arms minus closed reference, per family (E2). |
| `tables/E1_*.csv`, `E2_*.csv`, `E3_*.csv`, `E4_*.csv` | Outcome, paired, rediscovery, and proposer tables per experiment. Tables VI to IX in the article are built from these. |
| `data/runs.csv` | One row per run (1208 rows): experiment, family, arm, seed, margin, outcome, costs, admissions, rediscovery flags and minutes, proposer accounting, wall time. |
| `data/curves.csv` | Per-cycle curves per run. |
| `data/proposals.jsonl` | Every stage-3 attempt with proposer, kind, gate, verdict, and search accounting. |
| `data/admitted.jsonl` | Every admitted object with its clauses or composition and the minute of admission. |
| `data/summary.txt` | The text report that `report.py` prints. |
| `data/log.txt` | Campaign log. |
| `meta/v3_manuscript.md` | The v3 text with `{{...}}` placeholders; the single source for the docx. |
| `meta/OPEN_SPACE_SPEC.md` | Specification of the open intervention space (levels, grammar, gates, withholding, proposers, campaign design). |

## What changed from v2 to v3 (section by section)

Abstract: rewritten around the open intervention space; the three closing claims now carry the measured numbers (rediscovery 47 to 54 percent of incidents; 43 percent less burned area against the closed adaptive reference over eight families; two proposers admitted through the same gates at a similar rate).

I. Introduction: contribution paragraphs replaced (open space over a verified basis; rediscovery as the test of openness; end-to-end measurement on families built for the question). Two search proposers compared, language-model proposer carried through the same interface.

II. Related Work and Background: new II.B closed and open decision spaces, II.C mechanisms that open a space, II.D wildfire and disaster DSS with learned and generative components; references [35] to [52] added.

IV.F: replaced by a pointer; the staged adaptation text moved to the new Section V.

V. The Open Intervention Space (new): levels of novelty and the grammar, staged adaptation, proposers and the trial context, verification gates, learning from the simulator.

VI. Evaluation Design (new): A criteria and B hindcast kept from v2; C scenario families with Table V (S1 to S8); D arms, experiments E1 to E4, and measures.

VII. Results: A rediscovery (Table VI, Fig. 9); B closed and open space across families (Table VII, Fig. 10); C proposers and the composition prior (Table VIII); D gate margin (Table IX), timeliness, scalability; E compliance with the criteria.

VIII. Discussion and IX. Conclusion: rewritten to the measured results.

## Main results written into the article

E1 (library withheld, S6 to S8, 30 seeds): an equivalent of at least one withheld tactic was admitted in 15/16/11 of 30 runs (evolutionary) and 18/15/16 (random). Drafting was rediscovered most often (9 to 11 of 30 per cell), coating next (2 to 6), counter-fire least (1 to 4). Paired burned area against F5+Ev: evolutionary 44, 26, 45 ha less; random 41, 17, 28 ha less; open arm better in 81 of 90 pairs (evolutionary).

E2 (library available, S1 to S8, 12 seeds): F5 and F5+Ev differ by at most 12 ha on any family. Open/Evo burned less than F5+Ev on all eight families (59, 46, 15, 22, 22, 55, 29, 37 ha), pooled 43 percent less; Open/Random better on seven of eight, pooled 34 percent. Extinction within the cap 49 to 73 percent. Gain of the same size on S1 to S5 (43 percent) as on S6 to S8 (44 percent).

E3 (prior ablation, 30 seeds): the prior is used only by the evolutionary proposer; the random arm is unchanged by it and acts as a control. With the prior, the evolutionary proposer proposed 222 sited interventions (128 without), admitted 2.1 per run (1.4), spent 121 forecasts per admission (126), and burned 19 ± 22 ha less on S8 on the same seeds (better in 21 of 30 pairs), S6 and S7 within noise.

E4 (G5 margin 1e-4, 1e-3, 1e-2; 10 seeds per margin): admission rate 37, 29, 19 percent; forecasts per admission 116, 171, 312; sited interventions per run 1.9, 1.3, 0.5; runs with a rediscovery 47, 40, 17 percent; mean burned area 56, 57, 59 ha.

Timing: a proposer call with its 36 shadow forecasts cost at most 4 s of wall time on the two-core reference machine.

Not run: the language-model arm (needs an interactive model session per run). The text says so in VI.D, VII.C, and VIII.

## How to regenerate

All commands from `03_Codes/`.

    # campaign (E1, E3 at 30 seeds; E2 at 12; E4 at 10 per margin)
    bash experiments/open_space/run_all.sh          # E1, E3, then report
    bash experiments/open_space/run_rest.sh         # E2, E4, then report
    python experiments/open_space/report.py --out experiments/open_space/out
    python experiments/open_space/figures.py --out experiments/open_space/out
    python experiments/open_space/write_article_v3.py \
        --v2 ../02_Article/ieee_systems/DisasterAware_IEEESystems_v2.docx \
        --manuscript ../02_Article/ieee_systems/meta/v3_manuscript.md \
        --data experiments/open_space/out \
        --figures experiments/open_space/out/figures \
        --out-dir ../02_Article/ieee_systems

`write_article_v3.py` prints any unfilled placeholder; the current build prints none.

Optional language-model arm on a machine with the `claude` CLI on the path:

    python experiments/open_space/campaign.py --exp E1 --arms LLM --seeds 201-210 --workers 1 --out experiments/open_space/out_llm
    python experiments/open_space/report.py --out experiments/open_space/out_llm

Note on `report.py`: E4 is now split by margin (arm label `Evo@<margin>`); proposal records written before this change carry no margin field and are attributed by walking `runs.csv` and `proposals.jsonl` together. New runs write the margin into each proposal record.

## Still open before submission

Page count in Word (target 12 for a regular paper; trim VII.B or the E2 table if needed). Supplementary material: `data/` can be zipped as the ledger. Cover letter and title page in this folder are from v1 and need the new abstract.
