# DisasterAware, IEEE Systems Journal, v4 (benefits of DisasterAware, thesis-based)

Date: 2026-10-05. v4 is built on `DisasterAware_IEEESystems_v2.docx` (not on v3). v3 (open intervention space with search proposers) is set aside; its related-work paragraphs and references [35]-[52] are carried over in condensed form.

## Files

| Path | What it is |
|---|---|
| `DisasterAware_IEEESystems_v4.docx` | v2 with every v4 change as a tracked change (author "Reviewer"). |
| `DisasterAware_IEEESystems_v4_clean.docx` | The same with all changes accepted. |
| `DisasterAware_IEEESystems_v4_preview.pdf` | LibreOffice rendering, 13 pages there (Word sets tighter, expect about 12). Equation numbers and the old reference block render differently in LibreOffice; that is a rendering artifact. |
| `figures/fig11_ladder_burned.png` | Fig. 11, burned area per family and configuration, 95 % CI. Replaces the v2 image behind the Fig. 11 text box. |
| `figures/fig12_curves.png` | Mean burned area against time, S2 and S4 (not in the document; available). |
| `tables/v4_outcomes.csv`, `v4_paired.csv`, `v4_fill.json` | Table V data, paired deltas, and the placeholder values written into the text. |
| `meta/v4_manuscript.md` | The v4 text with `{{...}}` placeholders; single source for the docx. |
| `code/results_v4.py`, `code/write_article_v4.py` | Also written to `03_Codes/experiments/article_v4/`. |

## What changed from v2

Abstract: open decision space named as the central contribution; the language model named as the proposer; closing claims from the paired campaign (extinction within the episode, affected population, burned area against no action, agent scaling).

I. Introduction: contribution paragraphs rewritten around paired measurement against three references (no DSS, static doctrine, self-tuning doctrine).

II. Related Work and Background: II.B replaced by B closed and open decision spaces, C mechanisms that open a space, D wildfire and disaster DSS with learned and generative components (condensed from v3); old C, D, E become E, F, G.

V.C: experimental setup synced to the thesis (72 by 50 cells at 30 m, 2 min step, 12 min cycle, 24 min horizon, 12 h cap).

V.D: four configurations T0 / F5 / F5+Ev / DisasterAware on paired worlds, outcome at episode end; Table V rebuilt (burned area per family with CI, extinction share, affected population, Jphys, Jtotal); Fig. 11 replaced; one sentence states which proposer fed stage 3 in the campaign.

V.E (was the second "D"): products paragraph kept from the recorded runs with the language-model proposer, with rule-base growth (8.8 rules) and the 52 % adaptation share added. Headings fixed to D, E, F, G.

V.G compliance, VI discussion, VII conclusion: numbers and wording synced.

## Data source of this build

`--source thesis`: `03_Codes/experiments/out_n50/ladder_runs.csv`, 1000 runs, 50 worlds per family, episodes to 12 h. The DisasterAware column of this campaign was fed by the deterministic stand-in proposer (`genai_source = template`), and the text says so. The language-model proposer ran only in the recorded 160 runs (products, no paired outcome).

Numbers written: burned area 250.8 / 84.7 / 83.0 / 76.1 ha; extinguished 0 / 10 / 46 / 52 %; affected population 5775 / 498 / 396 / 358; Jphys 0.596 / 0.160 / 0.151 / 0.140; Jtotal (0.526) / 0.193 / 0.176 / 0.168; DisasterAware minus F5+Ev 7 ± 4 ha (better in 153 of 250 paired worlds); S4 to S5 +4 %.

## Rebuilding with the language-model campaign

After `03_Codes/experiments/open_space/run_llm.bat` has produced `experiments/open_space/out_llm/runs.csv` (arm LLM) and the T0 / F5 / F5Ev rows exist in `experiments/open_space/out/runs.csv` (E2, seeds 201-212), run from `03_Codes/`:

    python experiments/article_v4/results_v4.py --source open --da-arm LLM \
        --runs experiments/open_space/out_llm/runs.csv \
        --extra experiments/open_space/out/runs.csv \
        --out experiments/article_v4/out_llm
    python experiments/article_v4/write_article_v4.py \
        --v2 ../02_Article/ieee_systems/DisasterAware_IEEESystems_v2.docx \
        --manuscript ../02_Article/ieee_systems/meta/v4_manuscript.md \
        --results experiments/article_v4/out_llm --source open \
        --out-dir ../02_Article/ieee_systems --name DisasterAware_IEEESystems_v5

The proposer sentence switches to the language-model wording automatically with `--source open`. The V.E products paragraph and the hindcast, sensitivity, timeliness, and greedy paragraphs do not depend on the campaign.

## Still open

Word page count (target 12). Cover letter and title page (still TSMC-era). Submission PDF from Word. Supplementary data: tables/ and the campaign csv.
