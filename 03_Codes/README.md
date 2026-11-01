# DisasterAware

Enhanced Decision Support System for Wildfire Disaster Response and
Management: a grid-based discrete-time hybrid fire spread engine and the
concept-driven decision support system that runs on top of it, behind a
single application.

## Running

    run_dashboard.bat        the DisasterAware application, the only entry point

First install:

    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt

## Language model for stage 3 (Claude or ChatGPT)

Stage 3 runs on a chat subscription through the vendor CLI, never on an API key.

| Provider | CLI | One-time setup | Login |
|---|---|---|---|
| `claude` | Claude Code | installed by `run_dashboard.bat` | `claude-login.bat` (Pro/Max) |
| `codex` | OpenAI Codex | Node.js LTS, then `npm i -g @openai/codex` (`codex-login.bat` does it) | `codex-login.bat` (ChatGPT Plus/Pro) |

Sessions are stored in `.claude-config` and `.codex-config` in this folder.

* Dashboard: pick the provider and model in the sidebar (GenAI ③). Layer 4 Settings adds reasoning effort, a fallback, a login check and a button that saves the choice for the batch runs.
* Batch runs: every run script calls `llm_env.bat`, which reads `llm.cfg`. Edit it with `llm-select.bat`, check it with `llm-status.bat`. Without `llm.cfg` the scripts use `claude:claude-opus-5`.
* Variables: `DSS_GENAI_PROVIDER` (claude | codex), `DSS_GENAI_MODEL`, `DSS_GENAI_EFFORT`, `DSS_GENAI_FALLBACK` (e.g. `codex:gpt-5.5`, used when the selected provider reports a usage or rate limit).
* Provenance: every proposal carries `_llm` (`provider:model`, column `llm` in `proposals.jsonl`), and strict batch runs append one line per call to `genai_calls.jsonl`, including any fallback switch. A campaign that switched model midway shows up there and should be rerun on one model.

## Folder layout

```
03_Codes/
  run_dashboard.bat        starts the application
  requirements.txt         dependencies
  MAP_RULES.md             the realism rules generate_landscape obeys

  app/                     the Streamlit interface
    streamlit_app.py         every page (Simulation, Map editor, Data
                             layers, Parameters, GIS import, Validation,
                             System Description)
    system_description.py    the System Description page: the full
                             mathematical documentation of architecture,
                             simulator, DSS and cost

  disaster_phyengine/      the physics: transition operator, Rothermel
                           spread, suppression, intensity, cost, rewind,
                           GIS import, scenarios, validation metrics
  dss/                     the decision layer: observation and confidence,
                           concept space, rule base, cost and quality
                           gates, the three adaptation stages, the
                           coordinator, the learned store, logging

  validation/              hindcast validation against recorded fires
    auto_validate.py         the core: data download, blind run, scores.
                             The Validation page loads this file directly,
                             and it also runs from the command line
    thesis_validation.py     the same pipeline over the four thesis cases,
                             assembled into one reportable table
    make_scenario_figures.py the scenario figures
    export_generated_rules.py  what the DSS wrote or changed, as a Word
                             table built from the learned store
    cache/                   downloaded real data, per case (not in git)
    runs/                    run archives: report.json, log, maps, frames
                             (not in git)

  experiments/             thesis material. Nothing here is imported by
                           the application; the dependency runs one way
    scenario.py              the fixed testbed world every study shares
    offline_proposer.py      deterministic stand-in for the generative
                             stage, so a study runs offline and twice
                             alike
    docx_track.py            tracked-changes primitives for the scripts
                             that write into the thesis
    omml.py                  Office Math builder: equations in the
                             thesis's own numbered-table style
    fill_ch4_example.py      the worked cost and quality example
    fill_ch4_attention.py    the attention threshold correction
    jth_probe.py             traces the satisficing bound over many cycles
    cost_probe.py            traces the cost terms
    plot_thresholds*.py      the threshold figures, from the probe output
    make_*_drawio.py         block diagrams as drawio documents
    out/                     the measurements the thesis numbers come from

  logs/                     what a run leaves behind, and the DSS's own
                            memory: dss_generated_state.json holds every
                            rule and concept the adaptation stages
                            produced. Not in git, and not disposable
  maps/                     saved maps
  tests/                    the pytest suite: engine and decision layer
```

## Pages

- **Simulation** runs the fire: conditions (wind compass, EMC moisture,
  step length), ignition, rewind, and the cost panel broken into its J
  terms.
- **Map editor** generates and resizes a map, and paints fuel, roads,
  assets and elevation on a 2D canvas, with 3D as preview only.
- **Data layers** shows the same fields as the System Description, plus
  operational diagnostics such as Byram intensity.
- **Parameters** exposes every parameter, with literature defaults
  (Anderson 1982, Scott and Burgan 2005, Rothermel 1972).
- **GIS import** builds a map from real DEM and fuel rasters.
- **Validation** runs an automatic hindcast against recorded fires
  (Copernicus DEM, ESA WorldCover, ERA5, NASA FIRMS), with live
  monitoring, a wind uncertainty ensemble and a run archive. The method,
  the metrics and the protocol are documented inside the page.
- **System Description** is the full mathematical documentation of the
  model, with search and a table of contents.

## Tests

    .venv\Scripts\python -m pytest tests -q
