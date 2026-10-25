# Revision runs (referee round, IEEE Systems Journal)

Open two windows and run `run_window_A.bat` (K1, K2, K3, 16 workers; `run_window_A.bat 8` for 8) and `run_window_B.bat` (K4) at the same time.
No language-model calls; everything is CPU. Resumable: finished rows are skipped.

| Step | Script | Output | Pairs with | Referee point |
|---|---|---|---|---|
| K1 | `open_space/campaign.py --exp E2P --arms F40Ev` | `final/t13_full_doctrine/runs.csv` | t1_reference F5Ev, t6_voda | weak closed reference (5 rules) |
| K2 | `k2_predefined.py --mode canon / own` | `k2_predefined/runs.csv`, `k2_doctrine.json` | t1_reference, t6_voda | is the gain discovery or just a richer doctrine |
| K3 | `k3_model_error.py` | `k3_model_error/runs.csv` | t1_reference, t10_benefit (removed=none) | same simulator plans and scores |
| K4 | `k4_hindcast.py --stop area / none` | `k4_hindcast/stop_*/hindcast_errors.csv` | Table IV | errors in metres and hours, area ratio, stopping rule |
| sum | `revision_tables.py` | `summary.md` | | |

K1 adds the arm `F40Ev` to `experiments/open_space/campaign.py` (profile `full`, same self-tuning as F5Ev).
K2 and K3 reuse the leave-one-out replay machinery (`benchmark/loo.py`, `campaign.run_once(script=...)`).
K3 patches `campaign.build_world` (perturbed live world) and `dss.evaluate.clone_sim` (every shadow
forecast restored to the nominal suppression parameters), so both arms plan on the nominal model.
K4 is `validation/thesis_validation.py` with 50 seeds, `area_bias` and `stop_hours` added to the kept
metrics, and the area-matched stop injected into `auto_validate.run_case` when `--stop area`.

Population note: `pop_affected` in every runs.csv is the sum of population DENSITY (persons/km2) over
burned cells; persons = value x 0.0009 at 30 m cells. `revision_tables.py` reports persons.
