# Validating DisasterAware against real fire data

## 0. Fully automatic route (recommended first step)

    run_validation.bat YOURKEY

downloads everything for a documented Turkish fire (Manavgat 2021 by
default) and produces the metric report and the agreement map without any
manual GIS work: Copernicus GLO-30 DEM + ESA WorldCover fuel + real hourly
ERA5 weather + NASA FIRMS satellite detections (ignition and burned
footprint). YOURKEY is a free FIRMS MAP_KEY:
https://firms.modaps.eosdis.nasa.gov/api/map_key/
`run_validation.bat demo` runs the offline self-test of the pipeline.
For a referee-grade ground truth, replace the FIRMS footprint with the
official EFFIS/EMS perimeter: `--burned perimeter.tif` (Sec. 1).

## 1. What counts as ground truth

For wildfire spread models the accepted ground truth is a **documented
historical fire**:

1. **Final burned-area perimeter** (the primary ground truth) — the mapped
   polygon of what actually burned. Sources:
   - **EFFIS / Copernicus burnt area product** (Europe + Türkiye):
     https://forest-fire.emergency.copernicus.eu/ (Data & services > Burnt
     areas, downloadable shapefiles per event/season).
   - **Copernicus EMS Rapid Mapping** delineation products for named
     activations (e.g. the 2021 Antalya-Manavgat and Muğla-Marmaris fires):
     https://emergency.copernicus.eu/
   - **MTBS** (https://www.mtbs.gov/) for US fires, widely used in
     FARSITE/Cell2Fire validation papers.
2. **Fire progression in time** (secondary, for temporal validation) —
   VIIRS/MODIS active fire detections with timestamps from **NASA FIRMS**:
   https://firms.modaps.eosdis.nasa.gov/download/ (CSV archive; each
   detection has lat/lon, date, time). Grouping detections into 6-12 h bins
   gives observed arrival times to compare with the simulated
   `first_ignition_step` field (`t_ign` layer in the dashboard).

## 2. Input data for the hindcast

| Input | Source | Maps to |
|---|---|---|
| Elevation (DEM 30 m) | SRTM GL1 or Copernicus GLO-30 via https://portal.opentopography.org/ | `G_elev` (slope/aspect derived by Horn's method) |
| Fuel map | CORINE Land Cover 2018, https://land.copernicus.eu/ | CLC codes -> fuel classes via `validation.CORINE_TO_FUEL` |
| Weather | ERA5 hourly (10 m u/v wind, 2 m T, 2 m dewpoint) via https://cds.climate.copernicus.eu/ or MGM station records | `W_ws, W_wd` per step (`wind.csv`); T/RH -> dead fuel moisture (EMC) |
| Ignition point & start time | official incident reports; cross-check with the first FIRMS detection | `U_Ign` |
| Fire duration | incident reports (time to containment of the main run) | `--hours` |

Preprocessing (QGIS or any GIS): reproject everything to the local UTM zone,
clip to a bounding box that covers the burned area with margin, resample to
the simulation grid (30 m recommended), export as GeoTIFF. Rasterize the
observed perimeter polygon to the same grid (burned = 1). The script also
accepts plain `.npy` arrays already on the grid.

## 3. Running the hindcast

    python examples/validate_real_case.py \
        --dem dem.tif --fuel clc.tif --corine --burned observed.tif \
        --nx 300 --ny 200 --cell 30 --step-minutes 30 --hours 36 \
        --ignite 142,88 --wind-csv wind.csv --moisture 0.06 --seeds 5

- The run is **blind**: no parameter may be tuned on the case being scored.
- Ember spotting is stochastic, so the script runs several seeds and
  reports mean +/- sd.
- Outputs: `validation_report.json` (all metrics) and
  `validation_report.png` — the agreement map (green = correctly burned,
  red = simulated only / false alarm, blue = observed only / missed).

## 4. Metrics and acceptance targets

| Metric | Definition | Typical published values |
|---|---|---|
| Sorensen-Dice | 2\|AnB\|/(\|A\|+\|B\|) | 0.7-0.9 calibrated (Cell2Fire); 0.5-0.7 uncalibrated semi-empirical |
| Jaccard / IoU | \|AnB\|/\|AuB\| | > 0.5 considered good agreement |
| Hit rate (POD) | \|AnB\|/\|B\| | > 0.7 |
| False alarm ratio | \|A\\B\|/\|A\| | < 0.3 |
| Area bias | \|A\|/\|B\| | 0.8-1.2 |
| Front error mean / p90 | perimeter-to-perimeter distance | O(1-3 cells) |

## 5. Protocol that convinces reviewers

1. **Calibration/validation split.** Pick >= 3-4 documented fires. Calibrate
   the free parameters (fuel table scaling, theta_ign, alpha_s) on ONE fire
   only; freeze them; hindcast the remaining fires blind and report those
   scores. Never report the calibration fire as validation.
2. **Multi-seed uncertainty.** Report mean +/- sd over >= 5 seeds per fire.
3. **Sensitivity analysis.** +-25% on wind speed, moisture and theta_ign;
   show Dice stays in a reasonable band (model robustness).
4. **Baselines.** Compare against (a) a no-wind isotropic run and (b)
   published FARSITE/Cell2Fire scores on comparable cases, to show skill.
5. **Temporal check.** Compare simulated arrival times (t_ign layer)
   against FIRMS detection bins for at least one fire.
6. Present per-fire: the agreement map, the metric table, and the weather
   trace used. Archive the exact input rasters and the JSON reports for
   reproducibility.

## 6. Suggested Turkish case studies

- **Manavgat (Antalya), July-August 2021** — very large, well documented,
  EFFIS perimeter and Copernicus EMS delineations available, strong wind
  driven runs (good test of the anisotropy model).
- **Marmaris (Muğla), 2021** — complementary terrain (steep, pine).
- **Canakkale, August 2023** — recent, EFFIS mapped, ERA5 available.

Pipeline self-check: the repository test suite includes a closed-loop check
(simulate -> treat the result as "observed" -> re-simulate -> Dice ~ 1), so
any score obtained on real data reflects the model, not the tooling.
