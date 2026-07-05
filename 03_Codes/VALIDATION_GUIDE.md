# Wildfire model validation guide

How to check that the DisasterAware simulator behaves correctly, in two layers:
internal (already automated) and external (against real fire data).

## 1. Internal validation (already in the test suite)

`pytest -q` runs invariant checks that any correct fire model must pass:

- fuel never goes negative and never increases; intensity stays in [0, 1]
- rate of spread increases with wind and with slope (Rothermel trend)
- at moisture above the extinction point the fire does not propagate
- zero wind gives isotropic (symmetric) spread; strong wind gives directional
- firebreaks and water stop the fire; roads are non flammable
- suppression reduces burned area; the city scenario burns structures
- deterministic: same inputs and seed reproduce the same run

These confirm the physics is self consistent. External data is still needed to
confirm the model matches reality.

## 2. External validation (against a real fire)

The idea: reconstruct a documented fire (terrain, fuel, weather, ignition point,
time), run the simulator, and compare the simulated burned area / perimeter /
arrival time to the observed one.

### 2.1 What to download

Fire perimeter or burned area (the ground truth):
- EFFIS / GWIS (Europe and Turkey): https://effis.jrc.ec.europa.eu (burnt area
  polygons, daily perimeters)
- MTBS (USA): https://www.mtbs.gov (burned area boundaries and severity)
- NIFC Open Data (USA): https://data-nifc.opendata.arcgis.com (fire perimeters)

Fire progression over time (arrival time):
- NASA FIRMS active fire (MODIS / VIIRS): https://firms.modaps.eosdis.nasa.gov
  Download the CSV of detections; each row has a location and an acquisition
  time, so you get the fire front position through time.

Terrain (DEM):
- SRTM 30 m or Copernicus DEM (GLO-30): https://portal.opentopography.org

Fuel / land cover:
- LANDFIRE fuel model rasters (USA): https://landfire.gov
- CORINE Land Cover (Europe/Turkey): https://land.copernicus.eu — map land cover
  classes to the internal fuel classes (or use the Anderson / Scott and Burgan
  mapping in `fuels_standard.py`).

Weather during the fire (wind, temperature, humidity):
- ERA5 reanalysis (hourly): https://cds.climate.copernicus.eu
- or the nearest meteorological station records.

Pick one well documented case (for example a Mediterranean fire from EFFIS, or
the 2018 Camp Fire in California via MTBS/NIFC) that has all of the above.

### 2.2 How to prepare the case

1. Clip the DEM and fuel rasters to the fire's bounding box.
2. Import them with the GIS import page (DEM plus optional fuel raster). Slope
   and aspect are derived with Horn's method; everything is resampled onto the
   simulation grid.
3. Set the wind speed and direction from the weather record; set fuel moisture
   from humidity (the EMC option), and the ignition point at the reported origin.
4. Rasterise the observed final perimeter to the same grid as a boolean burned
   mask (1 inside the perimeter).

### 2.3 How to score it

Run the simulation to the same duration as the real fire, then:

```python
from disasteraware import validation
metrics = validation.validate_run(sim, observed_burned_mask)
print(metrics)   # jaccard, dice, hit_rate, false_alarm, front position error
```

Interpretation (typical acceptance in the literature):
- Jaccard / IoU and Dice close to 1 mean the burned shapes match well; Dice
  above ~0.6 is already a good cellular-automaton result.
- hit_rate high and false_alarm low means the fire went where it really went.
- front position error (metres) small means the perimeter is in the right place.
- Also compare the burned-area-over-time curve and, using FIRMS times, the
  arrival-time map (`sim.first_ignition_step`) against observed detection times.

### 2.4 Calibration

If the match is poor, tune the calibration parameters (Parameters tab):
`theta_ign`, `w0`, `aniso_wind_full`, the fuel `r_base` and `b_base`, and the
suppression gains, or switch on the elliptical spread and spotting modes. Kose
et al. and Cell2Fire calibrate the spread coefficients by minimising the error
between simulated and observed progression; the same can be done here with the
`validation` metrics as the objective.
